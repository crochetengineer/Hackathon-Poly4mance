"""
Bayesian fit (MAP + adaptive Metropolis) for one System B candidate.

Imported by predict.py (run_bayesian). Can also be run alone, after the
classical fit for that model exists in scratch/:
    python3 baseline/bayesian_system_b.py M2

Parameterisation: theta = (ln k, Ea). Likelihood is pipeline_common.log_likelihood
(Gaussian for quantified values, Phi((LOQ - mu)/sigma) for censored Q).
Prior: uniform Ea on EA_BOUNDS; for k either log-uniform (default) or uniform
on K_BOUNDS, set by BAYES_SETTINGS["k_prior"].

MAP: L-BFGS-B on -log posterior, started from the best classical multistart
solutions (scratch/system_B_<M>_weighted_censored_all_starts.csv).

Laplace: numerical Hessian at the MAP gives a Gaussian approximation of the
posterior, used to start the walkers and as their initial proposal covariance.

MCMC: independent random-walk Metropolis walkers, each run as its own job so
walkers can use separate CPU cores (predict.py runs every walker of both models
in one process pool). Every walker does a fixed number of steps (so results
do not depend on machine speed): the first adapt_steps adapt its proposal
(covariance from its own history, scale tuned towards 23 % acceptance), then
the proposal is frozen and the remaining steps are kept. A time cap stops a
walker early only on a machine too slow for the budget; that is reported. Longer chains per walker, rather than more walkers, are what
bring split-R-hat down. A walker that does not finish adapting, or keeps too
few steps, is rejected. Only if the remaining walkers pass the convergence check
(max split-R-hat < RHAT_OK) are their draws written as posterior draws;
otherwise they are saved separately as unconverged and predictions fall back to
draws from the Laplace approximation, explicitly labelled as such.

Outputs (scratch/):
    system_B_<M>_bayesian_posterior_draws.csv   (only if converged; read by predict.py, suggest.py)
    system_B_<M>_bayesian_unconverged_draws.csv (not converged: for inspection only)
    system_B_<M>_bayesian_laplace_draws.csv     (not converged: approximate fallback for predictions)
    system_B_<M>_bayesian_parameter_summary.csv
    system_B_<M>_bayesian_rhat.csv
    system_B_<M>_bayesian_map_starts.csv
    system_B_<M>_bayesian_run_summary.csv
    system_B_<M>_bayesian_trace_plots.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import pipeline_common as pc  # noqa: E402
from pipeline_common import CANDIDATES  # noqa: E402


class Posterior:
    def __init__(self, model, settings):
        self.model = model
        self.settings = settings
        self.names = list(CANDIDATES[model].param_names)
        self.is_k = np.array([pc.is_rate(n) for n in self.names])
        self.lower = np.where(self.is_k, np.log(pc.K_BOUNDS[0]), pc.EA_BOUNDS[0])
        self.upper = np.where(self.is_k, np.log(pc.K_BOUNDS[1]), pc.EA_BOUNDS[1])
        self.step = np.where(self.is_k, settings["log_k_step"], settings["ea_step"])
        self.runs = pc.get_runs()
        self.n_evals = 0

    def to_params(self, theta):
        theta = np.asarray(theta, dtype=float)
        values = np.where(self.is_k, np.exp(theta), theta)
        return {n: float(v) for n, v in zip(self.names, values)}

    def from_params(self, params):
        values = np.array([float(params[n]) for n in self.names])
        return np.where(self.is_k, np.log(values), values)

    def to_physical(self, chains):
        out = np.array(chains, dtype=float, copy=True)
        out[..., self.is_k] = np.exp(out[..., self.is_k])
        return out

    def in_bounds(self, theta):
        return bool(np.all(theta >= self.lower) and np.all(theta <= self.upper))

    def log_prior(self, theta):
        if not self.in_bounds(theta):
            return -np.inf
        if self.settings["k_prior"] == "uniform":
            return float(np.sum(theta[self.is_k]))  # Jacobian of k = exp(theta)
        return 0.0

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        self.n_evals += 1
        ll = pc.log_likelihood(self.model, self.runs, self.to_params(theta))
        return lp + ll if np.isfinite(ll) else -np.inf


# =============================================================================
# MAP
# =============================================================================

def map_start_points(post, n_starts):
    starts = []
    path = pc.all_starts_file(post.model)
    if path.exists():
        table = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
        key = "neg2_log_likelihood" if "neg2_log_likelihood" in table.columns else "weighted_deviance"
        table = table[table[key].notna()].sort_values(key)
        cols = [f"{n}_fitted" for n in post.names]
        if all(c in table.columns for c in cols):
            for _, row in table.dropna(subset=cols).iterrows():
                starts.append(post.from_params({n: row[f"{n}_fitted"] for n in post.names}))
                if len(starts) >= n_starts:
                    break
    best_file = pc.best_parameters_file(post.model)
    if best_file.exists():
        baseline = post.from_params(pc.load_best_parameters(post.model))
        if not any(np.allclose(baseline, s) for s in starts):
            starts.insert(0, baseline)
    if not starts:
        raise FileNotFoundError(
            f"No classical fit found for {post.model}; run the multistart fit first."
        )
    margin = 1e-6 * (post.upper - post.lower)
    return [np.clip(s, post.lower + margin, post.upper - margin) for s in starts]


def find_map(post):
    s = post.settings
    starts = map_start_points(post, s["n_map_starts"])
    box = pc.TimeBox(s["map_time_budget_s"])
    best_theta, best_lp, records = None, -np.inf, []

    for number, theta0 in enumerate(starts, start=1):
        start_lp = post(theta0)
        tracker = {"theta": theta0.copy(), "lp": start_lp}
        if start_lp > best_lp:
            best_theta, best_lp = theta0.copy(), start_lp

        if box.expired:
            records.append({"start_number": number, "optimiser_success": False,
                            "message": "skipped (time budget)", "log_posterior": start_lp})
            continue

        # Optimise in step-scaled coordinates so ln k and Ea are comparably scaled.
        def objective(u):
            box.check()
            theta = u * post.step
            value = post(theta)
            if value > tracker["lp"]:
                tracker.update(theta=theta.copy(), lp=value)
            return -value if np.isfinite(value) else 1e30

        bounds = list(zip(post.lower / post.step, post.upper / post.step))
        try:
            res = minimize(objective, theta0 / post.step, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": s["map_maxiter"]})
            success, message = bool(res.success), str(res.message)
        except pc.BudgetExceeded:
            success, message = False, "time budget reached"

        theta_fit, lp_fit = tracker["theta"], tracker["lp"]
        record = {"start_number": number, "optimiser_success": success,
                  "message": message, "log_posterior": lp_fit}
        for n, v0, v in zip(post.names, post.to_params(theta0).values(),
                            post.to_params(theta_fit).values()):
            record[f"{n}_start"], record[f"{n}_fitted"] = v0, v
        records.append(record)
        print(f"  [bayes] {post.model} MAP start {number}/{len(starts)} | "
              f"log posterior = {lp_fit:.4f} | success = {success}", flush=True)

        if lp_fit > best_lp:
            best_theta, best_lp = theta_fit.copy(), lp_fit

    if best_theta is None or not np.isfinite(best_lp):
        raise RuntimeError(f"MAP search failed for {post.model}.")
    return best_theta, best_lp, pd.DataFrame(records)


# =============================================================================
# Laplace approximation (warm start for MCMC)
# =============================================================================

def laplace_covariance(post, theta_map, map_lp):
    """
    Inverse of the numerical Hessian of -log posterior at the MAP, in theta
    space. Costs about 2d^2 posterior evaluations. Returns None if the Hessian
    is not usable (non-finite, not positive definite, or MAP on a bound).
    """
    d = len(theta_map)
    h = 0.1 * post.step

    def f(delta):
        value = post(theta_map + delta)
        return -value if np.isfinite(value) else np.nan

    f0 = -map_lp
    hess = np.empty((d, d))
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h[i]
        hess[i, i] = (f(ei) - 2.0 * f0 + f(-ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(d)
            ej[j] = h[j]
            hess[i, j] = hess[j, i] = (f(ei + ej) - f(ei - ej) - f(-ei + ej) + f(-ei - ej)) / (4 * h[i] * h[j])
    if not np.all(np.isfinite(hess)):
        return None
    try:
        cov = np.linalg.inv(hess)
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return None
    # Guard against absurdly wide directions (flat/unidentified parameters).
    max_sd = 0.25 * (post.upper - post.lower)
    sd = np.sqrt(np.diag(cov))
    scale = np.minimum(1.0, max_sd / sd)
    return cov * np.outer(scale, scale)


# =============================================================================
# MCMC, organised as prepare -> independent walker jobs -> finalize
# =============================================================================

RHAT_OK = 1.1


def prepare(model, settings=None):
    """MAP + Laplace. Returns a small picklable dict used to build walker jobs."""
    model = str(model).upper()
    settings = {**pc.BAYES_SETTINGS, **(settings or {})}
    post = Posterior(model, settings)
    pc.banner(f"BAYESIAN FIT: {model}  (k prior: {settings['k_prior']})", "-")

    theta_map, map_lp, map_table = find_map(post)
    print(f"  [bayes] {model} MAP log posterior = {map_lp:.4f}", flush=True)

    d = len(theta_map)
    laplace = laplace_covariance(post, theta_map, map_lp)
    if laplace is not None:
        proposal = (2.38 ** 2 / d) * laplace
        init_chol = np.linalg.cholesky(laplace)
        print(f"  [bayes] {model} Laplace sd: "
              + ", ".join(f"{n}={v:.3g}" for n, v in zip(post.names, np.sqrt(np.diag(laplace)))),
              flush=True)
    else:
        proposal = np.diag(post.step ** 2)
        init_chol = np.diag(settings["init_scale"] * post.step)
        print(f"  [bayes] {model} Hessian unusable (bound or flat direction); default proposal",
              flush=True)

    return {"model": model, "settings": settings, "names": post.names,
            "theta_map": theta_map, "map_lp": float(map_lp), "map_table": map_table,
            "proposal": proposal, "init_chol": init_chol, "laplace_ok": laplace is not None,
            "laplace_cov": laplace}


def walker_jobs(preps, wall_budget_s, n_workers):
    """
    One job per walker across all prepared models: a fixed number of walkers per
    model, each doing a fixed number of steps, so results do not depend on the
    machine. Each walker also gets a time cap, set so the whole batch stops
    within about wall_budget_s even on a slow machine (the run report flags it).
    """
    preps = [p for p in preps if p is not None]
    if not preps:
        return []
    per_model = preps[0]["settings"]["walkers_per_model"]
    n_jobs = per_model * len(preps)
    cap_s = wall_budget_s * min(n_workers, n_jobs) / n_jobs
    jobs = []
    for m, prep in enumerate(preps):
        rng = np.random.default_rng(prep["settings"]["seed"] + 1000 * m)
        d = len(prep["theta_map"])
        for w in range(per_model):
            start = prep["theta_map"] + prep["init_chol"] @ rng.standard_normal(d)
            jobs.append({"model": prep["model"], "settings": prep["settings"], "walker": w,
                         "start": start, "theta_map": prep["theta_map"],
                         "proposal": prep["proposal"], "seed": int(rng.integers(2 ** 31)),
                         "time_cap_s": cap_s})
    return jobs


def run_walker(job):
    """One adaptive-then-frozen Metropolis chain. Top-level so worker processes can run it."""
    s = job["settings"]
    post = Posterior(job["model"], s)
    rng = np.random.default_rng(job["seed"])
    d = len(job["start"])

    pos = np.array(job["start"], dtype=float)
    lp = post(pos) if post.in_bounds(pos) else -np.inf
    if not np.isfinite(lp):
        pos = np.array(job["theta_map"], dtype=float)
        lp = post(pos)

    cov = np.array(job["proposal"], dtype=float)
    chol = np.linalg.cholesky(cov)
    log_scale = 0.0
    box = pc.TimeBox(job["time_cap_s"])
    planned, adapt_steps = s["steps_per_walker"], s["adapt_steps"]

    chain, lps = [], []
    burn = None
    accepted_kept = 0
    step = 0
    while step < planned and not box.expired:
        proposal = pos + np.exp(log_scale) * (chol @ rng.standard_normal(d))
        accepted = False
        if post.in_bounds(proposal):
            value = post(proposal)
            if np.log(rng.uniform()) < value - lp:
                pos, lp, accepted = proposal, value, True
        chain.append(pos.copy())
        lps.append(lp)
        step += 1

        if burn is None:
            # Robbins-Monro: nudge the proposal scale towards 23 % acceptance.
            log_scale += (float(accepted) - 0.234) / max(step, 10) ** 0.6
            if step >= 20 * d and step % 25 == 0:
                window = np.asarray(chain[step // 2:])
                emp = np.cov(window, rowvar=False)
                candidate = (2.38 ** 2 / d) * emp + 1e-6 * np.diag(np.diag(cov) + 1e-12)
                try:
                    chol = np.linalg.cholesky(candidate)
                    cov = candidate
                    log_scale *= 0.5  # covariance now carries the scale
                except np.linalg.LinAlgError:
                    pass
            if step >= adapt_steps:
                burn = step
        else:
            accepted_kept += int(accepted)

    # A walker only contributes samples if it finished adapting (so its proposal
    # was frozen and the kept steps form a valid Markov chain) and kept enough
    # steps afterwards. Otherwise it is rejected, never salvaged.
    if burn is None:
        status = "rejected: adaptation not finished"
    elif len(chain) - burn < s["min_kept_steps"]:
        status = "rejected: too few kept steps"
    else:
        status = "ok"
    ok = status == "ok"
    kept = len(chain) - burn if ok else 0
    return {"model": job["model"], "walker": job["walker"], "status": status,
            "planned_steps": planned, "capped": step < planned,
            "chain": np.asarray(chain[burn:]) if ok else np.empty((0, d)),
            "lps": np.asarray(lps[burn:]) if ok else np.empty(0),
            "burn_steps": burn if burn is not None else len(chain), "total_steps": len(chain),
            "acceptance": accepted_kept / max(kept, 1), "n_evals": post.n_evals,
            "sim_stats": pc.sim_stats_line()}


def split_rhat(chains):
    n_walkers, n_steps, n_par = chains.shape
    half = n_steps // 2
    if half < 2:
        return np.full(n_par, np.nan)
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    n = split.shape[1]
    means = split.mean(axis=1)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    var_plus = (n - 1) / n * within + between / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_plus / within)


def save_trace_plots(model, names, chains_physical, path):
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.0 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for j, (axis, name) in enumerate(zip(axes, names)):
        for w in range(chains_physical.shape[0]):
            axis.plot(chains_physical[w, :, j], lw=0.6, alpha=0.6)
        axis.set_ylabel(name)
        if name.startswith("k"):
            axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Kept MCMC iteration")
    fig.suptitle(f"{model} posterior traces", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def laplace_draws(prep, n, rng):
    """Draws from the Laplace approximation N(MAP, H^-1), restricted to the prior bounds."""
    post = Posterior(prep["model"], prep["settings"])
    chol = np.linalg.cholesky(prep["laplace_cov"])
    d = len(prep["theta_map"])
    out = []
    for _ in range(50):
        z = prep["theta_map"] + (chol @ rng.standard_normal((d, 2 * n))).T
        inside = np.all((z >= post.lower) & (z <= post.upper), axis=1)
        out.extend(z[inside])
        if len(out) >= n:
            break
    if not out:
        return None
    return post.to_physical(np.asarray(out[:n]))


def finalize(prep, walker_results):
    """
    Combine one model's walkers and decide which draws predictions may use:
      converged MCMC   -> posterior draws file              (source "posterior")
      not converged    -> MCMC draws saved separately as "unconverged" (not used),
                          plus Laplace-approximation draws  (source "laplace")
      no usable Laplace either                              (source "none")
    """
    model, names, s = prep["model"], prep["names"], prep["settings"]
    post = Posterior(model, s)
    prefix = pc.SCRATCH_DIR / f"system_B_{model}_bayesian"
    results = [r for r in walker_results if r is not None]
    ok = sorted([r for r in results if r["status"] == "ok"], key=lambda r: r["walker"])
    rejected = {}
    for r in results:
        if r["status"] != "ok":
            rejected[r["status"]] = rejected.get(r["status"], 0) + 1
    for status, count in rejected.items():
        print(f"  [bayes] {model}: {count} walker(s) {status}; their samples are discarded",
              flush=True)

    length, max_rhat, rhat = 0, np.nan, np.full(len(names), np.nan)
    summary = pd.DataFrame({"parameter": names})
    if len(ok) >= 2:
        length = min(len(r["chain"]) for r in ok)
        kept = np.stack([r["chain"][-length:] for r in ok])            # (W, L, d), theta space
        kept_lps = np.stack([r["lps"][-length:] for r in ok])
        rhat = split_rhat(kept)
        max_rhat = float(np.max(rhat)) if np.all(np.isfinite(rhat)) else np.nan
        physical = post.to_physical(kept)
        flat = physical.reshape(-1, physical.shape[-1])
        flat_lp = kept_lps.reshape(-1)
        thin = max(1, len(flat) // s["max_stored_draws"])
        draws = pd.DataFrame(flat[::thin], columns=names)
        draws["log_posterior"] = flat_lp[::thin]
        summary = pd.DataFrame([{
            "parameter": n,
            "map": post.to_params(prep["theta_map"])[n],
            "posterior_mean": float(np.mean(flat[:, j])),
            "posterior_median": float(np.median(flat[:, j])),
            "posterior_sd": float(np.std(flat[:, j], ddof=1)),
            "posterior_q05": float(np.quantile(flat[:, j], 0.05)),
            "posterior_q95": float(np.quantile(flat[:, j], 0.95)),
            "split_rhat": float(rhat[j]),
        } for j, n in enumerate(names)])
        save_trace_plots(model, names, physical, f"{prefix}_trace_plots.png")
    # Converged only if EVERY parameter's R-hat is finite and below the threshold
    # (an undefined R-hat, e.g. a parameter stuck on a bound, fails the gate).
    converged = bool(len(ok) >= 2 and np.all(np.isfinite(rhat)) and np.all(rhat < RHAT_OK))

    if converged:
        draws.to_csv(pc.posterior_draws_file(model), index=False)
        source = "posterior"
    else:
        if len(ok) >= 2:
            draws.to_csv(pc.unconverged_draws_file(model), index=False)
        fallback = (laplace_draws(prep, s["laplace_draws"], np.random.default_rng(s["seed"]))
                    if prep.get("laplace_cov") is not None else None)
        if fallback is not None:
            pd.DataFrame(fallback, columns=names).to_csv(pc.laplace_draws_file(model), index=False)
            source = "laplace"
        else:
            source = "none"

    summary.to_csv(f"{prefix}_parameter_summary.csv", index=False)
    pd.DataFrame({"parameter": names, "split_rhat": rhat}).to_csv(f"{prefix}_rhat.csv", index=False)
    prep["map_table"].to_csv(f"{prefix}_map_starts.csv", index=False)

    run_summary = {
        "model": model,
        "k_prior": s["k_prior"],
        "map_log_posterior": prep["map_lp"],
        "laplace_warm_start": prep["laplace_ok"],
        "walkers_run": len(results),
        "walkers_used": len(ok),
        "walkers_rejected": len(results) - len(ok),
        "kept_steps_per_walker": int(length),
        "walkers_capped_by_time": int(sum(r.get("capped", False) for r in results)),
        "mean_acceptance_rate": float(np.mean([r["acceptance"] for r in ok])) if ok else np.nan,
        "max_split_rhat": max_rhat,
        "converged": converged,
        "prediction_draws": source,
        "log_posterior_evaluations": int(sum(r["n_evals"] for r in results)),
    }
    pd.DataFrame([run_summary]).to_csv(f"{prefix}_run_summary.csv", index=False)

    rhat_text = f"{max_rhat:.3f}" if np.isfinite(max_rhat) else "undefined for some parameter"
    print(f"  [bayes] {model}: {len(ok)}/{len(results)} walkers used x {length} kept steps | "
          f"max split R-hat {rhat_text} | {'converged' if converged else 'NOT converged'}",
          flush=True)
    if len(ok) >= 2:
        print(summary.to_string(index=False), flush=True)
    if not converged:
        what = {"laplace": "Laplace-approximation draws (approximate posterior)",
                "none": "the classical fit plus measurement noise only"}[source]
        kept_note = ("its draws are saved as unconverged and NOT used" if len(ok) >= 2
                     else "fewer than two usable walkers, so no MCMC draws were kept")
        print(f"  [bayes] WARNING {model}: MCMC not converged (R-hat threshold {RHAT_OK}); "
              f"{kept_note}. Predictions use {what}. "
              "Raise steps_per_walker (and the mcmc_time_budget_s cap) in "
              "pipeline_common.BAYES_SETTINGS.", flush=True)
    return {"model": model, "summary": summary, "run_summary": run_summary}


def run_bayesian(model_name, settings=None):
    """Standalone: prepare, run this model's walkers in parallel, finalize."""
    prep = prepare(model_name, settings)
    jobs = walker_jobs([prep], prep["settings"]["mcmc_time_budget_s"], pc.N_WORKERS)
    results = pc.map_parallel(run_walker, jobs, "bayes")
    return finalize(prep, results)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 baseline/bayesian_system_b.py M2")
        sys.exit(1)
    run_bayesian(sys.argv[1])


if __name__ == "__main__":
    main()