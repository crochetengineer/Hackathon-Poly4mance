"""
Maximum-likelihood multistart fit for one System B candidate.

Imported by predict.py (run_fit). Can also be run alone:
    python3 baseline/fit_system_b_weighted_multistart.py M2

Objective: the exact negative log-likelihood of the observation model,
written as a least-squares problem so the trust-region solver can be used.
For every quantified point two residuals are returned,

    z = (mu - y) / sigma(mu)              and    sqrt(2 log(sigma(mu) / a_s)),

and for every censored Q one residual, sqrt(-2 log Phi((LOQ - mu) / sigma)).
Because sigma >= a_s (the absolute error), the second term is always real, and

    sum of squares = -2 log L + constant,

so minimising it is maximum likelihood. (Plain weighted least squares drops
the log sigma term, which biases fits towards larger predicted concentrations
when the noise is heteroscedastic.) The optimiser works in (log10 k, Ea/100).

Outputs (scratch/), consumed by bayesian_system_b.py, bootstrap_system_b.py
and replicate.py:
    system_B_<M>_weighted_censored_all_starts.csv
    system_B_<M>_weighted_censored_best_parameters.csv
    system_B_<M>_weighted_censored_summary.csv
    system_B_<M>_weighted_censored_profiles.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.special import log_ndtr  # noqa: E402

import pipeline_common as pc  # noqa: E402
from pipeline_common import CANDIDATES, Conditions, simulate  # noqa: E402


LARGE_RESIDUAL = 1e3


# =============================================================================
# Residuals
# =============================================================================

def residual_parts(model, params, runs):
    """
    Returns (z, log_sigma_terms, censored_terms):
      z                standardised residuals of quantified points
      log_sigma_terms  sqrt(2 log(sigma / a_s)) for the same points
      censored_terms   sqrt(-2 log Phi((LOQ - mu)/sigma)) for censored points
    """
    observed = set(CANDIDATES[model].observed)
    sims = pc.simulate_runs(model, runs, params)
    z_parts, ls_parts, c_parts = [], [], []
    for run_id, run in runs.items():
        sim = sims[run_id]
        for name, block in run["blocks"].items():
            if name not in observed:
                continue
            mu = pc.predicted_at(sim, name, block)
            sig = pc.sigma(name, mu)
            cens = block["censored"]
            if cens.any():
                logp = np.maximum(log_ndtr((pc.Q_LOQ - mu[cens]) / sig[cens]), -1e6)
                c_parts.append(np.sqrt(-2.0 * logp))
            unc = ~cens
            if unc.any():
                z_parts.append((mu[unc] - block["values"][unc]) / sig[unc])
                absolute = pc.NOISE_MODEL[name][1]
                ls_parts.append(np.sqrt(2.0 * np.maximum(np.log(sig[unc] / absolute), 0.0)))
    parts = [np.concatenate(p) if p else np.zeros(0) for p in (z_parts, ls_parts, c_parts)]
    if not all(np.all(np.isfinite(p)) for p in parts):
        raise FloatingPointError("non-finite residuals")
    return parts


def n_residuals(model, runs):
    observed = set(CANDIDATES[model].observed)
    total = 0
    for run in runs.values():
        for name, block in run["blocks"].items():
            if name in observed:
                n_cens = int(block["censored"].sum())
                total += 2 * (len(block["censored"]) - n_cens) + n_cens
    return total


def residual_vector(model, params, runs):
    """Least-squares residuals whose sum of squares is -2 log L + constant."""
    return np.concatenate(residual_parts(model, params, runs))


def safe_residuals(model, params, runs, n_resid):
    try:
        return residual_vector(model, params, runs)
    except Exception:
        return np.full(n_resid, LARGE_RESIDUAL)


def weighted_deviance(model, params, runs):
    """The old weighted least-squares deviance (sum z^2 + censored terms), for reference."""
    try:
        z, _, c = residual_parts(model, params, runs)
        return float(z @ z + c @ c)
    except Exception:
        return np.inf


# =============================================================================
# Multistart
# =============================================================================

def make_starts(model, n_starts, rng):
    names = CANDIDATES[model].param_names
    baseline = np.array([-2.0 if pc.is_rate(n) else 0.55 for n in names])
    starts = [baseline]
    for _ in range(n_starts - 1):
        starts.append(
            np.array(
                [
                    rng.uniform(np.log10(1e-5), np.log10(0.3)) if pc.is_rate(n)
                    else rng.uniform(0.15, 1.10)
                    for n in names
                ]
            )
        )
    return starts


def bound_flags(z, lo, hi, tol=1e-3):
    span = hi - lo
    return (z - lo) < tol * span, (hi - z) < tol * span


def fit_multistart(model, settings, runs):
    names = CANDIDATES[model].param_names
    lo, hi = pc.z_bounds(model)
    rng = np.random.default_rng(settings["seed"])
    starts = make_starts(model, settings["n_starts"], rng)
    n_resid = n_residuals(model, runs)
    box = pc.TimeBox(settings["time_budget_s"])

    records = []
    best = None
    capped = False

    for start_number, z0 in enumerate(starts, start=1):
        if box.expired and best is not None:
            print(f"  [fit] {model}: time budget reached after {start_number - 1} starts", flush=True)
            capped = True
            break

        tracker = {"cost": np.inf, "z": None}

        def fun(z):
            box.check()
            r = safe_residuals(model, pc.from_z(model, z), runs, n_resid)
            cost = float(r @ r)
            if cost < tracker["cost"]:
                tracker.update(cost=cost, z=np.array(z, copy=True))
            return r

        z0 = np.clip(z0, lo + 1e-9, hi - 1e-9)
        try:
            res = least_squares(fun, z0, bounds=(lo, hi), method="trf",
                                max_nfev=settings["max_nfev"], x_scale=1.0)
            z_fit, success, message, nfev = res.x, bool(res.success), res.message, int(res.nfev)
            if tracker["z"] is not None and tracker["cost"] < float(res.fun @ res.fun):
                z_fit = tracker["z"]
        except pc.BudgetExceeded:
            z_fit, success, message, nfev = tracker["z"], False, "time budget reached", np.nan
            capped = True
        except Exception as error:  # noqa: BLE001
            z_fit, success, message, nfev = tracker["z"], False, f"{type(error).__name__}: {error}", np.nan

        if z_fit is None:
            records.append({"start_number": start_number, "optimiser_success": False,
                            "message": message, "n_function_evaluations": nfev,
                            "neg2_log_likelihood": np.inf, "weighted_deviance": np.inf})
            print(f"  [fit] {model} start {start_number:02d} | FAILED | {message}", flush=True)
            continue

        fit_params = pc.from_z(model, z_fit)
        neg2ll = -2.0 * pc.log_likelihood(model, runs, fit_params)
        deviance = weighted_deviance(model, fit_params, runs)
        at_lo, at_hi = bound_flags(z_fit, lo, hi)
        start_params = pc.from_z(model, z0)

        record = {"start_number": start_number, "optimiser_success": success,
                  "message": message, "n_function_evaluations": nfev,
                  "neg2_log_likelihood": neg2ll, "weighted_deviance": deviance}
        for i, n in enumerate(names):
            record[f"{n}_start"] = start_params[n]
            record[f"{n}_fitted"] = fit_params[n]
            record[f"{n}_at_lower_bound"] = bool(at_lo[i])
            record[f"{n}_at_upper_bound"] = bool(at_hi[i])
        records.append(record)

        print(f"  [fit] {model} start {start_number:02d} | -2logL = {neg2ll:12.4f} "
              f"| success = {success}", flush=True)

        if np.isfinite(neg2ll) and (best is None or neg2ll < best["neg2ll"]):
            best = {"z": z_fit, "neg2ll": neg2ll, "deviance": deviance,
                    "start_number": start_number, "success": success,
                    "message": message, "nfev": nfev}

    if best is None:
        raise RuntimeError(f"All optimisation starts failed for {model}.")

    best["capped"] = capped
    best["starts_completed"] = len(records) - (1 if capped and records and
                                               records[-1]["message"] == "time budget reached" else 0)
    best["starts_planned"] = len(starts)
    return best, pd.DataFrame(records), lo, hi


# =============================================================================
# Outputs
# =============================================================================

def save_plot(model, params):
    df = pc.load_system_b_data()
    run_ids = sorted(df["run_id"].unique())
    fig, axes = plt.subplots(len(run_ids), 1, figsize=(10, 3.6 * len(run_ids)))
    axes = np.atleast_1d(axes)
    colours = {"A": "tab:blue", "B": "tab:orange", "P": "tab:green", "Q": "tab:red"}

    for axis, run_id in zip(axes, run_ids):
        run_df = df[df["run_id"] == run_id]
        first = run_df.iloc[0]
        dense = np.linspace(0.0, float(run_df["time_min"].max()), 300).tolist()
        try:
            sim = pc.safe_simulate(model, Conditions(float(first["temperature_C"]), float(first["A0_M"]),
                                             float(first["B0_M"]), dense), params, timeout=30.0)
        except Exception:
            sim = None
        for sp in pc.SPECIES:
            sd = run_df[run_df["species"] == sp]
            unc, cen = sd[~sd["censored"]], sd[sd["censored"]]
            if not unc.empty:
                axis.plot(unc["time_min"], unc["value"], "o", color=colours[sp],
                          mfc="white", ms=4, label=f"{sp} measured")
            if not cen.empty:
                axis.plot(cen["time_min"], np.full(len(cen), pc.Q_LOQ), "v",
                          color=colours[sp], mfc="white", ms=5, label="Q < LOQ")
            if sim is not None:
                axis.plot(sim["time_min"], sim[sp], "-", color=colours[sp], lw=1.5,
                          label=f"{sp} fitted")
        axis.axhline(pc.Q_LOQ, color="tab:red", ls=":", lw=1.0, alpha=0.7)
        axis.set_title(f"{run_id}: {float(first['temperature_C']):.0f} °C, "
                       f"A0 = {float(first['A0_M']):.2f} M, B0 = {float(first['B0_M']):.2f} M")
        axis.set_xlabel("Time (min)")
        axis.set_ylabel("Concentration (mol/L)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=3)

    fig.suptitle(f"System B: weighted, censored-data fit — {model}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = pc.SCRATCH_DIR / f"system_B_{model}_weighted_censored_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_fit(model_name, settings=None, make_plot=True):
    """
    Fit one model, save scratch outputs, and return a result dictionary.
    Safe to import and call from predict.py.
    """
    model = str(model_name).upper()
    if model not in CANDIDATES:
        raise ValueError(f"Unknown model {model}; choose from {', '.join(CANDIDATES)}")
    settings = {**pc.FIT_SETTINGS, **(settings or {})}
    runs = pc.get_runs()
    names = CANDIDATES[model].param_names

    best, all_results, lo, hi = fit_multistart(model, settings, runs)
    params = pc.from_z(model, best["z"])
    at_lo, at_hi = bound_flags(best["z"], lo, hi)
    log_lik = pc.log_likelihood(model, runs, params)

    all_results.sort_values("neg2_log_likelihood", na_position="last").reset_index(drop=True) \
        .to_csv(pc.all_starts_file(model), index=False)

    pd.DataFrame([{"parameter": n, "value": params[n], "at_lower_bound": bool(at_lo[i]),
                   "at_upper_bound": bool(at_hi[i])} for i, n in enumerate(names)]) \
        .to_csv(pc.best_parameters_file(model), index=False)

    df = pc.load_system_b_data()
    pd.DataFrame([{
        "model": model,
        "n_starts_run": len(all_results),
        "best_start_number": best["start_number"],
        "best_weighted_deviance": best["deviance"],
        "neg2_log_likelihood": -2.0 * log_lik,
        "n_parameters": len(names),
        "n_observations_used": pc.n_observations(model, runs),
        "n_measurements_total": len(df),
        "n_censored_measurements": int(df["censored"].sum()),
        "best_optimiser_success": best["success"],
        "best_message": best["message"],
    }]).to_csv(pc.SCRATCH_DIR / f"system_B_{model}_weighted_censored_summary.csv", index=False)

    plot_file = save_plot(model, params) if make_plot else None

    return {
        "model": model,
        "parameters": params,
        "weighted_deviance": float(best["deviance"]),
        "neg2_log_likelihood": float(-2.0 * log_lik),
        "n_parameters": len(names),
        "n_observations": pc.n_observations(model, runs),
        "success": bool(best["success"]),
        "message": best["message"],
        "best_start_number": int(best["start_number"]),
        "at_bounds": [n for i, n in enumerate(names) if at_lo[i] or at_hi[i]],
        "capped": bool(best["capped"]),
        "starts_completed": int(best["starts_completed"]),
        "starts_planned": int(best["starts_planned"]),
        "all_results": all_results,
        "plot_file": plot_file,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 baseline/fit_system_b_weighted_multistart.py M2")
        sys.exit(1)
    out = run_fit(sys.argv[1])
    pc.banner(f"SYSTEM B WEIGHTED + CENSORED RESULTS: {out['model']}")
    print(f"Weighted deviance: {out['weighted_deviance']:.6f}")
    print(f"-2 log L:          {out['neg2_log_likelihood']:.6f}")
    print(f"Optimiser success: {out['success']} ({out['message']})")
    for name, value in out["parameters"].items():
        flag = "  [AT BOUND]" if name in out["at_bounds"] else ""
        print(f"  {name:>4s} = {value:.6g}{flag}")


if __name__ == "__main__":
    main()