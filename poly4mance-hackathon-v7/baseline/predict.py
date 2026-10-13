"""
just predict  ->  python3 baseline/predict.py

Stage 1  classical weighted + censored multistart fit of every candidate
         (fit_system_b_weighted_multistart.run_fit)
Stage 2  AICc ranking on -2 log L, Akaike weights over all six candidates,
         write submission/hypotheses.yaml
Stage 3  Bayesian MAP + Laplace, then parallel MCMC walkers for the top N models
         (bayesian_system_b.prepare / run_walker / finalize)
Stage 4  model-averaged posterior-predictive point predictions and 90 %
         intervals for data/heldout_conditions.csv -> submission/predictions.csv

Optional hand-written justifications: baseline/hypothesis_notes.yaml
    M4: "Mechanistic note that will be appended to M4's justification."
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import pipeline_common as pc
from pipeline_common import CANDIDATES, Conditions, simulate
from pipeline_common import SUBMISSION, ensure_submission_dir
from fit_system_b_weighted_multistart import run_fit
import bayesian_system_b as bayes

RUN_REPORT = []   # filled by each stage; written by pc.write_run_report at the end
HYPOTHESES_FILE = Path(SUBMISSION) / "hypotheses.yaml"
PREDICTIONS_FILE = Path(SUBMISSION) / "predictions.csv"
NOTES_FILE = pc.BASELINE_DIR / "hypothesis_notes.yaml"

MECHANISMS = {
    "M1": "A + B -> P ; A + B -> Q (parallel impurity from reactants)",
    "M2": "A + B -> I ; I -> P ; I -> Q (intermediate branches to P or Q)",
    "M3": "A + B -> P ; P -> Q (impurity from product degradation)",
    "M4": "A + B -> I ; I -> P ; I + A -> Q (second-order impurity route consuming A)",
    "M5": "A + B -> P ; A + P -> Q (product reacts with A)",
    "M6": "A -> I ; I + B -> P ; I -> Q (A activates first, then branches)",
}


# =============================================================================
# Stage 1: classical fits
# =============================================================================

def _fit_worker(model):
    """Runs in a worker process; must be a top-level function."""
    try:
        result = run_fit(model)
        result.pop("all_results", None)
        result["sim_stats"] = pc.sim_stats_line()
        return result
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def _bayes_prepare_worker(model):
    """MAP + Laplace for one model, in a worker process."""
    try:
        return bayes.prepare(model)
    except Exception as error:  # noqa: BLE001
        return {"model": model, "error": f"{type(error).__name__}: {error}"}


def fit_all_models():
    pc.banner("[predict] STAGE 1: CLASSICAL FITS FOR ALL CANDIDATE MODELS")
    fitted = {}
    results = pc.run_parallel(_fit_worker, pc.ALL_MODELS, "predict")
    for model in pc.ALL_MODELS:
        result = results[model] or {"error": "worker returned nothing"}
        if "error" in result:
            print(f"[predict] WARNING: {model} fit failed ({result['error']}); it gets zero weight.",
                  flush=True)
            continue
        if not result["success"]:
            print(f"[predict] note: {model} best start did not report convergence "
                  f"({result['message']}); using best point found.", flush=True)
        fitted[model] = result
        print(f"[predict] {model}: -2logL = {result['neg2_log_likelihood']:.3f}, "
              f"weighted deviance = {result['weighted_deviance']:.3f} | {result['sim_stats']}",
              flush=True)
    if not fitted:
        raise RuntimeError("No candidate model could be fitted.")
    RUN_REPORT.append({"stage": "classical fit starts (all models)",
                       "planned": sum(f["starts_planned"] for f in fitted.values()),
                       "completed": sum(f["starts_completed"] for f in fitted.values()),
                       "capped": any(f["capped"] for f in fitted.values())})
    return fitted


# =============================================================================
# Stage 2: ranking and hypotheses.yaml
# =============================================================================

def add_information_criteria(fitted):
    for info in fitted.values():
        k, n = info["n_parameters"], info["n_observations"]
        neg2ll = info["neg2_log_likelihood"]
        info["aic"] = neg2ll + 2.0 * k
        info["aicc"] = info["aic"] + (2.0 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else np.inf
        info["bic"] = neg2ll + k * np.log(n)
    best = min(info["aicc"] for info in fitted.values())
    raw = {m: np.exp(-0.5 * (info["aicc"] - best)) for m, info in fitted.items()}
    total = sum(raw.values())
    for m, info in fitted.items():
        info["delta_aicc"] = info["aicc"] - best
        info["akaike_weight"] = raw[m] / total
    return fitted


def evidence_label(delta):
    if delta < 2:
        return "statistically indistinguishable from the best model on these data (dAICc < 2)"
    if delta < 4:
        return "substantial support (2 <= dAICc < 4)"
    if delta < 10:
        return "weak support (4 <= dAICc < 10)"
    return "essentially no support (dAICc >= 10)"


def load_notes():
    if NOTES_FILE.exists():
        with NOTES_FILE.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def write_hypotheses(fitted, n_keep):
    pc.banner("[predict] STAGE 2: MODEL RANKING")
    fitted = add_information_criteria(fitted)
    ranked = sorted(fitted, key=lambda m: (fitted[m]["aicc"], m))
    selected = ranked[:n_keep]

    sel_raw = np.array([fitted[m]["akaike_weight"] for m in selected])
    prediction_weights = dict(zip(selected, (sel_raw / sel_raw.sum()).tolist()))
    notes = load_notes()
    near_best = [m for m in ranked if fitted[m]["delta_aicc"] < 2.0]

    entries = []
    for rank, model in enumerate(ranked, start=1):
        info = fitted[model]
        text = (f"dAICc = {info['delta_aicc']:.2f}, Akaike weight = {info['akaike_weight']:.3f}; "
                f"{evidence_label(info['delta_aicc'])}.")
        if info["at_bounds"]:
            text += f" Parameters at a bound: {', '.join(info['at_bounds'])} (weakly identified or reaction switched off)."
        if model in notes:
            text += " " + str(notes[model]).strip()
        entries.append({
            "rank": rank,
            "name": model,
            "mechanism": MECHANISMS.get(model, ""),
            "weight": round(float(info["akaike_weight"]), 6),
            "justification": text,
            "neg2_log_likelihood": float(info["neg2_log_likelihood"]),
            "weighted_deviance": float(info["weighted_deviance"]),
            "n_parameters": int(info["n_parameters"]),
            "n_observations": int(info["n_observations"]),
            "aic": float(info["aic"]),
            "aicc": float(info["aicc"]),
            "bic": float(info["bic"]),
            "delta_aicc": float(info["delta_aicc"]),
            "parameters": {k: float(v) for k, v in info["parameters"].items()},
            "parameters_at_bounds": list(info["at_bounds"]),
            "selected_for_bayesian": model in selected,
            "selected_weight": float(prediction_weights.get(model, 0.0)),
            "supported_weight": float(prediction_weights.get(model, 0.0)),
        })

    # Models that failed to fit still appear, ranked last with zero weight.
    for model in pc.ALL_MODELS:
        if model not in fitted:
            entries.append({"rank": len(entries) + 1, "name": model,
                            "mechanism": MECHANISMS.get(model, ""), "weight": 0.0,
                            "justification": "Fit failed; no weight assigned.",
                            "selected_for_bayesian": False, "selected_weight": 0.0,
                            "supported_weight": 0.0})

    payload = {
        "hypothesis_set": "System B (MT-419 coupling) candidate mechanisms",
        "weights_are": "Akaike (AICc) weights over all candidates, from -2 log L of the "
                       "weighted, censored-data maximum-likelihood fits",
        "best_model_by_aicc": ranked[0],
        "indistinguishable_from_best": near_best,
        "selected_models": selected,
        "method": {
            "classical_fit": "maximum likelihood: heteroscedastic Gaussian + left-censored Q, multistart",
            "ranking_metric": "AICc on -2 log L",
            "bayesian_stage": f"MAP + adaptive Metropolis on top {n_keep} models",
            "prediction_weights": "Akaike weights renormalised over the retained models",
        },
        "all_models": entries,
    }
    with HYPOTHESES_FILE.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    for entry in entries:
        marker = "  <-- Bayesian + prediction" if entry["selected_for_bayesian"] else ""
        print(f"  {entry['rank']}. {entry['name']}: weight = {entry['weight']:.4f}, "
              f"dAICc = {entry.get('delta_aicc', float('nan')):.3f}{marker}", flush=True)
    print(f"[predict] wrote {HYPOTHESES_FILE}", flush=True)
    return selected, prediction_weights


# =============================================================================
# Stage 3: Bayesian fits
# =============================================================================

def run_bayesian_stage(selected):
    """
    1. MAP + Laplace for each retained model (in parallel).
    2. Every walker of every model as its own job in one process pool, so all
       CPU cores sample at once and each walker gets a long chain.
    3. Combine each model's walkers: R-hat, summaries, draws file.
    """
    pc.banner("[predict] STAGE 3: BAYESIAN FITS FOR RETAINED MODELS")
    # Remove draws from any previous run so a failed fit can't silently reuse them.
    for model in selected:
        for stale in (pc.posterior_draws_file(model), pc.unconverged_draws_file(model),
                      pc.laplace_draws_file(model)):
            if stale.exists():
                stale.unlink()

    preps = {}
    for model, prep in pc.run_parallel(_bayes_prepare_worker, selected, "predict").items():
        if not prep or "error" in prep:
            reason = prep["error"] if prep else "worker returned nothing"
            print(f"[predict] WARNING: Bayesian MAP failed for {model} ({reason}); "
                  "predictions for it fall back to the classical fit + noise.", flush=True)
        else:
            preps[model] = prep

    jobs = bayes.walker_jobs(list(preps.values()), pc.BAYES_SETTINGS["mcmc_time_budget_s"],
                             pc.N_WORKERS)
    if jobs:
        per_model = len(jobs) // len(preps)
        steps = jobs[0]["settings"]["steps_per_walker"]
        print(f"[predict] MCMC: {per_model} walkers per model x {steps} steps "
              f"({jobs[0]['settings']['adapt_steps']} adaptation), {len(jobs)} jobs on "
              f"{min(pc.N_WORKERS, len(jobs))} cores, time cap {jobs[0]['time_cap_s']:.0f} s per walker",
              flush=True)
    walker_results = pc.map_parallel(bayes.run_walker, jobs, "predict")
    done = [r for r in walker_results if r is not None]
    RUN_REPORT.append({"stage": "MCMC steps (all walkers)",
                       "planned": sum(j["settings"]["steps_per_walker"] for j in jobs),
                       "completed": sum(r["total_steps"] for r in done),
                       "capped": any(r.get("capped") for r in done) or len(done) < len(jobs)})

    completed = []
    for model, prep in preps.items():
        mine = [r for r in walker_results if r is not None and r["model"] == model]
        try:
            summary = bayes.finalize(prep, mine)["run_summary"]
        except Exception as error:  # noqa: BLE001
            print(f"[predict] WARNING: Bayesian fit failed for {model} ({error}); "
                  "predictions for it fall back to the classical fit + noise.", flush=True)
            continue
        print(f"[predict] {model}: {summary['walkers_used']}/{summary['walkers_run']} walkers x "
              f"{summary['kept_steps_per_walker']} steps, R-hat {summary['max_split_rhat']:.3f} "
              f"({'converged' if summary['converged'] else 'NOT converged'}); "
              f"predictions will use: {summary['prediction_draws']}", flush=True)
        completed.append(model)
    return completed


# =============================================================================
# Stage 4: held-out predictions
# =============================================================================

def simulate_heldout(model, run_df, params, times):
    first = run_df.iloc[0]
    cond = Conditions(temperature_c=float(first["temperature_C"]), A0=float(first["A0_M"]),
                      B0=float(first["B0_M"]), times_min=[float(t) for t in times])
    sim = pc.safe_simulate(model, cond, params)
    return np.maximum(sim[pc.SPECIES].to_numpy(dtype=float), 0.0)  # (n_times, 4)


def mean_profiles(model, run_df, times, draws, fallback_params, box):
    # (sets mean_profiles.capped = True if its time cap cut the draws short)
    """Noise-free model predictions, shape (n_draws, n_times, 4)."""
    names = list(CANDIDATES[model].param_names)
    profiles = []
    if draws is not None and not draws.empty:
        for _, row in draws.iterrows():
            if box.expired and len(profiles) >= 20:
                mean_profiles.capped = True
                break
            try:
                out = simulate_heldout(model, run_df, {n: float(row[n]) for n in names}, times)
            except Exception:
                continue
            if np.all(np.isfinite(out)):
                profiles.append(out)
    if profiles:
        return np.asarray(profiles), True
    return simulate_heldout(model, run_df, fallback_params, times)[None, ...], False


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, v))


DETAILED_PREDICTIONS_FILE = pc.SCRATCH_DIR / "system_B_predictions_detailed.csv"
PREDICTION_SOURCES_FILE = pc.SCRATCH_DIR / "system_B_prediction_sources.csv"


def build_predictions(fitted, selected, weights, settings):
    """
    For every held-out condition, species and time:
      * underlying concentration mu: simulated from parameter draws, >= 0 by physics;
      * measurement y ~ N(mu, sigma(mu)^2): the observation model, NOT clipped,
        so its interval keeps the specified Gaussian shape (it can dip below 0);
      * models mixed by their renormalised Akaike weights.
    prediction = weighted mean of mu (the expected measurement equals the
    expected concentration). lower90/upper90 = 5th/95th percentiles of y
    (interval_target "measurement", default) or of mu ("concentration").
    Both interval types go to scratch/system_B_predictions_detailed.csv.
    """
    pc.banner("[predict] STAGE 4: HELD-OUT POSTERIOR PREDICTIONS")
    heldout = pd.read_csv(pc.find_heldout_file())
    rng = np.random.default_rng(settings["seed"])
    target = settings["interval_target"]
    if target not in ("measurement", "concentration"):
        raise ValueError(f"interval_target must be 'measurement' or 'concentration', not {target!r}")

    draws_by_model, sources = {}, []
    labels = {"posterior": "converged posterior draws",
              "laplace": "Laplace-approximation draws (MCMC not converged; approximate)",
              "none": "classical fit + measurement noise only (no parameter uncertainty; "
                      "intervals too narrow)"}
    for model in selected:
        path, source = pc.prediction_draws_file(model)
        if path is not None:
            draws = pd.read_csv(path)
            take = min(settings["posterior_draws_per_model"], len(draws))
            draws_by_model[model] = draws.sample(n=take, random_state=settings["seed"])
        else:
            draws_by_model[model], take = None, 0
        sources.append({"model": model, "weight": weights[model], "draw_source": source,
                        "draws_used": take})
        print(f"[predict] {model} (weight {weights[model]:.3f}): {labels[source]}"
              + (f", {take} draws" if take else ""), flush=True)
    pd.DataFrame(sources).to_csv(PREDICTION_SOURCES_FILE, index=False)

    groups = list(heldout.groupby("run_id", sort=True))
    per_job_s = settings["time_budget_s"] / max(len(groups) * len(selected), 1)
    mean_profiles.capped = False

    rows, detailed = [], []
    for run_id, run_df in groups:
        times = np.sort(run_df["time_min"].astype(float).unique())
        first = run_df.iloc[0]
        print(f"[predict] {run_id}: T = {float(first['temperature_C']):.1f} C, "
              f"A0 = {float(first['A0_M']):.3f} M, B0 = {float(first['B0_M']):.3f} M", flush=True)

        mus, ys, ws = [], [], []
        for model in selected:
            mu, from_draws = mean_profiles(model, run_df, times, draws_by_model[model],
                                           fitted[model]["parameters"], pc.TimeBox(per_job_s))
            if not from_draws:
                mu = np.repeat(mu, settings["noise_draws_without_posterior"], axis=0)
            mu = np.maximum(mu, 0.0)                      # concentrations are non-negative
            y = np.empty_like(mu)
            for j, sp in enumerate(pc.SPECIES):           # measurements: unclipped Gaussian
                y[..., j] = rng.normal(mu[..., j], pc.sigma(sp, mu[..., j]))
            mus.append(mu)
            ys.append(y)
            ws.append(np.full(len(mu), weights[model] / len(mu)))

        mu_all, y_all = np.concatenate(mus), np.concatenate(ys)
        w = np.concatenate(ws)
        w = w / w.sum()

        for i, t in enumerate(times):
            for j, sp in enumerate(pc.SPECIES):
                m, y = mu_all[:, i, j], y_all[:, i, j]
                point = float(np.sum(w * m))
                c_lo, c_hi = weighted_quantile(m, w, 0.05), weighted_quantile(m, w, 0.95)
                y_lo, y_hi = weighted_quantile(y, w, 0.05), weighted_quantile(y, w, 0.95)
                lower, upper = (y_lo, y_hi) if target == "measurement" else (c_lo, c_hi)
                upper = max(upper, lower + 1e-12)
                rows.append({"run_id": run_id, "time_min": round(float(t), 4), "species": sp,
                             "prediction": f"{point:.6g}", "lower90": f"{lower:.6g}",
                             "upper90": f"{upper:.6g}"})
                detailed.append({"run_id": run_id, "time_min": float(t), "species": sp,
                                 "prediction": point,
                                 "concentration_lower90": c_lo, "concentration_upper90": c_hi,
                                 "measurement_lower90": y_lo, "measurement_upper90": y_hi,
                                 "prob_measurement_below_loq":
                                     float(np.sum(w * (y < pc.Q_LOQ))) if sp == "Q" else np.nan})

    pd.DataFrame(detailed).to_csv(DETAILED_PREDICTIONS_FILE, index=False)
    planned = sum(s_["draws_used"] for s_ in sources) * len(groups)
    RUN_REPORT.append({"stage": "posterior-predictive simulations", "planned": planned,
                       "completed": "all" if not mean_profiles.capped else "partial",
                       "capped": bool(mean_profiles.capped)})
    build_predictions.sources = sources
    print(f"[predict] intervals in predictions.csv describe: {target} "
          f"(both types in {DETAILED_PREDICTIONS_FILE.name})", flush=True)
    return pd.DataFrame(rows, columns=["run_id", "time_min", "species",
                                       "prediction", "lower90", "upper90"])


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_submission_dir()
    settings = pc.PREDICT_SETTINGS
    box = pc.TimeBox(1e9)
    pc.banner("# SYSTEM B PREDICTION PIPELINE" + ("  [FAST MODE]" if pc.FAST else ""), "#")
    # Fail before any fitting if an input is missing.
    heldout_file = pc.find_heldout_file()
    if not pc.SYSTEM_B_FILE.is_file():
        raise FileNotFoundError(f"System B measurements not found at {pc.SYSTEM_B_FILE}.")
    print(f"[predict] held-out conditions: {heldout_file.relative_to(pc.PROJECT_ROOT)}", flush=True)
    stage_times = {}

    t = pc.TimeBox(1e9)
    fitted = fit_all_models()
    stage_times["1 classical fits"] = t.elapsed

    t = pc.TimeBox(1e9)
    selected, weights = write_hypotheses(fitted, settings["n_models_to_keep"])
    stage_times["2 ranking"] = t.elapsed

    t = pc.TimeBox(1e9)
    run_bayesian_stage(selected)
    stage_times["3 Bayesian"] = t.elapsed

    t = pc.TimeBox(1e9)
    predictions = build_predictions(fitted, selected, weights, settings)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    stage_times["4 predictions"] = t.elapsed

    pc.banner("[predict] PIPELINE COMPLETE", "#")
    for stage, seconds in stage_times.items():
        print(f"[predict] stage {stage:<18s} {seconds:7.1f} s", flush=True)
    seconds_by_stage = [stage_times["1 classical fits"], stage_times["3 Bayesian"],
                        stage_times["4 predictions"]]
    for row, sec in zip(RUN_REPORT, seconds_by_stage):
        row["seconds"] = sec
    pc.write_run_report("predict", RUN_REPORT)

    print("[predict] UNCERTAINTY SOURCE PER MODEL (see scratch/system_B_prediction_sources.csv):",
          flush=True)
    for src in build_predictions.sources:
        meaning = {"posterior": "converged MCMC posterior",
                   "laplace": "Laplace approximation (MCMC did not converge)",
                   "none": "NO parameter uncertainty - intervals are too narrow"}[src["draw_source"]]
        print(f"  {src['model']}: {src['draw_source']:<9s} {meaning}", flush=True)
    if any(src["draw_source"] == "none" for src in build_predictions.sources):
        print("[predict] WARNING: at least one model has no parameter uncertainty in its "
              "predictions. Check its Bayesian stage before submitting.", flush=True)
    print(f"[predict] {len(predictions)} predictions -> {PREDICTIONS_FILE}", flush=True)
    print(f"[predict] hypotheses -> {HYPOTHESES_FILE}", flush=True)
    print(f"[predict] wall time {box.elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()