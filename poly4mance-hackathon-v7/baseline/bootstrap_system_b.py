"""
Parametric bootstrap of the retained System B models.

Imported by suggest.py (run_bootstrap). Can also be run alone, after
`just predict` has written the classical fits:
    python3 baseline/bootstrap_system_b.py M2 M4

For each model:
  1. take its classical best fit (scratch/system_B_<M>_weighted_censored_best_parameters.csv),
  2. simulate the historical sampling schedule (same runs, same times, Q only
     where Q was assayed), add noise from the shared noise model and re-apply
     Q censoring at the LOQ,
  3. refit by maximum likelihood (L-BFGS-B in log10 k / Ea/100 space),
  4. keep the refitted parameters as one bootstrap draw.

The spread of draws approximates parameter uncertainty and feeds the design
scoring in suggest.py. The whole stage is time-boxed (BOOTSTRAP_SETTINGS).

Outputs (scratch/):
    system_B_bootstrap_parameter_samples.csv
    system_B_bootstrap_diagnostics.csv
    system_B_bootstrap_parameter_uncertainty_summary.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import pipeline_common as pc
from pipeline_common import CANDIDATES

CANONICAL_PARAMETER_SAMPLES = pc.SCRATCH_DIR / "system_B_bootstrap_parameter_samples.csv"
CANONICAL_DIAGNOSTICS = pc.SCRATCH_DIR / "system_B_bootstrap_diagnostics.csv"
CANONICAL_SUMMARY = pc.SCRATCH_DIR / "system_B_bootstrap_parameter_uncertainty_summary.csv"

FAILED_OBJECTIVE = 1e50


def make_artificial_runs(runs, generating_sims, rng):
    artificial = {}
    for run_id, run in runs.items():
        sim = generating_sims[run_id]
        blocks = {}
        for name, block in run["blocks"].items():
            mu = pc.predicted_at(sim, name, block)
            values = rng.normal(mu, pc.sigma(name, mu))
            censored = values < pc.Q_LOQ if name == "Q" else np.zeros(len(values), dtype=bool)
            values = np.where(censored, np.nan, values)
            blocks[name] = {"time_index": block["time_index"], "censored": censored, "values": values}
        artificial[run_id] = {**run, "blocks": blocks}
    return artificial


def refit(model, data_runs, x_start, bounds, maxiter, box):
    tracker = {"fun": np.inf, "x": None}

    def objective(z):
        box.check()
        ll = pc.log_likelihood(model, data_runs, pc.from_z(model, z))
        value = -ll if np.isfinite(ll) else FAILED_OBJECTIVE
        if value < tracker["fun"]:
            tracker.update(fun=value, x=np.array(z, copy=True))
        return value

    try:
        res = minimize(objective, x_start, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-10})
        return tracker["x"], tracker["fun"], bool(res.success), str(res.message)
    except pc.BudgetExceeded:
        return tracker["x"], tracker["fun"], False, "time budget reached"


def run_bootstrap(models, settings=None):
    """Run the parametric bootstrap and return the accepted-sample DataFrame."""
    settings = {**pc.BOOTSTRAP_SETTINGS, **(settings or {})}
    models = [m.upper() for m in models]
    runs = pc.get_runs()
    records, diagnostics, summary = [], [], []

    pc.banner(f"PARAMETRIC BOOTSTRAP: {', '.join(models)} "
              f"({settings['n_bootstrap']} replicates each, budget {settings['time_budget_s']} s)")

    per_model_budget = settings["time_budget_s"] / max(len(models), 1)
    report = []

    for model_number, model in enumerate(models):
        names = CANDIDATES[model].param_names
        data_rng = np.random.default_rng(settings["seed"] + model_number)
        start_rng = np.random.default_rng(100 + model_number)
        original = pc.load_best_parameters(model)
        lo, hi = pc.z_bounds(model)
        bounds = list(zip(lo, hi))
        x_original = np.clip(pc.to_z(model, original), lo + 1e-9, hi - 1e-9)
        generating = pc.simulate_runs(model, runs, original)
        box = pc.TimeBox(per_model_budget)
        done, capped = 0, False

        for replicate in range(1, settings["n_bootstrap"] + 1):
            if box.expired:
                print(f"  [bootstrap] {model}: time budget reached after {replicate - 1} replicates",
                      flush=True)
                capped = True
                break

            data_runs = make_artificial_runs(runs, generating, data_rng)
            x, fun, success, message = refit(model, data_runs, x_original, bounds,
                                             settings["maxiter"], box)

            # One perturbed restart if the first attempt produced nothing usable.
            if (x is None or fun >= FAILED_OBJECTIVE) and not box.expired:
                perturbed = np.clip(
                    x_original + start_rng.normal(0.0, [0.25 if pc.is_rate(n) else 0.10 for n in names]),
                    lo + 1e-6, hi - 1e-6)
                x, fun, success, message = refit(model, data_runs, perturbed, bounds,
                                                 settings["maxiter"], box)

            # A refit cut off by the time budget is not a converged estimate: reject it
            # rather than let a half-finished optimisation distort the spread.
            accepted = (x is not None and np.isfinite(fun) and fun < FAILED_OBJECTIVE
                        and message != "time budget reached")
            if message == "time budget reached":
                capped = True
            else:
                done += 1
            bound_hits = []
            if accepted:
                bound_hits = [n for n, v, (a, b) in zip(names, x, bounds)
                              if min(v - a, b - v) < 1e-3 * (b - a)]
                records.append({"model": model, "replicate": replicate,
                                "negative_log_likelihood": float(fun),
                                "optimiser_success": success,
                                "bound_hits": ", ".join(bound_hits),
                                **pc.from_z(model, x)})

            diagnostics.append({"model": model, "replicate": replicate,
                                "status": "accepted" if accepted else "FAILED",
                                "optimiser_success": success, "message": message,
                                "best_nll": float(fun) if accepted else np.nan,
                                "bound_hits": ", ".join(bound_hits)})
            print(f"  [bootstrap] {model} {replicate}/{settings['n_bootstrap']} "
                  f"{'accepted' if accepted else 'FAILED'} ({message})", flush=True)

        report.append({"stage": f"bootstrap {model} replicates", "planned": settings["n_bootstrap"],
                       "completed": done, "capped": capped, "seconds": box.elapsed})
        model_rows = [r for r in records if r["model"] == model]
        for n in names:
            values = np.array([r[n] for r in model_rows], dtype=float)
            if values.size == 0:
                continue
            summary.append({"model": model, "parameter": n,
                            "original_estimate": float(original[n]),
                            "bootstrap_median": float(np.median(values)),
                            "percentile_5": float(np.quantile(values, 0.05)),
                            "percentile_95": float(np.quantile(values, 0.95)),
                            "accepted_replicates": int(values.size)})

    samples = pd.DataFrame(records)
    samples.to_csv(CANONICAL_PARAMETER_SAMPLES, index=False)
    pd.DataFrame(diagnostics).to_csv(CANONICAL_DIAGNOSTICS, index=False)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(CANONICAL_SUMMARY, index=False)

    print("\n  [bootstrap] parameter uncertainty summary:", flush=True)
    print(summary_frame.to_string(index=False) if not summary_frame.empty
          else "  no accepted bootstrap fits", flush=True)
    run_bootstrap.last_report = report
    return samples


if __name__ == "__main__":
    run_bootstrap(sys.argv[1:] or ["M2", "M4"])