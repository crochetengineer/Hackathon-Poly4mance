"""
just suggest  ->  python3 baseline/suggest.py [--reuse-bootstrap]

1. Read the two retained models from submission/hypotheses.yaml (written by
   `just predict`).
2. Run the parametric bootstrap for them (bootstrap_system_b.run_bootstrap).
   If a model has too few accepted bootstrap fits, top up with its Bayesian
   posterior draws from `just predict`.
3. Score a grid of candidate experiments by how strongly the two models'
   predictions separate under parameter uncertainty.
4. Pick a diverse, budget-feasible campaign and write submission/campaign.yaml.

Speed: each (temperature, A0, B0) condition is simulated once per parameter
set on the union of all candidate sampling schedules, then every schedule is
scored by sub-setting those rows.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr

import pipeline_common as pc
from pipeline_common import CANDIDATES, Conditions, simulate
from pipeline_common import SUBMISSION, ensure_submission_dir
from bootstrap_system_b import CANONICAL_PARAMETER_SAMPLES, run_bootstrap


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

HYPOTHESES_FILE = Path(SUBMISSION) / "hypotheses.yaml"
CAMPAIGN_FILE = Path(SUBMISSION) / "campaign.yaml"

REACTION_VOLUME_L = 0.005
A_MOLAR_MASS_G_PER_MOL = 250.0

MIN_EXPERIMENTS = 3
MAX_EXPERIMENTS = 10
TARGET_N_EXPERIMENTS = 5
MAX_TOTAL_SAMPLES = 60
MAX_TOTAL_A_G = 5.0
T_RANGE = (50.0, 120.0)
CONC_RANGE = (0.1, 1.5)
MAX_DURATION_MIN = 960.0

if pc.FAST:
    TEMPERATURES_C = [60.0, 90.0, 110.0]
    A0_GRID = [0.2, 0.75, 1.5]
    B0_GRID = [0.2, 0.75, 1.5]
else:
    TEMPERATURES_C = [55.0, 65.0, 75.0, 85.0, 95.0, 105.0, 115.0]
    A0_GRID = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]
    B0_GRID = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]

TIME_GRIDS = [
    [0.0, 15.0, 45.0, 90.0, 180.0, 360.0],
    [0.0, 30.0, 60.0, 120.0, 240.0],
    [0.0, 60.0, 120.0, 240.0, 480.0],
    [0.0, 120.0, 240.0, 480.0, 960.0],
]
UNION_TIMES = sorted({t for grid in TIME_GRIDS for t in grid})
GRID_INDEX = [[UNION_TIMES.index(t) for t in grid] for grid in TIME_GRIDS]

TEMPERATURE_DIFFERENCE_MIN = 15.0
A0_DIFFERENCE_MIN = 0.15
B0_DIFFERENCE_MIN = 0.15

SORT_COLUMNS = [
    "max_detection_mismatch",
    "detection_mismatch_total",
    "robust_q_gap_q10",
    "robust_q_gap_median",
    "q_gap_total",
    "full_gap_total",
]


# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------

def a_mass_g(a0_m):
    return float(REACTION_VOLUME_L * float(a0_m) * A_MOLAR_MASS_G_PER_MOL)


def models_from_hypotheses():
    if not HYPOTHESES_FILE.exists():
        raise FileNotFoundError(f"Missing {HYPOTHESES_FILE}. Run `just predict` first.")
    with HYPOTHESES_FILE.open("r", encoding="utf-8") as handle:
        hypotheses = yaml.safe_load(handle) or {}
    selected = [str(m).upper() for m in hypotheses.get("selected_models", [])]
    if len(selected) < 2:
        ranked = sorted(hypotheses.get("all_models", []), key=lambda e: e.get("rank", 99))
        selected = [str(e["name"]).upper() for e in ranked[:2]]
    if len(selected) < 2:
        raise RuntimeError("Need two retained models in hypotheses.yaml to design a campaign.")
    return selected[:2], hypotheses


def parameter_sets(model, settings):
    names = list(CANDIDATES[model].param_names)
    rows = pd.DataFrame()
    if CANONICAL_PARAMETER_SAMPLES.exists() and CANONICAL_PARAMETER_SAMPLES.stat().st_size > 0:
        table = pd.read_csv(CANONICAL_PARAMETER_SAMPLES)
        if "model" in table.columns:
            rows = table.loc[table["model"].astype(str).str.upper() == model]
            rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=names)[names]
    source = f"{len(rows)} bootstrap"

    if len(rows) < settings["min_bootstrap_sets"]:
        path, kind = pc.prediction_draws_file(model)
        if path is not None:
            extra = pd.read_csv(path)[names].dropna()
            need = settings["max_parameter_sets_per_model"] - len(rows)
            extra = extra.sample(n=min(need, len(extra)), random_state=settings["seed"])
            rows = pd.concat([rows, extra], ignore_index=True)
            source += f" + {len(extra)} {kind}"

    if rows.empty:
        rows = pd.DataFrame([pc.load_best_parameters(model)])[names]
        source = "point estimate only (no uncertainty)"

    if len(rows) > settings["max_parameter_sets_per_model"]:
        rows = rows.sample(n=settings["max_parameter_sets_per_model"], random_state=settings["seed"])
    print(f"[suggest] {model}: {len(rows)} parameter sets ({source})", flush=True)
    return rows.to_dict("records")


# ---------------------------------------------------------
# Scoring
# ---------------------------------------------------------

def simulate_sets(model, sets, temperature_c, a0_m, b0_m):
    """Return array (n_sets_ok, n_union_times, 4) of A, B, P, Q."""
    cond = Conditions(temperature_c=float(temperature_c), A0=float(a0_m), B0=float(b0_m),
                      times_min=list(UNION_TIMES))
    out = []
    for params in sets:
        try:
            sim = pc.safe_simulate(model, cond, params)
        except Exception:
            continue
        arr = np.maximum(sim[pc.SPECIES].to_numpy(dtype=float), 0.0)
        if arr.shape[0] == len(UNION_TIMES) and np.all(np.isfinite(arr)):
            out.append(arr)
    return np.asarray(out)


def score_design(arr_a, arr_b, temperature_c, a0_m, b0_m, schedule, index):
    a, b = arr_a[:, index, :], arr_b[:, index, :]
    q_a, q_b = a[..., 3], b[..., 3]

    q_gap_total = float(np.sum(np.abs(q_a.mean(axis=0) - q_b.mean(axis=0))))
    full_gap_total = float(np.sum(np.abs(a.reshape(len(a), -1).mean(axis=0)
                                         - b.reshape(len(b), -1).mean(axis=0))))

    # Probability that an actual Q MEASUREMENT reads at or above the LOQ, including
    # assay noise: P(y >= LOQ | mu) = Phi((mu - LOQ) / sigma_Q(mu)), averaged over
    # each model's parameter sets. (Comparing noise-free mu with the LOQ would
    # ignore that a true value just above the LOQ is often reported below it.)
    detect_a = np.mean(ndtr((q_a - pc.Q_LOQ) / pc.sigma("Q", q_a)), axis=0)
    detect_b = np.mean(ndtr((q_b - pc.Q_LOQ) / pc.sigma("Q", q_b)), axis=0)
    mismatch = np.abs(detect_a - detect_b)

    n_pairs = min(len(q_a), len(q_b))
    pairwise = np.mean(np.abs(q_a[:n_pairs] - q_b[:n_pairs]), axis=1)
    q_detectable_any = bool(np.any((detect_a > 0.05) | (detect_b > 0.05)))

    return {
        "temperature_C": float(temperature_c),
        "A0_M": float(a0_m),
        "B0_M": float(b0_m),
        "sample_times_min": [float(t) for t in schedule],
        "n_timepoints": int(len(schedule)),
        "a_mass_g": a_mass_g(a0_m),
        "q_gap_total": q_gap_total,
        "full_gap_total": full_gap_total,
        "detection_mismatch_total": float(np.sum(mismatch)),
        "max_detection_mismatch": float(np.max(mismatch)),
        "robust_q_gap_q10": float(np.quantile(pairwise, 0.10)),
        "robust_q_gap_median": float(np.quantile(pairwise, 0.50)),
        "robust_q_gap_q90": float(np.quantile(pairwise, 0.90)),
        "q_detectable_any": q_detectable_any,
        "feasible": q_detectable_any,
    }


def rank_candidate_experiments(model_a, model_b, sets_a, sets_b, settings):
    conditions = [(t, a0, b0) for t in TEMPERATURES_C for a0 in A0_GRID for b0 in B0_GRID]
    # Shuffle so that, if the time budget truncates scoring, the scored subset
    # still spans the whole design space rather than only low temperatures.
    order = np.random.default_rng(settings["seed"]).permutation(len(conditions))
    box = pc.TimeBox(settings["time_budget_s"])

    print(f"[suggest] scoring {len(conditions)} conditions x {len(TIME_GRIDS)} schedules "
          f"(budget {settings['time_budget_s']} s)", flush=True)

    rows = []
    done = 0
    capped = False
    for k in order:
        if box.expired:
            print(f"[suggest] time budget reached: scored {done}/{len(conditions)} conditions", flush=True)
            capped = True
            break
        t, a0, b0 = conditions[k]
        arr_a = simulate_sets(model_a, sets_a, t, a0, b0)
        arr_b = simulate_sets(model_b, sets_b, t, a0, b0)
        done += 1
        if len(arr_a) == 0 or len(arr_b) == 0:
            continue
        for schedule, index in zip(TIME_GRIDS, GRID_INDEX):
            rows.append(score_design(arr_a, arr_b, t, a0, b0, schedule, index))
        if done % 50 == 0:
            print(f"[suggest] scored {done}/{len(conditions)} conditions", flush=True)

    rank_candidate_experiments.report = {"stage": "design grid conditions", "planned": len(conditions),
                                         "completed": done, "capped": capped,
                                         "seconds": box.elapsed}
    if not rows:
        raise RuntimeError("No candidate design could be scored.")

    ranking = pd.DataFrame(rows).sort_values(
        by=["feasible"] + SORT_COLUMNS, ascending=[False] * (1 + len(SORT_COLUMNS)),
        ignore_index=True)
    return ranking


# ---------------------------------------------------------
# Campaign selection (same logic as before)
# ---------------------------------------------------------

def same_design(row1, row2):
    return (
        float(row1["temperature_C"]) == float(row2["temperature_C"])
        and float(row1["A0_M"]) == float(row2["A0_M"])
        and float(row1["B0_M"]) == float(row2["B0_M"])
        and tuple(row1["sample_times_min"]) == tuple(row2["sample_times_min"])
    )


def already_selected(candidate_row, selected_rows):
    return any(same_design(candidate_row, row) for row in selected_rows)


def campaign_budget_ok(selected_rows, candidate_row):
    rows = list(selected_rows) + [candidate_row]
    return (sum(int(r["n_timepoints"]) for r in rows) <= MAX_TOTAL_SAMPLES
            and sum(float(r["a_mass_g"]) for r in rows) <= MAX_TOTAL_A_G)


def pick_first_matching(df, selected_rows, predicate):
    for _, row in df.iterrows():
        if already_selected(row, selected_rows) or not campaign_budget_ok(selected_rows, row):
            continue
        if predicate(row):
            return row
    return None


def choose_diverse_campaign(ranking):
    feasible = ranking.loc[ranking["feasible"]].copy()
    if feasible.empty:
        print("[suggest] WARNING: no design predicts detectable Q under either model; "
              "ranking all designs instead.", flush=True)
        feasible = ranking.copy()
    feasible = feasible.sort_values(by=SORT_COLUMNS, ascending=[False] * len(SORT_COLUMNS),
                                    ignore_index=True)

    selected = [feasible.iloc[0]]
    base = selected[0]

    for predicate in (
        lambda r: abs(float(r["temperature_C"]) - float(base["temperature_C"])) >= TEMPERATURE_DIFFERENCE_MIN,
        lambda r: abs(float(r["A0_M"]) - float(base["A0_M"])) >= A0_DIFFERENCE_MIN,
        lambda r: abs(float(r["B0_M"]) - float(base["B0_M"])) >= B0_DIFFERENCE_MIN,
    ):
        if len(selected) >= TARGET_N_EXPERIMENTS:
            break
        row = pick_first_matching(feasible, selected, predicate)
        if row is not None:
            selected.append(row)

    def new_conditions(r):
        return min(abs(float(r["temperature_C"]) - float(e["temperature_C"]))
                   + abs(float(r["A0_M"]) - float(e["A0_M"]))
                   + abs(float(r["B0_M"]) - float(e["B0_M"])) for e in selected) > 0.0

    while len(selected) < TARGET_N_EXPERIMENTS:
        row = pick_first_matching(feasible, selected, new_conditions)
        if row is None:
            break
        selected.append(row)

    # Guarantee the minimum campaign size, relaxing the diversity rule if needed.
    while len(selected) < MIN_EXPERIMENTS:
        row = pick_first_matching(ranking, selected, lambda r: True)
        if row is None:
            break
        selected.append(row)

    return selected


def validate_campaign(experiments):
    problems = []
    n = len(experiments)
    if not MIN_EXPERIMENTS <= n <= MAX_EXPERIMENTS:
        problems.append(f"{n} experiments (allowed {MIN_EXPERIMENTS}-{MAX_EXPERIMENTS})")
    total_samples = sum(len(e["sample_times_min"]) for e in experiments)
    total_a = sum(e["a_mass_g"] for e in experiments)
    if total_samples > MAX_TOTAL_SAMPLES:
        problems.append(f"{total_samples} samples > {MAX_TOTAL_SAMPLES}")
    if total_a > MAX_TOTAL_A_G + 1e-9:
        problems.append(f"{total_a:.3f} g of A > {MAX_TOTAL_A_G} g")
    for e in experiments:
        if not T_RANGE[0] <= e["temperature_C"] <= T_RANGE[1]:
            problems.append(f"{e['id']}: temperature out of range")
        for key in ("A0_M", "B0_M"):
            if not CONC_RANGE[0] <= e[key] <= CONC_RANGE[1]:
                problems.append(f"{e['id']}: {key} out of range")
        if max(e["sample_times_min"]) > MAX_DURATION_MIN:
            problems.append(f"{e['id']}: sampling beyond 16 h")
    if problems:
        raise ValueError("Campaign violates constraints: " + "; ".join(problems))
    return total_samples, total_a


def build_campaign(selected, model_a, model_b):
    experiments = []
    for i, row in enumerate(selected, start=1):
        times = [float(t) for t in row["sample_times_min"]]
        experiments.append({
            "id": f"E{i}",
            "temperature_C": float(row["temperature_C"]),
            "A0_M": float(row["A0_M"]),
            "B0_M": float(row["B0_M"]),
            "duration_min": float(max(times)),
            "sample_times_min": times,
            "n_samples": len(times),
            "assay_species": list(pc.SPECIES),
            "a_mass_g": round(float(row["a_mass_g"]), 4),
            "discrimination_scores": {c: float(row[c]) for c in SORT_COLUMNS},
        })
    total_samples, total_a = validate_campaign(experiments)
    return {
        "campaign": "System B confirmatory model-discrimination campaign",
        "discriminating_between": [model_a, model_b],
        "supported_models": [model_a, model_b],
        "design_basis": (
            f"Screening heuristic, not a formal expected-information criterion. Grid search over "
            f"temperature, A0, B0 and sampling schedule; each design is ranked by how differently "
            f"{model_a} and {model_b} predict the probability that a Q measurement (including assay "
            "noise) reads above the LOQ, then by the separation of their Q and full profiles, "
            "across parameter-uncertainty draws (parametric bootstrap, topped up with posterior or "
            "Laplace draws if needed). Diversity rules then force contrasts in T, A0 and B0."
        ),
        "scoring_is": "screening heuristic",
        "constraints": {"max_experiments": MAX_EXPERIMENTS, "max_total_samples": MAX_TOTAL_SAMPLES,
                        "max_total_A_g": MAX_TOTAL_A_G, "reaction_volume_mL": 5.0},
        "totals": {"n_experiments": len(experiments), "total_samples": total_samples,
                   "total_A_g": round(total_a, 4)},
        "experiments": experiments,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Design the System B confirmatory campaign.")
    parser.add_argument("--reuse-bootstrap", action="store_true",
                        help="Skip the bootstrap and reuse the samples already in scratch/.")
    args = parser.parse_args()

    ensure_submission_dir()
    settings = pc.SUGGEST_SETTINGS
    box = pc.TimeBox(1e9)
    pc.banner("# SYSTEM B CAMPAIGN DESIGN" + ("  [FAST MODE]" if pc.FAST else ""), "#")

    (model_a, model_b), _ = models_from_hypotheses()
    print(f"[suggest] discriminating {model_a} vs {model_b}", flush=True)

    if args.reuse_bootstrap and CANONICAL_PARAMETER_SAMPLES.exists():
        print("[suggest] reusing existing bootstrap samples", flush=True)
    else:
        run_bootstrap([model_a, model_b])

    sets_a = parameter_sets(model_a, settings)
    sets_b = parameter_sets(model_b, settings)

    ranking = rank_candidate_experiments(model_a, model_b, sets_a, sets_b, settings)
    ranking_file = pc.SCRATCH_DIR / f"system_B_{model_a}_{model_b}_bootstrap_ranked_experiments.csv"
    ranking.to_csv(ranking_file, index=False)

    selected = choose_diverse_campaign(ranking)
    campaign = build_campaign(selected, model_a, model_b)
    with CAMPAIGN_FILE.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(campaign, handle, sort_keys=False, allow_unicode=True)

    pc.banner("[suggest] CAMPAIGN", "#")
    for e in campaign["experiments"]:
        print(f"  {e['id']}: {e['temperature_C']:.0f} C, A0 = {e['A0_M']:.2f} M, "
              f"B0 = {e['B0_M']:.2f} M, samples at {e['sample_times_min']} min, "
              f"A = {e['a_mass_g']:.3f} g", flush=True)
    t = campaign["totals"]
    print(f"  totals: {t['n_experiments']} experiments, {t['total_samples']} samples, "
          f"{t['total_A_g']:.3f} g A", flush=True)
    print(f"[suggest] ranking -> {ranking_file}", flush=True)
    print(f"[suggest] campaign -> {CAMPAIGN_FILE}", flush=True)
    report = list(getattr(run_bootstrap, "last_report", []))
    report.append(rank_candidate_experiments.report)
    pc.write_run_report("suggest", report)
    print(f"[suggest] wall time {box.elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()