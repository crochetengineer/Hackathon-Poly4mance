"""Shared helpers for System B: paths, data loading, a fast ODE simulator, and a
fallback fit/predict.

Improved from the original naive baseline helper. The public names are unchanged
(ROOT, DATA_B, HELDOUT, SUBMISSION, BASELINE_MODEL, OBSERVED, load_system_b, _sim,
fit_baseline, predict, ensure_submission_dir), so anything that imported the old
module still works. What changed:

* Censored Q values ("<LOQ" in the value column, or censored=True) are kept as
  censored observations instead of being silently dropped.
* _sim / simulate_fast integrate each candidate's own rhs() with LSODA, a stiff
  solver. A drop-in replacement for candidates.simulate that stays fast in stiff
  parameter regions (fast intermediates at high temperature). The main pipeline
  (baseline/pipeline_common.py) checks that simulate_fast agrees with the
  official simulate before using it.
* fit_baseline uses the noise model (relative + absolute error), a censored
  likelihood term for Q < LOQ, log10(k) / (Ea/100) scaling and several starts.
* predict gives noise-model 90 % intervals instead of fixed +/-50 % bands.

Pure numpy/scipy/pandas -- nothing outside the pinned packages.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import log_ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
from candidates import CANDIDATES  # noqa: E402

DATA_B = os.path.join(ROOT, "data", "system_B", "measurements.csv")
HELDOUT = os.path.join(ROOT, "data", "heldout_conditions.csv")
SUBMISSION = os.path.join(ROOT, "submission")
FIGURES = os.path.join(SUBMISSION, "figures")

BASELINE_MODEL = "M1"     # model used by fit_baseline()/predict() when none is given
OBSERVED = ["A", "B", "P", "Q"]

# Observation model: sigma = sqrt((rel * concentration)^2 + abs^2)
Q_LOQ = 3.0e-5
NOISE_MODEL = {
    "A": (0.03, 3.5e-3),
    "B": (0.03, 3.5e-3),
    "P": (0.03, 3.5e-3),
    "Q": (0.15, 2.2e-5),
}
Z90 = 1.6448536269514722  # two-sided 90 % normal quantile

K_BOUNDS = (1e-6, 1.0)     # rate constants at the 75 C reference temperature
EA_BOUNDS = (5.0, 150.0)   # activation energies, kJ/mol


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def sigma(species: str, concentration) -> np.ndarray:
    """Measurement standard deviation for one species at the given concentration(s)."""
    rel, absolute = NOISE_MODEL[species]
    c = np.maximum(np.asarray(concentration, dtype=float), 0.0)
    return np.sqrt((rel * c) ** 2 + absolute ** 2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_system_b_frame(path: str = DATA_B) -> pd.DataFrame:
    """
    Long-format System B measurements with a clean boolean `censored` column and
    a numeric `value` column (NaN for censored rows).

    A row is censored if `censored` is true OR its value is written as "<LOQ"
    (or any value starting with "<").
    """
    df = pd.read_csv(path)
    if "system" in df.columns:
        df = df[df["system"].astype(str) == "B"].copy()

    raw = df["value"].astype(str).str.strip()
    below_loq_text = raw.str.startswith("<")
    if "censored" in df.columns:
        flagged = df["censored"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        df["censored"] = flagged | below_loq_text
    else:
        df["censored"] = below_loq_text

    df["value"] = pd.to_numeric(df["value"].where(~df["censored"]), errors="coerce")
    df["time_min"] = df["time_min"].astype(float)
    return df[df["censored"] | df["value"].notna()].reset_index(drop=True)


def load_system_b() -> list[dict]:
    """
    One dict per run:
        run_id, T (C), A0, B0,
        obs[species]      = (times, values)  quantified points only (same shape as before)
        censored[species] = times at which the species was reported < LOQ
    """
    df = load_system_b_frame()
    runs = []
    for rid, g in df.groupby("run_id", sort=True):
        obs, censored = {}, {}
        for sp, gg in g.groupby("species"):
            quantified = gg[~gg["censored"]]
            obs[sp] = (quantified["time_min"].to_numpy(float), quantified["value"].to_numpy(float))
            censored[sp] = gg.loc[gg["censored"], "time_min"].to_numpy(float)
        runs.append({"run_id": rid, "T": float(g["temperature_C"].iloc[0]),
                     "A0": float(g["A0_M"].iloc[0]), "B0": float(g["B0_M"].iloc[0]),
                     "obs": obs, "censored": censored})
    return runs


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _sim(model, params, T, A0, B0, times) -> dict[str, np.ndarray]:
    """
    Integrate one candidate with LSODA (stiff-capable) and return
    {species: concentrations at `times`}, in the order `times` was given.
    Raises RuntimeError if the solver fails.
    """
    cand = CANDIDATES[model]
    times = np.asarray(times, dtype=float)
    unique_times, inverse = np.unique(times, return_inverse=True)
    y0 = np.array([A0 if s == "A" else B0 if s == "B" else 0.0 for s in cand.species], dtype=float)
    t_end = float(unique_times.max()) if unique_times.size else 0.0

    if t_end <= 0.0:
        values = np.repeat(y0[:, None], unique_times.size, axis=1)
    else:
        sol = solve_ivp(lambda t, y: cand.rhs(y, params, T), (0.0, t_end), y0,
                        t_eval=unique_times, method="LSODA", rtol=1e-7, atol=1e-10)
        if not sol.success or sol.y.shape[1] != unique_times.size:
            raise RuntimeError(f"{model} integration failed: {sol.message}")
        values = sol.y

    return {s: np.clip(values[i][inverse], 0.0, None) for i, s in enumerate(cand.species)}


def simulate_fast(model, conditions, params) -> pd.DataFrame:
    """
    Drop-in replacement for candidates.simulate(model, conditions, params):
    a DataFrame with `time_min` plus one column per species.
    """
    times = np.asarray(conditions.times_min, dtype=float)
    sim = _sim(model, params, float(conditions.temperature_c), float(conditions.A0),
               float(conditions.B0), times)
    frame = pd.DataFrame(sim)
    frame.insert(0, "time_min", times)
    return frame


# ---------------------------------------------------------------------------
# Fallback fit and prediction
# ---------------------------------------------------------------------------

def _unpack(names, z):
    return {n: float(10.0 ** z[i] if n.startswith("k") else 100.0 * z[i]) for i, n in enumerate(names)}


def fit_baseline(model: str = BASELINE_MODEL, n_starts: int = 4, seed: int = 0,
                 max_nfev: int = 300) -> dict[str, float]:
    """
    Weighted, censored-aware least-squares fit of one candidate.

    Residuals: (mu - y) / sigma(mu) for quantified points, and
    sqrt(-2 log Phi((LOQ - mu)/sigma)) for Q < LOQ. Optimised in
    (log10 k, Ea/100) space from one fixed and n_starts - 1 random starts.
    """
    runs = load_system_b()
    cand = CANDIDATES[model]
    names = cand.param_names
    observed = set(getattr(cand, "observed", OBSERVED))

    n_resid = sum(len(r["obs"][sp][0]) + len(r["censored"].get(sp, []))
                  for r in runs for sp in r["obs"] if sp in observed)

    def resid(z):
        params = _unpack(names, z)
        out = []
        try:
            for r in runs:
                all_times = np.unique(np.concatenate(
                    [r["obs"][sp][0] for sp in r["obs"]] + [r["censored"][sp] for sp in r["censored"]]))
                sim = _sim(model, params, r["T"], r["A0"], r["B0"], all_times)
                position = {t: i for i, t in enumerate(all_times)}
                for sp in r["obs"]:
                    if sp not in observed or sp not in sim:
                        continue
                    ts, vs = r["obs"][sp]
                    mu = sim[sp][[position[t] for t in ts]]
                    out.append((mu - vs) / sigma(sp, mu))
                    cens_t = r["censored"].get(sp, np.array([]))
                    if cens_t.size:
                        mu_c = sim[sp][[position[t] for t in cens_t]]
                        logp = np.maximum(log_ndtr((Q_LOQ - mu_c) / sigma(sp, mu_c)), -1e6)
                        out.append(np.sqrt(-2.0 * logp))
            r_all = np.concatenate(out) if out else np.zeros(n_resid)
            return r_all if np.all(np.isfinite(r_all)) else np.full(n_resid, 1e3)
        except Exception:
            return np.full(n_resid, 1e3)

    lb = np.array([np.log10(K_BOUNDS[0]) if n.startswith("k") else EA_BOUNDS[0] / 100 for n in names])
    ub = np.array([np.log10(K_BOUNDS[1]) if n.startswith("k") else EA_BOUNDS[1] / 100 for n in names])
    rng = np.random.default_rng(seed)
    starts = [np.array([-2.0 if n.startswith("k") else 0.6 for n in names])]
    starts += [np.array([rng.uniform(-5.0, -0.5) if n.startswith("k") else rng.uniform(0.2, 1.1)
                         for n in names]) for _ in range(max(n_starts - 1, 0))]

    best = None
    for z0 in starts:
        sol = least_squares(resid, np.clip(z0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub),
                            max_nfev=max_nfev)
        if best is None or sol.cost < best.cost:
            best = sol
    return _unpack(names, best.x)


def predict(params: dict[str, float], model: str = BASELINE_MODEL,
            rel_halfwidth: float | None = None) -> pd.DataFrame:
    """
    Point predictions for the held-out conditions with 90 % measurement
    intervals from the noise model (mu +/- 1.645 sigma, not clipped at zero:
    a measurement near zero can read negative even though mu cannot). This ignores parameter and
    model uncertainty, so it is a floor on interval width; pass rel_halfwidth
    to widen every interval by an extra +/- rel_halfwidth * mu.
    """
    cond = pd.read_csv(HELDOUT)
    rows = []
    for rid, g in cond.groupby("run_id", sort=True):
        times = np.sort(g["time_min"].astype(float).unique())
        T, A0, B0 = float(g["temperature_C"].iloc[0]), float(g["A0_M"].iloc[0]), float(g["B0_M"].iloc[0])
        sim = _sim(model, params, T, A0, B0, times)
        for sp in OBSERVED:
            mu = sim.get(sp, np.zeros_like(times))
            half = Z90 * sigma(sp, mu)
            if rel_halfwidth:
                half = half + rel_halfwidth * mu
            for t, v, h in zip(times, mu, half):
                lo, hi = v - h, v + h   # measurement interval: Gaussian, not clipped
                rows.append({"run_id": rid, "time_min": round(float(t), 4), "species": sp,
                             "prediction": f"{v:.6g}", "lower90": f"{lo:.6g}", "upper90": f"{hi:.6g}"})
    return pd.DataFrame(rows, columns=["run_id", "time_min", "species", "prediction",
                                       "lower90", "upper90"])


def ensure_submission_dir() -> str:
    """Create submission/ and submission/figures/; return the submission path."""
    os.makedirs(SUBMISSION, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    return SUBMISSION