"""
Shared paths, settings, data preparation and likelihood for the System B pipeline.

    just predict   -> baseline/predict.py
                      fit all 6 candidates (fit_system_b_weighted_multistart.run_fit)
                      -> rank by AICc, write submission/hypotheses.yaml
                      -> Bayesian MAP + MCMC on the top models (bayesian_system_b.run_bayesian)
                      -> posterior-predictive held-out predictions
    just suggest   -> baseline/suggest.py
                      parametric bootstrap of the top-2 models (bootstrap_system_b.run_bootstrap)
                      -> score candidate designs -> submission/campaign.yaml
    just replicate -> baseline/replicate.py (figures)

Every baseline module imports this first, so src/ is always on sys.path and
every stage uses the same noise model, bounds, censoring rule and likelihood.

RUNTIME: the judges' limit is 10 minutes for all four commands together.
Every expensive stage below has a wall-clock budget (seconds). Budgets are
caps, not targets: a stage that finishes early just finishes. Time
`just predict && just suggest && just replicate` on a machine similar to the
judges' and shrink the budgets if you are close to the limit.

Set SYSTEM_B_FAST=1 for a quick smoke test of the whole chain.
"""

import os
import signal
import sys
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BASELINE_DIR = PROJECT_ROOT / "baseline"

for _path in (SRC_DIR, BASELINE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from candidates import CANDIDATES, Conditions, simulate  # noqa: E402,F401


def _import_reference():
    """
    Import the repo's `reference` module (SUBMISSION, ensure_submission_dir)
    wherever it lives in the project. If it can't be found, fall back to
    submission/ at the project root so the pipeline still runs.
    """
    def _try_import():
        sys.modules.pop("reference", None)  # drop a namespace-package false hit
        try:
            import reference  # noqa: F401
        except ModuleNotFoundError:
            return None
        return reference if hasattr(reference, "SUBMISSION") else None

    found = _try_import()
    if found is not None:
        return found

    skip = {".git", ".venv", "venv", "env", "node_modules", "site-packages", "__pycache__",
            "scratch", "submission"}
    for path in sorted(PROJECT_ROOT.rglob("reference*")):
        if any(part in skip for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        if path.name == "reference.py":
            folder = path.parent
        elif path.is_dir() and path.name == "reference" and (path / "__init__.py").exists():
            folder = path.parent
        else:
            continue
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
        found = _try_import()
        if found is not None:
            return found

    print("[pipeline] note: no `reference` module found; writing to submission/ at the "
          "project root", flush=True)

    class _FallbackReference:
        SUBMISSION = PROJECT_ROOT / "submission"

        @staticmethod
        def ensure_submission_dir():
            _FallbackReference.SUBMISSION.mkdir(parents=True, exist_ok=True)
            return _FallbackReference.SUBMISSION

    return _FallbackReference


_reference = _import_reference()
SUBMISSION = _reference.SUBMISSION
ensure_submission_dir = _reference.ensure_submission_dir

SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_B_FILE = PROJECT_ROOT / "data" / "system_B" / "measurements.csv"
HELDOUT_FILE = PROJECT_ROOT / "data" / "heldout_conditions.csv"   # where the brief says it is


def find_heldout_file():
    """
    Locate the held-out conditions CSV. Order: the SYSTEM_B_HELDOUT environment
    variable, data/heldout_conditions.csv, then any *heldout*.csv under data/
    (preferring data/system_B/). Raises a clear error listing what is in data/
    if nothing is found, so predict.py can fail before any fitting.
    """
    override = os.environ.get("SYSTEM_B_HELDOUT")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path
        raise FileNotFoundError(f"SYSTEM_B_HELDOUT points to {path}, which does not exist.")
    if HELDOUT_FILE.is_file():
        return HELDOUT_FILE
    data_dir = PROJECT_ROOT / "data"
    found = sorted(data_dir.rglob("*heldout*.csv"), key=lambda p: ("system_B" not in str(p), str(p)))
    if found:
        if len(found) > 1:
            print("[pipeline] several held-out files found; using "
                  f"{found[0].relative_to(PROJECT_ROOT)}", flush=True)
        return found[0]
    listing = (sorted(str(p.relative_to(PROJECT_ROOT)) for p in data_dir.rglob("*") if p.is_file())
               if data_dir.exists() else [])
    raise FileNotFoundError(
        "Held-out conditions file not found. Expected data/heldout_conditions.csv (as in the brief) "
        "or any *heldout*.csv under data/. Files in data/: "
        + (", ".join(listing) if listing else "(data/ is missing or empty)")
        + ". Copy the file from the original repository, or set SYSTEM_B_HELDOUT=path/to/file.csv."
    )

ALL_MODELS = [m for m in ["M1", "M2", "M3", "M4", "M5", "M6"] if m in CANDIDATES]
SPECIES = ["A", "B", "P", "Q"]

# -----------------------------------------------------------------------------
# Observation model (one definition used everywhere)
# -----------------------------------------------------------------------------
Q_LOQ = 3.0e-5
NOISE_MODEL = {
    "A": (0.03, 3.5e-3),
    "B": (0.03, 3.5e-3),
    "P": (0.03, 3.5e-3),
    "Q": (0.15, 2.2e-5),
}
LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)

# Physical bounds shared by the classical fit, the Bayesian prior and the bootstrap.
K_BOUNDS = (1e-6, 1.0)      # rate constant at 75 C
EA_BOUNDS = (5.0, 150.0)    # kJ/mol

# -----------------------------------------------------------------------------
# Runtime settings
# -----------------------------------------------------------------------------
FAST = os.environ.get("SYSTEM_B_FAST", "0") == "1"


def _pick(full, fast):
    return fast if FAST else full


# Worker processes for the 6 classical fits and the 2 Bayesian fits.
# Override with SYSTEM_B_WORKERS=1 to run serially.
N_WORKERS = int(os.environ.get("SYSTEM_B_WORKERS", min(os.cpu_count() or 1, 6)))

# Per-simulation timeout. Stiff parameter regions (fast intermediate decay at
# high T) can make one ODE solve take seconds or minutes, and no stage budget
# can interrupt a solve in progress. Any single solve slower than
# max(SIM_TIMEOUT_FLOOR_S, SIM_TIMEOUT_MULTIPLIER x recent median) is aborted
# and treated as a failed (infinitely unlikely) parameter set.
SIM_TIMEOUT_FLOOR_S = 0.5
SIM_TIMEOUT_MULTIPLIER = 25.0

FIT_SETTINGS = {
    "n_starts": _pick(8, 3),
    "max_nfev": _pick(600, 120),
    "time_budget_s": _pick(30, 6),     # per model, 6 models
    "seed": 20261010,
}

BAYES_SETTINGS = {
    "n_map_starts": _pick(3, 1),
    "map_maxiter": _pick(200, 30),
    "map_time_budget_s": _pick(15, 4),
    # Sampling: every walker is its own job and does a FIXED number of steps, so
    # results do not depend on machine speed (reproducible). mcmc_time_budget_s is
    # only a safety cap on the whole batch; if it is hit, the run report says so.
    "walkers_per_model": 4,
    "steps_per_walker": _pick(2400, 120),
    "adapt_steps": _pick(400, 40),     # adapted then discarded; the rest are kept
    "mcmc_time_budget_s": _pick(300, 20),
    "min_kept_steps": _pick(100, 15),  # walkers keeping fewer steps are rejected
    "max_stored_draws": 8000,          # thin the draws CSV to at most this many rows
    "laplace_draws": 2000,             # fallback draws if MCMC does not converge
    "init_scale": 0.5,                 # walker spread when no Laplace fit, in proposal steps
    "log_k_step": 0.08,                # default proposal sd for ln k
    "ea_step": 0.8,                    # default proposal sd for Ea (kJ/mol)
    # "log-uniform": flat in ln k (scale-free; standard for rate constants)
    # "uniform":     flat in k (adds +ln k Jacobian; pushes weakly identified k upward)
    "k_prior": "log-uniform",
    "seed": 20261012,
}

PREDICT_SETTINGS = {
    "n_models_to_keep": 2,
    "posterior_draws_per_model": _pick(200, 30),
    "noise_draws_without_posterior": 400,
    "time_budget_s": _pick(45, 10),    # posterior-predictive simulation, total
    # What lower90/upper90 in predictions.csv describe:
    #   "measurement":   a new noisy measurement y ~ N(mu, sigma(mu)^2) (can be < 0)
    #   "concentration": the underlying concentration mu (always >= 0)
    # Both are written to scratch/system_B_predictions_detailed.csv.
    "interval_target": "measurement",
    "seed": 20261012,
}

BOOTSTRAP_SETTINGS = {
    # Fixed number of replicates for reproducibility; time_budget_s is a safety cap.
    "n_bootstrap": _pick(20, 3),
    "maxiter": _pick(300, 40),
    "time_budget_s": _pick(200, 15),   # safety cap, total, split across models
    "seed": 20261010,
}

SUGGEST_SETTINGS = {
    "max_parameter_sets_per_model": _pick(30, 4),
    "min_bootstrap_sets": 5,           # below this, top up with posterior draws
    "time_budget_s": _pick(150, 15),   # design-grid scoring (safety cap; full grid normally fits)
    "seed": 20261012,
}


# -----------------------------------------------------------------------------
# Time budget helper
# -----------------------------------------------------------------------------
class BudgetExceeded(Exception):
    """Raised from inside an objective function when its time budget runs out."""


class TimeBox:
    def __init__(self, seconds):
        self.seconds = float(seconds)
        self.start = time.monotonic()

    @property
    def elapsed(self):
        return time.monotonic() - self.start

    @property
    def expired(self):
        return self.elapsed > self.seconds

    def check(self):
        if self.expired:
            raise BudgetExceeded()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_system_b_data():
    df = pd.read_csv(SYSTEM_B_FILE)
    if "system" in df.columns:
        df = df[df["system"].astype(str) == "B"].copy()
    # Censored if flagged in a `censored` column OR written as "<LOQ" in `value`.
    below_loq_text = df["value"].astype(str).str.strip().str.startswith("<")
    if "censored" in df.columns:
        flagged = df["censored"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        df["censored"] = flagged | below_loq_text
    else:
        df["censored"] = below_loq_text
    df["value"] = pd.to_numeric(df["value"].where(~df["censored"]), errors="coerce")
    df["time_min"] = df["time_min"].astype(float)
    df = df[df["censored"] | df["value"].notna()].copy()

    bad = df[df["censored"] & (df["species"] != "Q")]
    if not bad.empty:
        raise ValueError("Only Q is expected to be censored in System B data.")
    return df


def prepare_runs(df):
    """
    Turn the long-format measurement table into per-run blocks:

        runs[run_id] = {
            "conditions": Conditions(... times_min = sorted unique run times),
            "times": array of those times,
            "blocks": {species: {"time_index", "censored", "values"}},
        }

    time_index points into the simulator output rows (one row per time).
    """
    runs = {}
    for run_id, group in df.groupby("run_id", sort=True):
        times = np.sort(group["time_min"].astype(float).unique())
        time_index = np.searchsorted(times, group["time_min"].to_numpy(dtype=float))
        species = group["species"].astype(str).to_numpy()
        censored = group["censored"].to_numpy(dtype=bool)
        values = group["value"].to_numpy(dtype=float)

        blocks = {}
        for name in np.unique(species):
            mask = species == name
            blocks[str(name)] = {
                "time_index": time_index[mask],
                "censored": censored[mask],
                "values": values[mask],
            }

        first = group.iloc[0]
        runs[run_id] = {
            "conditions": Conditions(
                temperature_c=float(first["temperature_C"]),
                A0=float(first["A0_M"]),
                B0=float(first["B0_M"]),
                times_min=times.tolist(),
            ),
            "times": times,
            "blocks": blocks,
        }
    return runs


@lru_cache(maxsize=1)
def get_runs():
    return prepare_runs(load_system_b_data())


def n_observations(model, runs=None):
    runs = runs or get_runs()
    observed = set(CANDIDATES[model].observed)
    return int(
        sum(
            len(block["time_index"])
            for run in runs.values()
            for name, block in run["blocks"].items()
            if name in observed
        )
    )


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
def is_rate(name):
    return name.startswith("k")


def to_z(model, params):
    """Optimiser space: log10 k and Ea/100 (both O(1))."""
    return np.array(
        [
            np.log10(float(params[n])) if is_rate(n) else float(params[n]) / 100.0
            for n in CANDIDATES[model].param_names
        ],
        dtype=float,
    )


def from_z(model, z):
    return {
        n: (10.0 ** float(v) if is_rate(n) else 100.0 * float(v))
        for n, v in zip(CANDIDATES[model].param_names, z)
    }


def z_bounds(model):
    lo, hi = [], []
    for n in CANDIDATES[model].param_names:
        if is_rate(n):
            lo.append(np.log10(K_BOUNDS[0]))
            hi.append(np.log10(K_BOUNDS[1]))
        else:
            lo.append(EA_BOUNDS[0] / 100.0)
            hi.append(EA_BOUNDS[1] / 100.0)
    return np.array(lo), np.array(hi)


def best_parameters_file(model):
    return SCRATCH_DIR / f"system_B_{model}_weighted_censored_best_parameters.csv"


def all_starts_file(model):
    return SCRATCH_DIR / f"system_B_{model}_weighted_censored_all_starts.csv"


def posterior_draws_file(model):
    return SCRATCH_DIR / f"system_B_{model}_bayesian_posterior_draws.csv"


def unconverged_draws_file(model):
    return SCRATCH_DIR / f"system_B_{model}_bayesian_unconverged_draws.csv"


def laplace_draws_file(model):
    return SCRATCH_DIR / f"system_B_{model}_bayesian_laplace_draws.csv"


def prediction_draws_file(model):
    """
    Parameter draws predictions should use for a model, best first:
    converged posterior > Laplace approximation > none. Returns (path, source).
    """
    if posterior_draws_file(model).exists():
        return posterior_draws_file(model), "posterior"
    if laplace_draws_file(model).exists():
        return laplace_draws_file(model), "laplace"
    return None, "none"


def load_best_parameters(model):
    path = best_parameters_file(model)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `just predict` first (it fits every model)."
        )
    table = pd.read_csv(path)
    column = "value" if "value" in table.columns else "best_fitted_value"
    return {str(r["parameter"]): float(r[column]) for _, r in table.iterrows()}


# -----------------------------------------------------------------------------
# Simulation + likelihood
# -----------------------------------------------------------------------------
def sigma(species, mu):
    rel, absolute = NOISE_MODEL[species]
    mu = np.maximum(np.asarray(mu, dtype=float), 0.0)
    return np.sqrt((rel * mu) ** 2 + absolute ** 2)


class SimulationTimeout(Exception):
    """A single ODE solve exceeded its time limit."""


SIM_STATS = {"calls": 0, "seconds": 0.0, "timeouts": 0, "failures": 0}
_RECENT_DURATIONS = deque(maxlen=200)
_HAS_ALARM = hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")


def _raise_timeout(signum, frame):
    raise SimulationTimeout()


def current_sim_timeout():
    if not _RECENT_DURATIONS:
        return max(SIM_TIMEOUT_FLOOR_S, 5.0)  # generous until we know typical speed
    return max(SIM_TIMEOUT_FLOOR_S, SIM_TIMEOUT_MULTIPLIER * float(np.median(_RECENT_DURATIONS)))


_FAST_SIM_OK = {}
USE_FAST_SIM = os.environ.get("SYSTEM_B_FAST_SIM", "1") != "0"


def _agree(official, fast):
    cols = [c for c in SPECIES if c in official.columns and c in fast.columns]
    if len(official) != len(fast) or not cols:
        return False
    a = official[cols].to_numpy(dtype=float)
    b = fast[cols].to_numpy(dtype=float)
    scale = np.maximum(np.abs(a).max(axis=0), 1e-12)
    return bool(np.all(np.abs(a - b) <= 1e-3 * np.abs(a) + 1e-3 * scale + 1e-12))


def _simulator_for(model, conditions, params):
    """
    Use reference.simulate_fast (LSODA on the candidate's own rhs) when it
    exists and reproduces candidates.simulate for this model; otherwise use the
    official simulate. Checked once per model per process.
    Disable with SYSTEM_B_FAST_SIM=0.
    """
    fast = getattr(_reference, "simulate_fast", None)
    if not USE_FAST_SIM or fast is None:
        return simulate
    if model not in _FAST_SIM_OK:
        try:
            ok = _agree(simulate(model, conditions, params), fast(model, conditions, params))
        except Exception:  # noqa: BLE001
            ok = False
        _FAST_SIM_OK[model] = ok
        if not ok:
            print(f"[pipeline] {model}: fast simulator does not match candidates.simulate; "
                  "using the official simulator", flush=True)
    return fast if _FAST_SIM_OK[model] else simulate


def safe_simulate(model, conditions, params, timeout=None):
    """
    simulate() with a wall-clock limit (Linux/macOS, main thread; on Windows the
    limit is skipped). Raises SimulationTimeout or the simulator's own error.
    """
    simulator = _simulator_for(model, conditions, params)
    limit = current_sim_timeout() if timeout is None else timeout
    use_alarm = _HAS_ALARM and limit > 0 and threading.current_thread() is threading.main_thread()
    start = time.perf_counter()
    previous = None
    ok = False
    try:
        if use_alarm:
            previous = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, limit)
        result = simulator(model, conditions, params)
        ok = True
        return result
    except SimulationTimeout:
        SIM_STATS["timeouts"] += 1
        raise
    except Exception:
        SIM_STATS["failures"] += 1
        raise
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        duration = time.perf_counter() - start
        SIM_STATS["calls"] += 1
        SIM_STATS["seconds"] += duration
        if ok:
            _RECENT_DURATIONS.append(duration)


def sim_stats_line():
    s = SIM_STATS
    mean_ms = 1000.0 * s["seconds"] / max(s["calls"], 1)
    return (f"{s['calls']} simulations, {s['seconds']:.1f} s total, {mean_ms:.1f} ms mean, "
            f"{s['timeouts']} timed out, {s['failures']} failed")


def map_parallel(function, items, label):
    """
    Run function(item) for each item in worker processes (N_WORKERS) and return
    the results as a list in input order. A job that raises returns None (and
    is reported) without affecting the others. Falls back to serial if only one
    worker is available or the process pool itself cannot run.
    """
    items = list(items)

    def serial():
        out = []
        for item in items:
            try:
                out.append(function(item))
            except Exception as error:  # noqa: BLE001
                print(f"[{label}] job failed: {type(error).__name__}: {error}", flush=True)
                out.append(None)
        return out

    if N_WORKERS <= 1 or len(items) <= 1:
        return serial()
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    workers = min(N_WORKERS, len(items))
    print(f"[{label}] running {len(items)} jobs on {workers} worker processes", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(function, item) for item in items]
            out = []
            for future in futures:
                try:
                    out.append(future.result())
                except BrokenProcessPool:
                    raise
                except Exception as error:  # noqa: BLE001
                    print(f"[{label}] job failed: {type(error).__name__}: {error}", flush=True)
                    out.append(None)
            return out
    except (BrokenProcessPool, OSError, PermissionError) as error:
        print(f"[{label}] process pool unavailable ({error}); running serially", flush=True)
        return serial()


def run_parallel(function, items, label):
    """map_parallel keyed by item: returns {item: result}. Items must be hashable."""
    items = list(items)
    return dict(zip(items, map_parallel(function, items, label)))


def simulate_runs(model, runs, params):
    sims = {}
    for run_id, run in runs.items():
        sim = safe_simulate(model, run["conditions"], params)
        if len(sim) != len(run["times"]):
            raise RuntimeError(
                f"Simulator returned {len(sim)} rows for {len(run['times'])} times."
            )
        sims[run_id] = sim
    return sims


def predicted_at(sim, species, block):
    return np.maximum(sim[species].to_numpy(dtype=float)[block["time_index"]], 0.0)


def log_likelihood(model, runs, params):
    """
    Gaussian likelihood for quantified values, left-censored likelihood
    P(Q_obs < LOQ) = Phi((LOQ - mu)/sigma) for censored Q.
    Returns -inf if the simulation fails or is non-finite.
    """
    observed = set(CANDIDATES[model].observed)
    try:
        sims = simulate_runs(model, runs, params)
    except Exception:
        return -np.inf

    total = 0.0
    for run_id, run in runs.items():
        sim = sims[run_id]
        for name, block in run["blocks"].items():
            if name not in observed:
                continue
            mu = predicted_at(sim, name, block)
            if not np.all(np.isfinite(mu)):
                return -np.inf
            sig = sigma(name, mu)
            cens = block["censored"]
            if cens.any():
                total += float(np.sum(log_ndtr((Q_LOQ - mu[cens]) / sig[cens])))
            unc = ~cens
            if unc.any():
                z = (block["values"][unc] - mu[unc]) / sig[unc]
                total += float(np.sum(-0.5 * z ** 2 - np.log(sig[unc]) - LOG_SQRT_2PI))
    return total if np.isfinite(total) else -np.inf


def write_run_report(command, rows):
    """
    rows: list of dicts with stage, planned, completed, capped (bool), seconds.
    Writes scratch/system_B_run_report_<command>.csv and prints a reproducibility
    verdict: outputs are reproducible across machines only if no time cap was hit.
    """
    frame = pd.DataFrame(rows)
    path = SCRATCH_DIR / f"system_B_run_report_{command}.csv"
    frame.to_csv(path, index=False)
    capped = [r["stage"] for r in rows if r.get("capped")]
    print("\n" + "-" * 72, flush=True)
    for r in rows:
        flag = "  <-- TIME CAP HIT" if r.get("capped") else ""
        print(f"[{command}] {r['stage']:<34s} planned {str(r.get('planned', '')):>6s} | "
              f"done {str(r.get('completed', '')):>6s} | {r.get('seconds', float('nan')):6.1f} s{flag}",
              flush=True)
    if capped:
        print(f"[{command}] REPRODUCIBILITY: NOT guaranteed - time caps hit in: {', '.join(capped)}. "
              "Results depend on machine speed; raise those budgets.", flush=True)
    else:
        print(f"[{command}] REPRODUCIBILITY: every stage completed its planned work "
              "(same inputs and seeds give the same outputs).", flush=True)
    print("-" * 72, flush=True)
    return frame


def banner(title, char="="):
    print("\n" + char * 72, flush=True)
    print(title, flush=True)
    print(char * 72, flush=True)


def benchmark():
    """
    python3 baseline/pipeline_common.py
    Times one full likelihood evaluation (all System B runs) per model at a
    typical parameter set and at a stiff corner, so you can see how fast your
    simulator is and whether stiff regions are the problem.
    """
    runs = get_runs()
    t_max = max(r["conditions"].temperature_c for r in runs.values())
    print(f"{len(runs)} runs, {sum(len(r['times']) for r in runs.values())} sample times, "
          f"max T = {t_max:.0f} C, {N_WORKERS} worker processes available\n")
    print(f"{'model':6s} {'typical (s)':>12s} {'stiff corner (s)':>18s}")
    for model in ALL_MODELS:
        names = CANDIDATES[model].param_names
        typical = {n: (0.01 if is_rate(n) else 60.0) for n in names}
        stiff = {n: (K_BOUNDS[1] if is_rate(n) else EA_BOUNDS[1]) for n in names}
        first_k = next(n for n in names if is_rate(n))
        stiff[first_k] = 1e-3  # slow first step, fast everything else
        if "Ea" + first_k[1:] in stiff:
            stiff["Ea" + first_k[1:]] = 60.0
        timings = []
        for params in (typical, stiff):
            start = time.perf_counter()
            try:
                for run in runs.values():
                    safe_simulate(model, run["conditions"], params, timeout=30.0)
                timings.append(f"{time.perf_counter() - start:.3f}")
            except SimulationTimeout:
                timings.append("> 30 (timeout)")
            except Exception as error:  # noqa: BLE001
                timings.append(f"error: {type(error).__name__}")
        print(f"{model:6s} {timings[0]:>12s} {timings[1]:>18s}")
    fast = [m for m, ok in _FAST_SIM_OK.items() if ok]
    print(f"\nFast LSODA simulator (reference.simulate_fast) in use for: "
          f"{', '.join(fast) if fast else 'none (official simulate only)'}")
    print("One MCMC step costs (walkers x typical time). If 'typical' is above ~0.05 s,\n"
          "lower the budgets/sample counts in this file; if only 'stiff' is slow, the\n"
          "per-simulation timeout is what keeps the pipeline bounded.")


if __name__ == "__main__":
    benchmark()