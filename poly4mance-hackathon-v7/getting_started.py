"""Start here: load the data, meet the candidate library, see the four deliverables.

Run it from anywhere in the repo:

    python getting_started.py

It prints a tour of what you have been given and writes two orientation figures to
`scratch/`. Nothing here touches a solution -- you have the six candidate ODE models and
the released data, and that is all you need.

Run `just install` first so that `from candidates import ...` resolves.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs; no display needed on the lab machines

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SCRATCH = ROOT / "scratch"


def load_measurements(path: Path) -> pd.DataFrame:
    """Read a measurements file, splitting `value` into a float and a censoring flag.

    `value` arrives as either a number or the literal string `<LOQ`, so the raw column is
    object dtype and `astype(float)` on it will raise. `conc` is NaN for censored rows --
    censored means the true value lies in [0, LOQ), which is information, not a missing
    number, so do not fill it with zero when you come to fit.
    """
    df = pd.read_csv(path)
    df["censored"] = df["censored"].astype(str).str.lower().eq("true")
    df["conc"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def describe_system_b(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("System B -- the blind system, the one you are scored on")
    print("=" * 72)

    design = df.drop_duplicates("run_id")[["run_id", "temperature_C", "A0_M", "B0_M"]]
    design = design.assign(ratio=lambda d: (d.A0_M / d.B0_M).round(2)).set_index("run_id")
    print("\nreleased design:")
    print(design.to_string())

    print(f"\nspecies present: {sorted(df.species.unique())}")
    print(f"rows: {len(df)}   runs: {df.run_id.nunique()}")

    counts = df.pivot_table(index="species", columns="run_id", values="time_min", aggfunc="count")
    print("\nmeasurements per species per run:")
    print(counts.fillna(0).astype(int).to_string())
    print("\nQ has fewer rows than A/B/P: the trace assay is run on a subset of timepoints.")
    print("An absent Q row is a GAP. A present row with value '<LOQ' is a CENSORED")
    print("measurement -- the true value is in [0, LOQ). They are not the same thing.")

    q = df[df.species == "Q"]
    print(f"\nQ rows: {len(q)}   of which censored (<LOQ): {int(q.censored.sum())}")
    if (~q.censored).any():
        print(f"smallest quantified Q: {q.loc[~q.censored, 'conc'].min():.3e} mol/L")


def describe_candidates() -> None:
    from candidates import CANDIDATES  # after `just install`, or add src/ to PYTHONPATH

    print("\n" + "=" * 72)
    print("The six candidates")
    print("=" * 72)
    for name, cand in CANDIDATES.items():
        hidden = "I" in cand.species
        print(f"{name}: species={cand.species}  params={cand.param_names}"
              f"{'   <- has an unobserved intermediate' if hidden else ''}")
    print("\nEvery candidate shares one interface: simulate(model, conditions, params).")
    print("Rate constants are k at the 75 C reference temperature plus an activation")
    print("energy Ea in kJ/mol. Estimating the real values is your job.")


def plot_orientation(system_a: pd.DataFrame, system_b: pd.DataFrame) -> list[Path]:
    """Two figures: the released System B profiles, and one simulated candidate."""
    from candidates import Conditions, simulate

    SCRATCH.mkdir(exist_ok=True)
    written: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for sp, ax in zip(["A", "P", "Q"], axes):
        for run_id, g in system_b[system_b.species == sp].groupby("run_id"):
            g = g.dropna(subset=["conc"])
            ax.plot(g.time_min, g.conc, "o-", ms=3, label=run_id)
        ax.set_title(f"System B: {sp}")
        ax.set_xlabel("time [min]")
        ax.set_ylabel(f"[{sp}] mol/L")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=7, title="run")
    fig.tight_layout()
    out = SCRATCH / "system_B_released.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out)

    # Illustrative parameters only -- these are NOT the values that generated the data.
    cond = Conditions(temperature_c=75.0, A0=1.0, B0=0.25, times_min=np.linspace(0, 720, 120))
    params = {"k1": 0.011, "Ea1": 58.0, "k2": 0.007, "Ea2": 52.0, "k3": 8e-4, "Ea3": 66.0}
    sim = simulate("M2", cond, params)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for sp in [c for c in ["A", "B", "I", "P"] if c in sim.columns]:
        axes[0].plot(sim.time_min, sim[sp], label=sp)
    axes[0].set_title("M2 simulated (illustrative params)")
    axes[0].legend(fontsize=8)
    axes[1].plot(sim.time_min, sim["Q"], color="crimson")
    axes[1].set_title("M2 simulated: Q")
    for ax in axes:
        ax.set_xlabel("time [min]")
        ax.set_ylabel("mol/L")
    fig.tight_layout()
    out = SCRATCH / "M2_simulated.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out)

    if not system_a.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for sp, ax in zip(["P", "Q"], axes):
            for run_id, g in system_a[system_a.species == sp].groupby("run_id"):
                ax.plot(g.time_min, g.conc, "o-", ms=3, label=run_id)
            ax.set_title(f"System A: {sp}")
            ax.set_xlabel("time [min]")
            ax.set_ylabel(f"[{sp}] mol/L")
        axes[0].legend(fontsize=7, title="run")
        fig.tight_layout()
        out = SCRATCH / "system_A_reference.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)

    return written


def print_deliverables() -> None:
    print("\n" + "=" * 72)
    print("What you have to hand in")
    print("=" * 72)
    print("1. submission/hypotheses.yaml   the six candidates ranked, with weights")
    print("2. just predict  -> submission/predictions.csv   point + 90% interval,")
    print("                    every species, every held-out row")
    print("3. just suggest  -> submission/campaign.yaml     the experiments you would")
    print("                    run next, inside the budget in the handout")
    print("4. a presentation, every figure of which comes out of `just replicate`")
    print("\nThe brief is UCL_Hackathon_Brief.pdf -- read it before you write any code.")
    print("data/README.md has the schema, the noise model, the LOQ and censoring")
    print("convention, and the campaign budget.")
    print("\nbaseline/ already runs all four recipes, badly, so you never lose an evening")
    print("to plumbing. Beat it with reasoning.")


def main() -> None:
    system_b = load_measurements(DATA / "system_B" / "measurements.csv")
    system_a = load_measurements(DATA / "system_A" / "measurements.csv")

    describe_system_b(system_b)
    describe_candidates()

    print("\nSystem A is the disclosed reference (data/system_A/MECHANISM.md). Validate your")
    print("whole workflow there first. If it cannot recover System A, it will not recover")
    print("System B.")

    for path in plot_orientation(system_a, system_b):
        print(f"wrote {path.relative_to(ROOT)}")

    print_deliverables()


if __name__ == "__main__":
    main()
