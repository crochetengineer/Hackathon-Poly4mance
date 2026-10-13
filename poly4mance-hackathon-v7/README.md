# Reaction Mechanism Discovery Challenge

A student modelling challenge: infer which of six candidate reaction networks produced a
blind kinetic dataset, predict held-out conditions with calibrated intervals, and design a
confirmatory experimental campaign inside a materials budget.

**The brief is the handout: [`UCL_Hackathon_Brief.pdf`](UCL_Hackathon_Brief.pdf)** — read it
first. This README is the repo tour and reference (install, layout, the candidate library);
`data/README.md` is the data reference (schema, LOQ, censoring, noise model, budget). Then run `python getting_started.py`.

## Quickstart

```bash
just install     # deps (five pinned packages) + the local `candidates` package
just predict     # -> submission/predictions.csv
just suggest      # -> submission/campaign.yaml
just replicate    # -> submission/figures/*
```

(No `just`? The four recipes in `justfile` are one line each — run the `python ...` command.)

## Layout

```
data/
  system_A/    disclosed reference reaction (MECHANISM.md) + measurements.csv
  system_B/    the blind system -- measurements.csv (one species is never measured)
  heldout_conditions.csv   conditions to predict; README.md documents schema, LOQ, budget
src/candidates/   the six candidate models (M1-M6) and one simulate() interface
baseline/        a naive-but-working reference solution behind the four just recipes
getting_started.py   orientation: prints a tour, writes figures to scratch/
submission/      your four deliverables land here (a template ships)
tests/           sanity tests for the candidate library
```

## The candidate library

```python
from candidates import CANDIDATES, Conditions, simulate
cond = Conditions(temperature_c=75.0, A0=1.0, B0=1.0, times_min=[0, 60, 120, 240, 480])
# illustrative parameters only -- estimating the real ones is your job
df = simulate("M2", cond, {"k1":0.012,"Ea1":58,"k2":0.006,"Ea2":52,"k3":7e-4,"Ea3":66})
```

Rate constants are given at a **reference temperature of 75 °C** plus an activation energy
(kJ/mol); concentrations are mol/L, time is minutes. See `data/README.md` for the data schema,
the LOQ / censoring convention, and the campaign budget.

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml` (pinned in `requirements.txt`).
No network access is required at run time.
