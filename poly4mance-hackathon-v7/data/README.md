# Data — Reaction Mechanism Discovery Challenge

Two reaction systems. **System A** is a disclosed worked reference
(`system_A/MECHANISM.md`); **System B** is the blind MT-419 step you are scored on.

## File schema (long format)

Every measurement file has exactly these columns, one measurement per row:

| column | meaning |
|--------|---------|
| `system` | `A` or `B` |
| `run_id` | experiment identifier (e.g. `B3`) |
| `temperature_C` | isothermal run temperature [°C] |
| `A0_M` | initial concentration of A [mol/L] |
| `B0_M` | initial concentration of B [mol/L] |
| `time_min` | sample time [min] |
| `species` | one of `A`, `B`, `P`, `Q` |
| `value` | concentration [mol/L], or the string `<LOQ` if censored |
| `unit` | always `mol/L` |
| `censored` | `true` / `false` |

No wide format, no per-run files with different columns. Files:

- `system_A/measurements.csv` — A, B, P, Q, all measured, no censoring.
- `system_B/measurements.csv` — the blind system (see below).
- `heldout_conditions.csv` — conditions only (`run_id, temperature_C, A0_M, B0_M, time_min`);
  predict A, B, P, Q at each row. Held-out conditions are not confined to the region covered by
  the released runs; some lie outside it. Intervals sized for interpolation alone will not be
  well calibrated.

**A missing row is not a censored row.** Q is assayed at only a subset of timepoints (see
below), so at un-assayed times there is simply *no Q row* — a gap, not a measurement. A
censored Q is different: the row **is** present, with `value = <LOQ` and `censored = true`.
Do not read an absent Q as `<LOQ`, and do not read a `<LOQ` as zero.

## System B — what is (and isn't) there

- **At least one species is never measured.** It is not a column and not a row. Infer that it
  exists, and what it does, from its effect on the species you can see.
- **Concentrations** are reported for A, B, P, Q in mol/L.
- **Noise model (use this exact form for weighted fitting and interval calibration).** Each
  measurement has standard deviation `sigma = sqrt((rel*c)^2 + abs^2)` — a relative part plus
  an absolute floor, so near-zero concentrations are *not* known to arbitrary precision:

  | species | rel | abs [mol/L] |
  |---------|-----|-------------|
  | A, B, P | 0.03 | 3.5e-3 |
  | Q       | 0.15 | 2.2e-5 |

  The two absolute floors reflect **two different assays**: A, B, P are quantified by UPLC-UV
  against an internal standard, whose repeatability floor at these concentrations (0.1–1 M) is
  ~3.5 mM; Q is quantified by a separate, far more sensitive UPLC-MS trace method with an
  absolute floor of ~2.2e-5 mol/L, i.e. near its own limit of quantitation. The ~150×
  difference is method sensitivity, not an inconsistency.

  For the main components the two terms are **equal when `rel*c = abs`, i.e. at c = 3.5e-3 /
  0.03 = 0.117 M**: above 0.117 M the 3 % relative part dominates; below it the 3.5 mM absolute
  floor does. That crossover is why a model that is off by ~0.1 mol/L on P at low conversion
  reads as ~25σ rather than ~5σ — weight your residuals accordingly.
- **Sparse Q:** the trace genotoxic assay is expensive, so **Q is sampled at fewer timepoints
  than A, B, P** (5 of the ~10 times in each run). You will have Q at only some of the times.
- **Limit of quantitation (LOQ):** Q measurements below **3.0e-5 mol/L** are reported as
  `<LOQ` with `censored = true`. A censored value is left-censored — the true value lies in
  `[0, LOQ)`, not zero. Treat it as information, not a gap.
- The released runs vary **temperature only** (single initial charge).
- All five historical runs were charged at A:B ≈ 4:1, [A]₀ = 1.0 M — the ratio carried over
  from the discovery route.

## The impurity specification

Q is genotoxic. The specification is a hard limit, not a yield trade-off:

```
Q  <=  15 ppm w/w relative to P  in the isolated API
```

## Campaign budget (for any experiments you propose)

You may request further experiments, run as a single block, within:

| limit | value |
|-------|-------|
| total A consumed | **5.0 g** |
| experiments | **3–10** |
| analytical samples (timepoints) | **60** |
| vessel | 5 mL, single temperature per run |
| temperature | 50–120 °C |
| [A]₀, [B]₀ | 0.1–1.5 M, set independently |
| duration | up to 16 h (960 min) |

Cost model: an experiment consumes `5 mL/1000 · [A]₀ · 250 g/mol` of A (A has MW 250 g/mol),
so a run at 1.0 M costs 1.25 g and a run at 0.25 M costs 0.31 g. One analytical sample is one
timepoint, quantified for all observable species. The campaign scorer uses this same cost model.
