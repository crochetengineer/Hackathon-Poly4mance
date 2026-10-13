"""The single simulation interface shared by all six candidates.

Pure scipy: this is the closed library the challenge is built on. It exposes the models and
the ``simulate`` call, and nothing about the true parameter values.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from candidates.models import CANDIDATES, Candidate


@dataclass(frozen=True)
class Conditions:
    """A single isothermal batch experiment.

    temperature_c : held constant for the whole run [degC]
    A0, B0        : initial concentrations [mol/L] (P, Q, I all start at 0)
    times_min     : sample times [min]; the first should be 0.0
    """

    temperature_c: float
    A0: float
    B0: float
    times_min: Sequence[float] = field(default_factory=lambda: (0.0,))


def simulate(model: str, conditions: Conditions, params: Mapping[str, float]) -> pd.DataFrame:
    """Integrate ``model`` at ``conditions`` with ``params`` and return concentrations.

    Returns a wide DataFrame: a ``time_min`` column plus one column per species in the model
    (mol/L), sampled at ``conditions.times_min``. All six models share this signature.

    Tolerances are set well below the 3% measurement noise (rtol 1e-9); the integrator is the
    accuracy bottleneck for nothing here.
    """
    if model not in CANDIDATES:
        raise KeyError(f"unknown model {model!r}; choose from {list(CANDIDATES)}")
    cand: Candidate = CANDIDATES[model]

    y0 = np.array([conditions.A0 if s == "A" else conditions.B0 if s == "B" else 0.0
                   for s in cand.species], dtype=float)
    times = np.asarray(conditions.times_min, dtype=float)
    t_end = float(times.max())

    def fun(_t, y):
        return cand.rhs(y, params, conditions.temperature_c)

    sol = solve_ivp(fun, (0.0, t_end), y0, t_eval=times, method="LSODA",
                    rtol=1e-9, atol=1e-12, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"integration failed for {model}: {sol.message}")

    out = {"time_min": times}
    for i, s in enumerate(cand.species):
        out[s] = np.clip(sol.y[i], 0.0, None)  # concentrations are non-negative
    return pd.DataFrame(out)
