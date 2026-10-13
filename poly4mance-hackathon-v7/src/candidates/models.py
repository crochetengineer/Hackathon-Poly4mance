"""The six candidate reaction networks for Reaction Mechanism Discovery Challenge.

Why this module exists: students choose between exactly these six named mechanisms, so the
networks are fixed and closed. Each is an ODE right-hand side in **molar concentration
(mol/L)** with **time in minutes**. Rate constants are Arrhenius, parameterised by the rate
constant at a reference temperature of 75 degC plus an activation energy (kJ/mol) -- not by a
pre-exponential factor, because the pre-exponential is badly scaled and wrecks the fit.

Stoichiometry and rate laws are taken verbatim from the brief:

    M1  A + B -> P ; A + B -> Q          impurity direct from reactants, parallel to product
    M2  A + B -> I ; I -> P ; I -> Q     reactive intermediate branches to product or impurity
    M3  A + B -> P ; P -> Q              impurity by degradation of product, in series
    M4  A + B -> I ; I -> P ; I + A -> Q intermediate; second-order impurity route consumes A
    M5  A + B -> P ; A + P -> Q          product reacts further with A to give impurity
    M6  A -> I ; I + B -> P ; I -> Q     A activates first; activated species branches

Species labels are A, B, P, Q, I. I is a genuine chemical intermediate in M2/M4/M6; it is a
modelled state here even though it is never *observed* in the System B data.

Every reaction is elementary: its rate law order equals its stoichiometry. All rate
constants use the ``k@75C`` + ``Ea`` convention below.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

R_GAS = 8.314          # J/mol/K
T_REF_C = 75.0         # reference temperature for the k_ref parameterisation
T_REF_K = T_REF_C + 273.15


def arrhenius(k_ref: float, ea_kj: float, temperature_c: float) -> float:
    """k(T) from its value at 75 degC and an activation energy.

    Reference-temperature form (well conditioned): only the *ratio* of temperatures enters,
    so k_ref carries the units (1/min for first order, L/mol/min for second order) and Ea is
    in kJ/mol. Equivalent to a pre-exponential Arrhenius law but far better scaled for fitting.
    """
    t_k = temperature_c + 273.15
    return k_ref * np.exp(-(ea_kj * 1000.0 / R_GAS) * (1.0 / t_k - 1.0 / T_REF_K))


@dataclass(frozen=True)
class Candidate:
    """A named reaction network: its species, its fit parameters, and its ODE RHS."""

    name: str
    species: list[str]                       # state order for the ODE
    observed: list[str]                       # species a detector could see (excludes I)
    param_names: list[str]                    # (k, Ea) pair per reaction
    rhs: Callable[[np.ndarray, dict[str, float]], np.ndarray] = field(repr=False)

    @property
    def latent(self) -> list[str]:
        return [s for s in self.species if s not in self.observed]


# --- rate-constant helpers: unpack (k_ref, Ea) params at a given temperature ---------------
def _rate_consts(params: Mapping[str, float], temperature_c: float, n: int) -> list[float]:
    """Return [k1(T), ..., kn(T)] from params {'k1','Ea1',...,'kn','Ean'}."""
    return [arrhenius(params[f"k{i}"], params[f"Ea{i}"], temperature_c) for i in range(1, n + 1)]


# --- the six right-hand sides. State vectors follow Candidate.species order. ----------------
def _rhs_m1(y, k):  # species A,B,P,Q ; k=[k1,k2]
    A, B, P, Q = y
    r1, r2 = k[0] * A * B, k[1] * A * B
    return np.array([-(r1 + r2), -(r1 + r2), r1, r2])


def _rhs_m2(y, k):  # A,B,I,P,Q ; k1:A+B->I, k2:I->P, k3:I->Q
    A, B, I, P, Q = y
    r1, r2, r3 = k[0] * A * B, k[1] * I, k[2] * I
    return np.array([-r1, -r1, r1 - r2 - r3, r2, r3])


def _rhs_m3(y, k):  # A,B,P,Q ; k1:A+B->P, k2:P->Q
    A, B, P, Q = y
    r1, r2 = k[0] * A * B, k[1] * P
    return np.array([-r1, -r1, r1 - r2, r2])


def _rhs_m4(y, k):  # A,B,I,P,Q ; k1:A+B->I, k2:I->P, k3:I+A->Q
    A, B, I, P, Q = y
    r1, r2, r3 = k[0] * A * B, k[1] * I, k[2] * I * A
    return np.array([-r1 - r3, -r1, r1 - r2 - r3, r2, r3])


def _rhs_m5(y, k):  # A,B,P,Q ; k1:A+B->P, k2:A+P->Q
    A, B, P, Q = y
    r1, r2 = k[0] * A * B, k[1] * A * P
    return np.array([-r1 - r2, -r1, r1 - r2, r2])


def _rhs_m6(y, k):  # A,B,I,P,Q ; k1:A->I, k2:I+B->P, k3:I->Q
    A, B, I, P, Q = y
    r1, r2, r3 = k[0] * A, k[1] * I * B, k[2] * I
    return np.array([-r1, -r2, r1 - r2 - r3, r2, r3])


def _make(name, species, observed, n_rxn, rhs_raw) -> Candidate:
    param_names = [p for i in range(1, n_rxn + 1) for p in (f"k{i}", f"Ea{i}")]

    def rhs(y, params, temperature_c, _rhs_raw=rhs_raw, _n=n_rxn):
        return _rhs_raw(y, _rate_consts(params, temperature_c, _n))

    return Candidate(name=name, species=species, observed=observed,
                     param_names=param_names, rhs=rhs)


CANDIDATES: dict[str, Candidate] = {
    "M1": _make("M1", ["A", "B", "P", "Q"], ["A", "B", "P", "Q"], 2, _rhs_m1),
    "M2": _make("M2", ["A", "B", "I", "P", "Q"], ["A", "B", "P", "Q"], 3, _rhs_m2),
    "M3": _make("M3", ["A", "B", "P", "Q"], ["A", "B", "P", "Q"], 2, _rhs_m3),
    "M4": _make("M4", ["A", "B", "I", "P", "Q"], ["A", "B", "P", "Q"], 3, _rhs_m4),
    "M5": _make("M5", ["A", "B", "P", "Q"], ["A", "B", "P", "Q"], 2, _rhs_m5),
    "M6": _make("M6", ["A", "B", "I", "P", "Q"], ["A", "B", "P", "Q"], 3, _rhs_m6),
}

MODEL_NAMES = list(CANDIDATES.keys())
