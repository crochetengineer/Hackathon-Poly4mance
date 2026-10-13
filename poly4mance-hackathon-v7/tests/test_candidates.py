"""Student-side sanity tests for the closed candidate library. Pure scipy, no truth-side tools."""
import numpy as np

from candidates import CANDIDATES, MODEL_NAMES, Conditions, arrhenius, simulate


def test_all_models_present():
    assert set(MODEL_NAMES) == {"M1", "M2", "M3", "M4", "M5", "M6"}


def test_simulate_shapes_and_species():
    cond = Conditions(75.0, 1.0, 1.0, [0, 30, 60, 120, 240])
    p = {"k1": 0.01, "Ea1": 60, "k2": 0.005, "Ea2": 60, "k3": 1e-4, "Ea3": 60}
    for m, cand in CANDIDATES.items():
        df = simulate(m, cond, {k: p[k] for k in cand.param_names})
        assert list(df["time_min"]) == [0, 30, 60, 120, 240]
        for sp in cand.species:
            assert sp in df.columns
            assert (df[sp] >= -1e-9).all()
        assert ("I" in cand.species) == (m in {"M2", "M4", "M6"})


def test_arrhenius_reference():
    assert abs(arrhenius(0.05, 65.0, 75.0) - 0.05) < 1e-12
    assert arrhenius(0.05, 65.0, 85.0) > 0.05


def test_reactant_consumption_and_product_formation():
    cond = Conditions(85.0, 1.0, 1.0, np.linspace(0, 400, 20))
    p = {"k1": 0.05, "Ea1": 60, "k2": 0.01, "Ea2": 60, "k3": 0.007, "Ea3": 60}
    for m, cand in CANDIDATES.items():
        df = simulate(m, cond, {k: p[k] for k in cand.param_names})
        assert df["A"].iloc[-1] <= df["A"].iloc[0] + 1e-9
        assert df["P"].iloc[-1] >= df["P"].iloc[0] - 1e-9
