"""candidates: the closed candidate library for Reaction Mechanism Discovery Challenge.

Exposes the six candidate reaction networks (M1-M6) and one shared ``simulate`` interface.
Pure numpy/scipy; no true parameter values and nothing that touches the truth-side toolchain.
"""
from candidates.models import CANDIDATES, MODEL_NAMES, Candidate, arrhenius
from candidates.simulate import Conditions, simulate

__all__ = ["CANDIDATES", "MODEL_NAMES", "Candidate", "arrhenius", "Conditions", "simulate"]
