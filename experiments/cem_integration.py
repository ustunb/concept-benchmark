"""Backwards-compatible re-export shim.

All implementations have moved to ``experiments.baselines``:

- ``experiments.baselines.cem``      — CEM wrapper and training
- ``experiments.baselines.probcbm``  — ProbCBM wrapper and training
- ``experiments.baselines.ecbm``     — ECBM implementation and training
- ``experiments.baselines._common``  — shared utilities, backbones, base class
"""
from experiments.baselines import *  # noqa: F401, F403
from experiments.baselines._common import (  # noqa: F401 — private names used by tests
    _CEMDependencies,
    _OfficialBenchmarkModelBase,
    _PredictionCache,
    _ensure_local_cem_checkout_on_path,
)
