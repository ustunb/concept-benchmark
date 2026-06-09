"""Shared enums and type aliases for concept_benchmark."""

from __future__ import annotations

from enum import StrEnum


class CBMTrainingMode(StrEnum):
    """How a CBM front-end (label predictor) is trained relative to the backend.

    Independent: front-end fits on ground-truth concepts ``C``. Composed CBM
    evaluates on backend-predicted concepts at inference time.

    Sequential: front-end fits on backend-predicted concepts. Both training
    and evaluation flow through the backend.
    """

    Independent = "independent"
    Sequential = "sequential"
