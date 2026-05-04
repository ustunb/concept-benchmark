"""Tests for pipeline helper classes (InterventionSettings, FEOnProbs)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class TestInterventionSettings:
    def test_defaults(self):
        from scripts.robot_pipeline import InterventionSettings

        s = InterventionSettings(seed=42, budgets=[1, 3])
        assert s.intervention_accuracy == 0.9
        assert s.intervention_strategy == "up_to_k"
        assert s.intervention_threshold == 1.0
        assert s.intervention_expert == ""
        assert s.intervention_llm is None

    def test_all_fields(self):
        from scripts.robot_pipeline import InterventionSettings

        s = InterventionSettings(
            seed=1,
            budgets=[1, 2, 5],
            intervention_accuracy=0.8,
            intervention_threshold=0.5,
            intervention_strategy="exactly_k",
            intervention_expert="llm",
            intervention_llm={"provider": "gemini"},
            run_dir="/tmp/test",
        )
        assert s.seed == 1
        assert s.budgets == [1, 2, 5]
        assert s.intervention_accuracy == 0.8
        assert s.intervention_strategy == "exactly_k"


class TestFEOnProbs:
    def _make_fe(self, k=4, seed=42):
        from scripts.robot_pipeline import FEOnProbs

        rng = np.random.default_rng(seed)
        clf = LogisticRegression(random_state=42, max_iter=500)
        # Train on logit-transformed features (matching FEOnProbs usage)
        P = rng.random((50, k)).astype(np.float32)
        P = np.clip(P, 1e-6, 1 - 1e-6)
        Z = np.log(P / (1 - P))
        y = rng.integers(0, 2, size=50)
        clf.fit(Z, y)
        return FEOnProbs(clf), k

    def test_shape(self):
        fe, k = self._make_fe()
        P = np.random.default_rng(0).random((10, k)).astype(np.float32)
        proba = fe.predict_proba(P)
        assert proba.shape == (10, 2)

    def test_clips_extremes(self):
        fe, k = self._make_fe()
        # Edge cases: probabilities at 0 and 1
        P = np.zeros((3, k), dtype=np.float32)
        P[1, :] = 1.0
        P[2, :] = 0.5
        proba = fe.predict_proba(P)
        assert not np.any(np.isnan(proba))
        assert proba.shape == (3, 2)

    def test_fast_path_disabled(self):
        from scripts.robot_pipeline import FEOnProbs

        assert FEOnProbs._kflip_fast_path is True
