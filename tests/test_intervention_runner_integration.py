
import numpy as np
import pytest

from concept_benchmark.intervention import (
    ConceptInterventionRunner,
    InterventionBatch,
    InterventionConfig,
    RandomInterventionStrategy,
)

class ConstFE:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32)
    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        # Repeat the fixed distribution for each row
        return np.tile(self._probs[None, :], (X.shape[0], 1))

class DummyCBM:
    def __init__(self, fe):
        self.front_end_model = fe
        self.concept_detector = None
    # runner may call this for conceptual safeguards; define a trivial version anyway
    def _propagate_predict_proba_mc(self, concept_probs):
        return self.front_end_model.predict_proba((concept_probs >= 0.5).astype(np.float32))

def test_runner_random_strategy_shapes_and_mask_applied():
    # Simple sanity test of the runner without relying on conceptual safeguards specifics.
    P = np.array([[0.6, 0.4, 0.2],
                  [0.9, 0.1, 0.1]], dtype=np.float32)
    T = np.zeros_like(P, dtype=np.float32)
    y = np.array([0, 1], dtype=int)

    model = DummyCBM(ConstFE([0.6, 0.4]))
    runner = ConceptInterventionRunner(model)

    cfg = InterventionConfig(
        concept_budget=2,
        instance_budget=1,
        max_concepts_per_instance=2,
        random_state=0,
        shuffle_candidates=False,
    )

    # Use RandomInterventionStrategy just to exercise the path
    from concept_benchmark.intervention import RandomInterventionStrategy
    strat = RandomInterventionStrategy()

    res = runner.run(strat, cfg, dataset=None, concept_proba=P, concept_true=T, labels=y)
    assert res.mask.shape == P.shape
    assert res.C_intervened.shape == P.shape
    assert np.isfinite(res.y_prob_before).all() and np.isfinite(res.y_prob_after).all()

    # Budget logic: at most 2 concept edits in total and max 2 per single selected instance
    assert int(res.mask.sum()) <= 2
