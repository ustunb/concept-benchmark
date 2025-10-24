
import numpy as np
import pytest

from concept_benchmark.kflip import KFlipInterventionStrategy
from concept_benchmark.intervention import InterventionBatch, InterventionConfig
from types import SimpleNamespace

class XORFrontEnd:
    """Binary XOR of first two bits for class. Returns soft probs around 0.6 for the predicted class."""
    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        y = (X[:, :2].sum(axis=1) % 2).astype(int)
        # confidence 0.6 on the predicted class
        out[np.arange(X.shape[0]), 1-y] = 0.4
        out[np.arange(X.shape[0]), y] = 0.6
        return out

class DummyCBM:
    def __init__(self, fe):
        self.front_end_model = fe
        self.concept_detector = None

def _run_kflip(P, k, threshold=0.0, instance_budget=None, select_only_abstained=False, tau=None, random_state=0, limit_subsets=None):
    P = np.asarray(P, dtype=np.float32)
    n = P.shape[0]
    # we never use C_true in KFlip.propose, but InterventionBatch requires shape
    batch = InterventionBatch(C_pred=P, C_true=np.zeros_like(P), y_true=None)
    cfg = InterventionConfig(
        max_concepts_per_instance=int(k),
        score_threshold=float(threshold),
        instance_budget=instance_budget,
        select_only_abstained=bool(select_only_abstained),
        tau=tau,
        random_state=random_state,
        shuffle_candidates=False,
    )
    strat = KFlipInterventionStrategy(batch_size=1024, limit_subsets=limit_subsets)
    cbm = DummyCBM(XORFrontEnd())
    prop = strat.propose(cbm, batch, cfg)
    return prop

def test_kflip_single_sample_k1_probability_and_subset():
    # One sample, two concepts. XOR frontend. Base Z = [1, 0] -> class 1
    P = np.array([[0.9, 0.1]], dtype=np.float32)
    prop = _run_kflip(P, k=1, threshold=0.0)
    # mask should select exactly one concept in the only selected instance
    assert prop.mask.shape == P.shape
    sel = prop.selected_instances
    assert sel is not None and sel.size == 1 and int(sel[0]) == 0
    chosen = np.nonzero(prop.mask[0])[0]
    assert chosen.size == 1
    # For XOR with base [1,0], flipping either c0->0 or c1->1 flips the label.
    # Prob(c0=0)=0.1, Prob(c1=1)=0.1, so best mass = 0.1
    flip_prob = float(prop.details["flip_prob"][0])
    assert pytest.approx(flip_prob, rel=0.0, abs=1e-6) == 0.1
    # Destination class should be 0
    assert int(prop.details["best_label"][0]) == 0

def test_kflip_k2_picks_high_mass_subset():
    # Three concepts. Majority >=2 votes -> class 1. Use XORFrontEnd on first two bits still ok:
    # To control label rule for 3 bits, we craft a FE that uses majority on first 3 bits.
    class Majority3FE:
        def predict_proba(self, X):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X[None, :]
            ones = (X[:, :3] > 0.5).sum(axis=1)
            y = (ones >= 2).astype(int)
            out = np.zeros((X.shape[0], 2), dtype=np.float32)
            out[np.arange(X.shape[0]), 1-y] = 0.4
            out[np.arange(X.shape[0]), y] = 0.6
            return out

    cbm = DummyCBM(Majority3FE())
    P = np.array([[0.6, 0.6, 0.1]], dtype=np.float32)  # base Z=[1,1,0] -> class 1
    batch = InterventionBatch(C_pred=P, C_true=np.zeros_like(P), y_true=None)
    cfg = InterventionConfig(max_concepts_per_instance=2, score_threshold=0.0, shuffle_candidates=False)
    strat = KFlipInterventionStrategy(batch_size=4096)
    prop = strat.propose(cbm, batch, cfg)
    # best subset should be {0,1} with flip mass 0.64
    bs = prop.details["best_subset"][0]
    assert set(bs) == {0,1}
    prob = float(prop.details["flip_prob"][0])
    assert pytest.approx(prob, rel=0.0, abs=1e-6) == 0.64

def test_kflip_threshold_and_instance_budget_selection_order_is_unsorted():
    # Three samples with different flip probabilities, budget selects first matching indices by default.
    P = np.array([
        [0.99, 0.99],  # low flip prob ~0.01
        [0.1, 0.5],    # high flip prob ~0.9
        [0.8, 0.2],    # medium flip prob ~0.2
    ], dtype=np.float32)
    prop = _run_kflip(P, k=1, threshold=0.15, instance_budget=1, random_state=0)
    # Both idx 1 and 2 exceed threshold. Without sorting, first index kept is 1.
    assert prop.selected_instances.size == 1
    assert int(prop.selected_instances[0]) == 1

def test_kflip_respects_global_concept_budget_and_per_instance_limit():
    # Two samples. Each has best subset of size 2. But concept_budget=1 caps total edits.
    class FE2:
        def predict_proba(self, X):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X[None, :]
            # Predict class 1 when at least one of first 2 bits is 1
            y = ((X[:, :2] > 0.5).any(axis=1)).astype(int)
            out = np.zeros((X.shape[0], 2), dtype=np.float32)
            out[np.arange(X.shape[0]), 1-y] = 0.4
            out[np.arange(X.shape[0]), y] = 0.6
            return out
    cbm = DummyCBM(FE2())
    P = np.array([[0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5]], dtype=np.float32)
    batch = InterventionBatch(C_pred=P, C_true=np.zeros_like(P), y_true=None)
    cfg = InterventionConfig(
        max_concepts_per_instance=2,
        score_threshold=0.0,
        concept_budget=1,         # total cap
        shuffle_candidates=False,
    )
    prop = KFlipInterventionStrategy(batch_size=4096).propose(cbm, batch, cfg)
    # Only one True in the entire mask due to concept_budget=1
    assert int(prop.mask.sum()) == 1

def test_kflip_select_only_abstained_filters_candidates():
    P = np.array([[0.9, 0.1]], dtype=np.float32)
    prop = _run_kflip(P, k=1, threshold=0.0, select_only_abstained=True, tau=0.2, random_state=0)
    # XORFrontEnd returns conf=0.6 -> inside [0.2,0.8] -> abstain -> selected
    assert prop.selected_instances is not None
    assert prop.selected_instances.size == 1 and int(prop.selected_instances[0]) == 0
