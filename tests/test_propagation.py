import itertools
import numpy as np

from concept_benchmark.models import ConceptBasedModel, ConceptDetector


def _naive_enumeration_aggregate(
    model: ConceptBasedModel, concept_probs: np.ndarray
) -> np.ndarray:
    """Compute aggregated label probabilities by explicit enumeration.

    Args:
        model: Trained ConceptBasedModel (front_end_model used here).
        concept_probs: Array of shape (N, C) with concept probabilities.

    Returns:
        Array of shape (N, K) with aggregated label probabilities.
    """
    N, C = concept_probs.shape
    out = []
    combos = list(itertools.product([0, 1], repeat=C))
    # Precompute front-end probabilities per combo for reuse
    combos_arr = np.array(combos, dtype=np.int32)
    y_combo = model.front_end_model.predict_proba(combos_arr)  # (M, K)

    for i in range(N):
        p = concept_probs[i]
        agg = np.zeros(y_combo.shape[1], dtype=np.float64)
        for j, c in enumerate(combos_arr):
            w = np.prod((p**c) * ((1.0 - p) ** (1 - c)))
            agg += y_combo[j] * w
        out.append(agg)
    return np.vstack(out)


def test_exact_matches_naive_on_small_k(tabular_train_valid):
    # Small number of concepts so exact enumeration is feasible
    train, valid, d, k = tabular_train_valid

    model = ConceptBasedModel(
        concept_detector=ConceptDetector(embedding_model=None),
        propagate=True,
        mc_mode="exact",
        mc_exact_threshold=4096,
    )

    model.fit(
        train_dataset=train,
        valid_dataset=valid,
        freeze=False,
        concept_fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
        front_fit_params={"max_iter": 200},
        calibrate=False,
    )

    # Exact propagation via model
    proba_exact = model.predict_proba(valid, propagate=True)
    assert proba_exact.shape[0] == len(valid)
    assert np.all(np.isfinite(proba_exact))

    # Naive explicit enumeration aggregation
    concept_probs = model.concept_detector.predict(valid, calibrate=False)
    proba_naive = _naive_enumeration_aggregate(model, concept_probs)

    # Should match very closely (up to tiny numeric differences)
    assert np.allclose(proba_exact, proba_naive, atol=1e-9, rtol=1e-5)


def test_mc_matches_exact_on_small_k(tabular_train_valid):
    # Use small number of concepts so exact enumeration is feasible
    train, valid, d, k = tabular_train_valid

    # Build model with propagation enabled; no embedding model for speed
    model = ConceptBasedModel(
        concept_detector=ConceptDetector(embedding_model=None),
        propagate=True,
        mc_mode="exact",  # start with exact
        mc_exact_threshold=4096,  # default; 2**k will be small here
    )

    # Train concept detector (few epochs) and front-end
    model.fit(
        train_dataset=train,
        valid_dataset=valid,
        freeze=False,
        concept_fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
        front_fit_params={"max_iter": 200},
        calibrate=False,
    )

    # Exact propagation probabilities
    proba_exact = model.predict_proba(valid, propagate=True)
    assert proba_exact.shape[0] == len(valid)
    assert np.all(np.isfinite(proba_exact))

    # Switch to Monte Carlo mode with a fixed seed and large sample budget
    model._mc_mode = "mc"
    model._random_state = 123
    model._mc_samples = 4096
    model._mc_max_samples = 4096
    model._mc_chunk_size = 1024
    model._mc_tol = 0.0  # force using full sample budget

    proba_mc = model.predict_proba(valid, propagate=True)

    assert proba_mc.shape == proba_exact.shape
    assert np.all(np.isfinite(proba_mc))

    # Monte Carlo estimate should be close to exact enumeration
    # Allow small deviation due to sampling variance
    assert np.allclose(proba_mc, proba_exact, atol=2e-2, rtol=0.0)
