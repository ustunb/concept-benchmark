"""Determinism tests: same seed must produce identical results.

Each test runs a pipeline component twice with the same seed and verifies
that the outputs are bit-for-bit identical.  Marked ``@pytest.mark.slow``
because they involve data generation and/or model training.
"""
from __future__ import annotations

import numpy as np
import pytest

SEED = 777

# ── Sudoku data generation ─────────────────────────────────────────


@pytest.mark.slow
def test_sudoku_data_deterministic():
    """Same seed produces identical sudoku boards and concepts."""
    from concept_benchmark.synthetic.sudoku import create_sudoku_dataset

    kwargs = dict(
        n=3,
        n_samples=20,
        valid_ratio=0.5,
        max_corrupt=3,
        data_type="tabular",
        seed=SEED,
    )

    ds1 = create_sudoku_dataset(**kwargs)
    ds2 = create_sudoku_dataset(**kwargs)

    np.testing.assert_array_equal(ds1.X, ds2.X)
    np.testing.assert_array_equal(ds1.C, ds2.C)
    np.testing.assert_array_equal(ds1.y, ds2.y)


# ── Robot image generation ─────────────────────────────────────────


TINY_CONCEPTS = {
    "head_shape": ["square", "round"],
    "body_shape": ["square", "round"],
    "has_knees": ["false", "true"],
    "has_elbows": ["false", "true"],
    "has_antennae": ["false", "true"],
    "ears_shape": ["square", "triangle"],
    "mouth_type": ["closed", "open"],
    "hand_shape": ["round_circle", "edgy_square"],
    "foot_shape": ["flat_trapezoid", "pointy_4sided"],
}


@pytest.mark.slow
def test_robot_image_data_deterministic(tmp_path):
    """Same seed produces identical robot image datasets."""
    from torchvision import transforms
    from concept_benchmark.synthetic.robot import create_robot_image_dataset

    def _gen(subdir):
        ds = create_robot_image_dataset(
            concepts=TINY_CONCEPTS,
            samples_per_instance=1,
            size="small",
            output_directory=tmp_path / subdir,
            draw=True,
            color_mode="greyscale",
            model="'glorp' if row['mouth_type']=='open' else 'drent'",
            model_type="deterministic",
            seed=SEED,
        )
        ds.transform = transforms.Compose([transforms.ToTensor()])
        return ds

    ds1 = _gen("run1")
    ds2 = _gen("run2")

    np.testing.assert_array_equal(ds1.C, ds2.C)
    np.testing.assert_array_equal(ds1.y, ds2.y)


# ── Robot text generation ──────────────────────────────────────────


@pytest.mark.slow
def test_robot_text_data_deterministic():
    """Same seed produces identical robot text datasets."""
    from concept_benchmark.synthetic.robot import create_robot_text_dataset
    from concept_benchmark.synthetic.helper.robot_catalog import generate_robot_catalog

    def _gen():
        catalog_df, _ = generate_robot_catalog(
            concepts=TINY_CONCEPTS,
            num_robots=10,
            resolution=8,
            output_directory="/tmp/unused",
            draw=False,
            color_mode="greyscale",
        )
        catalog_df["label"] = "glorp"
        catalog_df.loc[catalog_df.index[:5], "label"] = "drent"
        concept_cols = [c for c in TINY_CONCEPTS if c in catalog_df.columns]
        return create_robot_text_dataset(
            source=catalog_df,
            concept_cols=concept_cols,
            label_col="label",
            label_map={"drent": 0, "glorp": 1},
            variants_per_row=2,
            rng_seed=SEED,
        )

    ds1 = _gen()
    ds2 = _gen()

    np.testing.assert_array_equal(ds1.X, ds2.X)
    np.testing.assert_array_equal(ds1.C, ds2.C)
    np.testing.assert_array_equal(ds1.y, ds2.y)


# ── CBM training determinism ──────────────────────────────────────


@pytest.mark.slow
def test_cbm_training_deterministic():
    """Same seed + same data → identical model predictions."""
    from concept_benchmark.data import ConceptDataset
    from concept_benchmark.utils import set_deterministic_seed
    from experiments.models import ConceptBasedModel, ConceptDetector

    rng = np.random.default_rng(SEED)
    n, k = 40, 4
    X = rng.random((n, 8)).astype(np.float32)
    C = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y = np.tile([0, 1], n // 2 + 1)[:n].astype(np.int32)
    rng.shuffle(y)
    meta = {
        "classes": ["c0", "c1"],
        "concepts": [f"z{i}" for i in range(k)],
        "data_type": "tabular",
    }

    fit_kwargs = dict(
        freeze_backbone=False,
        concept_embed_params={"device": "cpu", "batch_size": 8, "num_workers": 0},
        concept_fit_params={
            "epochs": 3,
            "lr": 1e-3,
            "patience": 2,
            "device": "cpu",
            "batch_size": 8,
            "num_workers": 0,
        },
    )

    def _train():
        set_deterministic_seed(SEED)
        ds = ConceptDataset(X=X.copy(), C=C.copy(), y=y.copy(), meta=meta)
        ds.sample(test_size=0.2, val_size=0.2, stratify=ds.y, seed=SEED)
        cd = ConceptDetector()
        cbm = ConceptBasedModel(concept_detector=cd)
        cbm.fit(
            train_dataset=ds.training,
            valid_dataset=ds.validation,
            **fit_kwargs,
        )
        return cbm.predict(ds.test), cbm.concept_detector.predict(ds.test)

    preds1, concepts1 = _train()
    preds2, concepts2 = _train()

    np.testing.assert_array_equal(preds1, preds2)
    np.testing.assert_allclose(concepts1, concepts2, atol=1e-6)


# ── Intervention determinism ──────────────────────────────────────


@pytest.mark.slow
def test_intervention_deterministic():
    """Same seed → identical intervention masks and post-intervention predictions."""
    from concept_benchmark.data import ConceptDatasetSample
    from experiments.intervention import (
        ConceptInterventionRunner,
        InterventionConfig,
        RandomInterventionStrategy,
    )
    from experiments.models import ConceptBasedModel, FrontEndModel

    n, k = 30, 5
    rng = np.random.default_rng(SEED)
    X = rng.random((n, 8)).astype(np.float32)
    C = rng.random((n, k)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    meta = {
        "classes": ["c0", "c1"],
        "concepts": [f"z{i}" for i in range(k)],
        "data_type": "tabular",
    }

    def _run():
        sample = ConceptDatasetSample(X=X.copy(), C=C.copy(), y=y.copy(), meta=meta)
        fe = FrontEndModel()
        fe.fit((C > 0.5).astype(float), y)
        model = ConceptBasedModel(label_predictor=fe)
        runner = ConceptInterventionRunner(model)
        config = InterventionConfig(max_concepts_per_instance=2, random_state=SEED)
        strat = RandomInterventionStrategy()
        return runner.run(strat, config, sample, concept_proba=C.copy())

    r1 = _run()
    r2 = _run()

    np.testing.assert_array_equal(r1.mask, r2.mask)
    np.testing.assert_array_equal(r1.y_pred_after, r2.y_pred_after)
    np.testing.assert_allclose(r1.y_prob_after, r2.y_prob_after, atol=1e-6)
    np.testing.assert_allclose(r1.C_intervened, r2.C_intervened, atol=1e-6)
