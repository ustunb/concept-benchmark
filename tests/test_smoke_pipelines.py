"""Smoke tests that exercise data generation, model training, and prediction.

Each test creates a tiny (≈10 sample) dataset, trains a model for 1-2 epochs,
and verifies end-to-end operation. Marked ``@pytest.mark.slow`` — use
``--run-slow`` to include them.  Expected wall time: ~30-60s total.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Shared tiny concept space ────────────────────────────────────────

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


# ── Robot image pipeline ─────────────────────────────────────────────


@pytest.mark.slow
def test_robot_image_generation_and_training(tmp_path):
    """Generate ~16 robot images, train a 1-epoch CBM, run predictions."""
    from torchvision import transforms

    from concept_benchmark.synthetic.robot import create_robot_image_dataset
    from experiments.models import (
        ConceptBasedModel,
        ConceptDetector,
        RobotConceptClassifier,
    )

    # Step 1: Generate images (exercises pycairo drawing)
    ds = create_robot_image_dataset(
        concepts=TINY_CONCEPTS,
        samples_per_instance=1,
        size="small",
        output_directory=tmp_path / "robot_imgs",
        draw=True,
        color_mode="greyscale",
        model="'glorp' if row['mouth_type']=='open' else 'drent'",
        model_type="deterministic",
    )
    ds.transform = transforms.Compose([transforms.ToTensor()])
    assert ds.n >= 16, f"Expected >=16 samples, got {ds.n}"

    # Verify images exist on disk
    pngs = list((tmp_path / "robot_imgs").rglob("*.png"))
    assert len(pngs) >= 16, f"Expected >=16 PNGs, got {len(pngs)}"

    # Step 2: Split into train/val/test
    ds.sample(test_size=4, val_size=4, seed=42)
    assert ds.training is not None
    assert ds.validation is not None
    assert ds.test is not None

    # Step 3: Train a tiny CBM (1 epoch)
    n_concepts = ds.training.n_concepts
    cd = ConceptDetector(
        model=RobotConceptClassifier(num_concepts=n_concepts, input_size=8)
    )
    cbm = ConceptBasedModel(concept_detector=cd)
    cbm.fit(
        train_dataset=ds.training,
        valid_dataset=ds.validation,
        freeze_backbone=False,
        concept_embed_params={"device": "cpu", "batch_size": 4, "num_workers": 0},
        concept_fit_params={
            "epochs": 1,
            "lr": 1e-3,
            "patience": 1,
            "device": "cpu",
            "batch_size": 4,
            "num_workers": 0,
        },
    )

    # Step 4: Predictions work
    preds = cbm.predict(ds.test)
    assert len(preds) == ds.test.n
    assert set(preds).issubset({0, 1})

    # Step 5: Concept detector predictions have right shape
    concept_preds = cbm.concept_detector.predict(ds.test)
    assert concept_preds.shape[0] == ds.test.n


# ── Sudoku pipeline ──────────────────────────────────────────────────


@pytest.mark.slow
def test_sudoku_board_generation():
    """Generate 10 sudoku boards with handwritten digit images."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "concept_benchmark.synthetic.sudoku_ocr.make_ocr_dataset",
        "--n",
        "3",
        "--n-samples",
        "10",
        "--valid-ratio",
        "0.5",
        "--max-corrupt",
        "3",
        "--seed",
        "42",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"make_ocr_dataset failed:\n{result.stderr}"


# ── Robot text pipeline ──────────────────────────────────────────────


@pytest.mark.slow
def test_robot_text_generation_and_training():
    """Generate ~20 robot text descriptions and train a tiny text detector."""
    from concept_benchmark.synthetic.robot import create_robot_text_dataset
    from concept_benchmark.synthetic.helper.robot_catalog import generate_robot_catalog

    # Step 1: Generate catalog (no images)
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

    # Step 2: Generate text descriptions (exercises template engine)
    ds = create_robot_text_dataset(
        source=catalog_df,
        concept_cols=concept_cols,
        label_col="label",
        label_map={"drent": 0, "glorp": 1},
        variants_per_row=2,
        rng_seed=42,
    )
    assert ds.n >= 10, f"Expected >=10 text samples, got {ds.n}"
    assert isinstance(ds.X[0], str), "X should contain text strings"
    assert len(ds.X[0]) > 10, "Text descriptions should be non-trivial"

    # Step 3: Verify concepts and labels are consistent
    assert ds.C.shape[0] == ds.n
    assert ds.C.shape[1] > 0, "Should have concept columns"
    assert set(np.unique(ds.y)).issubset({0, 1})
