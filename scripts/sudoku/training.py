"""Sudoku pipeline — dataset setup and model training."""
from __future__ import annotations

import copy
import logging
import platform

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from concept_benchmark.utils import (
    compute_accuracy,
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
    set_deterministic_seed,
)
from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.ext.fileutils import load, save
from experiments.models import (
    ConceptBasedModel,
    ConceptDetector,
)

logger = logging.getLogger(__name__)


# ── Stage: setup_dataset ──────────────────────────────────────────────

def setup_dataset(config: SudokuBenchmarkConfig) -> None:
    """Generate sudoku dataset (image + tabular + OCR sidecar)."""
    from concept_benchmark.synthetic.sudoku_ocr.make_ocr_dataset import (
        generate_sudoku_pipeline_data,
    )

    generate_sudoku_pipeline_data(config)


# ── Stage: train_ocr ──────────────────────────────────────────────────

def train_ocr(config: SudokuBenchmarkConfig) -> None:
    """Train OCR digit recognizer."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m", "concept_benchmark.synthetic.sudoku_ocr.train_ocr_fast",
        "--seed", str(config.seed),
        "--max-corrupt", str(config.max_cell_swaps),
    ]
    subprocess.run(cmd, check=True)


# ── Stage: train_cs ───────────────────────────────────────────────────

def train_cs(
    config: SudokuBenchmarkConfig,
    data=None,
) -> ConceptBasedModel:
    """Train a concept supervision model (concept detector + frontend).

    Returns the trained CBM.
    """
    set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        tab_dir = config.get_dataset_path(data_type="tabular")
        data = load(tab_dir / "sudoku_dataset.pkl")
        # Fractional split with stratification: ensures balanced valid/invalid boards
        data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=config.seed)

    # Missingness can be applied here if needed:
    # data.sample_concept_missingness(p=0.2, mechanism="mcar", rng=config.seed)

    _macos = platform.system() == "Darwin"
    loader_config = {
        "device": device,
        "batch_size": config.batch_size,
        "num_workers": 0 if _macos else 12,
        "pin_memory": not _macos,
    }

    from experiments.models import (
        GroupPoolingConceptSudokuCNN as SudokuConceptModel,
    )

    model = SudokuConceptModel()
    cd = ConceptDetector(model=model)
    cbm = ConceptBasedModel(concept_detector=cd, should_propagate=True)
    cbm.fit(
        train_dataset=data.training,
        valid_dataset=data.validation,
        freeze_backbone=False,
        concept_embed_params={"shuffle": False, **loader_config},
        concept_fit_params={
            "epochs": config.cs_epochs,
            "lr": 1e-3,
            "patience": config.cs_patience,
            **loader_config,
        },
    )

    test_pred = cbm.predict(data.test)
    logger.info("Test Accuracy: %s", np.mean(test_pred == data.test.y))

    save(cbm, config.get_model_path("cs", data_type="tabular"), overwrite=True)
    return cbm


# ── Stage: train_dnn ──────────────────────────────────────────────────

def train_dnn(
    config: SudokuBenchmarkConfig,
    data=None,
) -> dict:
    """Train an end-to-end DNN baseline for sudoku.

    Returns the best state_dict.
    """
    set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        tab_dir = config.get_dataset_path(data_type="tabular")
        data = load(tab_dir / "sudoku_dataset.pkl")
        data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=config.seed)

    from experiments.models import SudokuValidatorCNN as DNNSudokuModel

    model = DNNSudokuModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader_config = get_loader_config()
    train_loader = data.training.loader(shuffle=True, **loader_config)
    valid_loader = data.validation.loader(shuffle=False, **loader_config)
    test_loader = data.test.loader(shuffle=False, **loader_config)

    model.to(device)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        model.train()
        for X, _, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                batch_loss = criterion(outputs.squeeze(), y.float())
                val_loss_sum += batch_loss.item()
                val_batches += 1
        avg_val_loss = val_loss_sum / max(val_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if config.patience > 0 and epochs_no_improve >= config.patience:
                logger.info(
                    "Early stopping at epoch %d with best val loss %.6f",
                    epoch + 1, best_val_loss,
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_acc = compute_accuracy(model, train_loader, device=device)
    valid_acc = compute_accuracy(model, valid_loader, device=device)
    test_acc = compute_accuracy(model, test_loader, device=device)
    logger.info("Training Accuracy: %.2f%%", train_acc * 100)
    logger.info("Validation Accuracy: %.2f%%", valid_acc * 100)
    logger.info("Test Accuracy: %.2f%%", test_acc * 100)

    weights = best_state_dict if best_state_dict is not None else model.state_dict()
    save(weights, config.get_model_path("dnn", data_type="tabular"), overwrite=True)
    return weights
