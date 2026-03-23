"""Training and evaluation utilities (repo-only)."""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from concept_benchmark.data import ConceptDatasetSample

logger = logging.getLogger(__name__)


def determine_device() -> torch.device:
    """Determine the best available compute device.

    Respects ``PYTORCH_DEVICE`` env var to override auto-detection.
    """
    override = os.environ.get("PYTORCH_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_accuracy(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    """Compute binary classification accuracy (threshold 0.5)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, _, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = (outputs.squeeze() > 0.5).long()
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total > 0 else 0


def get_loader_config() -> dict:
    """Return DataLoader kwargs safe for the current platform."""
    _macos = platform.system() == "Darwin"
    return {
        "batch_size": 32,
        "num_workers": 0 if _macos else 12,
        "pin_memory": not _macos,
    }


def patch_macos_dataloader() -> None:
    """Force num_workers=0 on macOS to avoid MPS/fork hangs.

    Uses a real subclass so that third-party libraries (e.g. timm) can
    subclass ``torch.utils.data.DataLoader`` after the patch.
    """
    if platform.system() != "Darwin":
        return

    import torch.utils.data as _tud
    import concept_benchmark.data as _cb_data

    _OrigDataLoader = _tud.DataLoader

    # Already patched
    if getattr(_OrigDataLoader, "_macos_patched", False):
        return

    class _SafeDataLoader(_OrigDataLoader):
        _macos_patched = True

        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs["pin_memory"] = False
            super().__init__(*args, **kwargs)

    _tud.DataLoader = _SafeDataLoader
    _cb_data.DataLoader = _SafeDataLoader


# ── DNN training ─────────────────────────────────────────────────────


def train_dnn(
    model: nn.Module,
    train_dataset: ConceptDatasetSample,
    val_dataset: ConceptDatasetSample,
    test_dataset: ConceptDatasetSample,
    device: torch.device,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    loader_config: dict | None = None,
) -> float:
    """Train a binary classifier with early stopping. Returns test accuracy.

    Parameters
    ----------
    model : nn.Module
        A PyTorch module that outputs a single sigmoid probability per sample.
    train_dataset : ConceptDatasetSample
        Training split.
    val_dataset : ConceptDatasetSample
        Validation split (for early stopping).
    test_dataset : ConceptDatasetSample
        Test split (for final evaluation).
    device : torch.device
        Torch device to use.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate for Adam.
    patience : int
        Early stopping patience (0 disables).
    loader_config : dict, optional
        Extra kwargs forwarded to ``dataset.loader()``
        (e.g. ``batch_size``, ``num_workers``).

    Returns
    -------
    float
        Test accuracy in [0, 1].
    """
    if loader_config is None:
        loader_config = get_loader_config()

    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = train_dataset.loader(shuffle=True, **loader_config)
    val_loader = val_dataset.loader(shuffle=False, **loader_config)
    test_loader = test_dataset.loader(shuffle=False, **loader_config)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(epochs):
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
            for X, _, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                batch_loss = criterion(outputs.squeeze(), y.float())
                val_loss_sum += batch_loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return compute_accuracy(model, test_loader, device)


# ── Alignment ────────────────────────────────────────────────────────


def run_alignment(
    concept_based_model,
    train_dataset,
    test_dataset,
    monotonicity_constraints: dict[str, int],
    save_path: Path | None = None,
) -> dict:
    """Run alignment: retrain frontend with sign constraints, compare to original.

    Parameters
    ----------
    concept_based_model : ConceptBasedModel
        Trained ConceptBasedModel.
    train_dataset : ConceptDatasetSample
        Training split (for retraining the frontend).
    test_dataset : ConceptDatasetSample
        Test split (for evaluation).
    monotonicity_constraints : dict
        ``{concept_name: sign}`` where sign is +1 (positive weight) or
        -1 (negative weight).
    save_path : Path, optional
        Optional path to save results as JSON.

    Returns
    -------
    dict
        Dict with ``original_accuracy``, ``aligned_accuracy``,
        ``accuracy_change``, ``predictions_changed``, ``aligned_weights``.
    """
    from experiments.alignment import retrain_aligned

    # Use ground-truth concepts for training (matching the paper where both
    # original and aligned frontends are trained on GT labels).
    # Test uses predicted concepts (binarised at 0.5, matching cbm.predict()).
    concept_preds_train = train_dataset.C.astype(np.float32)
    concept_preds_test = concept_based_model.concept_detector.predict(
        test_dataset
    ).astype(np.float32)

    stats = retrain_aligned(
        concept_preds_train=concept_preds_train,
        y_train=train_dataset.y.astype(int),
        concept_preds_test=concept_preds_test,
        y_test=test_dataset.y.astype(int),
        concept_names=list(test_dataset.concepts),
        original_frontend=concept_based_model.label_predictor,
        monotonicity_constraints=monotonicity_constraints,
    )

    logger.info("\n=== Alignment Results ===")
    logger.info("  Original accuracy: %.4f", stats["original_accuracy"])
    logger.info("  Aligned accuracy:  %.4f", stats["aligned_accuracy"])
    logger.info("  Accuracy change:   %+.4f", stats["accuracy_change"])
    logger.info("  Predictions changed: %d", stats["predictions_changed"])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types for JSON serialization
        serializable = {
            k: (
                v
                if not isinstance(v, dict)
                else {kk: float(vv) for kk, vv in v.items()}
            )
            for k, v in stats.items()
        }
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("  Saved to %s", save_path)

    return stats
