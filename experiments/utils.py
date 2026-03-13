"""Training and evaluation utilities (repo-only)."""

from __future__ import annotations

import json
import logging
import os
import platform
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

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


def get_loader_config(device: torch.device | None = None) -> dict:
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


# ── Alignment ────────────────────────────────────────────────────────


def run_alignment(
    cbm,
    train_dataset,
    test_dataset,
    monotonicity_constraints: Dict[str, int],
    save_path: Optional[Path] = None,
) -> dict:
    """Run alignment: retrain frontend with sign constraints, compare to original.

    Args:
        cbm: Trained ConceptBasedModel.
        train_dataset: Training split (for retraining the frontend).
        test_dataset: Test split (for evaluation).
        monotonicity_constraints: ``{concept_name: sign}`` where sign is
            +1 (positive weight) or -1 (negative weight).
        save_path: Optional path to save results as JSON.

    Returns:
        Dict with original_accuracy, aligned_accuracy, accuracy_change,
        predictions_changed, aligned_weights.
    """
    from experiments.alignment import retrain_aligned

    # Use ground-truth concepts for training (matching the paper where both
    # original and aligned frontends are trained on GT labels).
    # Test uses predicted concepts (binarised at 0.5, matching cbm.predict()).
    h_train = train_dataset.C.astype(np.float32)
    h_test = (cbm.concept_detector.predict(test_dataset) > 0.5).astype(np.float32)

    stats = retrain_aligned(
        h_train=h_train,
        y_train=train_dataset.y.astype(int),
        h_test=h_test,
        y_test=test_dataset.y.astype(int),
        concept_names=list(test_dataset.concepts),
        original_frontend=cbm.front_end_model,
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
