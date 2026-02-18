import os
import platform
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from concept_benchmark.paths import data_dir, results_dir

# -- Force num_workers=0 on macOS to avoid MPS/fork hangs ----------------
if platform.system() == "Darwin":
    import torch.utils.data as _tud
    import concept_benchmark.data as _cb_data
    _OrigDataLoader = _tud.DataLoader

    def _safe_dataloader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return _OrigDataLoader(*args, **kwargs)

    _tud.DataLoader = _safe_dataloader
    _cb_data.DataLoader = _safe_dataloader
# -------------------------------------------------------------------------

MISSING_TYPES = ["mcar", "mnar"]

DEFAULT_SUDOKU_SETTINGS = {
    'data_name': 'sudoku',
    "n": 3,
    "n_samples": 1000,
    "valid_ratio": 0.5,
    "max_corrupt": 9,
    "seed": 171,
    "temp_train_data_path": data_dir / "sudoku" / "multimodal_m_21" / "tabular" / "sudoku_dataset.pkl",
    "epochs": 20,
    "patience": 5,
    "concept_missing_mech": "none"
}

def get_dataset_file(
    data_type: str,
    n: int,
    n_samples: int,
    max_corrupt: int,
    seed: int,
    **kwargs
) -> Path:
    """Get the directory path for the dataset based on its parameters."""
    filename = f"sudoku_{data_type}_n{n}_ns{n_samples}_mc{max_corrupt}_seed{seed}"

    return data_dir / "sudoku" / filename

def get_model_file(
    data_type: str,
    n: int,
    max_corrupt: int,
    concept_missing: float = 0.0,
    concept_missing_mech: str = "none",
    model_class: str = "cbm",
    **kwargs
) -> Path:
    """Get the file path for the model based on its parameters."""
    filename = f"sudoku_{model_class}_{data_type}_n{n}_mc{max_corrupt}"

    if concept_missing_mech is not None and concept_missing > 0.0:
        filename += f"_cm{concept_missing_mech}{concept_missing}"

    return results_dir / f"{filename}.model"

def get_results_fule(
    data_type: str,
    n: int,
    max_corrupt: int,
    concept_missing: float = 0.0,
    concept_missing_mech: str = "none",
    model_class: str = "cbm",
    **kwargs
) -> Path:
    """Get the file path for the results based on its parameters."""
    filename = f"sudoku_{model_class}_{data_type}_n{n}_mc{max_corrupt}"

    if concept_missing_mech is not None and concept_missing > 0.0:
        filename += f"_cm{concept_missing_mech}{concept_missing}"

    return results_dir / f"{filename}.results"
    

def determine_device():
    """Determine the device to be used for computations.

    Respects PYTORCH_DEVICE env var (e.g. ``PYTORCH_DEVICE=cpu``) to override
    auto-detection.  MPS on macOS can hang with image workloads, so setting
    ``PYTORCH_DEVICE=cpu`` is a safe fallback.
    """
    override = os.environ.get("PYTORCH_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, _, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = (outputs.squeeze() > 0.5).long()  # Thresholding at 0.5
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total > 0 else 0


class AndConceptSudokuValidator(nn.Module):
    """Return 1 only if all 27 concept inputs are active."""

    def __init__(self, num_concepts: int = 27, threshold: float = 0.5) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.threshold = float(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.num_concepts:
            raise ValueError(
                f"Expected {self.num_concepts} concepts, got {x.shape[-1]}."
            )
        active = x > self.threshold
        valid = active.all(dim=-1, keepdim=True)
        return valid.float()


class AndConceptFrontEndModel:
    """Front-end wrapper that applies an AND rule over 27 concept inputs."""

    def __init__(self, num_concepts: int = 27, threshold: float = 0.5) -> None:
        self.num_concepts = int(num_concepts)
        self.threshold = float(threshold)

    def fit(self, C: np.ndarray, y: np.ndarray, fit_params: dict | None = None) -> None:
        """No-op; the AND rule is fixed."""
        return None

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        if C.shape[-1] != self.num_concepts:
            raise ValueError(
                f"Expected {self.num_concepts} concepts, got {C.shape[-1]}."
            )
        active = C > self.threshold
        valid = active.all(axis=-1).astype(np.float32)
        return np.stack([1.0 - valid, valid], axis=1)

    def predict(self, C: np.ndarray) -> np.ndarray:
        return self.predict_proba(C).argmax(axis=1)
