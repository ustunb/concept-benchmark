import torch
from pathlib import Path

from concept_benchmark.paths import results_dir

MISSING_TYPES = ["mcar", "mnar"]

DEFAULT_SUDOKU_SETTINGS = {
    'data_name': 'sudoku',
    "n": 3,
    "n_samples": 5000,
    "valid_ratio": 0.5,
    "max_corrupt": 21,
    "data_type": "tabular",
    "seed": 42,
}

def get_dataset_file(
    data_type: str,
    n: int,
    max_corrupt: int,
    **kwargs
) -> Path:
    """Get the file path for the dataset based on its parameters."""
    filename = f"sudoku_{data_type}_n{n}_mc{max_corrupt}"

    return results_dir / f"{filename}.data"

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
    """Determine the device to be used for computations."""
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