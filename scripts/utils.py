import torch
import numpy as np
from pathlib import Path

from concept_benchmark.paths import results_dir

CONCEPT_NOISE = np.arange(0, 0.35, 0.05).round(2)
CONCEPT_MISSING = np.arange(0.05, 0.35, 0.05).round(2)
MISSING_TYPES = ["mcar", "mnar"]

DIFFICULTY = {
    'easy': 1.0,
    'medium': 0.8,
    'hard': 0.6,
}

DEFAULT_SUDOKU_SETTINGS = {
    'data_name': 'sudoku',
    "n": 3,
    "n_samples": 5000,
    "valid_ratio": 0.5,
    "max_corrupt": 21,
    "data_type": "tabular",
    "seed": 42,
    "target_accuracy": 1.0,
    "concept_noise": 0.0,
}

DEFAULT_ROBOT_SETTINGS = {
    'data_name': 'robot',
    'n': 1,
    'draw': False,
    'output_directory': results_dir / 'robots_large',
    'concepts' : {
        "foot_shape": [
            "flat_4sided",
            "flat_5sided",
            "flat_lshaped",
            "pointy_3sided",
            "pointy_4sided",
            "pointy_6sided",
        ],
        "body_shape": ["square", "round"],  # no subtypes (could add)
        "head_shape": ["square", "round"],  # no subtypes (could add)
        #
        "has_elbows": [True, False],  # all round
        "has_knees": [True, False],
        "has_antennae": [True, False],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": [
            "round_circle",
            "round_oval",
            "round_oval2",
            "edgy_triangle",
            "edgy_square",
            "edgy_trapezoid",
        ],
    },
    # 'irrelevant_features': ['has_antennae'],  # take antennae out, as they are too hard to detect
    'model': "'glorp' if (int(row['body_shape']=='square') + int(row['foot_shape']=='pointy') - 2 >= 0) else 'drent'",
    'model_type': 'deterministic', 
    'size': 'large',  
    'color_mode': 'color',  
    'data_type': 'image',
    'target_accuracy': 1.0,
    'concept_noise': 0.0,
}

def get_dataset_file(
    data_name: str,
    data_type: str,
    n: int,
    concept_noise: float = 0.0,
    target_accuracy: float = 1.0,
    blur: dict = None,
    **kwargs
) -> Path:
    """Get the file path for the dataset based on its parameters."""
    if data_name == "sudoku":
        filename = f"sudoku_{data_type}_{n**2}_{kwargs.get('max_corrupt')}_{concept_noise}_{target_accuracy}"
    elif data_name == "robot":
        filename = f"robot_{data_type}_{n}_{concept_noise}_{target_accuracy}"
        if blur:
            filename += f"_blur_{'_'.join(blur['parts'])}_{blur['radius']}"
            if blur.get('expand_mask_px'):
                filename += f"_expand{blur['expand_mask_px']}"
            if blur.get('feather_mask_px'):
                filename += f"_feather{blur['feather_mask_px']}"
        if kwargs.get('collapse') is False:
            filename += "_ncollapse"

    return results_dir / f"{filename}.data"

def get_model_file(
    data_name: str,
    data_type: str,
    model_type: str,
    n: int,
    concept_noise: float = 0.0,
    concept_missing: float = 0.0,
    concept_missing_mech: str = "none",
    target_accuracy: float = 1.0,
    **kwargs
) -> Path:
    """Get the file path for the model based on its parameters."""
    if data_name == "sudoku":
        filename = f"sudoku_{data_type}_{model_type}_{n**2}_{kwargs.get('max_corrupt')}"
    elif data_name == "robot":
        filename = f"robot_{data_type}_{model_type}_{n}"
        
    if data_name != "sudoku" or model_type == "cd":
        filename += f"_{concept_noise}"
        if concept_missing:
            filename += f"_{concept_missing_mech}_{concept_missing}"

    if model_type in {"fe", "dnn"}:
        filename += f"_{target_accuracy}"
        
    filename += ".model"

    return results_dir / filename
    

def determine_device():
    """Determine the device to be used for computations."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
