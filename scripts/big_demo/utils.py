import torch
from pathlib import Path

from concept_benchmark.paths import results_dir

CONCEPT_NOISE = 0.05
CONCET_MISSING = 0.05

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
    'concepts': {
        'head_shape': ['square', 'round'],
        'body_shape': ['square', 'round'],
        'has_knees': ['false', 'true'],
        'has_elbows': ['false', 'true'],
        'has_antennae': ['false', 'true'],
        'ears_shape': ['square', 'triangle'],
        'mouth_type': ['closed', 'open'],
        'hand_shape': ['round_circle', 'round_oval', 'round_oval2',
                        'edgy_triangle', 'edgy_square', 'edgy_trapezoid'],
        'foot_shape': ['flat_4sided', 'flat_5sided', 'flat_lshaped',
                        'pointy_3sided', 'pointy_4sided', 'pointy_6sided'],
    },
    'irrelevant_features': ['has_antennae'],  # take antennae out, as they are too hard to detect
    'model': "'glorp' if (int(row['body_shape']=='square') + int(row['foot_shape']=='pointy') - 2 >= 0) else 'drent'",
    'model_type': 'deterministic', 
    'size': 'large',  
    'color_mode': 'color',  
    'data_type': 'image',
    'target_accuracy': 1,
    'concept_noise': 0.0,
}

def get_dataset_file(
    data_name: str,
    data_type: str,
    n: int,
    concept_noise: float = 0.0,
    target_accuracy: float = 1.0,
    **kwargs
) -> Path:
    """Get the file path for the dataset based on its parameters."""
    if data_name == "sudoku":
        filename = f"sudoku_{data_type}_{n**2}_{kwargs.get('max_corrupt')}_{concept_noise}_{target_accuracy}.data"
    elif data_name == "robot":
        filename = f"robot_{data_type}_{n}_{concept_noise}_{target_accuracy}.data"

    return results_dir / filename

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
        
    filename += f"_{concept_noise}"

    if concept_missing:
        filename += f"_{concept_missing_mech}_{concept_missing}"

    filename += f"_{target_accuracy}.model"

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