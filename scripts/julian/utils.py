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


def create_skewed_splits(dataset, skew_specs, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25, rng=None):
    """
    Skew training by ensuring minimum representation of specific concept patterns.

    Args:
        skew_specs: List of dicts, each with 'concepts' (dict of concept:value) and 'min_fraction' (float)
                   e.g., [{'concepts': {'body_shape': 0, 'foot_shape_3sided': 1}, 'min_fraction': 0.4},
                          {'concepts': {'body_shape': 0, 'foot_shape_4sided': 1}, 'min_fraction': 0.4}]
    """
    if rng is None:
        rng = np.random.default_rng()

    # print y labels all
    print("Overall class distribution in full dataset:")
    unique, counts = np.unique(dataset.y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    for cls, cnt in class_dist.items():
        print(f"  Class {cls}: {cnt} samples ({cnt / len(dataset.y):.1%})")

    total_size = len(dataset.C)
    desired_train_size = int(total_size * train_fraction)
    print("Desired training size:", desired_train_size)

    # Find indices matching each specification
    spec_indices = []
    for spec in skew_specs:
        mask = np.ones(total_size, dtype=bool)
        for concept_name, target_value in spec['concepts'].items():
            concept_idx = dataset.concepts.index(concept_name)
            mask &= (dataset.C[:, concept_idx] == target_value)
        spec_indices.append(np.where(mask)[0])

    train_indices = []
    used = set()
    for spec, indices in zip(skew_specs, spec_indices):
        needed = int(desired_train_size * spec['min_fraction'])
        available = [i for i in indices if i not in used]
        rng.shuffle(available)
        take = available[:min(needed, len(available))]
        train_indices.extend(take)
        used.update(take)
        print(f"Added {len(take)} for spec {spec['concepts']} (wanted {needed})")

    # Fill remaining slots with any unused samples
    remaining_slots = desired_train_size - len(train_indices)
    if remaining_slots > 0:
        unused = [i for i in range(total_size) if i not in used]
        rng.shuffle(unused)
        train_indices.extend(unused[:remaining_slots])
        print(f"Filled {min(remaining_slots, len(unused))} remaining slots")

    print("\n=== Debugging: Sample robots from each spec ===")
    for spec, indices in zip(skew_specs, spec_indices):
        print(f"\nSpec {spec['concepts']}: {len(indices)} total samples")
        print("Sample of 10 robots:")
        sample_indices = indices[:10] if len(indices) >= 10 else indices

        for sample_idx in sample_indices:
            robot_features = {}
            for i, concept_name in enumerate(dataset.concepts):
                robot_features[concept_name] = int(dataset.C[sample_idx, i])
                robot_features["class"] = int(dataset.y[sample_idx])
            print(f"  Robot {sample_idx}: {robot_features}")


    train_indices = np.array(train_indices)
    rng.shuffle(train_indices)

    # Validation and test from what's left
    remaining = np.array([i for i in range(total_size) if i not in train_indices])
    rng.shuffle(remaining)

    val_size = int(len(remaining) * val_fraction / (val_fraction + test_fraction))
    val_indices = remaining[:val_size]
    test_indices = remaining[val_size:]

    train_mask = np.zeros(total_size, dtype=bool)
    train_mask[train_indices] = True
    val_mask = np.zeros(total_size, dtype=bool)
    val_mask[val_indices] = True
    test_mask = np.zeros(total_size, dtype=bool)
    test_mask[test_indices] = True

    dataset.training = dataset._full.filter(indices=train_mask)
    dataset.validation = dataset._full.filter(indices=val_mask)
    dataset.test = dataset._full.filter(indices=test_mask)

    return dataset.training, dataset.validation, dataset.test