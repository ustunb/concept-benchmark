import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from concept_benchmark.paths import results_dir, data_dir


IDEAL_DROP = [
    "foot_shape_flat_rounded",
    "foot_shape_pointy_trapezoid",
    "foot_shape_pointy_3sided",
    "foot_shape_flat_lshaped",
    "foot_shape_pointy_4sided",
    "foot_shape_pointy_square",
    "foot_shape_pointy_rounded",
    "foot_shape_flat_5sided",
    "foot_shape_flat_square",
    "foot_shape_flat_trapezoid",
]

SUBCONCEPT_DROP = [
    "foot_shape_flat_rounded",
    "foot_shape_pointy_trapezoid",
    'foot_shape_pointy_3sided', 
    'foot_shape_flat_lshaped',
    'foot_shape'
]

DEFAULT_ROBOT_SETTINGS = {
    "data_type": "image",
    "samples_per_instance": 4,
    "draw": 0,
    "output_directory": data_dir / "robot_images",
    "size": "medium",
    "color_mode": "color",
    "train_dnn": 0,
    "seed": 1002,
    "model": "'glorp' if (int(row['mouth_type']=='closed') + int(row['foot_shape']=='pointy'))>= 3 else 'drent'",
    "test_size": 10000,
    "train_skew_size": 3800,
    "knows_concepts": False,
    "concepts": {
                "head_shape": ["square", "round"],
                "body_shape": ["square", "round"],
                "has_knees": ["false", "true"],
                "has_elbows": ["false", "true"],
                "has_antennae": ["false", "true"],
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
                "foot_shape": [
                    "flat_trapezoid",
                    "flat_rounded",
                    "flat_square",
                    "flat_5sided",
                    "flat_lshaped",
                    "pointy_trapezoid",
                    "pointy_rounded",
                    "pointy_square",
                    "pointy_3sided",
                    "pointy_4sided",
                ],
            },
    # "subconcepts": ["foot_shape_subtype"],
    "additional_features": ["foot_shape_subtype"],
    "spurious_features": ["has_elbows", "hand_shape"],
    "subconcept": False,
    "drop_concepts": IDEAL_DROP,
    "model_type": "stochastic",
    "scalar": 1.0,
    "intercept": 3,
    "weights": {"mouth_type": 5, "foot_shape": 10},
    "skew_specs": [
                     {'concepts': {'foot_shape_pointy_square': 1}, 'min_fraction': 0.005},
                     {'concepts': {'foot_shape_pointy_rounded': 1}, 'min_fraction': 0.005},
                     {'concepts': {'foot_shape_pointy_4sided': 1}, 'min_fraction': 0.49},
                     {'concepts': {'foot_shape_flat_square': 1}, 'min_fraction': 0.005},
                     {'concepts': {'foot_shape_flat_trapezoid': 1}, 'min_fraction': 0.005},
                     {'concepts': {'foot_shape_flat_5sided': 1}, 'min_fraction': 0.49},
                     ],
}

INPUT_MAP = {
    "large": 600,
    "medium": 32,
    "small": 8,
}

INTERVENTION_SETTINGS = {
    "budget": [1, 3],
    "intervention_accuracy": 0.9,
    "intervention_threshold": 0.2,
}

def get_dataset_file(
    data_type: str,
    samples_per_instance: int,
    subconcept: bool,
    **kwargs
) -> Path:
    """Get the file path for the dataset based on its parameters."""
    filename = f"robot_{data_type}_{samples_per_instance}"

    if subconcept:
        filename += "_subconcept"
    else:
        filename += "_ideal"

    return results_dir / f"{filename}.data"

def get_model_file(
    data_type: str,
    model_type: str,
    samples_per_instance: int,
    subconcept: bool,
    model_class: str,
    **kwargs
) -> Path:
    """Get the file path for the model based on its parameters."""
    filename = f"robot_{data_type}_{model_type}_{samples_per_instance}"

    if subconcept:
        filename += "_subconcept"
    else:
        filename += "_ideal"
        
    filename += f"_{model_class}.model"

    return results_dir / filename

def get_results_file(
    data_type: str,
    model_type: str,
    subconcept: bool,
    model_class: str = "cbm",
    **kwargs
) -> Path:
    """Get the file path for the results based on its parameters."""
    filename = f"robot_{data_type}_{model_type}"

    if model_class == "cbm":
        if subconcept:
            filename += "_subconcept"
        else:
            filename += "_ideal"
        
    filename += f"_{model_class}_results.csv"

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


class RobotClassifierCNN(nn.Module):
    def __init__(self, num_classes=1, input_size=224):
        super(RobotClassifierCNN, self).__init__()
        
        # --- Feature Extractor ---
        # Input images are assumed to be 3-channel RGB
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves the dimensions
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves the dimensions again
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.backbone = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            nn.ReLU(),
            self.pool3,
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            dummy_out = self.backbone(dummy)
            feature_size = dummy_out.view(1, -1).size(1)

        # self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc1 = nn.Linear(feature_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass through the feature extractor
        x = self.backbone(x)
        
        # Flatten the feature maps for the classifier
        x = torch.flatten(x, 1)
        
        # Pass through the classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # For binary classification, we apply a sigmoid function to the output
        return torch.sigmoid(x)

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
