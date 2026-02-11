# Concept Benchmark

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange)](https://github.com/ustunb/concept-benchmark)

A framework for generating synthetic concept-bottleneck benchmarks and evaluating concept-based models (CBMs). The package provides configurable dataset generators, model architectures, training pipelines, and evaluation tools for studying concept alignment, interventions, and interpretability.

## Installation

Requires Python >= 3.10.

**Quick start** -- clone the repo and run the install script:

```bash
git clone <repo-url>
cd concept-benchmark
./install.sh
source venv/bin/activate
```

The script creates a virtual environment, installs the package in editable mode, and pulls all dependencies (torch, torchvision, transformers, scikit-learn, cvxpy, etc.).

**Manual install** (if you prefer your own venv or conda environment):

```bash
pip install -e .
```

---

## Overview

The framework follows a common pipeline across all benchmarks:

1. **Design a benchmark** -- define concepts, labels, and data generation parameters
2. **Generate a dataset** -- produce a `ConceptDataset` with features `X`, concepts `C`, and labels `y`
3. **Train a concept detector** -- learn to predict concepts from raw inputs (images, text, tabular)
4. **Train a frontend model** -- learn to predict labels from (predicted) concepts
5. **Evaluate** -- measure concept accuracy, label accuracy, intervention effectiveness, and human alignment

```
Raw Input (X) --> [Concept Detector] --> Predicted Concepts (C_hat) --> [Frontend] --> Label (y_hat)
```

### ConceptDataset

All benchmarks produce a `ConceptDataset` object:

```python
from concept_benchmark.data import ConceptDataset

data = ConceptDataset(X, C, y, meta)

# Generate cross-validation folds and split
data.generate_cvindices(seed=0)
data.split("K05N01", fold_num_validation=4, fold_num_test=5)

# Access splits
data.training.X, data.training.C, data.training.y
data.validation.X, data.validation.C, data.validation.y
data.test.X, data.test.C, data.test.y
```

---

## Sudoku Benchmark

Generate synthetic Sudoku boards with row/column/block validity concepts. The task is binary classification: valid vs. invalid board.

### Creating a Sudoku Dataset

```python
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset, image_transform

data = create_sudoku_dataset(
    n=3,                    # block size (3 -> 9x9 board)
    n_samples=1000,         # number of samples
    valid_ratio=0.5,        # fraction of valid boards
    max_corrupt=3,          # max corruptions per invalid board
    data_type="image",      # "image" or "tabular"
    seed=42,
    transform=image_transform,
    dataset_name="my_sudoku",
)
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n` | Block size (`n=3` for 9x9, `n=4` for 16x16) | `3` |
| `n_samples` | Total number of samples | `1000` |
| `valid_ratio` | Fraction of valid boards (0.0 to 1.0) | `0.5` |
| `max_corrupt` | Max corruption actions per invalid board | `3` |
| `data_type` | `"image"` or `"tabular"` | `"image"` |
| `transform` | Feature transform function (see below) | `None` (flatten) |
| `add_cell_digit_concepts` | Add per-cell digit indicator concepts | `False` |
| `positions_subset` | Subset of (row, col) positions for cell concepts | `None` (all) |
| `digits_subset` | Subset of digits for cell concepts | `None` (all) |

#### Concepts

For an NxN board (N = n*n), the base concepts are:

- **Row validity**: `row_valid_1` ... `row_valid_N` -- whether each row contains all digits exactly once
- **Column validity**: `col_valid_1` ... `col_valid_N`
- **Block validity**: `block_valid_1` ... `block_valid_N`

Total: 3N base concepts. Optionally, per-cell-digit concepts can be appended.

#### Feature Transforms

Four built-in transforms control how boards are represented as features:

```python
from concept_benchmark.synthetic.sudoku import (
    default_transform,     # flatten to 1D vector (N^2,)
    onehot_transform,      # one-hot encoding (N, N, N)
    histogram_transform,   # per-unit digit histograms (3N, N)
    image_transform,       # rendered PNG images
)
```

**Image transform options** (passed as `kwargs` to `create_sudoku_dataset`):

| Option | Description | Default |
|--------|-------------|---------|
| `cell_px` | Pixel size per cell | `40` |
| `font_size` | Digit font size | `10` |
| `handwriting` | Apply sketch effect for handwritten style | `False` |
| `radius`, `sigma`, `angle` | Handwriting Gaussian blur parameters | varies |

#### CLI Usage

```bash
# 9x9 tabular, 10k samples
python scripts/run_sudoku.py --n 3 --n-samples 10000 --data-type tabular --save-dir out/tabular

# 9x9 image dataset
python scripts/run_sudoku.py --data-type image --dataset_name demo --n-samples 2000 --cell-px 24

# One-hot encoded, saved as PyTorch tensor
python scripts/run_sudoku.py --transform onehot --save-dir out/onehot --save-format pt
```

### Training a Sudoku Model

```python
from concept_benchmark.models import ConceptDetector, FrontEndModel
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset, image_transform
from transformers import ViTModel
import torch

# 1. Create dataset
data = create_sudoku_dataset(n=3, n_samples=1000, data_type="image", transform=image_transform)
data.generate_cvindices(seed=42)
data.split("K05N01", fold_num_validation=4, fold_num_test=5)

# 2. Set up ViT backbone
class ViTWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
    def forward(self, x):
        return self.vit(pixel_values=x).last_hidden_state[:, 0, :]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = ViTWrapper().to(device)

# 3. Train concept detector
cd = ConceptDetector(embedding_model=embed_model)
cd.fit(data.training, data.validation, freeze=True,
       embed_params={"device": device},
       fit_params={"epochs": 10, "device": "cpu", "hidden_size": 360})

# 4. Evaluate concepts
c_pred = cd.predict(data.test, embed_params={"device": device}) > 0.5
print("Concept accuracy:", (c_pred == data.test.C).mean(axis=0))

# 5. Train frontend (concepts -> label)
fe = FrontEndModel()
fe.fit(data.training.C, data.training.y)
preds = fe.predict(c_pred.astype(float))
print(f"Label accuracy: {(preds == data.test.y).mean():.4f}")
```

### Sudoku Model Architectures

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| **ViT + MLP head** | ViT-Base-Patch16-224 backbone with per-concept MLP heads | Image data (224x224) |
| **ConceptDetector (default)** | Auto-builds MLP head on top of any embedding model | General purpose |
| **JointConceptModel** | Wraps any backbone + head into a single module | Custom architectures |

---

## Robot Benchmark

Generate synthetic robot images (or text descriptions) with configurable visual features as concepts. The task is binary classification between two fictional species: **glorp** (1) and **drent** (0).

### Creating a Robot Dataset

```python
from concept_benchmark.synthetic.robot import create_robot_image_dataset

data = create_robot_image_dataset(
    concepts={
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "foot_shape": ["flat_trapezoid", "flat_rounded", "pointy_3sided", "pointy_4sided"],
    },
    model="'glorp' if int(row['has_knees']=='true') + int(row['foot_shape']=='pointy') >= 2 else 'drent'",
    model_type="deterministic",    # or "stochastic"
    size="large",                  # "large" (600px), "medium" (32px)
    samples_per_instance=1,
    draw=True,                     # render robot PNG images
    output_directory="./data/robot_images",
    color_mode="color",
    seed=42,
)
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `concepts` | Dict mapping feature names to possible values | *required* |
| `model` | Python expression defining the label rule (evaluated per row) | *required* |
| `model_type` | `"deterministic"` or `"stochastic"` | `"deterministic"` |
| `size` | Image resolution: `"large"` (600px), `"medium"` (32px) | `"large"` |
| `samples_per_instance` | Color variations per robot configuration (max 108) | `1` |
| `num_robots` | Total robots (default: all concept combinations x samples) | `None` |
| `draw` | Whether to render PNG images | `False` |
| `spurious_features` | Features excluded from concept matrix (kept for drawing) | `None` |
| `additional_features` | Extra features to track (e.g. subconcepts) | `None` |
| `color_mode` | `"color"` or `"greyscale"` | `"color"` |

#### Available Robot Features

| Feature | Values |
|---------|--------|
| `head_shape` | `square`, `round` |
| `body_shape` | `square`, `round` |
| `has_knees` | `true`, `false` |
| `has_elbows` | `true`, `false` |
| `has_antennae` | `true`, `false` |
| `ears_shape` | `square`, `triangle` |
| `mouth_type` | `closed`, `open` |
| `hand_shape` | `round_circle`, `round_oval`, `round_oval2`, `edgy_triangle`, `edgy_square`, `edgy_trapezoid` |
| `foot_shape` | `flat_trapezoid`, `flat_rounded`, `flat_square`, `flat_5sided`, `flat_lshaped`, `pointy_trapezoid`, `pointy_rounded`, `pointy_square`, `pointy_3sided`, `pointy_4sided` |

#### Label Models

**Deterministic**: A Python expression evaluated per robot row:
```python
model="'glorp' if (int(row['mouth_type']=='closed') + int(row['has_knees']=='true')) >= 2 else 'drent'"
```

**Stochastic**: Uses a logistic function to produce probabilistic labels:
```python
model_type="stochastic",
logit_scalar=4.2,
logit_intercept=-2,
logit_weights={"mouth_type": 5, "foot_shape": 8, "has_knees": -5},
```

#### Text Modality

Robots can also be represented as text descriptions:

```python
from concept_benchmark.synthetic.robot import create_synthetic_dataset

data = create_synthetic_dataset(data_type="text", source=image_dataset, ...)
```

See `scripts/run_robot_demos.py` for the full text pipeline.

### Running the Robot Pipeline

The main robot pipeline is in `scripts/robot_image_training.py`. It runs the full CBM workflow:

```python
from scripts.robot_image_training import main, settings

# Configure (see settings dict in the script for all options)
settings["concepts"] = { ... }
settings["model"] = "..."
settings["seed"] = 42

results = main(settings)
```

#### Pipeline Stages

The `main()` function executes these stages in order:

1. **Data Definition** -- generates the robot dataset, creates train/val/test splits with optional concept skewing, applies label noise and concept missingness
2. **Concept Detector Training** -- trains a CNN or ViT model to predict concepts from images
3. **Frontend Training** -- fits a logistic regression (or constrained CVXPY model) mapping concepts to labels
4. **Intervention Testing** -- simulates human corrections to concept predictions using K-Flip strategy
5. **Alignment Testing** -- re-trains the frontend with human-specified monotonicity and prediction constraints
6. **Results Saving** -- saves meta JSON, metrics JSON, confusion matrices, and model checkpoints

#### Advanced Settings

| Setting | Description |
|---------|-------------|
| `skew_concept` | Force minimum representation of specific concept patterns in training |
| `label_noise_rate` | Fraction of training labels to randomly flip |
| `missingness` | Concept missingness mode: `"complete"`, `"mcar"`, `"mar"`, `"mnar"` |
| `missing_rate` | Fraction of concepts to mask |
| `subconcepts` | Expand coarse features into fine-grained subconcepts (e.g. `foot_shape` subtypes) |
| `drop_concepts` | Remove specific subconcepts from the concept matrix |
| `budget` | List of intervention budgets (number of concepts to correct per sample) |
| `intervention_accuracy` | Simulated human accuracy when intervening |
| `human_alignment` | Dict with `"signs"` (monotonicity) and `"features"` (prediction constraints) |

### Robot Model Architectures

| Architecture | Class | Description |
|-------------|-------|-------------|
| **CNN (small/large)** | `RobotConceptClassifier` | 3-layer CNN with per-concept linear heads. Adaptive pooling for different image sizes. |
| **ViT** | `RobotViTConceptClassifier` | ViT-Base-Patch16-224 backbone with per-concept linear heads (768-dim features). |
| **Simple CNN** | `RobotClassifierCNN` | 3-layer CNN with single binary output (for DNN baseline). |
| **Frontend (unconstrained)** | `FrontEndModel` | Sklearn logistic regression: concepts -> label. |
| **Frontend (constrained)** | `FrontEndModelCVXPY` | CVXPY-based logistic regression with monotonicity and prediction constraints. |

---

## Full Pipeline Comparison: Sudoku vs. Robot

While both benchmarks share the same core concept-bottleneck architecture (concept detector + frontend), the end-to-end pipelines differ significantly in scope and structure.

### Sudoku Pipeline

The sudoku pipeline is split into **two separate scripts** that are run independently:

1. **Dataset generation** (`scripts/run_sudoku.py`) -- a CLI tool that creates the dataset and saves it to disk (images + CSV, or `.npz`/`.pt` arrays). No training happens here.

2. **Training and evaluation** (`scripts/sudoku_train.py`) -- a standalone script that loads/creates a dataset, trains a ViT-based ConceptDetector, evaluates concept accuracy, and fits a FrontEndModel.

```bash
# Step 1: Generate dataset
python scripts/run_sudoku.py --n 3 --n-samples 5000 --data-type image --dataset_name my_exp

# Step 2: Train and evaluate (edit settings in sudoku_train.py or use as a template)
python scripts/sudoku_train.py
```

The sudoku pipeline uses standard 5-fold cross-validation splits and does not include skewing, missingness, interventions, or alignment testing. It is designed as a straightforward benchmark for concept detection accuracy.

### Robot Pipeline

The robot pipeline is a **single monolithic script** (`scripts/robot_image_training.py`) with a `main(settings)` function that runs the entire workflow end-to-end:

```
main(settings)
├── 1. Data Definition
│   ├── Generate robot dataset (create_synthetic_dataset)
│   ├── Create skewed train/val/test splits
│   ├── Apply label noise
│   └── Apply concept missingness (MCAR/MAR/MNAR)
├── 2. Concept Detector Training
│   ├── Train RobotConceptClassifier (CNN)
│   └── Test detector invariance
├── 3. Frontend Training
│   ├── Fit FrontEndModel or FrontEndModelCVXPY
│   └── Compute concept + label accuracy
├── 4. Intervention Testing
│   ├── Run K-Flip interventions at each budget level
│   └── Simulate human error rates
├── 5. Alignment Testing
│   ├── Re-train frontend with monotonicity constraints
│   └── Compare aligned vs. unaligned accuracy
└── 6. Results Saving
    ├── meta.json (settings, artifacts, splits)
    ├── metrics.json (accuracies, interventions, weights)
    ├── confusion.csv, catalog.csv
    └── detector + frontend checkpoints
```

#### Running a single experiment

```python
from scripts.robot_image_training import main, settings

settings["seed"] = 42
settings["model_type"] = "stochastic"
settings["run_name"] = "my_experiment"
results = main(settings)
```

#### Running a grid search (big demo table)

The `scripts/robot_grid_search.py` script automates large-scale experimentation by systematically varying:

- **Subconcept subsets** -- which fine-grained subconcepts (e.g., specific foot shapes) to include
- **Skew fractions** -- how to distribute subconcept prevalence in training (e.g., 49% pointy_4sided vs. 0.5% pointy_square)
- **Dropped concepts** -- which subconcepts to exclude from the concept matrix
- **Logit weights** -- stochastic label model parameters tuned to achieve target P(glorp) for each feature combination

Each parameter combination calls `main(settings)` and saves results to a separate run directory. These results are then aggregated into a table for analysis.

```python
from scripts.robot_grid_search import run_experiments_varying_footshape_subconcepts

# Loops over all valid subconcept combinations and runs main() for each
run_experiments_varying_footshape_subconcepts()
```

#### Running alignment studies

Alignment studies use the same `robot_image_training.py` `main()` function but focus on the `human_alignment` setting:

```python
settings["human_alignment"] = {
    "signs": {"has_knees": 1},          # force positive weight for has_knees
    "features": [                        # prediction constraints
        (["foot_shape_flat_square"], ["True"], ">=", 0.95),
    ]
}
results = main(settings)
# results["alignment"] contains accuracy change, predictions changed, aligned weights
```

The alignment stage re-trains the frontend with CVXPY-based constrained optimization, comparing the aligned model's accuracy against the unconstrained baseline.

### Key Differences Summary

| Aspect | Sudoku | Robot |
|--------|--------|-------|
| **Script structure** | Two separate scripts (generate + train) | Single `main(settings)` runs everything |
| **Concept detector** | ViT backbone + MLP heads | CNN (RobotConceptClassifier) |
| **Data splitting** | Standard 5-fold CV | Custom skewed splits with min-fraction constraints |
| **Label model** | Structural validity (deterministic) | Configurable expression (deterministic or stochastic) |
| **Concept missingness** | Not supported | MCAR, MAR, MNAR with configurable rates |
| **Label noise** | Not supported | Configurable noise rate |
| **Interventions** | Not built into pipeline | K-Flip with configurable budgets and human accuracy |
| **Alignment testing** | Not built into pipeline | Monotonicity + prediction constraints via CVXPY |
| **Grid search** | Manual | Automated via `robot_grid_search.py` |
| **Output** | Model checkpoint + printed metrics | Full run directory with JSON metrics, confusion matrices, checkpoints |

---

## Common Architecture

Both benchmarks share the same two-stage concept bottleneck architecture:

```
                    Stage 1                         Stage 2
Input (X) ----> [Concept Detector] ----> C_hat ----> [Frontend Model] ----> y_hat
                  (CNN / ViT)           (binary)      (Logistic Reg.)
```

### ConceptDetector

The universal concept detection wrapper:

```python
from concept_benchmark.models import ConceptDetector

cd = ConceptDetector(
    embedding_model=backbone,       # optional feature extractor (ViT, CNN, etc.)
    model=pre_built_model,          # or pass a pre-built joint model
    trainer=custom_trainer,         # pluggable training loop
    model_builder=custom_factory,   # or a factory function
)

cd.fit(train_data, valid_data,
       embed_params={"device": device},
       fit_params={"epochs": 10, "hidden_size": 360})

predictions = cd.predict(test_data, embed_params={"device": device})
```

### ConceptBasedModel

Combines a concept detector with a frontend for end-to-end inference:

```python
from concept_benchmark.models import ConceptBasedModel

cbm = ConceptBasedModel(
    concept_detector=cd,
    front_end_model=fe,
    propagate=False,    # Monte Carlo uncertainty propagation
)
```

### Interventions (K-Flip)

The K-Flip intervention strategy evaluates all subsets of concepts (up to size k) and selects the subset whose correction maximizes the probability of flipping the final prediction:

```python
from concept_benchmark.intervention import InterventionConfig, ConceptInterventionRunner
from concept_benchmark.kflip import KFlipInterventionStrategy

config = InterventionConfig(
    concept_budget=5,       # max concepts to intervene on globally
    max_concepts_per_instance=3,  # max per sample
    tau=0.2,                # flip probability threshold
)
strategy = KFlipInterventionStrategy(config=config, cbm=cbm)
runner = ConceptInterventionRunner(strategy=strategy)
results = runner.run(intervention_batch)
```

---

## Project Structure

```
concept-benchmark/
├── concept_benchmark/              # Main package
│   ├── data.py                     # ConceptDataset, ConceptDatasetSample
│   ├── models.py                   # ConceptDetector, FrontEndModel, CNN/ViT architectures
│   ├── train.py                    # DefaultConceptTrainer, training utilities
│   ├── intervention.py             # InterventionConfig, ConceptInterventionRunner
│   ├── kflip.py                    # KFlipInterventionStrategy
│   ├── metrics.py                  # Evaluation metrics
│   ├── cv.py                       # Cross-validation fold format (K{folds}N{rep})
│   ├── paths.py                    # Repo/data/results path helpers
│   └── synthetic/
│       ├── sudoku.py               # Sudoku dataset generator
│       ├── robot.py                # Robot dataset generator (image + text)
│       └── helper/
│           ├── sudoku_helper.py    # Board generation, corruption modes
│           ├── robot_catalog.py    # Robot feature catalog
│           ├── robot_draw.py       # Robot image rendering
│           └── textgen.py          # Text description generation
├── scripts/
│   ├── run_sudoku.py               # Sudoku dataset CLI
│   ├── sudoku_train.py             # Sudoku training script
│   ├── robot_image_training.py     # Robot full pipeline (data + train + eval)
│   ├── run_robot_demos.py          # Robot text/image demo runner
│   ├── robot_grid_search.py        # Automated experiment sweeps
│   ├── dataset_skewing.py          # Train/test splitting with concept skew
│   ├── robot_alignment.py          # Human alignment testing
│   ├── robot_interventions.py      # Intervention utilities
│   └── robot_utils.py              # Shared robot utilities
├── tests/                          # Unit tests
├── pyproject.toml                  # Project config (uv)
└── README.md
```

## Contributions

To add a new benchmark, create `concept_benchmark/synthetic/your_dataset.py`:

```python
from concept_benchmark.data import ConceptDataset

def create_synthetic_dataset(*args, **kwargs) -> ConceptDataset:
    # ... generate X, C, y, meta
    return ConceptDataset(X, C, y, meta)
```

Then open a pull request on GitHub.
