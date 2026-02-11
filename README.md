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

The label model determines how the binary species label (glorp vs. drent) is assigned to each robot. There are two types, and the typical workflow is to **start with deterministic** and then use the resulting concept-label relationships to calibrate the **stochastic** model.

**Deterministic**: A Python expression evaluated per robot row. Use this first to understand which concept patterns produce which labels:
```python
model="'glorp' if (int(row['mouth_type']=='closed') + int(row['has_knees']=='true')) >= 2 else 'drent'"
model_type="deterministic",
```

**Stochastic**: Uses a logistic function to produce probabilistic labels. The weights, scalar, and intercept are typically tuned based on the deterministic model's concept-label mapping so that the stochastic model targets a desired `P(glorp)` for each feature combination:
```python
model_type="stochastic",
logit_scalar=4.2,       # controls sharpness of the decision boundary
logit_intercept=-2,     # bias term
logit_weights={"mouth_type": 5, "foot_shape": 8, "has_knees": -5},
```

For example, you might first run the deterministic model to identify that `has_knees=true` and `foot_shape=pointy` predict glorp, then set `logit_weights` accordingly with positive values for glorp-predictive concepts and negative values for drent-predictive ones. The `logit_scalar` and `logit_intercept` are then adjusted to achieve the desired noise level around the decision boundary.

#### Text Modality

Robots can also be represented as natural language descriptions instead of images. The text pipeline generates descriptive paragraphs for each robot configuration using template-based rendering with concept-specific placeholders.

**Generating text data** from an existing image dataset:

```python
from concept_benchmark.synthetic.robot import create_robot_text_dataset

text_data = create_robot_text_dataset(
    source=image_dataset,         # ConceptDataset or DataFrame with robot catalog
    variants_per_row=3,           # text variations per robot instance
    include_color=True,           # include color descriptions
    text_mode="unstructured",     # "unstructured", "structured", or "llm"
    rng_seed=0,
)
```

**Text generation modes**:

| Mode | Description |
|------|-------------|
| `"unstructured"` | Template-based with natural language fillers and synonym variation (default) |
| `"structured"` | Simple fixed templates with minimal variation |
| `"llm"` | Uses an LLM API (Gemini, OpenAI, or Anthropic) to generate captions |

**Template system**: Templates are stored as JSONL files in `concept_benchmark/synthetic/helper/static/text_templates/`. Each line contains a `"when"` condition (matching robot features) and a `"text"` field with placeholders like `{HEAD_NAT}`, `{BODY_NAT}`, `{FEET_NAT}`, etc. that are filled with natural language descriptors (e.g., `{HEAD_NAT}` becomes "boxy" or "dome-like"). Available template sets:

- `HardCorpus.jsonl` -- main template set
- `HardCorpus_NoAnt.jsonl` -- templates with antennae concept redacted
- `HardCorpus_EarsGeneric.jsonl` -- generic (non-concept-bearing) ear descriptions
- `HardCorpus_FootGeneric.jsonl` -- generic foot descriptions

**Text concept detection** uses a transformer-based model (`TextConceptDetector`) that learns to predict robot concepts from text. The default backbone is `distilbert-base-uncased`, with optional alternatives (`bert-tiny`, `bert-mini`, `bert-small`). The detector is trained with per-concept binary classification heads and optimized thresholds via ROC analysis.

**Running the full text pipeline** is handled by `scripts/run_robot_demos.py`, which orchestrates:

1. Baseline DNN training (`scripts/robot_baseline.py`) -- trains a standard text classifier as a baseline
2. Concept detection (`scripts/gen_text_samples.py`) -- trains a `TextConceptDetector` on text descriptions
3. CBM evaluation -- builds a concept bottleneck model (text -> concepts -> label) and evaluates it
4. Intervention testing -- runs K-Flip interventions at specified budgets (e.g., `[0, 1, 2, 5, 10]`)

```bash
python scripts/run_robot_demos.py \
  --modality text \
  --text_model distilbert-base-uncased \
  --seed 1337 \
  --difficulty hard \
  --budgets 0,1,2,5,10 \
  --run_tag my_experiment
```

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

## Running Large-Scale Experiments

The codebase includes dedicated pipelines for running large-scale experiments that sweep over concept noise, target accuracy, and concept missingness. These orchestrated pipelines produce the paper's result tables.

### Big Demo Pipeline (`big-demo` branch)

The `big-demo` branch contains a unified pipeline for both sudoku and robot experiments in `scripts/big_demo/`. The entry point is a shell script that runs three stages:

```bash
scripts/big_demo/run_pipeline_and_evals.sh
```

This executes:

1. **Pipeline options** -- `pipeline_options.py` enumerates and runs all parameter combinations across dataset generation, DNN training, concept detector training, and frontend training
2. **Conceptual safeguards evaluation** -- `eval_conceptual_safeguards.py` evaluates intervention strategies on sudoku
3. **Score intervention evaluation** -- `eval_score_intervention.py` evaluates interventions on robot

The pipeline sweeps over these axes (configured in `scripts/big_demo/utils.py`):

| Axis | Values |
|------|--------|
| Concept noise | `0.0, 0.05, 0.10, ..., 0.30` |
| Target accuracy | `easy` (1.0), `medium` (0.8), `hard` (0.6) |
| Missingness mechanism | `none`, `mcar`, `mnar` |
| Missingness rate | `0.05, 0.10, ..., 0.30` |

You can also run stages selectively:

```bash
# Enumerate all commands (dry run)
python scripts/big_demo/pipeline_options.py --dataset sudoku robot

# Run only sudoku dataset setup + training
python scripts/big_demo/pipeline_options.py --execute --dataset sudoku --stages setup_dataset train_concept_detector train_front_end

# Run only robot
python scripts/big_demo/pipeline_options.py --execute --dataset robot
```

Each stage is a standalone script that accepts CLI arguments:

| Script | Purpose |
|--------|---------|
| `setup_sudoku_dataset.py` | Generate sudoku datasets with concept noise and target accuracy |
| `setup_robot_dataset.py` | Generate robot datasets with concept noise and target accuracy |
| `train_concept_detectors.py` | Train concept detectors (CNN for robot, CNN for sudoku) with missingness |
| `train_dnn.py` | Train DNN baselines |
| `train_front_end.py` | Train frontend models (concepts -> label) |
| `eval_conceptual_safeguards.py` | Evaluate conceptual safeguards intervention strategy |
| `eval_score_intervention.py` | Evaluate score-based intervention strategy |

### Sudoku Demo Pipeline (`scripts/sudoku_demo/`)

A modular sudoku-specific pipeline for generating OCR-based sudoku datasets and running concept-based experiments:

```bash
python scripts/sudoku_demo/pipeline.py --stages setup cs dnn intervene --seed 171
```

This runs individual scripts as subprocesses:

1. **Setup** (`make_ocr_dataset.py`) -- generates OCR sudoku datasets with handwriting-style digit rendering
2. **CS training** (`train_cs.py`) -- trains concept-based models on sudoku
3. **DNN training** (`train_dnn.py`) -- trains DNN baselines
4. **Intervention** (`intervene.py`) -- runs intervention experiments

Additional flags:

```bash
# Run only dataset generation
python scripts/sudoku_demo/pipeline.py --stages setup

# Continue past failures
python scripts/sudoku_demo/pipeline.py --stages setup cs dnn intervene --ignore-errors
```

Default settings are centralized in `scripts/sudoku_demo/utils.py` (board size `n=3`, 1000 samples, 20 training epochs, early stopping patience of 5).

### Robot Demo Pipeline (`scripts/robot_demo/`)

A modular robot-specific pipeline in `scripts/robot_demo/`. The orchestrator is:

```bash
python scripts/robot_demo/pipeline.py --stages setup cbm dnn intervene --seed 1014
```

This runs individual scripts as subprocesses:

1. **Setup** (`setup_dataset_robot.py`) -- generates robot datasets (ideal + subconcept versions)
2. **CBM training** (`train_cbm.py`) -- trains concept-based models with optional missingness (MCAR/MNAR)
3. **DNN training** (`train_dnn.py`) -- trains DNN baselines
4. **Intervention** (`intervene.py`) -- runs K-Flip interventions at thresholds `[0.2, 0.4]`
5. **Metrics** (`calc_metrics.py`) -- consolidates results into `robot_demo_results.csv`

Additional flags:

```bash
# Run with robot image drawing enabled
python scripts/robot_demo/pipeline.py --stages setup --draw

# Run with concept missingness
python scripts/robot_demo/pipeline.py --stages cbm intervene --missing

# Continue past failures
python scripts/robot_demo/pipeline.py --stages setup cbm dnn intervene --ignore-errors
```

Default settings are centralized in `scripts/robot_demo/utils.py` (`DEFAULT_ROBOT_SETTINGS`), including concepts, label model, image size, and seed.

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
│   ├── robot_utils.py              # Shared robot utilities
│   ├── robot_demo/                 # Modular robot experiment pipeline
│   │   ├── pipeline.py             # Orchestrator (setup, cbm, dnn, intervene)
│   │   ├── setup_dataset_robot.py  # Robot dataset generation
│   │   ├── train_cbm.py            # CBM training with missingness
│   │   ├── train_dnn.py            # DNN baseline training
│   │   ├── intervene.py            # K-Flip intervention experiments
│   │   ├── calc_metrics.py         # Results consolidation
│   │   └── utils.py                # Default settings and helpers
│   └── sudoku_demo/                # Modular sudoku experiment pipeline
│       ├── pipeline.py             # Orchestrator (setup, cs, dnn, intervene)
│       ├── make_ocr_dataset.py     # OCR sudoku dataset generation
│       ├── train_cs.py             # Concept-based model training
│       ├── train_dnn.py            # DNN baseline training
│       ├── intervene.py            # Intervention experiments
│       └── utils.py                # Default settings and helpers
├── fonts/                          # Font files for handwriting rendering
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
