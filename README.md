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
    image_transform,       # rendered PNG images (options: cell_px, font_size, handwriting)
)
```

### Running a Single Experiment

Run the full CBM pipeline (data generation + concept model training + DNN baseline + evaluation) in one command:

```bash
python scripts/sudoku_pipeline.py --n 3 --n-samples 1000 --seed 42 --epochs 20
```

For large-scale sweeps with OCR image datasets, see [Sudoku Demo Pipeline](#sudoku-demo-pipeline-scriptssudoku_demo) below.

### Sudoku Model Architectures

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| **ViT + MLP head** | ViT-Base-Patch16-224 backbone with per-concept MLP heads | Image data (224x224) |
| **GroupPoolingCNN** | CNN with group-pooling over sudoku units | Tabular data |

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

The label model determines how the binary species label (glorp vs. drent) is assigned to each robot.

**Deterministic**: A Python expression evaluated per robot row:
```python
model="'glorp' if (int(row['mouth_type']=='closed') + int(row['has_knees']=='true')) >= 2 else 'drent'"
model_type="deterministic",
```

**Stochastic**: Uses a logistic function to produce probabilistic labels. Weights are typically calibrated from a deterministic model's concept-label mapping:
```python
model_type="stochastic",
logit_scalar=4.2,       # controls sharpness of the decision boundary
logit_intercept=-2,     # bias term
logit_weights={"mouth_type": 5, "foot_shape": 8, "has_knees": -5},
```

#### Text Modality

Robots can also be represented as natural language descriptions instead of images.

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

**Template system**: Templates are JSONL files in `concept_benchmark/synthetic/helper/static/text_templates/`. Each line has a `"when"` condition and a `"text"` field with placeholders (e.g., `{HEAD_NAT}`, `{BODY_NAT}`) that are filled with natural language descriptors. Available template sets:

- `HardCorpus.jsonl` -- main template set
- `HardCorpus_NoAnt.jsonl` -- templates with antennae concept redacted
- `HardCorpus_EarsGeneric.jsonl` -- generic (non-concept-bearing) ear descriptions
- `HardCorpus_FootGeneric.jsonl` -- generic foot descriptions

**Text concept detection** uses a transformer-based model (`TextConceptDetector`) that learns to predict robot concepts from text. The default backbone is `distilbert-base-uncased`, with optional alternatives (`bert-tiny`, `bert-mini`, `bert-small`). The detector is trained with per-concept binary classification heads and optimized thresholds via ROC analysis.

### Running a Single Experiment

**Image pipeline** -- full CBM workflow (data generation, concept detector training, frontend training, intervention, alignment):

```bash
python scripts/robot_image_pipeline.py
```

**Text pipeline** -- text-modality CBM workflow (text generation, concept detection, evaluation, intervention):

```bash
python scripts/robot_text_pipeline.py
```

For large-scale sweeps over noise, missingness, and target accuracy, see [Robot Demo Pipeline](#robot-demo-pipeline-scriptsrobot_demo) below.

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

The codebase includes dedicated pipelines for running large-scale experiments that sweep over concept noise, target accuracy, and concept missingness.

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
│   ├── sudoku_pipeline.py          # Sudoku single-experiment pipeline
│   ├── robot_image_pipeline.py     # Robot image single-experiment pipeline
│   ├── robot_text_pipeline.py      # Robot text single-experiment pipeline
│   ├── utils/                      # Shared utilities for pipelines
│   ├── robot_demo/                 # Large-scale robot experiment pipeline
│   │   ├── pipeline.py             # Orchestrator (setup, cbm, dnn, intervene)
│   │   └── ...
│   └── sudoku_demo/                # Large-scale sudoku experiment pipeline
│       ├── pipeline.py             # Orchestrator (setup, cs, dnn, intervene)
│       └── ...
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
