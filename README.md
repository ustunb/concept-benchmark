# Concept Benchmark

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange)](https://github.com/ustunb/concept-benchmark)

A framework for generating synthetic benchmarks for [concept bottleneck models](https://arxiv.org/abs/2007.04612) (CBMs). Define concepts, generate data, train models, and evaluate interventions — all with ground-truth concept labels.

<p align="center">
  <img src="docs/assets/robot_banner.png" width="700" alt="Example robots from the robot benchmark">
</p>

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Built-in Benchmarks](#built-in-benchmarks)
4. [Running Experiments](#running-experiments)
5. [Configuration](#configuration)
6. [Creating Your Own Benchmark](#creating-your-own-benchmark)
7. [Citation](#citation)

## Installation

```bash
git clone <repo-url>
cd concept-benchmark
./install.sh
source venv/bin/activate
```

Or manually: `pip install -e .`

## Quick Start

Run a full benchmark (generate data, train models, evaluate interventions) in a few lines:

```python
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

# Run the full robot pipeline with paper defaults
robot.run()

# Or customize and run specific stages
cfg = RobotBenchmarkConfig(seed=42, epochs=20)
data = robot.setup_dataset(cfg)
cbm = robot.train_cbm(cfg, data)
results = robot.run_interventions(cfg, cbm, data)
```

```bash
# Or from the command line
cbm-benchmark robot --seed 1014 --stages setup cbm dnn intervene
cbm-benchmark sudoku --seed 171 --stages setup ocr cs dnn intervene selective
```

## Built-in Benchmarks

### Robot Classification

Classify synthetic robots into two species (**glorp** vs. **drent**) based on visual concepts like head shape, foot shape, and body proportions. Supports image and text modalities.

<p align="center">
  <img src="docs/assets/robot_concepts.png" width="400" alt="Robot with annotated concepts">
</p>

Each robot is defined by **9 configurable visual concepts**. The foot shape concept supports 10 subtypes (5 pointy, 5 flat) to test concept granularity:

<p align="center">
  <img src="docs/assets/robot_foot_shapes.png" width="600" alt="Robot foot shape variations">
</p>

- **Deterministic or stochastic** label models
- **Image** (32px or 600px) and **text** (template-based or LLM-generated) modalities
- Up to 921,600 unique robot instances with color variation

### Sudoku Validation

Determine whether a Sudoku board is valid. Concepts are the 27 row/column/block validity constraints. Boards can be rendered with handwritten digits, candidate annotations, and highlighted constraint regions:

<p align="center">
  <img src="docs/assets/sudoku_handwritten.png" width="400" alt="Sudoku board with handwritten digits and concept annotations">
</p>

- **27 ground-truth concepts** (9 rows + 9 columns + 9 blocks)
- **Tabular or image** representations (with optional handwriting rendering)
- Configurable board size, corruption level, and valid/invalid ratio

## Running Experiments

### CLI

The `cbm-benchmark` command runs full pipelines end-to-end:

```bash
# Robot: generate data, train CBM + DNN, run interventions
cbm-benchmark robot --seed 1014 --stages setup cbm dnn intervene

# Sudoku: generate data, train OCR + concept model + DNN, intervene, compute selective metrics
cbm-benchmark sudoku --seed 171 --stages setup ocr cs dnn intervene selective

# Robot with subconcept variant and no missingness sweep
cbm-benchmark robot --subconcept --no-missing --stages setup cbm intervene

# Load settings from a YAML file
cbm-benchmark robot --config my_config.yaml
```

### Python API

Each benchmark exposes a programmatic API via `concept_benchmark.benchmarks`:

```python
from concept_benchmark.benchmarks import robot, sudoku
from concept_benchmark.config import RobotBenchmarkConfig, SudokuBenchmarkConfig

# Run the full robot pipeline with default settings (seed=1014)
robot.run()

# Run specific stages
robot.run(stages=["setup", "cbm"])

# Run sudoku with custom config
cfg = SudokuBenchmarkConfig(seed=42, max_corrupt=21, epochs=30)
sudoku.run(cfg, stages=["setup", "ocr", "cs", "dnn", "intervene", "selective"])
```

Individual stages can also be called directly:

```python
from concept_benchmark.benchmarks.robot import setup_dataset, train_cbm, train_dnn, run_interventions
from concept_benchmark.config import RobotBenchmarkConfig

config = RobotBenchmarkConfig.default_ideal()
data = setup_dataset(config)          # generate + skew + save
cbm = train_cbm(config, data)         # train concept detector + frontend
dnn_weights = train_dnn(config, data) # train end-to-end DNN baseline
results_df = run_interventions(config, cbm, data)  # evaluate interventions
```

```python
from concept_benchmark.benchmarks.sudoku import (
    setup_dataset, train_ocr, train_cs, train_dnn,
    run_interventions, compute_selective_results,
)
from concept_benchmark.config import SudokuBenchmarkConfig

config = SudokuBenchmarkConfig(max_corrupt=21)
setup_dataset(config)                  # generate boards + images
train_ocr(config)                      # train digit recognizer
cs_model = train_cs(config)            # train concept supervision model
dnn_weights = train_dnn(config)        # train DNN baseline
interv_df = run_interventions(config)  # run conceptual safeguards
sel_df = compute_selective_results(config)  # selective accuracy at multiple thresholds
```

### Pipeline Stages

**Robot** stages: `setup`, `cbm`, `dnn`, `intervene`

| Stage | What it does |
|-------|-------------|
| `setup` | Generate robot images, apply skewed train/val/test splits |
| `cbm` | Train concept detector + frontend (+ MCAR/MNAR variants) |
| `dnn` | Train end-to-end DNN baseline |
| `intervene` | Run k-flip interventions at multiple budgets and thresholds |

**Sudoku** stages: `setup`, `ocr`, `cs`, `dnn`, `intervene`, `selective`

| Stage | What it does |
|-------|-------------|
| `setup` | Generate sudoku boards (tabular + image) |
| `ocr` | Train OCR digit recognizer on board images |
| `cs` | Train concept supervision model (concept detector + frontend) |
| `dnn` | Train end-to-end DNN baseline |
| `intervene` | Run conceptual safeguards interventions |
| `selective` | Compute selective accuracy/coverage at multiple target thresholds |

## Configuration

Benchmark settings are managed via typed dataclasses in `concept_benchmark.config`. Each config controls data generation, training hyperparameters, intervention settings, and file paths.

### `RobotBenchmarkConfig`

Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `seed` | `1014` | Random seed for reproducibility |
| `size` | `"medium"` | Image resolution: `"small"` (8px), `"medium"` (32px), `"large"` (600px) |
| `samples_per_instance` | `4` | Color variations per robot design |
| `model_type` | `"stochastic"` | Label model: `"deterministic"` or `"stochastic"` |
| `concepts` | 9 visual concepts | Dict mapping concept names to their possible values |
| `drop_concepts` | 10 foot subtypes | Concepts to merge (ideal = drop subtypes, subconcept = keep some) |
| `subconcept` | `False` | Use finer-grained foot shape concepts |
| `epochs` | `50` | Training epochs for CBM and DNN |
| `lr` | `1e-3` | Learning rate |
| `patience` | `10` | Early stopping patience |
| `intervention_budgets` | `[1, 3]` | Number of concepts to intervene on |
| `intervention_thresholds` | `[0.2, 0.4]` | Uncertainty thresholds for intervention |
| `concept_missing` | `0.0` | Fraction of concepts to mask (0 = none) |
| `concept_missing_mech` | `"none"` | Missingness mechanism: `"none"`, `"mcar"`, or `"mnar"` |

```python
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.benchmarks import robot

# Paper defaults
cfg = RobotBenchmarkConfig.default_ideal()       # ideal concept set
cfg = RobotBenchmarkConfig.default_subconcept()   # finer-grained foot concepts

# Custom experiment
cfg = RobotBenchmarkConfig(seed=42, epochs=100, lr=5e-4, size="large")
robot.run(cfg, stages=["setup", "cbm", "dnn", "intervene"])

# Missingness experiment
cfg = RobotBenchmarkConfig(concept_missing=0.2, concept_missing_mech="mcar")
robot.train_cbm(cfg)
```

### `SudokuBenchmarkConfig`

Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `seed` | `171` | Random seed |
| `n` | `3` | Board size (3 = standard 9x9 sudoku) |
| `n_samples` | `1000` | Number of sudoku boards to generate |
| `max_corrupt` | `9` | Max cells corrupted in invalid boards |
| `valid_ratio` | `0.5` | Fraction of valid boards |
| `data_type` | `"tabular"` | Data modality: `"tabular"` or `"image"` |
| `epochs` | `20` | DNN training epochs |
| `cs_epochs` | `100` | Concept supervision model training epochs |
| `cs_patience` | `20` | CS early stopping patience |
| `target_accuracy` | `0.9` | Target accuracy for selective classification |
| `concept_missing` | `0.0` | Fraction of concepts to mask |
| `concept_missing_mech` | `"none"` | Missingness mechanism |

```python
from concept_benchmark.config import SudokuBenchmarkConfig
from concept_benchmark.benchmarks import sudoku

# Paper defaults
cfg = SudokuBenchmarkConfig.default()

# Harder task (more corruption)
cfg = SudokuBenchmarkConfig(max_corrupt=21, cs_epochs=200)
sudoku.run(cfg, stages=["setup", "ocr", "cs", "dnn", "intervene", "selective"])
```

### Saving and Loading Configs

Configs can be serialized to YAML for reproducibility:

```python
cfg = RobotBenchmarkConfig(seed=42, epochs=100)
cfg.to_yaml("my_experiment.yaml")

# Later, or on another machine
loaded = RobotBenchmarkConfig.from_yaml("my_experiment.yaml")
robot.run(loaded)
```

Or passed via CLI: `cbm-benchmark robot --config my_experiment.yaml`

## Creating Your Own Benchmark

To add a new domain beyond robot and sudoku, you need three things:

1. **A data generator** that returns a `ConceptDataset(X, C, y, meta)` — features, binary concepts, labels, and metadata.
2. **A config dataclass** (like `RobotBenchmarkConfig`) to hold your experiment settings.
3. **A benchmark module** (like `concept_benchmark/benchmarks/robot.py`) with `setup_dataset()`, `train_cbm()`, `train_dnn()`, `run_interventions()`, and `run()` functions.

The core data object is `ConceptDataset`:

```python
from concept_benchmark.data import ConceptDataset
import numpy as np

X = np.random.randn(1000, 10).astype(np.float32)       # features
C = (np.random.rand(1000, 3) > 0.5).astype(np.int8)    # binary concepts
y = (C[:, 0] & C[:, 1]).astype(np.int32)                # labels

data = ConceptDataset(X, C, y, meta={
    "concepts": ["has_feature_a", "has_feature_b", "has_feature_c"],
    "classes": ["negative", "positive"],
    "data_type": "tabular",
})

data.generate_cvindices(seed=0)
data.split("K05N01", fold_num_validation=4, fold_num_test=5)
```

Once your data is in a `ConceptDataset`, the existing training and intervention infrastructure (`ConceptDetector`, `FrontEndModel`, `ConceptInterventionRunner`) works out of the box. See `concept_benchmark/benchmarks/robot.py` and `concept_benchmark/benchmarks/sudoku.py` as templates for wiring everything together into a pipeline.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{skirzynski2025concept,
  title={Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models},
  author={Skirzy\'{n}ski, Julian and Cheon, Harry and Kadekodi, Shreyas and Stewart, Meredith and Ustun, Berk},
  year={2025},
}
```
