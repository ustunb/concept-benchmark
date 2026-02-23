# Concept Benchmark

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/assets/robot_banner.png" width="700" alt="Example robots from the robot benchmark">
</p>

**Concept Benchmark** is a Python package for benchmarking [concept bottleneck models](https://arxiv.org/abs/2007.04612) (CBMs). It provides synthetic datasets with fully-specified ground-truth concept labels, enabling controlled evaluation of CBM pipelines: concept detection accuracy, intervention effectiveness, robustness to noisy or missing annotations, and alignment between learned and expected concept-label relationships. For more details, see [our paper](https://arxiv.org/abs/TODO).

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Benchmarks](#benchmarks)
4. [Configuration](#configuration)
5. [Creating Your Own Benchmark](#creating-your-own-benchmark)
6. [Citation](#citation)

## Installation

```bash
git clone https://github.com/ustunb/concept-benchmark.git
cd concept-benchmark
./install.sh
source venv/bin/activate
```

You can also install directly via pip:

```bash
pip install -e .
```

## Quick Start

### Python API

```python
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

# Generate robot images with ground-truth concept labels (subconcept variant)
cfg = RobotBenchmarkConfig.default_subconcept()
cfg.seed = 1014
data = robot.setup_dataset(cfg)

# Train a CBM: concept detector (image -> concepts) + frontend (concepts -> label)
cbm = robot.train_cbm(cfg, data)

# Evaluate k-flip interventions: correct the k most impactful concepts per sample
results = robot.run_interventions(cfg, cbm, data)
print(results[["budget", "accuracy"]].to_string(index=False))
```

### CLI

```bash
# Run the full robot pipeline (subconcept variant, matching paper defaults)
cbm-benchmark robot --seed 1014 --subconcept

# Run with intervention regimes
cbm-benchmark robot --seed 1014 --subconcept --regimes baseline expert

# Paper-matching exact-k intervention strategy
cbm-benchmark robot --seed 1014 --subconcept --strategy exact_k --regimes baseline expert

# Robot text modality
cbm-benchmark robot-text --seed 1337

# Sudoku pipeline
cbm-benchmark sudoku --seed 171

# Run specific pipeline stages only
cbm-benchmark robot --seed 1014 --subconcept --stages setup cbm dnn
```

For fully-commented examples, see [`scripts/demo_robot.py`](scripts/demo_robot.py) and [`scripts/demo_sudoku.py`](scripts/demo_sudoku.py).

## Benchmarks

### Robot Classification

Classify synthetic robots into two species (**glorp** vs. **drent**) based on visual concepts. The label depends on three concepts (mouth type, foot shape, and knee presence), while other concepts like elbow shape and hand shape are spurious -- correlated with the label but not causally related.

<p align="center">
  <img src="docs/assets/robot_concepts.png" width="400" alt="Robot with annotated concepts">
</p>

Each robot is defined by 9 visual concepts. The foot shape concept has 10 subtypes (5 pointy, 5 flat), which can be provided to the model at different levels of granularity to test how concept resolution affects performance:

<p align="center">
  <img src="docs/assets/robot_foot_shapes.png" width="600" alt="Robot foot shape variations">
</p>

**Modalities.** The robot benchmark supports two input modalities. In **image** mode (`cbm-benchmark robot`), robots are rendered as pixel images (32x32 or 600x600) using pycairo, with configurable color variations. In **text** mode (`cbm-benchmark robot-text`), robots are described in natural language generated from a template corpus with SHA-256 deterministic synonym selection, ensuring reproducibility. Both modalities share the same concept structure and label rules.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `1014` / `1337` | Random seed (image / text) |
| `size` | `"medium"` | Image resolution: `"small"` (8px), `"medium"` (32px), `"large"` (600px). Image only. |
| `model_type` | `"stochastic"` | Label model: `"deterministic"` or `"stochastic"` |
| `subconcept` | `False` | Use fine-grained foot shape subtypes instead of binary pointy/flat |
| `concept_missing` | `0.0` | Fraction of concept labels to mask during training |
| `concept_missing_mech` | `"none"` | Missingness mechanism: `"none"`, `"mcar"`, or `"mnar"` |
| `intervention_budgets` | `[1, 3]` | Number of concepts to intervene on per sample |
| `difficulty` | `"hard"` | Corpus difficulty (text only) |
| `generic_rate` | `0.7` | Fraction of test set using concept-ambiguous text (text only) |

The full list of parameters is documented in `RobotBenchmarkConfig` and `RobotTextBenchmarkConfig` (see [`concept_benchmark/config.py`](concept_benchmark/config.py)).

#### Intervention Regimes

Intervention regimes control **where concepts come from** and **who corrects them**, simulating different real-world annotation scenarios:

| Regime | Concept Source | Intervention Source | What It Tests |
|--------|---------------|---------------------|---------------|
| `baseline` | Ground truth | Ground truth | Upper bound: perfect concepts + perfect corrections |
| `expert` | Ground truth | Noisy human (80% acc) | Expert corrections with realistic error rates |
| `subjective` | Noisy CBM (20% label noise) | Noisy human (80% acc) | Subjective concept definitions + noisy corrections |
| `machine` | LFCBM on GT descriptions | Noisy human (80% acc) | Machine-discovered concepts (CLIP-based) |
| `llm` | LFCBM on LLM descriptions | LLM (Gemini) | LLM-generated concept definitions + LLM corrections |
| `clip` | LFCBM on CLIP keywords | LLM (Gemini) | CLIP keyword concepts + LLM corrections |

```bash
cbm-benchmark robot --seed 1014 --subconcept --regimes baseline expert subjective machine

# LLM/CLIP regimes require a Gemini API key
export GEMINI_API_KEY=your_key_here
cbm-benchmark robot --seed 1014 --subconcept --regimes llm clip
```

```python
from concept_benchmark.config import RobotBenchmarkConfig

cfg = RobotBenchmarkConfig.default_subconcept()
cfg.seed = 1014
cfg.intervention_regimes = ["baseline", "expert", "subjective"]
cfg.intervention_strategy = "exact_k"  # paper-matching strategy
```

**Notes:**
- `machine`, `llm`, and `clip` regimes only work with `--subconcept` (12 concepts must match LFCBM dimensions)
- `llm` and `clip` regimes require `GEMINI_API_KEY`

### Sudoku Validation

Determine whether a 9x9 sudoku board is valid. The 27 concepts correspond to the validity of each row, column, and 3x3 block -- a board is valid if and only if all 27 concepts are true. This creates a naturally conjunctive relationship between concepts and the label, in contrast to the disjunctive structure of the robot benchmark.

<p align="center">
  <img src="docs/assets/sudoku_handwritten.png" width="400" alt="Sudoku board with handwritten digits and concept annotations">
</p>

Boards can be represented as tabular data (81-cell integer vectors) or rendered as images with handwritten digits. The image pipeline includes an OCR stage that learns to read digits from rendered boards before passing them to the concept model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `171` | Random seed for reproducibility |
| `n_samples` | `1000` | Number of boards to generate |
| `max_corrupt` | `9` | Maximum cells corrupted in invalid boards (higher = harder) |
| `valid_ratio` | `0.5` | Fraction of valid boards |
| `handwriting` | `True` | Render digits in handwritten style |
| `target_accuracy` | `0.9` | Target accuracy for selective classification |

The full list of parameters is documented in `SudokuBenchmarkConfig` (see [`concept_benchmark/config.py`](concept_benchmark/config.py)).

## Configuration

All experiment settings are managed through typed dataclasses in [`concept_benchmark/config.py`](concept_benchmark/config.py). Configs can be serialized to YAML for reproducibility:

```python
from concept_benchmark.config import RobotBenchmarkConfig

# Customize any parameter
cfg = RobotBenchmarkConfig(seed=42, epochs=100, concept_missing=0.2, concept_missing_mech="mcar")

# Save and reload configs for reproducibility
cfg.to_yaml("my_experiment.yaml")
loaded = RobotBenchmarkConfig.from_yaml("my_experiment.yaml")
```

```bash
# Or pass a config file from the command line
cbm-benchmark robot --config my_experiment.yaml
```

### Pipeline Stages

Each benchmark runs a sequence of stages. You can select specific stages with `--stages`:

| Benchmark | Default Stages |
|-----------|---------------|
| Robot | `setup cbm dnn intervene align collect` |
| Sudoku | `setup ocr cs dnn intervene selective align collect` |
| Robot Text | `setup cbm dnn intervene align collect` |

```bash
# Run only data generation and CBM training
cbm-benchmark robot --seed 1014 --subconcept --stages setup cbm
```

## Creating Your Own Benchmark

To add a new benchmark domain, you need three components:

1. **A data generator** that produces a `ConceptDataset(X, C, y, meta)` -- features, binary concept labels, target labels, and metadata.
2. **A config dataclass** (like `RobotBenchmarkConfig`) to hold experiment settings.
3. **A benchmark module** (like `concept_benchmark/benchmarks/robot.py`) with stage functions.

The core data object is `ConceptDataset`:

```python
from concept_benchmark.data import ConceptDataset
import numpy as np

X = np.random.randn(1000, 10).astype(np.float32)       # features
C = (np.random.rand(1000, 3) > 0.5).astype(np.int8)    # binary concepts
y = (C[:, 0] & C[:, 1]).astype(np.int32)                # labels from concepts

data = ConceptDataset(X, C, y, meta={
    "concepts": ["has_feature_a", "has_feature_b", "has_feature_c"],
    "classes": ["negative", "positive"],
    "data_type": "tabular",
})

data.generate_cvindices(seed=0)
data.split("K05N01", fold_num_validation=4, fold_num_test=5)
```

Once your data is in a `ConceptDataset`, the existing training and intervention infrastructure (`ConceptDetector`, `FrontEndModel`, `ConceptInterventionRunner`) works out of the box. See the built-in benchmarks in `concept_benchmark/benchmarks/` as templates.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{skirzynski2025concept,
  title={Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models},
  author={Skirzy\'{n}ski, Julian and Cheon, Harry and Kadekodi, Shreyas and Stewart, Meredith and Ustun, Berk},
  year={2025},
}
```
