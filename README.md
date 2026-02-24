# Concept Benchmark

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/assets/logo.svg" width="400" alt="Concept Benchmark logo">
</p>

**Concept Benchmark** is a Python package for benchmarking [concept bottleneck models](https://arxiv.org/abs/2007.04612) (CBMs). It provides synthetic datasets with ground-truth concept labels, allowing users to vary concept granularity, annotation quality, and the labeling rule, and measure how each factor affects model performance and the value of interventions. The package includes two benchmarks -- robot classification (decision support) and Sudoku validation (automation) -- across image, text, and tabular modalities.

For more details, see [our paper](https://arxiv.org/abs/TODO).

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Benchmarks](#benchmarks)
4. [Citation](#citation)

## Installation

```bash
git clone https://github.com/ustunb/concept-benchmark.git
cd concept-benchmark
./install.sh
source venv/bin/activate
```

Verify the installation:

```bash
python -c "import concept_benchmark; print('OK')"
```

You can also install directly via pip (note: `./install.sh` also installs dev tools like `pytest` and `ruff`):

```bash
pip install -e .
```

## Quick Start

A CBM predicts concepts from inputs (e.g., "has pointy feet"), then predicts the label from those concepts. At test time, a user can correct mispredicted concepts -- this is called an *intervention*. The package lets you measure whether correcting *k* concepts improves the label prediction, and how that depends on concept quality and annotation noise.

```python
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

cfg = RobotBenchmarkConfig(seed=1014)
data = robot.setup_dataset(cfg)                # generate 32x32 robot images
cbm = robot.train_cbm(cfg, data)               # concept detectors + label predictor
dnn = robot.train_dnn(cfg, data)               # end-to-end baseline (no concepts)
results = robot.run_interventions(cfg, cbm, data)  # measure effect of corrections
```

The same pipeline runs from the CLI:

```bash
cbm-benchmark robot --seed 1014
cbm-benchmark sudoku --seed 171
```

See [`scripts/demo_robot.py`](scripts/demo_robot.py) and [`scripts/demo_sudoku.py`](scripts/demo_sudoku.py) for fully-commented examples.

### Configuration

Experiment settings are managed through typed dataclasses in [`concept_benchmark/config.py`](concept_benchmark/config.py). Configs can be serialized to YAML for reproducibility:

```python
from concept_benchmark.config import RobotBenchmarkConfig

cfg = RobotBenchmarkConfig(seed=42, epochs=100, concept_missing=0.2, concept_missing_mech="mcar")
cfg.to_yaml("my_experiment.yaml")
loaded = RobotBenchmarkConfig.from_yaml("my_experiment.yaml")
```

```bash
cbm-benchmark robot --config my_experiment.yaml
```

### Pipeline Stages

Each benchmark runs a sequence of stages. You can select specific stages with `--stages`:

| Stage | Benchmarks | What It Does |
|-------|-----------|--------------|
| `setup` | all | Generate synthetic dataset (images, text, or boards) |
| `ocr` | sudoku | Train a digit recognizer on rendered board images |
| `cbm` | robot, robot-text | Train concept detectors + label predictor |
| `cs` | sudoku | Train concept-supervised (CS) model |
| `dnn` | all | Train end-to-end baseline (input to label, no concepts) |
| `intervene` | all | Correct up to *k* predicted concepts per sample and measure label accuracy change |
| `selective` | sudoku | Compute selective accuracy and coverage at each confidence threshold |
| `align` | all | Retrain label predictor with sign constraints on concept weights and compare to unconstrained |
| `collect` | all | Aggregate per-stage results into a single CSV |

## Benchmarks

The package includes two benchmarks. **Robot classification** is a decision-support task where a human corrects the model's concept predictions to improve accuracy. **Sudoku validation** is an automation task where the system handles routine cases and defers uncertain ones to a human.

### Robot Classification

This benchmark targets decision-support settings where a human uses the model's concept predictions to improve their own decisions. The task is to predict the species of a fictional robot -- **Glorp** or **Drent** -- from its body features. Each robot has 9 binary features (mouth type, foot shape, knee presence, etc.). The ground-truth labeling rule, which features matter, and which are spurious are all configurable, modeling settings where the true relationship between features and labels is unknown. Available as image (`cbm-benchmark robot`) and text (`cbm-benchmark robot-text`) modalities.

<p align="center">
  <img src="docs/assets/robot_concepts.png" width="400" alt="Robot with annotated concepts">
</p>

The following example uses the subconcept variant (12 fine-grained foot subtypes instead of 7 coarse concepts), masks 20% of concept labels during training (MCAR), and tests whether imposing a sign constraint on the `has_knees` weight preserves or destroys the benefit of interventions.

```python
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig, SUBCONCEPT_DROP

cfg = RobotBenchmarkConfig(
    seed=1014,
    size="medium",                             # 32x32 pixel images
    model_type="stochastic",                   # stochastic labeling function
    subconcept=True,                           # 12 subconcepts instead of 7
    drop_concepts=list(SUBCONCEPT_DROP),        # which concepts to exclude
    spurious_features=["has_elbows", "hand_shape"],
    concept_missing=0.2,                       # mask 20% of concept labels
    concept_missing_mech="mcar",               # missing completely at random
    intervention_budgets=[1, 3],               # correct k=1 or k=3 concepts
    intervention_thresholds=[0.2],
    alignment_constraints={"has_knees": 1},    # force has_knees weight to be positive
)

data = robot.setup_dataset(cfg)
cbm = robot.train_cbm(cfg, data)
dnn = robot.train_dnn(cfg, data)

# correct up to k concept predictions per sample and measure label accuracy
results = robot.run_interventions(cfg, cbm, data)
print(results[["budget", "accuracy"]].to_string(index=False))

# retrain with the sign constraint and check whether interventions still help
align_stats = robot.align(cfg, cbm, data)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drop_concepts` | `IDEAL_DROP` | Which concepts to exclude. Two presets are provided (`IDEAL_DROP` for 7 coarse concepts, `SUBCONCEPT_DROP` for 12 fine-grained foot subtypes), or define your own. |
| `subconcept` | `False` | Shortcut that switches `drop_concepts` to `SUBCONCEPT_DROP`. |
| `model_rule` | see `config.py` | Python expression defining the labeling rule over concepts |
| `weights` | `{"mouth_type": 5, ...}` | Concept weights for the stochastic labeling function |
| `concept_missing` | `0.0` | Fraction of concept labels masked during training |
| `regimes` | `["baseline"]` | How interventions are performed: `baseline` (oracle), `expert` (noisy human), `subjective` (noisy concept labels + noisy human), `machine`/`llm`/`clip` (concepts discovered via [Label-Free CBM](https://arxiv.org/abs/2304.06129)). `llm`/`clip` require `GEMINI_API_KEY`. |

<details>
<summary>All parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `1014` / `1337` | Random seed (image / text) |
| `size` | `"medium"` | Image resolution: `"small"` (8px), `"medium"` (32px), `"large"` (600px). Image only. |
| `model_type` | `"stochastic"` | Labeling function: `"deterministic"` or `"stochastic"` |
| `concept_missing_mech` | `"none"` | Missingness mechanism: `"none"`, `"mcar"`, or `"mnar"` |
| `intervention_budgets` | `[1, 3]` | Number of concepts to correct per sample |
| `intervention_thresholds` | `[0.2, 0.4]` | Concept confidence thresholds that determine which concepts are candidates for intervention |
| `intervention_strategy` | `"kflip"` | `"kflip"` (up to *k* concepts) or `"exact_k"` (exactly *k*) |
| `alignment_constraints` | `{}` | Sign constraints on concept weights (e.g., `{"has_knees": 1}`). Retrains the label predictor and re-evaluates interventions. |
| `difficulty` | `"hard"` | Corpus difficulty (text only) |
| `generic_rate` | `0.7` | Fraction of test set using concept-ambiguous text (text only) |

</details>

See `RobotBenchmarkConfig` and `RobotTextBenchmarkConfig` in [`concept_benchmark/config.py`](concept_benchmark/config.py) for the full list.

> **Note:** The `llm` and `clip` regimes call the Gemini API at intervention time. Set your key before running:
> ```bash
> export GEMINI_API_KEY=your_key_here
> ```

### Sudoku Validation

This benchmark targets automation settings where the system handles routine cases and defers uncertain ones to a human. The task is to determine whether a 9x9 Sudoku board is valid, i.e., contains the digits 1-9 exactly once in each row, column, and block. The 27 concepts correspond to the validity of each row, column, and 3x3 block. A board is valid if and only if all 27 concepts are true (AND structure), so a single violated concept is enough to invalidate the board. When the model abstains, a human can verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty.

<p align="center">
  <img src="docs/assets/sudoku_handwritten.png" width="400" alt="Sudoku board with handwritten digits and concept annotations">
</p>

The following example generates 1000 boards with handwritten digits, corrupting up to 9 cells in invalid boards, and requires 95% accuracy on kept predictions.

```python
from concept_benchmark.benchmarks import sudoku
from concept_benchmark.config import SudokuBenchmarkConfig

cfg = SudokuBenchmarkConfig(
    seed=171,
    max_corrupt=9,
    handwriting=True,
    target_accuracy=0.95,
)

sudoku.setup_dataset(cfg)                   # generate boards with handwritten digits
sudoku.train_ocr(cfg)                       # train digit recognizer
cs_model = sudoku.train_cs(cfg)             # train concept-supervised model
results = sudoku.run_interventions(cfg, cs_model)  # measure effect of concept verification
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_corrupt` | `9` | Number of cells corrupted in invalid boards (higher values produce subtler errors) |
| `handwriting` | `True` | Render digits in handwritten style (adds an OCR stage) |
| `target_accuracy` | `0.9` | Minimum accuracy required on kept predictions |

<details>
<summary>All parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `171` | Random seed |
| `n_samples` | `1000` | Number of boards to generate |
| `valid_ratio` | `0.5` | Fraction of valid boards |
| `intervention_thresholds` | `[0.2, 0.4, 0.6, 0.8]` | Concept confidence thresholds that determine which concepts are candidates for verification |

</details>

See `SudokuBenchmarkConfig` in [`concept_benchmark/config.py`](concept_benchmark/config.py) for the full list.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{skirzynski2026concept,
  title={Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models},
  author={Skirzy\'{n}ski, Julian and Cheon, Harry and Kadekodi, Shreyas and Stewart, Meredith and Ustun, Berk},
  year={2026},
}
```
