# Concept Benchmark

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/assets/logo.svg" width="400" alt="Concept Benchmark logo">
</p>

[Concept bottleneck models](https://arxiv.org/abs/2007.04612) (CBMs) first predict human-interpretable concepts from raw inputs, then use those concepts to predict a label. This architecture lets users inspect and correct the model's reasoning at test time. However, CBM research relies on a small number of concept-annotated datasets that were not designed for this purpose, making it difficult to systematically evaluate new methods.

**Concept Benchmark** provides synthetic datasets where concepts and labels are fully specified during construction. Because we control the ground truth, we can vary concept granularity, annotation quality, and the labeling rule, and measure exactly how each factor affects CBM performance and the value of interventions.

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

A CBM has two learned components: a set of *concept detectors* that predict concepts from inputs (e.g., "has pointy feet"), and a *label predictor* that maps predicted concepts to the final label. At test time, a user can replace a predicted concept value with its true value -- this is called an *intervention*. The central question our benchmarks address is whether correcting *k* concepts actually improves the label prediction, and how that depends on factors like concept quality and annotation noise.

The following example runs the robot benchmark with default settings:

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

All experiment settings are managed through typed dataclasses in [`concept_benchmark/config.py`](concept_benchmark/config.py). Configs can be serialized to YAML for reproducibility:

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

We provide two benchmarks that represent two use cases where CBMs provide unique value. **Robot classification** models decision support: a human expert corrects the model's concept predictions to improve accuracy. **Sudoku validation** models automation: the system handles routine cases and defers uncertain ones to a human. In both cases, we control the ground truth, so we can measure exactly when and why interventions help or fail.

### Robot Classification

We consider a task to classify fictional robots as one of two species -- **Glorp** or **Drent** -- based on visual body features. The task is inspired by [Williams et al. (2010)](https://doi.org/10.1016/j.cogpsych.2010.01.002), who used these robots to study how humans discover new concepts. Each robot has 9 binary body features that serve as concepts. The label depends on three of them (mouth type, foot shape, and knee presence); the rest (e.g., elbow shape, hand shape) are present in the data but do not determine robot type.

<p align="center">
  <img src="docs/assets/robot_concepts.png" width="400" alt="Robot with annotated concepts">
</p>

Because we control the data generation, we can vary several aspects of the benchmark. We can change which concepts the model observes, the labeling rule, annotation quality (noise and missingness), and how interventions are performed (by an oracle, a noisy human expert, or a machine). We also support both image and text modalities (`cbm-benchmark robot` and `cbm-benchmark robot-text`).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drop_concepts` | `IDEAL_DROP` | Which concepts to exclude. Two presets are provided (`IDEAL_DROP` for 7 coarse concepts, `SUBCONCEPT_DROP` for 12 fine-grained foot subtypes), or define your own. |
| `subconcept` | `False` | Shortcut that switches `drop_concepts` to `SUBCONCEPT_DROP`. |
| `model_rule` | see `config.py` | Python expression that defines the labeling rule over concepts |
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

### Sudoku Validation

We consider the task of determining whether a 9x9 Sudoku board is valid -- i.e., contains the digits 1-9 exactly once in each row, column, and block. The 27 concepts correspond to the validity of each row, column, and 3x3 block. A board is valid if and only if all 27 concepts are true. This AND structure means that a single violated concept is enough to invalidate the board, which contrasts with the robot benchmark's weighted labeling rule.

<p align="center">
  <img src="docs/assets/sudoku_handwritten.png" width="400" alt="Sudoku board with handwritten digits and concept annotations">
</p>

This benchmark models an automation setting: the system handles routine cases and defers uncertain ones to a human reviewer. When the model abstains, a human can verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty. We measure selective accuracy (accuracy on kept predictions), coverage (fraction of predictions kept), and the cost of concept verifications. Because of the AND structure, each additional verification adds cost but yields only marginal coverage gains -- one incorrect concept is enough to fail the board.

We can vary the difficulty of the task (by controlling how many cells are corrupted in invalid boards), the input modality (tabular or handwritten digit images), and the reliability/coverage tradeoff (by setting a minimum accuracy threshold for kept predictions).

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

```python
from concept_benchmark.benchmarks import sudoku
from concept_benchmark.config import SudokuBenchmarkConfig

cfg = SudokuBenchmarkConfig(
    seed=171,
    max_corrupt=9,
    handwriting=True,
    target_accuracy=0.95,
)

sudoku.setup_dataset(cfg)
sudoku.train_ocr(cfg)
cs_model = sudoku.train_cs(cfg)
results = sudoku.run_interventions(cfg, cs_model)
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{skirzynski2026concept,
  title={Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models},
  author={Skirzy\'{n}ski, Julian and Cheon, Harry and Kadekodi, Shreyas and Stewart, Meredith and Ustun, Berk},
  year={2026},
}
```
