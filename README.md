# Concept Benchmark

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/assets/logo.svg" width="400" alt="Concept Benchmark logo">
</p>

**Concept Benchmark** is a Python package for benchmarking [concept bottleneck models](https://arxiv.org/abs/2007.04612) (CBMs). It provides synthetic datasets with fully-specified ground-truth concept labels, enabling controlled evaluation of CBM pipelines: concept detection accuracy, intervention effectiveness, robustness to noisy or missing annotations, and alignment between learned and expected concept-label relationships.

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

A CBM predicts concepts from inputs (e.g., "has pointy feet"), then predicts the label from those concepts. At test time, a human can correct mispredicted concepts — this is called an *intervention*. The key question is whether correcting *k* concepts actually improves the label prediction.

The example below runs this pipeline on the robot benchmark with default settings:

```python
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

cfg = RobotBenchmarkConfig(seed=1014)
data = robot.setup_dataset(cfg)                # generate 32x32 robot images
cbm = robot.train_cbm(cfg, data)               # concept detector + frontend
dnn = robot.train_dnn(cfg, data)               # end-to-end baseline
results = robot.run_interventions(cfg, cbm, data)  # test concept corrections
```

The same pipeline runs from the CLI in one command:

```bash
cbm-benchmark robot --seed 1014
cbm-benchmark sudoku --seed 171
```

For fully-commented examples, see [`scripts/demo_robot.py`](scripts/demo_robot.py) and [`scripts/demo_sudoku.py`](scripts/demo_sudoku.py).

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
| `cbm` | robot, robot-text | Train concept detector (input → concepts) + frontend (concepts → label) |
| `cs` | sudoku | Train concept-supervised (CS) model (concepts predicted from board features) |
| `dnn` | all | Train end-to-end neural network baseline (input → label, no concepts) |
| `intervene` | all | Evaluate concept interventions: correct up to *k* concept predictions and measure label accuracy change |
| `selective` | sudoku | Compute selective accuracy and coverage at each confidence threshold |
| `align` | all | Retrain label predictor with sign constraints on concept weights (e.g., force `has_knees` to be positive) and compare to unconstrained |
| `collect` | all | Aggregate per-stage results into a single CSV |

## Benchmarks

Two benchmarks, two use cases. **Robot classification** is a decision-support task: a human corrects the model's concept predictions. **Sudoku validation** is an automation task: the model handles easy cases and defers the rest. Both are configurable -- concept granularity, label rules, annotation noise, and intervention regimes are all parameters, not fixed choices.

### Robot Classification

Classify synthetic robots into two species (**glorp** vs. **drent**) based on visual concepts. By default, the label depends on three concepts (mouth type, foot shape, and knee presence), while other concepts like elbow shape and hand shape are spurious -- present in the data but not part of the label rule.

<p align="center">
  <img src="docs/assets/robot_concepts.png" width="400" alt="Robot with annotated concepts">
</p>

**Everything is configurable.** The default setup is just one point in a large configuration space. Each robot has 9 visual features, several with multiple subtypes (foot shape has 10, hand shape has 6). Concept granularity, the label rule, annotation quality, and intervention regimes are all parameters. Both image (`cbm-benchmark robot`) and text (`cbm-benchmark robot-text`) modalities are supported.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drop_concepts` | `IDEAL_DROP` | Concept names to exclude. Two presets are provided (`IDEAL_DROP`, `SUBCONCEPT_DROP`), or define your own subset. |
| `subconcept` | `False` | Convenience flag: switches `drop_concepts` to the `SUBCONCEPT_DROP` preset (12 fine-grained foot subtypes instead of 7 coarse concepts). |
| `model_rule` | see `config.py` | Python expression defining the label rule over concepts |
| `weights` | `{"mouth_type": 5, ...}` | Concept weights for the stochastic label model |
| `concept_missing` | `0.0` | Fraction of concept labels to mask during training |
| `regimes` | `["baseline"]` | Intervention regimes: `baseline` (perfect), `expert` (noisy human), `subjective` (noisy concepts), `machine`/`llm`/`clip` (discovered via [Label-Free CBM](https://arxiv.org/abs/2304.06129)). `llm`/`clip` require `GEMINI_API_KEY`. |

<details>
<summary>All parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `1014` / `1337` | Random seed (image / text) |
| `size` | `"medium"` | Image resolution: `"small"` (8px), `"medium"` (32px), `"large"` (600px). Image only. |
| `model_type` | `"stochastic"` | Label model: `"deterministic"` or `"stochastic"` |
| `concept_missing_mech` | `"none"` | Missingness mechanism: `"none"`, `"mcar"`, or `"mnar"` |
| `intervention_budgets` | `[1, 3]` | Number of concepts to intervene on per sample |
| `intervention_thresholds` | `[0.2, 0.4]` | Concept confidence thresholds for intervention candidates |
| `intervention_strategy` | `"kflip"` | `"kflip"` (up-to-k, tries sizes 1 through *k*) or `"exact_k"` (exactly *k* concepts) |
| `alignment_constraints` | `{}` | Sign constraints on concept weights (e.g., `{"has_knees": 1}`). Retrains frontend and re-evaluates interventions. |
| `difficulty` | `"hard"` | Corpus difficulty (text only) |
| `generic_rate` | `0.7` | Fraction of test set using concept-ambiguous text (text only) |

</details>

The full list of parameters is documented in `RobotBenchmarkConfig` and `RobotTextBenchmarkConfig` (see [`concept_benchmark/config.py`](concept_benchmark/config.py)).

> **Note:** The `llm` and `clip` regimes make live Gemini API calls at intervention time. Set your key before running:
> ```bash
> export GEMINI_API_KEY=your_key_here
> ```

The example below uses the **subconcept** variant (12 fine-grained foot shape features instead of the default 7 coarse concepts), masks 20% of concept labels during training (MCAR), and tests alignment constraints -- matching the paper's experimental setup.

```python
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig, SUBCONCEPT_DROP

cfg = RobotBenchmarkConfig(
    seed=1014,
    size="medium",                             # 32x32 pixel images
    model_type="stochastic",                   # stochastic label rule
    subconcept=True,                           # use 12 subconcepts (vs 7 ideal)
    drop_concepts=list(SUBCONCEPT_DROP),        # which concepts to exclude
    spurious_features=["has_elbows", "hand_shape"],
    concept_missing=0.2,                       # mask 20% of concept labels
    concept_missing_mech="mcar",               # missing completely at random
    intervention_budgets=[1, 3],               # intervene on k=1, k=3 concepts
    intervention_thresholds=[0.2],
    alignment_constraints={"has_knees": 1},    # test sign constraint on has_knees
)

# 1. Generate dataset: robot images with concept annotations and train/val/test splits
data = robot.setup_dataset(cfg)

# 2. Train CBM: concept detector (image -> concept probabilities) + frontend (concepts -> label)
cbm = robot.train_cbm(cfg, data)

# 3. Train DNN baseline: end-to-end model that bypasses concepts entirely
dnn = robot.train_dnn(cfg, data)

# 4. Intervene: for each test sample, correct up to k concept predictions and measure
#    whether the label prediction improves
results = robot.run_interventions(cfg, cbm, data)
print(results[["budget", "accuracy"]].to_string(index=False))

# 5. Alignment: retrain the frontend with a monotonicity constraint (has_knees must have
#    positive weight) and check whether interventions still help under the constraint
align_stats = robot.align(cfg, cbm, data)
```

### Sudoku Validation

Determine whether a 9x9 sudoku board is valid. The 27 concepts correspond to the validity of each row, column, and 3x3 block -- a board is valid if and only if all 27 concepts are true. This AND structure contrasts with the robot benchmark's disjunctive label rule.

<p align="center">
  <img src="docs/assets/sudoku_handwritten.png" width="400" alt="Sudoku board with handwritten digits and concept annotations">
</p>

**Automation use case.** The sudoku benchmark models an automation setting: the system handles routine cases and defers uncertain ones to a human. When the model abstains, interventions ask a human to verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty. The key metrics are selective accuracy (on kept predictions), coverage (fraction kept), and net work automated (coverage minus the cost of concept verifications). The AND structure means each additional verification adds cost but only marginal coverage gain, since a single incorrect concept fails the entire board.

**Configurability.** Task difficulty, input modality (tabular or handwritten images), and the reliability/coverage tradeoff are all parameters.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_corrupt` | `9` | Maximum cells corrupted in invalid boards (higher = harder to detect) |
| `handwriting` | `True` | Render digits in handwritten style (enables OCR pipeline) |
| `target_accuracy` | `0.9` | Minimum accuracy demanded on kept predictions |

<details>
<summary>All parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `171` | Random seed for reproducibility |
| `n_samples` | `1000` | Number of boards to generate |
| `valid_ratio` | `0.5` | Fraction of valid boards |
| `intervention_thresholds` | `[0.2, 0.4, 0.6, 0.8]` | Concept confidence thresholds for intervention candidates |

</details>

The full list of parameters is documented in `SudokuBenchmarkConfig` (see [`concept_benchmark/config.py`](concept_benchmark/config.py)).

The example below generates 1000 boards with handwritten digits, corrupting up to 9 cells in invalid boards. It demands 95% selective accuracy, so the model abstains on uncertain predictions and interventions ask a human to verify specific row/column/block concepts.

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
