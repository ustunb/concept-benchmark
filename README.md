# Concept Benchmark

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/assets/logo.svg" width="400" alt="Concept Benchmark logo">
</p>

**Concept Benchmark** is a Python package for benchmarking [concept bottleneck models](https://arxiv.org/abs/2007.04612) (CBMs). It provides synthetic datasets with ground-truth concept labels, allowing users to vary concept granularity, annotation quality, and the labeling rule, and measure how each factor affects model performance and the value of interventions. The package includes two benchmarks -- robot classification (decision support) and Sudoku validation (automation) -- across image, text, and tabular modalities.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Benchmarks](#benchmarks)
4. [CLI Reference](#cli-reference)
5. [Citation](#citation)

## Installation

```bash
git clone https://github.com/ustunb/concept-benchmark.git
cd concept-benchmark
./install.sh
source venv/bin/activate
```

Verify the installation:

```bash
python3 -c "import concept_benchmark; print('OK')"
```

<details>
<summary>Manual install / troubleshooting</summary>

Robot image generation requires the cairo C library. If `./install.sh` can't install it automatically, install it manually first:

```bash
# macOS
brew install cairo pkg-config

# Ubuntu / Debian
sudo apt-get install libcairo2-dev pkg-config python3-dev

# Fedora / RHEL
sudo dnf install cairo-devel pkg-config python3-devel
```

Then install the Python package:

```bash
pip install -e .
```

</details>

## Quick Start

A CBM predicts concepts from inputs (e.g., "has pointy feet"), then predicts the label from those concepts. At test time, a user can correct mispredicted concepts -- this is called an *intervention*. The package lets you measure whether correcting *k* concepts improves the label prediction, and how that depends on concept quality and annotation noise.

The fastest way to run the benchmark is from the command line. This generates data, trains models, runs interventions, and saves a results CSV — with automatic caching so repeated runs skip completed stages:

```bash
cbm-benchmark robot --seed 1014 --stages setup cbm dnn intervene collect
```

Results are saved to `results/robot_ideal_seed1014_2d0aa353_results.csv`. Filter to `model == "cbm"` and `threshold == 0.2` to see accuracy numbers.

The same pipeline from Python:

```python
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

cfg = RobotBenchmarkConfig(seed=1014)
robot.run(cfg, stages=["setup", "cbm", "dnn", "intervene", "collect"])
```

Under the hood, `robot.run()` calls individual functions that you can also use directly to inspect intermediate objects:

```python
import numpy as np
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

cfg = RobotBenchmarkConfig(seed=1014)
data = robot.setup_dataset(cfg)                # generate 32x32 robot images
cbm = robot.train_cbm(cfg, data)               # concept detectors + label predictor
dnn = robot.train_dnn(cfg, data)               # end-to-end baseline (no concepts)
results = robot.run_interventions(cfg, cbm, data)  # measure effect of corrections

# CBM baseline (no interventions)
cbm_acc = float(np.mean(cbm.predict(data.test) == data.test.y))
print(f"CBM (k=0): {cbm_acc:.4f}")
# Intervention gains at threshold=0.2
print(results.query("threshold == 0.2")[["budget", "accuracy"]].to_string(index=False))
```

Expected output:
```
CBM (k=0): 0.8673
 budget  accuracy
      1    0.9736
      3    0.9769
      7    0.9769
```

See [`scripts/demo_robot.py`](scripts/demo_robot.py) and [`scripts/demo_sudoku.py`](scripts/demo_sudoku.py) for fully-commented examples.


## Benchmarks

The package includes two benchmarks. **Robot classification** is a decision-support task where a human corrects the model's concept predictions to improve accuracy. **Sudoku validation** is an automation task where the system handles routine cases and defers uncertain ones to a human.

### Robot Classification

This benchmark targets decision-support settings where a human uses the model's concept predictions to improve their own decisions. The task is to predict the species of a fictional robot -- **Glorp** or **Drent** -- from its body features. Each robot has 9 binary features (mouth type, foot shape, knee presence, etc.). The default labeling rule is: Glorp if mouth is closed, foot is pointy, and robot has knees (all three); Drent otherwise. Which features matter and which are spurious are configurable, mimicking real-world settings where the true relationship between features and labels is unknown. Available as image (`cbm-benchmark robot`) and text (`cbm-benchmark robot-text`) modalities.

<p align="center">
  <img src="docs/assets/robot_concepts.png" width="400" alt="Robot with annotated concepts">
</p>

The following example uses the subconcept variant (which splits foot_shape into 5 fine-grained subtypes, yielding 12 concepts instead of the default 7), and tests whether imposing a sign constraint on the `has_knees` weight preserves or destroys the benefit of interventions.

```python
import numpy as np
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

cfg = RobotBenchmarkConfig(
    seed=1014,
    subconcept=True,                           # use fine-grained foot subtypes (12 instead of 7)
    intervention_budgets=[1, 3],               # correct k=1 or k=3 concepts per sample
    intervention_thresholds=[0.2],
    alignment_constraints={"has_knees": 1},    # force has_knees weight to be positive
)

data = robot.setup_dataset(cfg)
cbm = robot.train_cbm(cfg, data)
dnn = robot.train_dnn(cfg, data)
results = robot.run_interventions(cfg, cbm, data)
align_stats = robot.align(cfg, cbm, data)

cbm_acc = float(np.mean(cbm.predict(data.test) == data.test.y))
print(f"CBM (k=0): {cbm_acc:.4f}")
print(results[["budget", "accuracy"]].to_string(index=False))

from concept_benchmark.paths import results_dir
cfg.to_yaml(results_dir / "my_experiment.yaml")  # save config for CLI use
```

Expected output:
```
CBM (k=0): 0.7812
 budget  accuracy
      1    0.9212
      3    0.9439
     12    0.9439
```

To re-run this experiment from the CLI (with automatic caching):

```bash
cbm-benchmark robot --config results/my_experiment.yaml
```

The most important parameters used in the config above are listed below. For the full list, see `RobotBenchmarkConfig` in [`concept_benchmark/config.py`](concept_benchmark/config.py) or the fully-commented [`scripts/demo_robot.py`](scripts/demo_robot.py).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drop_concepts` | `IDEAL_DROP` | Which concepts to exclude. Two presets are provided: `IDEAL_DROP` for 7 coarse concepts (binary foot_shape), `SUBCONCEPT_DROP` for 12 concepts (5 fine-grained foot subtypes). |
| `subconcept` | `False` | Shortcut that switches `drop_concepts` to `SUBCONCEPT_DROP`. |
| `model_rule` | see `config.py` | Python expression defining the labeling rule. Default: Glorp if `(mouth_closed + foot_pointy + has_knees) >= 3`. |
| `weights` | `{"mouth_type": 5, "foot_shape": 8, "has_knees": -5}` | Concept weights for the stochastic labeling function. |
| `concept_missing` | `0.0` | Fraction of concept labels masked during training. |
| `regimes` | `["baseline"]` | How interventions are performed: `baseline` (oracle), `expert` (noisy human), `subjective` (noisy concept labels + noisy human), `machine`/`llm`/`clip` (concepts discovered via [Label-Free CBM](https://arxiv.org/abs/2304.06129)). |

<details>
<summary>Remaining parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `1014` / `1337` | Random seed (image / text) |
| `size` | `"medium"` | Image resolution: `"small"` (8px), `"medium"` (32px), `"large"` (600px). Image only. |
| `model_type` | `"stochastic"` | Labeling function: `"deterministic"` or `"stochastic"` |
| `concept_missing_mech` | `"none"` | Missingness mechanism: `"none"`, `"mcar"`, or `"mnar"` |
| `intervention_budgets` | `[1, 3]` | Number of concepts to correct per sample |
| `intervention_thresholds` | `[0.2, 0.4]` | Concepts whose predicted probability is within this distance of 0.5 are candidates for intervention |
| `intervention_strategy` | `"kflip"` | `"kflip"` (up to *k* concepts) or `"exact_k"` (exactly *k*) |
| `alignment_constraints` | `{}` | Sign constraints on concept weights (e.g., `{"has_knees": 1}`). Retrains the label predictor and re-evaluates interventions. |
| `difficulty` | `"hard"` | Corpus difficulty (text only) |
| `generic_rate` | `0.7` | Fraction of test set using concept-ambiguous text (text only) |

</details>

> **Note:** The `llm` and `clip` regimes call the Gemini API at intervention time. Set your key before running:
> ```bash
> export GEMINI_API_KEY=your_key_here
> ```

### Sudoku Validation

This benchmark targets automation settings where the system handles routine cases and defers uncertain ones to a human. The task is to determine whether a 9x9 Sudoku board is valid, i.e., contains the digits 1-9 exactly once in each row, column, and block. The 27 concepts correspond to the validity of each row, column, and 3x3 block. A board is valid if and only if all 27 concepts are true (AND structure), so a single violated concept is enough to invalidate the board. When the model abstains, a human can verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty.

<p align="center">
  <img src="docs/assets/sudoku_handwritten.png" width="400" alt="Sudoku board with handwritten digits and concept annotations">
</p>

The following example generates 1000 boards with handwritten digits, corrupting up to 9 cells in invalid boards. The concept-supervised (CS) model -- the Sudoku equivalent of a CBM -- predicts 27 binary concepts, then a label predictor determines board validity. The selective classification stage finds a confidence threshold that achieves at least 95% accuracy on kept predictions.

```python
from concept_benchmark.benchmarks import sudoku
from concept_benchmark.config import SudokuBenchmarkConfig

cfg = SudokuBenchmarkConfig(
    seed=171,
    max_corrupt=9,                             # cells corrupted in invalid boards
    handwriting=True,                          # render with handwritten digits
    target_accuracy=0.95,                      # minimum accuracy on kept predictions
)

sudoku.setup_dataset(cfg)                      # generate boards + handwritten digit images
sudoku.train_ocr(cfg)                          # train digit recognizer on cell crops
cs_model = sudoku.train_cs(cfg)                # concept-supervised model (27 concepts -> valid/invalid)
dnn = sudoku.train_dnn(cfg)                    # end-to-end baseline (no concepts)
results = sudoku.run_interventions(cfg, cs_model)
sel = sudoku.compute_selective_results(cfg)     # selective accuracy and coverage

# Filter to the target accuracy threshold
t95 = sel[sel["target_accuracy"] == 0.95]
print(t95[["model", "selective_acc", "selective_cov"]].to_string(index=False))

from concept_benchmark.paths import results_dir
cfg.to_yaml(results_dir / "my_experiment.yaml")  # save config for CLI use
```

Expected output:
```
model  selective_acc  selective_cov
  dnn       0.923077          0.065
   cs       0.959596          0.990
```

To re-run this experiment from the CLI (with automatic caching):

```bash
cbm-benchmark sudoku --config results/my_experiment.yaml
```

The most important parameters are listed below. For the full list, see `SudokuBenchmarkConfig` in [`concept_benchmark/config.py`](concept_benchmark/config.py) or the fully-commented [`scripts/demo_sudoku.py`](scripts/demo_sudoku.py).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_corrupt` | `9` | Number of cells corrupted in invalid boards (higher values produce subtler errors) |
| `handwriting` | `True` | Render digits in handwritten style (adds an OCR stage) |
| `target_accuracy` | `0.9` | Minimum accuracy required on kept predictions |

<details>
<summary>Remaining parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `171` | Random seed |
| `n_samples` | `1000` | Number of boards to generate |
| `valid_ratio` | `0.5` | Fraction of valid boards |
| `intervention_thresholds` | `[0.2, 0.4, 0.6, 0.8]` | Concept confidence thresholds that determine which concepts are candidates for verification |

</details>

## CLI Reference

All benchmarks are run via `cbm-benchmark <benchmark>`. Use `cbm-benchmark <benchmark> --help` to see all options. All outputs (datasets, model weights, intervention CSVs, summary tables) are saved under `results/`.

### Pipeline Stages

Each benchmark runs a sequence of stages. Use `--stages` to run a subset. The `setup` stage generates the synthetic dataset. The `collect` stage produces a single results table (e.g., `results/robot_ideal_seed1014_2d0aa353_results.csv`) with all accuracy numbers across models, intervention budgets, and alignment variants.

```bash
# retrain models on existing data (skip data generation)
cbm-benchmark robot --stages cbm dnn intervene align collect

# rerun interventions with different regimes (models already trained)
cbm-benchmark robot --subconcept --regimes baseline expert --stages intervene collect
```

| Benchmark | Stages (in order) |
|-----------|-------------------|
| `robot` | `setup` · `cbm` · `dnn` · `intervene` · `align` · `collect` |
| `sudoku` | `setup` · `ocr` · `cs` · `dnn` · `intervene` · `selective` · `align` · `collect` |
| `robot-text` | `setup` · `cbm` · `dnn` · `lfcbm` · `intervene` · `align` · `collect` |

### Flags

| Flag | Benchmarks | Description |
|------|-----------|-------------|
| `--seed` | all | Random seed (defaults: robot 1014, sudoku 171, robot-text 1337) |
| `--stages` | all | Which stages to run (default: all) |
| `--config` | all | Path to YAML config file. CLI flags like `--regimes` and `--strategy` can further override values loaded from the file. |
| `--subconcept` | robot | Use subconcept variant (12 concepts with fine-grained foot subtypes instead of 7 coarse) |
| `--regimes` | robot, robot-text | Intervention regimes: `baseline`, `expert`, `subjective`, `machine`, `llm`, `clip` |
| `--strategy` | robot, robot-text | `kflip` (up to *k*) or `exact_k` (exactly *k* concepts) |
| `--concept-missing` | robot | Fraction of concept labels to mask (e.g. `0.2`) |
| `--concept-missing-mech` | robot | Missingness mechanism: `none`, `mcar`, or `mnar` |
| `--force-setup` | all | Regenerate all data (images, boards) from scratch, even if cached |
| `--force-retrain` | robot | Retrain LFCBM/subjective models even if cached |
| `--lfcbm` | robot-text | Also run the Label-Free CBM variant |
| `--llm-api-key` | robot | API key for LLM provider (alternative to `GEMINI_API_KEY` env var) |
| `--dry-run` | all | Print configuration and exit without running |
| `-v` / `-q` | all | Verbose / quiet output |

## Citation

If you use this package in your research, please cite:

```bibtex
@article{skirzynski2026concept,
  title={Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models},
  author={Skirzy\'{n}ski, Julian and Cheon, Harry and Kadekodi, Shreyas and Stewart, Meredith and Ustun, Berk},
  year={2026},
}
```
