# Concept Benchmark

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/assets/logo.svg" width="400" alt="Concept Benchmark logo">
</p>

**Concept Benchmark** is a Python package for benchmarking [concept bottleneck models](https://arxiv.org/abs/2007.04612) (CBMs). It provides synthetic datasets with ground-truth concept labels, allowing users to vary concept granularity, annotation quality, and the labeling rule, and measure how each factor affects model performance and the value of interventions. The package includes two benchmarks -- robot classification (decision support) and Sudoku validation (automation) -- across image, text, and tabular modalities.

## Installation

The package requires the **cairo** graphics library. Install it first:

```bash
# macOS
brew install cairo pkg-config

# Ubuntu / Debian
sudo apt-get install libcairo2-dev pkg-config python3-dev

# Fedora / RHEL
sudo dnf install cairo-devel pkg-config python3-devel
```

Then install the package:

```bash
pip install concept-benchmark
```

Or install from source (includes training/evaluation code and pipeline scripts):

```bash
git clone https://github.com/ustunb/concept-benchmark.git
cd concept-benchmark
pip install -e ".[experiments]"
```

If you use [uv](https://docs.astral.sh/uv/), `uv sync` works too and also installs dev/docs dependencies.

> **Package vs. repo:** `pip install concept-benchmark` gives you dataset generation and exploration (`concept_benchmark/`). To train models, run interventions, and use the full evaluation pipelines, clone the repo and install with `.[experiments]` — the `experiments/` package is then importable directly (no `PYTHONPATH` needed).

> **Device support:** The package auto-detects the best available device (CUDA → MPS → CPU). Apple Silicon (MPS) is fully supported. Override with `export PYTORCH_DEVICE=cpu`.

Verify the installation:

```bash
python3 -c "import concept_benchmark; print('OK')"
```

## Quick Start

A concept bottleneck model (CBM) first predicts interpretable *concepts* from inputs (e.g., "has pointy feet"), then uses those concepts to predict the final label. This two-stage design lets users inspect and correct the model's reasoning at test time — an operation called an *intervention*. This package gives you synthetic datasets where the ground-truth concepts are known, so you can measure exactly how much interventions help under different conditions.

### Robot Classification

The robot benchmark classifies fictional robots — **Glorps** vs. **Drents** — from their body features. Generate a dataset with rendered images and concept annotations:

```python
from concept_benchmark import RobotDatasetGenerator

dataset = RobotDatasetGenerator(
    seed=1014,                # reproducibility
    subconcept=True,          # 12 fine-grained concepts (default: 7 coarse)
    model_type="stochastic",  # probabilistic labeling (or "deterministic")
    size="medium",            # image resolution: "small" (8px), "medium" (32px), "large" (600px)
    draw=True,                # render robot images (default) — set False to skip for quick exploration
).generate()

print(dataset.training.C.shape)   # (3800, 12) — concept annotations
print(dataset.training.concepts)
# ['head_shape', 'body_shape', 'has_knees', 'has_antennae', 'ears_shape',
#  'mouth_type', 'foot_shape_flat_trapezoid', 'foot_shape_flat_square',
#  'foot_shape_flat_5sided', 'foot_shape_pointy_rounded',
#  'foot_shape_pointy_square', 'foot_shape_pointy_4sided']
```

With `subconcept=True`, the 9 raw body features are expanded into 12 binary concepts — for example, `foot_shape` (6 values) becomes 6 one-hot columns. With `subconcept=False` (default), you get 7 coarse concepts instead. The `draw=True` flag renders each robot as a 32×32 PNG image stored in `dataset.training.X`.

Each split (`training`, `validation`, `test`) is a `ConceptDataset` with attributes `X` (images), `C` (concept matrix), and `y` (labels). Convert to a DataFrame to see what the data looks like:

```python
dataset.training.to_dataframe().head(2)
#    head_shape  body_shape  has_knees  ...  foot_shape_pointy_4sided  label  class
# 0           0           0          0  ...                         0      1  glorp
# 1           0           0          0  ...                         1      1  glorp
```

For interactive browsing with [Renumics Spotlight](https://github.com/Renumics/spotlight) (`pip install concept-benchmark[explore]`):

```python
dataset.training.explore()  # opens in the browser
```

<p align="center">
  <img src="docs/assets/robot_samples.png" width="600" alt="Sample Glorps and Drents with concept annotations">
</p>

**Train a CBM.** The repo includes the building blocks for a full concept bottleneck model — a concept detector (images → concepts) and a label predictor (concepts → label):

```python
import numpy as np
from concept_benchmark import RobotDatasetGenerator
from concept_benchmark.utils import set_deterministic_seed
from experiments.models import (
    ConceptDetector, FrontEndModel, ConceptBasedModel, RobotConceptClassifier,
)
from experiments.utils import determine_device, get_loader_config, patch_macos_dataloader

set_deterministic_seed(1014)
patch_macos_dataloader()
device = determine_device()
loader_config = get_loader_config(device)

dataset = RobotDatasetGenerator(seed=1014, subconcept=True, draw=True).generate()

# Step 1: train concept detector (images → concepts)
n_concepts = dataset.training.n_concepts
cd = ConceptDetector(model=RobotConceptClassifier(num_concepts=n_concepts, input_size=32))
cd.fit(dataset.training, dataset.validation,
       fit_params={"epochs": 50, "lr": 1e-3, "patience": 10, "device": str(device), **loader_config})

# Step 2: train label predictor (concepts → label)
fe = FrontEndModel()
fe.fit(dataset.training.C, dataset.training.y)

# Step 3: combine into a CBM and evaluate
cbm = ConceptBasedModel(concept_detector=cd, front_end_model=fe)
predictions = cbm.predict(dataset.test)
accuracy = np.mean(predictions == dataset.test.y)
print(f"CBM accuracy: {accuracy:.4f}")
# CBM accuracy: 0.7812
```

For a quick exploration using only the pip package, see [`examples/robot_quickstart.py`](examples/robot_quickstart.py). For the full neural CBM pipeline with interventions (requires cloning the repo), see [`examples/robot_pipeline_example.py`](examples/robot_pipeline_example.py).

### Sudoku Validation

The Sudoku benchmark determines whether a 9×9 board is valid. The 27 concepts correspond to the validity of each row, column, and 3×3 block — a board is valid if and only if *all* 27 are true:

```python
from concept_benchmark import SudokuDatasetGenerator

dataset = SudokuDatasetGenerator(
    seed=171,             # reproducibility
    n_samples=100,        # number of boards (renders images, ~35 s)
    max_corrupt=9,        # cells swapped in invalid boards (higher = subtler errors)
    valid_ratio=0.5,      # fraction of valid boards
).generate()

dataset.training.explore()        # interactive viewer with board images
print(dataset.training.C.shape)   # (60, 27) — 27 concept annotations
print(dataset.training.concepts)  # ['row_valid_1', 'row_valid_2', ..., 'block_valid_9']
```

Inspect the data — each row is one board, with a binary flag for each of the 27 structural checks:

```python
df = dataset.training.to_dataframe()
show_cols = list(dataset.training.concepts[:5]) + ["label"]
print(df[show_cols])
#      row_valid_1  row_valid_2  row_valid_3  row_valid_4  row_valid_5  label
# 0              1            1            1            1            1      1
# ..           ...          ...          ...          ...          ...    ...
# 301            1            0            0            1            1      0
```

<p align="center">
  <img src="docs/assets/sudoku_samples.png" width="600" alt="Sample valid and invalid Sudoku boards with handwritten digits">
</p>

For a quick exploration using only the pip package, see [`examples/sudoku_quickstart.py`](examples/sudoku_quickstart.py). For the full neural CS model pipeline with selective classification and interventions (requires cloning the repo), see [`examples/sudoku_pipeline_example.py`](examples/sudoku_pipeline_example.py).


### Benchmark Your Own Model

Have your own concept detector or label predictor? Wrap them to plug into the evaluation pipeline:

```python
import numpy as np
from experiments.models import ConceptDetector, FrontEndModel, ConceptBasedModel

# Wrap your concept detector — predict() receives a ConceptDatasetSample
class MyConceptDetector(ConceptDetector):
    def __init__(self, my_model):
        super().__init__()
        self._model = my_model

    def predict(self, dataset, **kwargs):
        """Return (N, n_concepts) float array in [0, 1]."""
        return self._model.predict_concept_probs(dataset.X)

# Wrap your label predictor — C is binary (0/1), not probabilities
class MyFrontEnd(FrontEndModel):
    def __init__(self, my_clf):
        super().__init__()
        self._clf = my_clf

    def predict(self, C):
        return self._clf.predict(C)

    def predict_proba(self, C):
        return self._clf.predict_proba(C)

# Assemble, evaluate, and run interventions
cbm = ConceptBasedModel(
    concept_detector=MyConceptDetector(my_concept_model),
    front_end_model=MyFrontEnd(my_classifier),
)
predictions = cbm.predict(test)
accuracy = np.mean(predictions == test.y)
# Expected: compare against built-in CBM accuracy (0.7812 for subconcept)
```

For the full guide — including running interventions, alignment, bypassing the concept detector, and baseline comparisons — see [`docs/benchmark-your-model.md`](docs/benchmark-your-model.md).

## Evaluation

The repo includes tools for evaluating CBMs beyond raw accuracy — interventions, alignment constraints, and selective classification. These require cloning the repo and running `uv sync`.

### Interventions

Interventions correct the model's concept predictions at test time. The `ConceptInterventionRunner` + `KFlipInterventionStrategy` evaluates subsets of up to *k* concepts per sample and selects the most impactful correction. Expected results (seed=1014):

| budget (k) | DNN | ideal (7 concepts) | subconcept (12 concepts) |
|------------|------|---------------------|--------------------------|
| 0 | 0.8746 | 0.8673 | 0.7812 |
| 1 | — | 0.9736 | 0.9212 |
| 3 | — | 0.9769 | 0.9439 |

The package supports six intervention regimes that simulate different annotation scenarios: `baseline` (oracle), `expert` (noisy human), `subjective` (noisy labels + noisy human), `machine`/`llm`/`clip` (machine-discovered concepts via [Label-Free CBM](https://arxiv.org/abs/2304.06129)). See [`docs/interventions.md`](docs/interventions.md) for details.

### Alignment

Alignment constraints force concept weights to match expected directions (e.g., `has_knees` → positive). The paper shows this preserves training accuracy but **destroys** intervention benefit — aligned subconcept CBM drops from +16% gain to -8% at k=3.

### Selective Classification

For Sudoku, the key metric is selective classification: the model abstains on uncertain predictions to achieve high accuracy on kept samples. The CS model is very stable across seeds (sel_acc ~0.97, coverage ~99%). The DNN is highly variable — about 40% of seeds produce random-chance performance.

### Reproducing Paper Results

```bash
python scripts/robot_pipeline.py --seed 1014 --subconcept   # see --help for all flags
python scripts/sudoku_pipeline.py --seed 171
```

Results are fully deterministic for a given seed when using `set_deterministic_seed()`. Training takes ~3 min per model on MPS (Apple Silicon), ~5–10 min on CPU. The pipeline supports flags for intervention regimes (`--regimes expert subjective machine`), concept missingness (`--concept-missing 0.2`), running specific stages (`--stages cbm dnn intervene`), and more.

For complete end-to-end examples, see [`examples/robot_pipeline_example.py`](examples/robot_pipeline_example.py) and [`examples/sudoku_pipeline_example.py`](examples/sudoku_pipeline_example.py).

## Benchmarks

### Robot Classification

This benchmark targets decision-support settings where a human uses the model's concept predictions to improve their own decisions. The task is to predict the species of a fictional robot -- **Glorp** or **Drent** -- from its body features. Each robot has 9 binary features (mouth type, foot shape, knee presence, etc.). The default labeling rule is: Glorp if mouth is closed, foot is pointy, and robot has knees (all three); Drent otherwise. The labeling function can be deterministic or stochastic (probabilistic), controlled via the `model_type` parameter. Which features matter and which are excluded (via `drop_concepts`) are configurable, mimicking real-world settings where the true relationship between features and labels is unknown. Available as image and text modalities.

<p align="center">
  <img src="docs/assets/robot_concepts.png" width="400" alt="Robot with annotated concepts">
</p>

> **Note:** `cd.predict()` returns concept **probabilities** in [0, 1], not binary predictions. `ConceptBasedModel.predict()` handles thresholding internally. For manual interventions, see [`docs/interventions.md`](docs/interventions.md).

All parameters below can be passed directly to `RobotDatasetGenerator()` or as CLI flags to `robot_pipeline.py`. For the full list, see [`concept_benchmark/config.py`](concept_benchmark/config.py).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_type` | `"image"` | `"image"` (render robot PNGs) or `"text"` (generate text descriptions). |
| `label_formula` | `{("mouth_type","closed"): 5, ("foot_shape","pointy"): 8, ("has_knees","true"): -5, "intercept": 2}` | Labeling function. Score = `Σ wᵢ · 1[fᵢ = vᵢ] + intercept`. |
| `model_type` | `"stochastic"` | `"deterministic"`: Glorp if score ≥ 0. `"stochastic"`: Glorp ~ Bernoulli(σ(scalar × score)). |
| `drop_concepts` | `IDEAL_DROP` | Which concepts to exclude. Two presets: `IDEAL_DROP` for 7 coarse concepts, `SUBCONCEPT_DROP` for 12 fine-grained concepts. |
| `concept_missing` | `0.0` | Fraction of concept labels masked during training. |
| `regimes` | `["baseline"]` | How interventions are performed: `baseline` (oracle), `expert` (noisy human), `subjective` (noisy concept labels + noisy human), `machine`/`llm`/`clip` (Label-Free CBM). |

<details>
<summary>Remaining parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subconcept` | `False` | Shortcut that switches `drop_concepts` to `SUBCONCEPT_DROP` (12 fine-grained concepts). |
| `seed` | `1014` / `1337` | Random seed (image / text) |
| `size` | `"medium"` | Image resolution: `"small"` (8px), `"medium"` (32px), `"large"` (600px). Image only. |
| `samples_per_instance` | `4` | Number of images per unique robot configuration. Total dataset size = unique configs × this value. |
| `color_mode` | `"color"` | `"color"` or `"grayscale"`. Image only. |
| `model_scalar` | `4.2` | Sigmoid temperature for stochastic labeling (higher = more deterministic) |
| `skew_specs` | (see config) | List of dicts specifying class-balance constraints for training data (e.g., minimum fraction of specific concept values). |
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

All parameters below can be passed directly to `SudokuDatasetGenerator()` or as CLI flags to `sudoku_pipeline.py`. For the full list, see [`concept_benchmark/config.py`](concept_benchmark/config.py).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_corrupt` | `9` | Number of cells corrupted in invalid boards (higher values produce subtler errors) |
| `data_type` | `"image"` | `"image"` evaluates on OCR-inferred digits (adds OCR stage); `"tabular"` evaluates on ground-truth digit values (no OCR). Training always uses ground-truth values. |
| `handwriting` | `True` | Render digits in handwritten style (only applies when `data_type="image"`) |
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

## Citation

If you use this package in your research, please cite:

```bibtex
@article{skirzynski2026concept,
  title={Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models},
  author={Skirzy\'{n}ski, Julian and Cheon, Harry and Kadekodi, Shreyas and Stewart, Meredith and Ustun, Berk},
  year={2026},
}
```
