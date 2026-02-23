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
4. [Creating Your Own Benchmark](#creating-your-own-benchmark)
5. [Citation](#citation)

## Installation

```bash
git clone https://github.com/ustunb/concept-benchmark.git
cd concept-benchmark
./install.sh
source venv/bin/activate
```

You can also install directly via pip (note: `./install.sh` also installs dev tools like `pytest` and `ruff`):

```bash
pip install -e .
```

## Quick Start

A CBM pipeline has three steps: (1) generate a dataset with ground-truth concept annotations, (2) train a concept bottleneck model that predicts concepts from inputs and labels from concepts, and (3) evaluate whether correcting ("intervening on") the model's concept predictions at test time improves label accuracy.

The example below runs this pipeline on the robot benchmark. It uses the **subconcept** variant (12 fine-grained foot shape features instead of the default 7 coarse concepts), masks 20% of concept labels during training (MCAR), and tests whether interventions help recover the lost accuracy.

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

The same pipeline runs from the CLI in one command:

```bash
cbm-benchmark robot --seed 1014 --subconcept
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
| `cs` | sudoku | Train concept-supervised model (concepts predicted from board features) |
| `dnn` | all | Train end-to-end neural network baseline (input → label, no concepts) |
| `intervene` | all | Evaluate concept interventions (k-flip for robot, conceptual safeguards for sudoku) |
| `selective` | sudoku | Compute selective accuracy and coverage at each confidence threshold |
| `align` | all | Retrain frontend with monotonicity constraints and compare to unconstrained |
| `collect` | all | Aggregate per-stage results into a single CSV |

Default sequences:

| Benchmark | Default Stages |
|-----------|---------------|
| Robot | `setup cbm dnn intervene align collect` |
| Sudoku | `setup ocr cs dnn intervene selective align collect` |
| Robot Text | `setup cbm dnn intervene align collect` |

## Benchmarks

### Robot Classification

Classify synthetic robots into two species (**glorp** vs. **drent**) based on visual concepts. The label depends on three concepts (mouth type, foot shape, and knee presence), while other concepts like elbow shape and hand shape are spurious -- present in the data but not part of the ground-truth label rule.

<p align="center">
  <img src="docs/assets/robot_concepts.png" width="400" alt="Robot with annotated concepts">
</p>

**Concept granularity.** Each robot has 9 visual features, several of which have multiple subtypes (foot shape has 10, hand shape has 6). The `drop_concepts` parameter controls which concepts the model sees. The package provides two pre-defined setups:

- **Ideal** (default, `IDEAL_DROP`): Drops all foot shape subtypes, keeping only the binary `foot_shape` (pointy vs. flat). This gives 7 concepts.
- **Subconcept** (`SUBCONCEPT_DROP`): Drops the parent `foot_shape` and 5 subtypes, keeping 5 foot subtype indicators. This gives 12 concepts.

These are not the only options -- you can define any concept granularity by customizing `drop_concepts`. For example, you could keep all 10 foot subtypes, or apply the same subtype expansion to hand shapes instead of feet, or drop concepts entirely to test with fewer ground-truth features:

```python
# Custom: keep all foot subtypes (no foot-related drops)
cfg.drop_concepts = []

# Custom: expand hand shapes instead of feet
cfg.drop_concepts = ["hand_shape_round_circle", "hand_shape_round_oval", "hand_shape_edgy_square"]

# Custom: minimal concept set (only the 3 causal features)
cfg.drop_concepts = ["head_shape", "body_shape", "has_antennae", "ears_shape",
                     "has_elbows", "hand_shape"] + list(IDEAL_DROP)
```

<p align="center">
  <img src="docs/assets/robot_foot_shapes.png" width="600" alt="Robot foot shape variations">
</p>

**Modalities.** The robot benchmark supports two input modalities. In **image** mode (`cbm-benchmark robot`), robots are rendered as pixel images (32x32 or 600x600) using pycairo, with configurable color variations. In **text** mode (`cbm-benchmark robot-text`), robots are described in natural language generated from a template corpus with SHA-256 deterministic synonym selection, ensuring reproducibility. Both modalities share the same concept structure and label rules.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `1014` / `1337` | Random seed (image / text) |
| `size` | `"medium"` | Image resolution: `"small"` (8px), `"medium"` (32px), `"large"` (600px). Image only. |
| `model_type` | `"stochastic"` | Label model: `"deterministic"` or `"stochastic"` |
| `drop_concepts` | `IDEAL_DROP` | List of concept names to exclude from the model. Controls concept granularity. |
| `subconcept` | `False` | Use pre-defined `SUBCONCEPT_DROP` and add `_subconcept` suffix to file paths |
| `concept_missing` | `0.0` | Fraction of concept labels to mask during training |
| `concept_missing_mech` | `"none"` | Missingness mechanism: `"none"`, `"mcar"`, or `"mnar"` |
| `intervention_budgets` | `[1, 3]` | Number of concepts to intervene on per sample |
| `intervention_thresholds` | `[0.2, 0.4]` | Concept confidence thresholds -- concepts with predicted probability within this distance of 0.5 are candidates for intervention |
| `intervention_strategy` | `"kflip"` | `"kflip"` (up-to-k) or `"exact_k"` (exactly k). See [Intervention Strategies](#intervention-strategies). |
| `difficulty` | `"hard"` | Corpus difficulty (text only) |
| `generic_rate` | `0.7` | Fraction of test set using concept-ambiguous text (text only) |

The full list of parameters is documented in `RobotBenchmarkConfig` and `RobotTextBenchmarkConfig` (see [`concept_benchmark/config.py`](concept_benchmark/config.py)).

#### Intervention Strategies

At each budget *k*, the intervention framework tries correcting concept predictions and picks the correction that most changes the model's output. Two strategies control how candidate corrections are enumerated:

- **`kflip`** (default): For each sample, enumerate all concept subsets of size 1 through *k*. The subset whose correction most changes the predicted label is applied. This is the standard "up-to-k" strategy -- a budget of *k*=3 will also consider correcting 1 or 2 concepts if that helps more.
- **`exact_k`**: Enumerate only subsets of exactly size *k*. At budget *k*=3, only 3-concept subsets are considered. Useful for ablation studies that need to isolate the effect of a specific budget.

```bash
# Default kflip (up-to-k) -- recommended
cbm-benchmark robot --seed 1014 --subconcept --regimes baseline expert

# Exact-k for ablation
cbm-benchmark robot --seed 1014 --subconcept --strategy exact_k --regimes baseline expert
```

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

The `machine`, `llm`, and `clip` regimes use a **Label-Free CBM** (LFCBM) to discover concepts from CLIP embeddings rather than ground-truth annotations. Each regime trains its own LFCBM from a concept description file and uses the resulting 12-concept space for both prediction and intervention -- independent of the ground-truth concept count. The `llm` and `clip` regimes require `GEMINI_API_KEY` (the pipeline makes live Gemini calls to judge concept presence in images at intervention time).

```bash
cbm-benchmark robot --seed 1014 --subconcept --regimes baseline expert subjective machine

# LLM/CLIP regimes require a Gemini API key
export GEMINI_API_KEY=your_key_here
cbm-benchmark robot --seed 1014 --subconcept --regimes llm clip
```

All regime-specific parameters (noise rates, accuracy, concept file paths, LLM provider/model) are documented in `RobotBenchmarkConfig` (see [`concept_benchmark/config.py`](concept_benchmark/config.py)).

#### Alignment Testing

A learned CBM may assign a concept the wrong sign -- e.g., the model learns "more knees → less likely glorp" when domain knowledge says the opposite. The `align` stage retrains the frontend with monotonicity constraints that enforce expected signs, then re-evaluates interventions to measure whether the constraint helps or hurts.

```python
# Constrain has_knees to have positive weight (more knees -> more glorp)
cfg.alignment_constraints = {"has_knees": 1}
```

### Sudoku Validation

Determine whether a 9x9 sudoku board is valid. The 27 concepts correspond to the validity of each row, column, and 3x3 block -- a board is valid if and only if all 27 concepts are true. This creates a naturally conjunctive relationship between concepts and the label, in contrast to the disjunctive structure of the robot benchmark.

<p align="center">
  <img src="docs/assets/sudoku_handwritten.png" width="400" alt="Sudoku board with handwritten digits and concept annotations">
</p>

Boards can be represented as tabular data (81-cell integer vectors) or rendered as images with handwritten digits. The image pipeline includes an OCR stage that learns to read digits from rendered boards before passing them to the concept model.

**Selective abstention.** Unlike the robot benchmark (which uses k-flip interventions to correct concept predictions), the sudoku benchmark evaluates **selective abstention**: the model makes predictions only when it is confident enough, and abstains (defers to a human or fallback) otherwise. A confidence threshold is chosen on the validation set so that kept predictions achieve at least `target_accuracy`. The two key metrics are:

- **Selective accuracy**: Accuracy on predictions the model chose to keep (high = reliable when it acts).
- **Coverage**: Fraction of samples the model chose to keep (high = less human workload).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `171` | Random seed for reproducibility |
| `n_samples` | `1000` | Number of boards to generate |
| `max_corrupt` | `9` | Maximum cells corrupted in invalid boards (higher = harder) |
| `valid_ratio` | `0.5` | Fraction of valid boards |
| `handwriting` | `True` | Render digits in handwritten style |
| `target_accuracy` | `0.9` | Minimum accuracy demanded on kept predictions; higher values mean more abstention but higher reliability |
| `intervention_thresholds` | `[0.2, 0.4, 0.6, 0.8]` | Concept confidence thresholds for intervention candidates |

The full list of parameters is documented in `SudokuBenchmarkConfig` (see [`concept_benchmark/config.py`](concept_benchmark/config.py)).

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
