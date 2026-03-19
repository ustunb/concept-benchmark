# Quick Start

A concept bottleneck model (CBM) first predicts interpretable *concepts* from inputs (e.g., "has pointy feet"), then uses those concepts to predict the final label. This two-stage design lets users inspect and correct the model's reasoning at test time — an operation called an *intervention*. This package gives you synthetic datasets where the ground-truth concepts are known, so you can measure exactly how much interventions help under different conditions.

## Robot Classification

The robot benchmark classifies fictional robots — **Glorps** vs. **Drents** — from their body features:

```python
from concept_benchmark import DatasetGenerator

dataset = DatasetGenerator(
    "robot",
    seed=1014,                       # reproducibility
    concept_preset="foot_subtypes",  # 12 fine-grained concepts (default: "ground_truth" = 7)
    use_stochastic_labels=True,      # probabilistic labeling (or False for deterministic)
    image_size="medium",             # "small" (8px), "medium" (32px, default), or "large" (600px)
    render_images=True,              # set False to skip image rendering for quick exploration
    label_formula={                  # scoring rule for class assignment
        "terms": {
            "mouth_type": {"value": "closed", "weight": 5.0},
            "foot_shape": {"value": "pointy", "weight": 8.0},
            "has_knees":  {"value": "true",   "weight": -5.0},
        },
        "intercept": 2.0,
        "temperature": 4.2,          # sigmoid temperature for stochastic labels
    },
).generate()

print(dataset.training.C.shape)   # (3800, 12) — concept annotations
print(dataset.training.concepts)
# ['head_shape', 'body_shape', 'has_knees', 'has_antennae', 'ears_shape',
#  'mouth_type', 'foot_shape_flat_trapezoid', 'foot_shape_flat_square',
#  'foot_shape_flat_5sided', 'foot_shape_pointy_rounded',
#  'foot_shape_pointy_square', 'foot_shape_pointy_4sided']
```

Inspect the data:

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

```{image} assets/robot_samples.png
:width: 600px
:align: center
:alt: Sample Glorps and Drents with concept annotations
```

Train a CBM — concept detector (images → concepts) and label predictor (concepts → label):

```python
import numpy as np
from concept_benchmark import DatasetGenerator
from concept_benchmark.utils import set_deterministic_seed
from experiments.models import (
    ConceptDetector, FrontEndModel, ConceptBasedModel, RobotConceptClassifier,
)
from concept_benchmark.utils import determine_device, get_loader_config, patch_macos_dataloader

set_deterministic_seed(1014)
patch_macos_dataloader()
device = determine_device()
loader_config = get_loader_config(device)

dataset = DatasetGenerator(
    "robot", seed=1014, concept_preset="foot_subtypes", render_images=True).generate()

# Step 1: train concept detector (images → concepts)
n_concepts = dataset.training.n_concepts
cd = ConceptDetector(model=RobotConceptClassifier(num_concepts=n_concepts, input_size=32))
cd.fit(dataset.training, dataset.validation,
       fit_params={"epochs": 50, "lr": 1e-3, "patience": 10, "device": str(device), **loader_config})

# Step 2: train label predictor (concepts → label)
fe = FrontEndModel()
fe.fit(dataset.training.C, dataset.training.y)

# Step 3: combine into a CBM and evaluate
cbm = ConceptBasedModel(concept_detector=cd, label_predictor=fe)
predictions = cbm.predict(dataset.test)
accuracy = np.mean(predictions == dataset.test.y)
print(f"CBM accuracy: {accuracy:.4f}")
# CBM accuracy: 0.7812
```

For a complete walkthrough including interventions and alignment, see `examples/robot_pipeline_example.py`.

## Sudoku Validation

The Sudoku benchmark determines whether a 9×9 board is valid. 27 concepts capture row, column, and block validity — a board is valid iff all 27 are true:

```python
from concept_benchmark import DatasetGenerator

dataset = DatasetGenerator(
    "sudoku",
    seed=171,             # reproducibility
    n_boards=1000,        # number of boards
    max_cell_swaps=9,     # cells swapped in invalid boards (higher = subtler errors)
    valid_board_ratio=0.5,  # fraction of valid boards
).generate()

print(dataset.training.C.shape)   # (600, 27) — 27 concept annotations
print(dataset.training.concepts)  # ['row_valid_1', 'row_valid_2', ..., 'block_valid_9']
```

Inspect the data:

```python
df = dataset.training.to_dataframe()
show_cols = list(dataset.training.concepts[:5]) + ["label"]
print(df[show_cols])
#      row_valid_1  row_valid_2  row_valid_3  row_valid_4  row_valid_5  label
# 0              1            1            1            1            1      1
# ..           ...          ...          ...          ...          ...    ...
# 301            1            0            0            1            1      0
```

```{image} assets/sudoku_samples.png
:width: 600px
:align: center
:alt: Sample Sudoku boards generated by the benchmark
```

For a complete walkthrough including selective classification and interventions, see `examples/sudoku_quickstart.py`.

## Full Experiment Pipelines

To reproduce the paper results — including all intervention regimes, alignment constraints, and selective classification — use the pipeline scripts (requires cloning the repo):

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes   # see --help for all flags
python scripts/sudoku_pipeline.py --seed 171
```
