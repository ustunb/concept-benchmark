# Quick Start

A concept bottleneck model (CBM) first predicts interpretable *concepts* from inputs (e.g., "has pointy feet"), then uses those concepts to predict the final label. This two-stage design lets users inspect and correct the model's reasoning at test time — an operation called an *intervention*. This package gives you synthetic datasets where the ground-truth concepts are known, so you can measure exactly how much interventions help under different conditions.

## Robot Classification

The robot benchmark classifies fictional robots — **Glorps** vs. **Drents** — from their body features:

```python
from concept_benchmark.robots import DatasetGenerator

dataset = DatasetGenerator(
    seed=1014,
    concept_preset="foot_subtypes",  # expand foot_shape into subtypes (default: "ground_truth")
    render_images=True,              # set False to skip rendering for quick exploration
).generate_splits()                  # the train/val/test split of the paper

print(dataset.train.C.shape)   # (3800, 12) — concept annotations
print(dataset.train.concepts)
# ['head_shape', 'body_shape', 'has_knees', 'has_antennae', 'ears_shape',
#  'mouth_type', 'foot_shape_flat_trapezoid', 'foot_shape_flat_square',
#  'foot_shape_flat_5sided', 'foot_shape_pointy_rounded',
#  'foot_shape_pointy_square', 'foot_shape_pointy_4sided']
```

`generate()` returns the unsplit dataset with every concept, for you to drop concepts and split as you choose. Robots are labeled with the `balanced` rule by default; pass `label_rule="sparse"` for the rare-class rule (see [Robot Classification](robot.md)).

Inspect the data:

```python
dataset.train.to_dataframe().head(2)
#    head_shape  body_shape  has_knees  ...  foot_shape_pointy_4sided  label  class
# 0           0           0          0  ...                         0      1  glorp
# 1           0           0          0  ...                         0      1  glorp
```

For interactive browsing with [Renumics Spotlight](https://github.com/Renumics/spotlight) (`pip install concept-benchmark[explore]`):

```python
dataset.train.explore()  # opens in the browser
```

```{image} assets/robot_samples.png
:width: 600px
:align: center
:alt: Sample Glorps and Drents with concept annotations
```

Train a CBM — concept detector (images → concepts) and label predictor (concepts → label):

```python
import numpy as np
from concept_benchmark.robots import DatasetGenerator
from concept_benchmark.utils import set_deterministic_seed
from experiments.models import (
    ConceptDetector, FrontEndModel, ConceptBasedModel, RobotConceptClassifier,
)

set_deterministic_seed(1014)

dataset = DatasetGenerator(
    seed=1014, concept_preset="foot_subtypes", render_images=True).generate_splits()

# Step 1: train concept detector (images → concepts)
n_concepts = dataset.train.n_concepts
cd = ConceptDetector(model=RobotConceptClassifier(num_concepts=n_concepts, input_size=32))
cd.fit(dataset.train, dataset.val,
       fit_params={"epochs": 50, "lr": 1e-3, "patience": 10})

# Step 2: train label predictor (concepts → label)
fe = FrontEndModel()
fe.fit(dataset.train.C, dataset.train.y)

# Step 3: combine into a CBM and evaluate
cbm = ConceptBasedModel(concept_detector=cd, label_predictor=fe)
predictions = cbm.predict(dataset.test)
accuracy = np.mean(predictions == dataset.test.y)
print(f"CBM accuracy: {accuracy:.4f}")
# about 0.78
```

Over the 10 seeds of the paper, the CBM reaches 84.5% with the true concepts and 77.6% with the human concepts before interventions, and 92.0% and 85.7% once every concept is corrected; the DNN reaches 88%. A single run differs from these means by a point or two, and across hardware.

For a complete walkthrough including interventions and alignment, see `examples/robot_pipeline_example.py`.

Visualize intervention results:

```python
# Plot accuracy vs intervention budget
import pandas as pd
from concept_benchmark.evaluation import plot_intervention_curve

# one results file per seed, as written by scripts/robot_pipeline.py
results = pd.concat(
    pd.read_csv(path).assign(seed=seed) for seed, path in results_by_seed.items()
)
fig, ax = plot_intervention_curve(results, group="model_family", baseline_accuracy=dnn_accuracies)
```

```{image} assets/intervention_curve.png
:width: 500px
:align: center
:alt: Accuracy against the intervention budget for four architectures
```

See [Evaluation Metrics and Plots](evaluation.md) for the other plots.

## Sudoku Validation

The Sudoku benchmark determines whether a 9×9 board is valid. 27 concepts capture row, column, and block validity — a board is valid iff all 27 are true:

```python
from concept_benchmark.sudoku import DatasetGenerator

dataset = DatasetGenerator(
    seed=171,             # reproducibility
    n_boards=1000,        # number of boards
    max_cell_swaps=9,     # cells swapped in invalid boards (higher = subtler errors)
    valid_board_ratio=0.5,  # fraction of valid boards
    render_images=False,  # set True to generate board images (slower)
).generate_splits()       # 60/20/20, stratified on the label

print(dataset.train.C.shape)   # (600, 27) — 27 concept annotations
print(dataset.train.concepts)  # ['row_valid_1', 'row_valid_2', ..., 'block_valid_9']
```

Inspect the data:

```python
df = dataset.train.to_dataframe()
show_cols = list(dataset.train.concepts[:5]) + ["label"]
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

For a complete walkthrough including selective classification and interventions, see `examples/sudoku_pipeline_example.py`.

## Full Experiment Pipelines

To run the full experiments — including concept and intervention sources, alignment constraints, and selective classification — use the pipeline scripts (requires cloning the repo):

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes   # see --help for all flags
python scripts/sudoku_pipeline.py --seed 171

# Add plot to generate figures from results
python scripts/robot_pipeline.py --seed 1014 --stages setup cbm dnn intervene align collect plot
```

The paper reports every result as a mean over 10 seeds (robots `1014`–`1023`, sudoku `171`–`180`); [`EXPERIMENTS.md`](https://github.com/ustunb/concept-benchmark/blob/main/EXPERIMENTS.md) lists the commands behind each experiment.

The datasets of the paper are also on the Hugging Face Hub ([`robots-true-concepts`](https://huggingface.co/datasets/juliannski/robots-true-concepts), [`robots-human-concepts`](https://huggingface.co/datasets/juliannski/robots-human-concepts), [`sudoku`](https://huggingface.co/datasets/juliannski/sudoku)).
