# Evaluation Metrics and Plots

The `concept_benchmark.benchmark` module provides metric functions and plotting utilities for evaluating concept bottleneck models. All metrics take numpy arrays and return floats. All plot functions return `(fig, ax)` and accept an optional `ax` parameter for composing multiple plots on one figure.

For formal definitions, see the paper: *Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models* (Skirzynski et al.).

## Metrics

### accuracy

Fraction of correct predictions.

```python
from concept_benchmark.benchmark import accuracy

acc = accuracy(y_pred, y_true)
```

### delta_accuracy

Improvement in accuracy from interventions: `accuracy(after) - accuracy(before)`.

```python
from concept_benchmark.benchmark import delta_accuracy

da = delta_accuracy(y_pred_after, y_pred_before, y_true)
```

### gain

Accuracy gain over a baseline model (e.g. a DNN): `accuracy(predictions) - baseline_accuracy`.

```python
from concept_benchmark.benchmark import gain

g = gain(y_pred, y_true, baseline_accuracy=0.8746)
```

### selective_accuracy

Accuracy computed only on samples where the model does not abstain. The model abstains when its confidence score falls below the given threshold.

```python
from concept_benchmark.benchmark import selective_accuracy

sel_acc = selective_accuracy(y_pred, y_true, confidence, threshold=0.5)
```

### coverage

Fraction of samples where the model's confidence meets the threshold (i.e. the model does not abstain).

```python
from concept_benchmark.benchmark import coverage

cov = coverage(confidence, threshold=0.5)
```

### net_work_automated

Net fraction of work automated after accounting for intervention cost: `coverage - mean(n_interventions / n_concepts)`. A value near 1 means most work is automated with few interventions. A value near 0 or negative means interventions cost more than they save.

```python
from concept_benchmark.benchmark import net_work_automated
import numpy as np

nwa = net_work_automated(
    confidence=confidence,
    threshold=0.95,
    n_interventions=np.array([0, 1, 2, 0, 3]),
    n_concepts=27,
)
```

## Plots

### plot_intervention_curve

Line plot of a metric (default: accuracy) vs intervention budget *k*. Optionally draws a horizontal dashed line for the DNN baseline.

```python
from concept_benchmark.benchmark import plot_intervention_curve
import pandas as pd

results = pd.DataFrame({
    "budget": [0, 1, 3, 7],
    "accuracy": [0.8673, 0.9734, 0.9767, 0.9767],
})
fig, ax = plot_intervention_curve(results, baseline_accuracy=0.8746)
fig.savefig("intervention_curve.png")
```

### plot_regime_comparison

Horizontal bar chart of mean delta-accuracy per intervention regime, with min/max error bars across budgets. Compares how different annotation sources (expert, subjective, machine, etc.) affect intervention benefit.

```python
from concept_benchmark.benchmark import plot_regime_comparison
import pandas as pd

regime_df = pd.DataFrame({
    "regime": ["baseline"] * 4 + ["expert"] * 4,
    "budget": [0, 1, 2, 5, 0, 1, 2, 5],
    "accuracy": [0.78, 0.92, 0.94, 0.94, 0.78, 0.88, 0.90, 0.89],
})
fig, ax = plot_regime_comparison(regime_df, budgets=[1, 2, 5])
```

### plot_concept_discovery

Clustered bar chart comparing ideal (ground-truth) vs subconcept accuracy across intervention budgets, with a DNN baseline line.

```python
from concept_benchmark.benchmark import plot_concept_discovery
import pandas as pd

ideal = pd.DataFrame({"budget": [0, 1, 3], "accuracy": [0.8673, 0.9734, 0.9767]})
subconcept = pd.DataFrame({"budget": [0, 1, 3], "accuracy": [0.7812, 0.9212, 0.9439]})
fig, ax = plot_concept_discovery(ideal, subconcept, dnn_accuracy=0.8746)
```

### plot_selective_classification

Grouped bar chart comparing DNN vs CBM on selective classification metrics (selective accuracy, coverage, net work automated).

```python
from concept_benchmark.benchmark import plot_selective_classification

dnn_metrics = {"selective_accuracy": 0.833, "coverage": 0.12}
cbm_metrics = {"selective_accuracy": 0.938, "coverage": 0.975}
fig, ax = plot_selective_classification(dnn_metrics, cbm_metrics)
```

### plot_alignment_comparison

Horizontal bar chart comparing CBM vs aligned CBM gain at a given intervention budget. Shows how alignment constraints affect intervention benefit.

```python
from concept_benchmark.benchmark import plot_alignment_comparison

results = {
    "ideal": {"cbm_gain": 0.102, "aligned_gain": -0.004},
    "subconcept": {"cbm_gain": 0.069, "aligned_gain": -0.080},
}
fig, ax = plot_alignment_comparison(results)
```
