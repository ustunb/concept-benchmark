# Evaluation Metrics and Plots

The `concept_benchmark.evaluation` module provides metric functions and plotting utilities for evaluating concept bottleneck models. All metrics take numpy arrays and return floats. All plot functions return `(fig, ax)` and accept an optional `ax` parameter for composing multiple plots on one figure.

For formal definitions, see the paper: *Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models* .

## Metrics

### accuracy

Fraction of correct predictions.

```python
from concept_benchmark.evaluation import accuracy

acc = accuracy(y_pred, y_true)
```

### delta_accuracy

Improvement in accuracy from interventions: `accuracy(after) - accuracy(before)`.

```python
from concept_benchmark.evaluation import delta_accuracy

da = delta_accuracy(y_pred_after, y_pred_before, y_true)
```

### gain

Accuracy gain over a baseline model (e.g. a DNN): `accuracy(predictions) - baseline_accuracy`.

```python
from concept_benchmark.evaluation import gain

g = gain(y_pred, y_true, baseline_accuracy=0.88)
```

### selective_accuracy

Accuracy computed only on samples where the model does not abstain. The model abstains when its confidence score falls below the given threshold.

```python
from concept_benchmark.evaluation import selective_accuracy

sel_acc = selective_accuracy(y_pred, y_true, confidence, threshold=0.5)
```

### coverage

Fraction of samples where the model's confidence meets the threshold (i.e. the model does not abstain).

```python
from concept_benchmark.evaluation import coverage

cov = coverage(confidence, threshold=0.5)
```

### abstention_threshold, decision_threshold, selective_at

The paper's selective-classification protocol. `decision_threshold` finds the cut on P(positive) with the highest accuracy; `abstention_threshold` finds the largest abstention threshold `t` (the model abstains on `t <= p <= 1 - t`) at which the kept predictions reach the target; `selective_at` scores a test set under both. Fit both thresholds on validation predictions.

```python
from concept_benchmark.evaluation import abstention_threshold, decision_threshold, selective_at

decision, _ = decision_threshold(y_val, p_val)
threshold, _ = abstention_threshold(y_val, p_val, target_accuracy=0.95, decision_threshold=decision)
selective_acc, cov = selective_at(y_test, p_test, threshold, decision)
```

`classwise_thresholds` and `selective_at_classwise` fit one threshold per predicted class instead (an appendix variant).

### intervention_metrics

What one intervention did, from the runner's result: accuracy and gain, how many predictions were intervened on and changed, how many concepts were confirmed and edited.

```python
from concept_benchmark.evaluation import intervention_metrics

metrics = intervention_metrics(result.mask, result.C_pred, result.C_intervened,
                               result.y_prob_before, result.y_prob_after, test.y, accuracy_before)
```

### net_work_automated

Net fraction of work automated after accounting for intervention cost: `coverage - mean(n_interventions / n_concepts)`. A value near 1 means most work is automated with few interventions. A value near 0 or negative means interventions cost more than they save.

```python
from concept_benchmark.evaluation import net_work_automated
import numpy as np

nwa = net_work_automated(
    confidence=confidence,
    threshold=0.95,
    n_interventions=np.array([0, 1, 2, 0, 3]),
    n_concepts=27,
)
```

## Plots

Plots take the results tables that the pipelines write (or your own, with the same columns). Rows from several runs are averaged and drawn with a standard-error band, so pass the concatenated results of all your seeds. The pipelines' `plot` stage draws the ones that apply to a run. The figures below use the 10-seed results of the paper.

Outcome plots show *what* happens:

| Function | Shows |
|----------|-------|
| `plot_intervention_curve(results, group=..., baseline_accuracy=...)` | The paper's decision-support panel: accuracy against the intervention budget *k*, one line per model or concept set, with the DNN as a dashed line |
| `plot_intervention_heatmap(results)` | Change in accuracy from interventions for each model, concept set and intervention source |
| `plot_alignment_comparison(results)` | Constrained against unconstrained model, before and after interventions |
| `plot_automation(results, n_instances, n_concepts, baseline_coverage=..., target_accuracy=...)` | The paper's automation panel: Coverage and NetWorkAutomated against the intervention budget, with the DNN's coverage and the target in the title |
| `plot_selective_classification(dnn_metrics, cbm_metrics)` | DNN against CBM on selective accuracy and coverage |
| `plot_concept_discovery(ideal_df, subconcept_df, dnn_accuracy)` | True against human concepts at each budget |
| `plot_model_comparison(results, dnn_accuracy)` | Models × concept sets at each budget |

Diagnostic plots show *why*:

| Function | Shows | Read it as |
|----------|-------|------------|
| `plot_concept_report(mask, concept_proba, concept_answers, concepts_true, concept_names)` | Per concept: detector accuracy, share of interventions, intervener accuracy | An intervener helps only if it is accurate on the concepts the model asks about |
| `plot_answer_reliance(results)` | Change in accuracy when interventions supply the model's own answers against the true values | Similar bars mean the model does not rely on its concept values |
| `plot_confidence(prob_positive, y_true, abstention_threshold)` | Predicted probabilities by class, with the band where the model abstains | Instances inside the band are deferred no matter how many concepts are checked |

### plot_intervention_curve

Line plot of a metric (default: accuracy) against the intervention budget *k*. `results` needs the columns `budget` and `accuracy`; rows that share a budget are averaged and drawn with a standard-error band. `group=` names a column that splits the rows into one line each (e.g. `"model_family"` or `"concept_source"`), and `baseline_accuracy` (one value, or one per run) draws the DNN as a dashed line.

```python
import pandas as pd
from concept_benchmark.evaluation import plot_intervention_curve

# one results file per seed, as written by scripts/robot_pipeline.py
results = pd.concat(
    pd.read_csv(path).assign(seed=seed) for seed, path in results_by_seed.items()
)
fig, ax = plot_intervention_curve(results, group="model_family", baseline_accuracy=dnn_accuracies)
fig.savefig("intervention_curve.png")
```

```{image} assets/intervention_curve.png
:width: 500px
:align: center
:alt: Accuracy against the intervention budget for four architectures
```

### plot_intervention_heatmap

Heatmap of the change in accuracy that interventions bring: each cell is the mean over runs of the accuracy at the budgets above zero minus the accuracy at budget zero. Rows are the values of `rows` (default `"model_family"`) and columns the combinations of `columns` (default `("concept_source", "intervention_source")`); a `seed` column separates runs.

```python
from concept_benchmark.evaluation import plot_intervention_heatmap

fig, ax = plot_intervention_heatmap(results)
```

```{image} assets/intervention_heatmap.png
:width: 800px
:align: center
:alt: Change in accuracy from interventions per model, concept set and intervention source
```

### plot_alignment_comparison

Bar chart of a constrained against an unconstrained model. `results` is a DataFrame with one row per run and the columns `concepts` (concept set), `model` and `accuracy_before`; an optional `accuracy_after` column adds a second panel with the accuracy after interventions. Several rows per concept set and model (e.g. one per `seed`) are averaged and drawn with standard-error bars.

```python
import pandas as pd
from concept_benchmark.evaluation import plot_alignment_comparison

results = pd.DataFrame({
    "concepts": ["true", "true", "human", "human"],
    "model": ["CBM", "Constrained CBM", "CBM", "Constrained CBM"],
    "accuracy_before": [0.845, 0.898, 0.776, 0.810],
    "accuracy_after": [0.920, 0.907, 0.857, 0.832],
})
fig, axes = plot_alignment_comparison(results)
```

```{image} assets/alignment.png
:width: 700px
:align: center
:alt: Constrained against unconstrained CBM before and after interventions
```

### plot_automation

Line plot of coverage and net work automated against the number of concept checks allowed. `results` needs the columns `budget`, `coverage_after` and `total_concept_checks`, as written by the sudoku pipeline; `baseline_coverage` optionally draws a model without interventions (e.g. the DNN) as a dashed line.

```python
from concept_benchmark.evaluation import plot_automation

fig, ax = plot_automation(results, n_instances=len(test), n_concepts=27, baseline_coverage=dnn_coverage, target_accuracy=0.95)
```

```{image} assets/automation.png
:width: 400px
:align: center
:alt: Coverage and net work automated against concept checks
```

### plot_selective_classification

Grouped bar chart comparing DNN vs CBM on selective classification metrics (selective accuracy, coverage, net work automated). Each argument maps a metric name to its value.

```python
from concept_benchmark.evaluation import coverage, plot_selective_classification, selective_accuracy

dnn_metrics = {
    "selective_accuracy": selective_accuracy(y_pred_dnn, y_true, confidence_dnn, threshold_dnn),
    "coverage": coverage(confidence_dnn, threshold_dnn),
}
cbm_metrics = {
    "selective_accuracy": selective_accuracy(y_pred_cbm, y_true, confidence_cbm, threshold_cbm),
    "coverage": coverage(confidence_cbm, threshold_cbm),
}
fig, ax = plot_selective_classification(dnn_metrics, cbm_metrics)
```

### plot_concept_discovery

Clustered bar chart comparing true against human concepts across intervention budgets, with a DNN baseline line. Both DataFrames need the columns `budget` and `accuracy`; `budgets` selects which budgets to show (default `[0, 1, 3]`).

```python
from concept_benchmark.evaluation import plot_concept_discovery

fig, ax = plot_concept_discovery(ideal_df, subconcept_df, dnn_accuracy=0.88)
```

### plot_model_comparison

Grouped bar chart comparing models × concept sets across budgets. `results` maps `(model_name, concept_set)` tuples, e.g. `("CBM", "true_concepts")`, to DataFrames with the columns `budget` and `accuracy`.

```python
from concept_benchmark.evaluation import plot_model_comparison

fig, ax = plot_model_comparison(
    {("CBM", "true_concepts"): ideal_df, ("CBM", "human_concepts"): subconcept_df},
    dnn_accuracy=0.88,
)
```

### plot_concept_report

Per concept: how accurate the detector is, how often the concept is intervened on, and how accurate the intervener is. A concept with a high share of interventions and a low intervener accuracy explains interventions that do not help. The inputs are the intervention records of the robot pipeline (`--dump-interventions DIR`), one `.npz` file per budget.

```python
import numpy as np
from concept_benchmark.evaluation import plot_concept_report

records = np.load(record_path)  # a file in the --dump-interventions directory
fig, ax = plot_concept_report(
    mask=records["mask"],                 # concepts intervened on
    concept_proba=records["C_pred"],      # predicted probabilities before interventions
    concept_answers=records["C_answer"],  # concept values after interventions
    concepts_true=records["C_true"],
    concept_names=list(records["concept_names"]),
)
```

```{image} assets/concept_report.png
:width: 420px
:align: center
:alt: Per-concept detector accuracy, share of interventions and intervener accuracy
```

### plot_answer_reliance

Change in accuracy when interventions supply the model's own answers against the true values. Interventions with the model's own answers change no concept value, so any change in accuracy comes from the intervention mechanism. `results` holds the rows of a robot pipeline run with `--intervention-sources self perfect` (columns `budget`, `accuracy`, `intervention_source` and the `group` column, default `"model_family"`).

```python
from concept_benchmark.evaluation import plot_answer_reliance

fig, ax = plot_answer_reliance(results, group="model_family")
```

```{image} assets/answer_reliance.png
:width: 380px
:align: center
:alt: Change in accuracy with the model's own answers against the true values
```

### plot_confidence

Histogram of the predicted probability of the positive class, with the band where the model abstains: a selective classifier abstains when the probability lies in `[t, 1 - t]`. The inputs are saved by the sudoku pipeline's optional `diagnose` stage.

```python
import numpy as np
from concept_benchmark.evaluation import plot_confidence

saved = np.load(confidence_path)  # the .npz file written by the diagnose stage
fig, ax = plot_confidence(
    prob_positive=saved["p_test"],
    y_true=saved["y_test"],
    abstention_threshold=threshold,  # the threshold t fitted on the validation set
)
```

```{image} assets/confidence.png
:width: 400px
:align: center
:alt: Predicted probability of a valid board with the abstention band
```
