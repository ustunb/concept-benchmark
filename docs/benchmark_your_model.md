# Benchmark Your Own Model

This guide shows how to evaluate your own concept bottleneck model on the benchmarks provided by this package. All examples below use the robot benchmark, but the same approach works for sudoku.

> **Prerequisite:** You need the full repository (not just `pip install concept-benchmark`) to run the pipeline scripts and examples.

## Getting data for your model

Generate a dataset and access it in the format your model expects:

```python
from concept_benchmark.robots import DatasetGenerator

dataset = DatasetGenerator(seed=1014, concept_preset="foot_subtypes", render_images=True).generate_splits()
train, val, test = dataset.train, dataset.val, dataset.test
```

Each split is a `ConceptDatasetSample` with these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Input features (images or tabular) |
| `C` | `np.ndarray` | Concept labels `(N, n_concepts)` |
| `y` | `np.ndarray` | Target labels `(N,)` |
| `concepts` | `list[str]` | Concept names |
| `n_concepts` | `int` | Number of concepts |
| `classes` | `list[str]` | Class names |
| `n` | `int` | Number of samples |

Access formats:

```python
# NumPy arrays (default)
X_train, C_train, y_train = train.X, train.C, train.y

# PyTorch DataLoader
loader = train.loader(batch_size=64, shuffle=True)
for x_batch, c_batch, y_batch in loader:
    ...

# Pandas DataFrame
df = train.to_dataframe()
```

## Wrapping your concept detector

The intervention runner requires a `ConceptDetector` subclass. Wrap your model:

```python
from experiments.models import ConceptDetector

class MyConceptDetector(ConceptDetector):
    def __init__(self, my_model):
        super().__init__()
        self._my_model = my_model

    def predict_proba(self, dataset, **kwargs):
        """Must return (N, n_concepts) float array in [0, 1]."""
        return self._my_model.predict_concept_probs(dataset.X)
```

**Key point:** Override `predict_proba()` — it receives a `ConceptDatasetSample`, not raw arrays. Access inputs via `dataset.X`. The base class `predict()` calls `predict_proba()` and thresholds at 0.5 automatically.

### Using a PyTorch module directly

If your model is already a PyTorch `nn.Module`, you can pass it directly instead of subclassing. Your module's `forward()` must:

- Accept a batched input tensor (e.g. `(B, 3, 32, 32)` for images)
- Return `(B, n_concepts)` **raw logits** (pre-sigmoid) — sigmoid is applied internally by `predict()`

The return value can also be a tuple (first element is used) or a dict with a `"logits"` key.

```python
cd = ConceptDetector(model=my_pytorch_module)
cd.fit(train, val, fit_params={"epochs": 50, "lr": 1e-3})
```

You can also split your model into a backbone + concept head using the `embedding_model` parameter. The detector will probe the backbone's output shape and attach an MLP head automatically:

```python
cd = ConceptDetector(embedding_model=my_backbone)
cd.fit(train, val, fit_params={"epochs": 50, "lr": 1e-3})
```

## Wrapping your label predictor

Subclass `FrontEndModel` and override `predict()` and `predict_proba()`:

```python
from experiments.models import FrontEndModel

class MyFrontEnd(FrontEndModel):
    def __init__(self, my_classifier):
        super().__init__()
        self._clf = my_classifier

    def predict(self, C):
        """C is (N, n_concepts) binary 0/1. Returns (N,) int labels."""
        return self._clf.predict(C)

    def predict_proba(self, C):
        """C is (N, n_concepts) binary 0/1. Returns (N, n_classes) probs."""
        return self._clf.predict_proba(C)
```

**Key point:** `C` is already binarized at 0.5 — binary 0/1 values, not probabilities.

For a simple logistic regression baseline, the built-in `FrontEndModel()` works out of the box:

```python
fe = FrontEndModel()
fe.fit(train.C, train.y)
```

## Assembling and evaluating

Combine your concept detector and label predictor into a `ConceptBasedModel`:

```python
from experiments.models import ConceptBasedModel

cbm = ConceptBasedModel(
    concept_detector=MyConceptDetector(my_concept_model),
    label_predictor=MyFrontEnd(my_classifier),
)

predictions = cbm.predict(test)
accuracy = np.mean(predictions == test.y)
print(f"CBM accuracy: {accuracy:.4f}")
```

<details>
<summary><strong>Running interventions on your model</strong></summary>

Use `ConceptInterventionRunner` to evaluate intervention benefit:

```python
from experiments.evaluate import intervention_table

results = intervention_table(cbm, test, budgets=(1, 3, "max"), seed=1014)
print(results[["budget", "accuracy"]])
```

`intervention_table` runs the paper's policy (`KFlipInterventionStrategy` through `ConceptInterventionRunner`) at
each budget and returns the table that `plot_intervention_curve` reads.

**Bypassing the concept detector:**
If you already have concept probabilities, pass them directly via `concept_proba=` to skip the concept detector:

```python
my_concept_probs = my_model.predict_concepts(test.X)  # your own call

result = runner.run(
    strategy=KFlipInterventionStrategy(),
    config=InterventionConfig(per_instance_budget=3, score_threshold=0.2),
    dataset=test,
    concept_proba=my_concept_probs,  # bypasses the concept detector
)
```

**`C` vs `base_concepts`:**
The runner uses `dataset.base_concepts` (clean concepts before noise) for ground-truth corrections, not `dataset.C` (which may have noise applied). Pass `concept_true=` to override:

```python
result = runner.run(
    strategy=KFlipInterventionStrategy(),
    config=InterventionConfig(per_instance_budget=3, score_threshold=0.2),
    dataset=test,
    concept_true=my_ground_truth_concepts,  # override ground truth
)
```

</details>

<details>
<summary><strong>Running alignment</strong></summary>

Retrain the label predictor with sign constraints on concept weights and compare accuracy:

```python
from experiments.utils import run_alignment

stats = run_alignment(
    concept_based_model=cbm,
    train_dataset=train,
    test_dataset=test,
    monotonicity_constraints={"has_knees": 1},  # require a non-negative weight
)
print(f"Original: {stats['original_accuracy']:.4f}")
print(f"Aligned:  {stats['aligned_accuracy']:.4f}")
print(f"Change:   {stats['accuracy_change']:+.4f}")
```

</details>

## Comparing to baselines

The repo also includes built-in wrappers for the official `cem` and `probcbm`
baselines from `mateoespinosa/cem`, and an `ecbm` baseline that works out of
the box. These wrappers expose the same practical surface used above:

- `predict(dataset)`
- `predict_proba(dataset, return_concepts=True)`
- intervention-time label recomputation from edited concepts

Every family runs on both benchmarks:

```bash
./scripts/install_cem_repo.sh
python scripts/robot_pipeline.py --seed 1014 --cbm-family cem
python scripts/robot_pipeline.py --seed 1014 --cbm-family probcbm
python scripts/robot_pipeline.py --seed 1014 --cbm-family ecbm
python scripts/sudoku_pipeline.py --seed 171 --cbm-family cem
```

Alignment remains on the original `cbm` path.

Use the same seeds for apples-to-apples comparison with the built-in models. Expected results for the robot benchmark (`concept_preset="foot_subtypes"`, balanced rule), as means over the 10 seeds of the paper (`1014`–`1023`):

| Model | k=0 | k=max (every concept corrected) |
|-------|-----|---------------------------------|
| Built-in CBM | 77.6% | 85.7% |
| DNN baseline | 88% | — |

A single run differs from these means by a point or two, and across hardware. Run the built-in pipeline to generate baseline numbers:

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --budgets 1 3 max
```

For metrics and plots to compare models, see [Evaluation Metrics and Plots](evaluation.md); for the commands behind each experiment of the paper, see [`EXPERIMENTS.md`](https://github.com/ustunb/concept-benchmark/blob/main/EXPERIMENTS.md).
