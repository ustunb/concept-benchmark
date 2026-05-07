# Benchmark Your Own Model

This guide shows how to evaluate your own concept bottleneck model on the benchmarks provided by this package. All examples below use the robot benchmark, but the same approach works for sudoku.

> **Prerequisite:** You need the full repository (not just `pip install concept-benchmark`) to run the pipeline scripts and examples.

## Getting data for your model

Generate a dataset and access it in the format your model expects:

```python
from concept_benchmark.robots import DatasetGenerator

dataset = DatasetGenerator(seed=1014, concept_preset="foot_subtypes", render_images=True).generate()
dataset.drop_concepts(["has_elbows", "hand_shape"])
dataset.sample(test_size=10000, val_size=0.2, train_size=3800, seed=1014)
train, val, test = dataset.train, dataset.validation, dataset.test
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

    def predict(self, dataset, **kwargs):
        """Must return (N, n_concepts) float array in [0, 1]."""
        return self._my_model.predict_concept_probs(dataset.X)
```

**Key point:** `predict()` receives a `ConceptDatasetSample`, not raw arrays. Access inputs via `dataset.X`.

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
from experiments.intervention import ConceptInterventionRunner, InterventionConfig
from experiments.kflip import KFlipInterventionStrategy

runner = ConceptInterventionRunner(model=cbm)

for k in [1, 3]:
    result = runner.run(
        strategy=KFlipInterventionStrategy(),
        config=InterventionConfig(
            max_concepts_per_instance=k,
            score_threshold=0.2,
        ),
        dataset=test,
    )
    acc_after = np.mean(result.y_pred_after == test.y)
    print(f"k={k}: accuracy={acc_after:.4f}")
```

**Bypassing the concept detector:**
If you already have concept probabilities, pass them directly via `concept_proba=` to skip `concept_detector.predict()`:

```python
my_concept_probs = my_model.predict_concepts(test.X)  # your own call

result = runner.run(
    strategy=KFlipInterventionStrategy(),
    config=InterventionConfig(max_concepts_per_instance=3, score_threshold=0.2),
    dataset=test,
    concept_proba=my_concept_probs,  # bypasses concept_detector.predict()
)
```

**`C` vs `base_concepts`:**
The runner uses `dataset.base_concepts` (clean concepts before noise) for ground-truth corrections, not `dataset.C` (which may have noise applied). Pass `concept_true=` to override:

```python
result = runner.run(
    strategy=KFlipInterventionStrategy(),
    config=InterventionConfig(max_concepts_per_instance=3, score_threshold=0.2),
    dataset=test,
    concept_true=my_ground_truth_concepts,  # override ground truth
)
```

</details>

<details>
<summary><strong>Running alignment</strong></summary>

Test whether sign-constraining concept weights preserves intervention benefit:

```python
from experiments.utils import run_alignment

stats = run_alignment(
    concept_based_model=cbm,
    train_dataset=train,
    test_dataset=test,
    monotonicity_constraints={"has_knees": 1},  # force positive weight
)
print(f"Original: {stats['original_accuracy']:.4f}")
print(f"Aligned:  {stats['aligned_accuracy']:.4f}")
print(f"Change:   {stats['accuracy_change']:+.4f}")
```

</details>

## Comparing to baselines

The repo also includes built-in wrappers for the official `cem` and `probcbm`
baselines from `mateoespinosa/cem`. These wrappers expose the same practical
surface used above:

- `predict(dataset)`
- `predict_proba(dataset, return_concepts=True)`
- intervention-time label recomputation from edited concepts

For the robot benchmark:

```bash
./scripts/install_cem_repo.sh
python scripts/robot_pipeline.py --seed 1014 --cbm-family cem
python scripts/robot_pipeline.py --seed 1014 --cbm-family probcbm
```

For sudoku, the current integration is intentionally narrow: `cem` and
`probcbm` are wired into the tabular pipeline path, while OCR/image-specific
selective/alignment flows remain on the original `cbm` implementation.

Use the same seed for apples-to-apples comparison with the built-in models. Expected results for the robot benchmark (seed=1014, concept_preset="foot_subtypes"):

| Model | k=0 | k=1 | k=3 | k=12 (max) |
|-------|-----|-----|-----|------------|
| Built-in CBM | 0.7812 | 0.9212 | 0.9439 | 0.9439 |
| DNN baseline | 0.8746 | — | — | — |

Run the built-in pipeline to generate baseline numbers:

```bash
./venv/bin/python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes
```

For the complete set of expected results across all regimes, see the [README](https://anonymous.4open.science/r/concept-benchmark-84D2#readme).
