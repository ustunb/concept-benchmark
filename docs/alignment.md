# Alignment

Alignment constraints force the label predictor's concept weights to match a user's prior expectations about concept-label relationships. For example, if a domain expert knows that "has knees" should positively predict the Glorp class, alignment constrains that weight to be positive during retraining.

## Why alignment matters

In standard CBM training, the label predictor (logistic regression on concept activations) learns weights freely from data. This can produce **counterintuitive** weights — e.g., `has_knees` getting a *negative* weight even when knees truly indicate Glorp — because the model exploits correlations among imperfect concept predictions.

The paper (Section 5.2) shows that alignment constraints:

- **Preserve** training accuracy — the aligned model performs comparably at k=0 (no interventions).
- **Destroy** intervention benefit — at k=3, the aligned subconcept model goes from +16% gain to -8% loss.

This happens because alignment forces the model into a weight configuration that is locally optimal for the training distribution but incompatible with ground-truth concept corrections at test time.

## Usage

After training a `ConceptBasedModel`, use `run_alignment()` to retrain the frontend with sign constraints and compare:

```python
from experiments.utils import run_alignment

results = run_alignment(
    concept_based_model=cbm,
    train_dataset=train,
    test_dataset=test,
    monotonicity_constraints={"has_knees": 1},  # force positive weight
)

print(f"Original accuracy: {results['original_accuracy']:.4f}")
print(f"Aligned accuracy:  {results['aligned_accuracy']:.4f}")
print(f"Accuracy change:   {results['accuracy_change']:+.4f}")
# Expected (seed=1014, subconcept): original 0.7812, aligned 0.7656 (-0.0156)
```

The `monotonicity_constraints` dict maps concept names to their required sign: `+1` for positive weight, `-1` for negative.

## Expected results

| Setup | CBM (k=0) | Aligned (k=0) | CBM (k=3) gain | Aligned (k=3) gain |
|-------|-----------|----------------|-----------------|---------------------|
| ideal (7 concepts) | 0.8673 | 0.8657 | +10.2% | -0.4% |
| subconcept (12 concepts) | 0.7812 | 0.7656 | +6.9% | -8.0% |

For a complete end-to-end example with training, interventions, and alignment, see [`examples/robot_pipeline_example.py`](https://github.com/ustunb/concept-benchmark/blob/main/examples/robot_pipeline_example.py).
