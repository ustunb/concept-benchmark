# Alignment

Alignment constraints force the label predictor's concept weights to match a user's prior expectations about concept-label relationships. For example, if a domain expert knows that "has knees" should positively predict the Glorp class, alignment constrains that weight to be positive during retraining.

## Why alignment matters

In standard CBM training, the label predictor (logistic regression on concept activations) learns weights freely from data. This can produce **counterintuitive** weights — e.g., `has_knees` getting a *negative* weight even when knees truly indicate Glorp — because the model exploits correlations among imperfect concept predictions.

The paper shows that an alignment constraint can:

- **Raise** accuracy before interventions — the constrained model no longer relies on a concept its detector reads at chance.
- **Remove** the benefit of interventions — after every concept is corrected, the unconstrained model is the more accurate one.

The constraint closes the pathway through which a corrected concept reaches the label.

## Usage

After training a `ConceptBasedModel`, use `run_alignment()` to retrain the frontend with sign constraints and compare:

```python
from experiments.utils import run_alignment

results = run_alignment(
    concept_based_model=cbm,
    train_dataset=train,
    test_dataset=test,
    monotonicity_constraints={"has_knees": 1},  # require a non-negative weight
)

print(f"Original accuracy: {results['original_accuracy']:.4f}")
print(f"Aligned accuracy:  {results['aligned_accuracy']:.4f}")
print(f"Accuracy change:   {results['accuracy_change']:+.4f}")
```

The `monotonicity_constraints` dict maps concept names to their required sign: `+1` for positive weight, `-1` for negative.

## Expected results

Mean accuracy over the 10 seeds of the paper (balanced rule):

| Concepts | CBM (k=0) | Constrained (k=0) | CBM (k=max) | Constrained (k=max) |
|----------|-----------|-------------------|-------------|---------------------|
| true (7 concepts) | 84.5% | 89.8% | 92.0% | 90.7% |
| human (12 concepts) | 77.6% | 81.0% | 85.7% | 83.2% |

```{image} assets/alignment.png
:width: 700px
:align: center
:alt: Constrained against unconstrained CBM before and after interventions
```

For a complete end-to-end example with training, interventions, and alignment, see [`examples/robot_pipeline_example.py`](https://github.com/ustunb/concept-benchmark/blob/main/examples/robot_pipeline_example.py).
