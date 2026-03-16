# Interventions

Interventions are the core benefit of concept bottleneck models: at test time, a user (or automated system) can inspect and correct the model's concept predictions before the final label is determined. This page explains how to perform interventions programmatically.

## Oracle interventions (manual approach)

The simplest way to run interventions is to directly manipulate concept predictions. After training a `ConceptDetector` and `FrontEndModel`, the workflow is:

1. Get concept probabilities from the detector
2. Identify the most uncertain concepts per sample
3. Replace them with ground-truth values
4. Re-predict with the label model

```python
import numpy as np
from experiments.models import ConceptDetector, FrontEndModel

# Assume cd and fe are already trained (see examples/robot_pipeline_example.py)

# Step 1: Get concept probabilities
# NOTE: cd.predict() returns probabilities in [0, 1], NOT binary predictions.
concept_probs = cd.predict(test)

# Step 2-3: For each sample, replace the k most uncertain concepts
# with ground-truth values
for k in [1, 3]:
    C_intervened = concept_probs.copy()
    uncertainty = np.abs(concept_probs - 0.5)  # distance from decision boundary
    for i in range(len(test)):
        most_uncertain = np.argsort(uncertainty[i])[:k]
        C_intervened[i, most_uncertain] = test.C[i, most_uncertain]

    # Step 4: Threshold to binary and predict
    C_binary = (C_intervened > 0.5).astype(np.float32)
    preds = fe.predict(C_binary)
    acc = np.mean(preds == test.y)
    print(f"k={k}: accuracy={acc:.4f}")
```

> **Important:** `ConceptDetector.predict()` returns **probabilities**, not binary predictions. This differs from the sklearn convention. To get binary predictions, threshold at 0.5: `binary = (cd.predict(dataset) > 0.5).astype(int)`.

> **Important:** `FrontEndModel.predict()` expects **binary** concept values (0/1), not probabilities. Always threshold before passing to the label predictor.

## Using the intervention API

For more complex intervention scenarios (budgets, strategies, batched evaluation), use the `ConceptInterventionRunner`:

```python
import numpy as np
from experiments.models import ConceptBasedModel
from experiments.intervention import ConceptInterventionRunner, InterventionConfig
from experiments.kflip import KFlipInterventionStrategy

# Combine detector and label predictor into a CBM
cbm = ConceptBasedModel(concept_detector=cd, front_end_model=fe)

# Configure the intervention
config = InterventionConfig(
    max_concepts_per_instance=3,  # correct up to 3 concepts per sample
    score_threshold=0.2,          # only intervene on concepts with
                                  # probability within 0.2 of 0.5
)

# Run interventions
runner = ConceptInterventionRunner(model=cbm)
strategy = KFlipInterventionStrategy()
result = runner.run(strategy, config, test)

# InterventionResult contains y_prob_before/after and y_pred_after
acc_before = np.mean(np.argmax(result.y_prob_before, axis=1) == test.y)
acc_after = np.mean(result.y_pred_after == test.y)
print(f"Accuracy before: {acc_before:.4f}")
print(f"Accuracy after:  {acc_after:.4f}")
print(f"Concepts corrected: {result.mask.sum()}")
```

### Key classes

- **`InterventionConfig`** — controls intervention budgets, thresholds, and per-instance caps
- **`KFlipInterventionStrategy`** — the default strategy: evaluates all subsets of up to *k* concepts per sample and selects the intervention that maximizes predicted confidence
- **`ConceptInterventionRunner`** — coordinates intervention execution and before/after evaluation

## Intervention regimes

The package supports six intervention regimes that simulate different real-world annotation scenarios. Each regime varies the concept source (how concepts are predicted) and the intervention source (who corrects them):

| Regime | Concepts from | Corrected by | Description |
|--------|--------------|-------------|-------------|
| **baseline** | Ground truth | Ground truth | Perfect oracle — upper bound on intervention benefit |
| **expert** | Ground truth | Noisy human (80% acc) | Realistic human annotator |
| **subjective** | Noisy CBM (20% label noise) | Noisy human (80% acc) | Concepts trained on noisy labels |
| **machine** | LFCBM (GT descriptions) | Noisy human (80% acc) | Machine-discovered concepts |
| **llm** | LFCBM (LLM descriptions) | LLM (Gemini) | Fully automated with LLM |
| **clip** | LFCBM (CLIP keywords) | LLM (Gemini) | Fully automated with CLIP |

Run regimes via the pipeline script:

```bash
python scripts/robot_pipeline.py --seed 1014 --subconcept \
    --regimes baseline expert subjective machine
```

For details on each regime, see the [Robot benchmark documentation](robot.md).
