# Interventions

Interventions are the core benefit of concept bottleneck models: at test time, a user (or automated system) can inspect and correct the model's concept predictions before the final label is determined. This page explains how to perform interventions programmatically.

## Default automated concepts and intervention answers (pip)

The paper's automated concept sets and interveners come with the package, for every robot image:

```python
from concept_benchmark.robots import DatasetGenerator, load_ai_concepts, load_intervention_answers

data = DatasetGenerator(seed=1014, concept_preset="foot_subtypes").generate_splits()
labels, names, texts = load_ai_concepts(data.train, "llm_written")  # or "machine_labeled", "clip_words"
answers = load_intervention_answers(data.test, "llm")               # or "expert" (80% accurate), "perfect"
```

| | pip | repo |
|---|---|---|
| Automated concept labels | the paper's three sets | your own concept lists |
| Intervention answers | `perfect`, `expert`, `llm` (Gemini 2.5 Flash-Lite) | other LLMs or prompts, the model's own answers |

LLM answers are one per robot image and concept, taken from the paper's runs. For an automated concept set, pass its names: `load_intervention_answers(data.test, "llm", concepts=names)`.

**How the automated labels were made, and how to use your own cutoff.** Each label comes from a score in [0, 1] (`load_ai_concept_scores`): the paper's label-free model (seed 1014) projects CLIP similarities onto the concepts, standardizes them over its training robots, and applies a sigmoid. The CBM is the label-free model itself and uses these scores. To train CEM, ProbCBM, and ECBM, which need 0/1 concept labels, the paper labeled a concept 1 when its score is at least 0.5, the training average, so all three trained on the same labels. That one cutoff makes about half the robots positive for every concept, also for rare ones (each foot type is on about 10% of robots), so these labels are noisy. To apply another cutoff, threshold the scores, for example at each concept's share of positives in the true labels:

```python
import numpy as np
from concept_benchmark.robots import load_ai_concept_scores

scores, names, _ = load_ai_concept_scores(data.train, "machine_labeled")
true = np.asarray(data.train.base_concepts)
share = np.array([true[:, data.train.concepts.index(name)].mean() for name in names])
cutoffs = [np.quantile(scores[:, j], 1 - s) for j, s in enumerate(share)]
labels = (scores >= cutoffs).astype(int)  # the same share of 1s as the true labels
```

## Oracle interventions (manual approach)

The simplest way to run interventions is to directly manipulate concept predictions. After training a `ConceptDetector` and `FrontEndModel`, the workflow is:

1. Get concept probabilities from the detector
2. Identify the most uncertain concepts per sample
3. Replace them with ground-truth values
4. Re-predict with the label model

```python
import numpy as np
from experiments.models import ConceptDetector, FrontEndModel

# cd and fe are a trained concept detector and label predictor, e.g. cbm.concept_detector and
# cbm.label_predictor of a CBM from experiments.evaluate.train_cbm

# Step 1: Get concept probabilities
concept_probs = cd.predict_proba(test)

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

> **Note:** `ConceptDetector.predict_proba()` returns probabilities; `ConceptDetector.predict()` thresholds them at 0.5 and returns binary values.

> **Important:** `FrontEndModel.predict()` expects **binary** concept values (0/1), not probabilities. Always threshold before passing to the label predictor.

## Using the intervention API

For more complex intervention scenarios (budgets, strategies, batched evaluation), use the `ConceptInterventionRunner`:

```python
import numpy as np
from experiments.models import ConceptBasedModel
from experiments.intervention import ConceptInterventionRunner, InterventionConfig
from experiments.kflip import KFlipInterventionStrategy

# Combine detector and label predictor into a CBM
cbm = ConceptBasedModel(concept_detector=cd, label_predictor=fe)

# Configure the intervention
config = InterventionConfig(
    per_instance_budget=3,  # correct up to 3 concepts per sample
    score_threshold=0.2,          # intervene when the label would change
                                  # with probability 0.2 or more
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

## Writing a custom strategy

You can implement your own intervention strategy by subclassing `InterventionStrategy` and implementing the `propose()` method.

### The intervention flow

When `ConceptInterventionRunner.run()` is called, it:

1. Builds an `InterventionBatch` from the dataset (concept predictions + ground truth)
2. Calls `strategy.propose(model, batch, config)` → returns a `StrategyProposal`
3. Applies the proposal's `mask` to replace predicted concepts with ground truth
4. Re-predicts labels with the corrected concepts
5. Returns an `InterventionResult` with before/after predictions

### Key data classes

**`InterventionBatch`** — the input your strategy receives:

| Field | Type | Description |
|-------|------|-------------|
| `C_pred` | `(N, C) float` | Predicted concept probabilities |
| `C_true` | `(N, C) float` | Ground-truth concept values |
| `y_true` | `(N,) int` or `None` | True labels (optional) |
| `n_samples` | `int` | Number of samples |
| `n_concepts` | `int` | Number of concepts |

**`StrategyProposal`** — what your strategy returns:

| Field | Type | Description |
|-------|------|-------------|
| `mask` | `(N, C) bool` | `True` = replace prediction with ground truth |
| `ordering_used` | array or `None` | Concept order applied (optional) |
| `selected_instances` | array or `None` | Instance indices that received interventions |
| `details` | `dict` | Additional metadata |

### Minimal example

Here's a strategy that intervenes on concepts closest to the 0.5 decision boundary:

```python
import numpy as np
from experiments.intervention import InterventionStrategy, StrategyProposal

class UncertaintyStrategy(InterventionStrategy):
    def __init__(self):
        super().__init__(name="uncertainty")

    def propose(self, model, batch, config):
        k = int(min(config.per_instance_limit(batch.n_concepts), batch.n_concepts))
        mask = np.zeros((batch.n_samples, batch.n_concepts), dtype=bool)

        # Rank concepts by uncertainty (closeness to 0.5)
        uncertainty = 0.5 - np.abs(batch.C_pred - 0.5)
        for i in range(batch.n_samples):
            top_k = np.argsort(uncertainty[i])[-k:]
            mask[i, top_k] = True

        return StrategyProposal(mask=mask)
```

Use it with the runner:

```python
result = runner.run(
    strategy=UncertaintyStrategy(),
    config=InterventionConfig(per_instance_budget=3, score_threshold=0.2),
    dataset=test,
)
```

### The `prepare()` hook

Override `prepare()` if your strategy needs a validation pass before inference — for example, to precompute a global concept ordering:

```python
class MyStrategy(InterventionStrategy):
    def prepare(self, model, batch, config):
        """Called once on the validation set before run()."""
        # Compute concept importance from validation data
        self._state["concept_order"] = compute_importance(model, batch)

    def propose(self, model, batch, config):
        order = self._state["concept_order"]
        # ... use precomputed order
```

Call `runner.prepare()` before `runner.run()` to trigger the hook:

```python
runner.prepare(strategy, config, validation_dataset=val)
result = runner.run(strategy, config, dataset=test)
```

## Concept sources and intervention sources

The robot pipeline varies two things independently: who annotates the concepts a model is trained on, and who answers when the model asks about a concept at test time.

| Concept source | Concepts |
|----------------|----------|
| **ground_truth** | The 7 true concepts |
| **human_concepts** | 12 concepts as a human annotator would list them (six foot subtypes) |
| **noisy_human_concepts** | The same concepts, annotated with 20% label noise (annotators who disagree) |
| **machine_annotation** | The human concepts, scored by CLIP (label-free CBM) |
| **llm_concepts** | Concepts written by an LLM, scored by CLIP |
| **clip_concepts** | Single words chosen by CLIP, scored by CLIP |

| Intervention source | Answers |
|---------------------|---------|
| **perfect** | The true concept values — the upper bound |
| **expert** | A simulated expert, correct 80% of the time on every concept |
| **llm** | An LLM that answers each concept from the image (Gemini by default) |
| **self** | The model's own thresholded predictions — changes no value, so it isolates the effect of the intervention mechanism |

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes \
    --concept-sources human_concepts machine_annotation \
    --intervention-sources perfect expert self --budgets 1 3 max
```

LLM interventions need an API key (`--llm-api-key` or `GEMINI_API_KEY`); `--llm-cache-only` reuses saved answers.

### Regimes

A regime names one pairing of concept source and intervention source, as in the first version of the benchmark:

| Regime | Concepts from | Corrected by | Description |
|--------|--------------|-------------|-------------|
| **baseline** | Human annotation | True values | Upper bound on the benefit of interventions |
| **expert** | Human annotation | Expert (80% accurate) | Realistic human annotator |
| **subjective** | Human annotation with 20% label noise | Expert (80% accurate) | Annotators who disagree on the concepts |
| **machine** | CLIP scores of the human concepts | Expert (80% accurate) | Automated annotation |
| **llm** | Concepts written by an LLM | LLM | Fully automated with an LLM |
| **clip** | Single words chosen by CLIP | LLM | Fully automated with CLIP |

Both robot pipelines accept `--regimes` as shorthand for the matching sources:

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --regimes baseline expert subjective
python scripts/robot_text_pipeline.py --seed 1337 --regimes baseline expert subjective
```

Two further options control how interventions are applied and recorded:

| Option | Description |
|--------|-------------|
| `--intervention-encoding` | What a label-free CBM reads after an intervention: `binary` (default), `percentile` or `binary_revealed` |
| `--dump-interventions DIR` | Save what was asked and answered for each budget (input to `plot_concept_report`) |

For a complete end-to-end example with training, interventions (through the `intervention_table` block, which runs `ConceptInterventionRunner` with the paper's policy) and alignment, see [`examples/robot_pipeline_example.py`](https://github.com/ustunb/concept-benchmark/blob/main/examples/robot_pipeline_example.py).
