# Robot Classification

This benchmark targets decision-support settings where a human uses the model's concept predictions to improve their own decisions. The task is to predict the species of a fictional robot — **Glorp** or **Drent** — from its body features. Each robot has 9 binary features (mouth type, foot shape, knee presence, etc.). The default labeling rule is: Glorp if mouth is closed, foot is pointy, and robot has knees (all three); Drent otherwise. The labeling function can be deterministic or stochastic (probabilistic), controlled via the `model_type` parameter. Which features matter and which are excluded (via `drop_concepts`) are configurable, mimicking real-world settings where the true relationship between features and labels is unknown. Available as image and text modalities.

```{image} assets/robot_concepts.png
:width: 400px
:align: center
:alt: Robot with annotated concepts
```

## Expected Results

Generate the paper dataset and train a CBM (see {doc}`quickstart` for the full training code):

```python
from concept_benchmark import RobotDatasetGenerator

dataset = RobotDatasetGenerator(
    seed=1014,
    subconcept=True,          # 12 concepts (default: 7 coarse)
    model_type="stochastic",  # probabilistic labeling
).generate()
```

Expected results with oracle interventions (seed=1014, subconcept):

| budget (k) | accuracy |
|------------|----------|
| 0 | 0.7812 |
| 1 | 0.9212 |
| 3 | 0.9439 |

Or run the entire pipeline — including interventions, alignment, and regime comparisons — from the command line:

```bash
python scripts/robot_pipeline.py --seed 1014 --subconcept
```

The pipeline supports additional flags for intervention regimes (`--regimes expert subjective machine`), concept missingness (`--concept-missing 0.2`), running specific stages (`--stages cbm dnn intervene`), and more. Run `--help` for the full list.

## Parameters

All parameters below can be passed directly to `RobotDatasetGenerator()` or as CLI flags to `robot_pipeline.py`. For the full list, see `concept_benchmark/config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_type` | `"image"` | `"image"` (render robot PNGs) or `"text"` (generate text descriptions). |
| `label_formula` | `{("mouth_type","closed"): 5, ("foot_shape","pointy"): 8, ("has_knees","true"): -5, "intercept": 2}` | Labeling function. Score = `Σ wᵢ · 1[fᵢ = vᵢ] + intercept`. |
| `model_type` | `"stochastic"` | `"deterministic"`: Glorp if score ≥ 0. `"stochastic"`: Glorp ~ Bernoulli(σ(scalar × score)). |
| `drop_concepts` | `IDEAL_DROP` | Which concepts to exclude. Two presets: `IDEAL_DROP` for 7 coarse concepts, `SUBCONCEPT_DROP` for 12 fine-grained concepts. |
| `concept_missing` | `0.0` | Fraction of concept labels masked during training. |
| `regimes` | `["baseline"]` | How interventions are performed: `baseline` (oracle), `expert` (noisy human), `subjective` (noisy concept labels + noisy human), `machine`/`llm`/`clip` (concepts discovered via [Label-Free CBM](https://arxiv.org/abs/2304.06129)). |

<details>
<summary>Remaining parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subconcept` | `False` | Shortcut that switches `drop_concepts` to `SUBCONCEPT_DROP` (12 fine-grained concepts). |
| `seed` | `1014` / `1337` | Random seed (image / text) |
| `size` | `"medium"` | Image resolution: `"small"` (8px), `"medium"` (32px), `"large"` (600px). Image only. |
| `samples_per_instance` | `4` | Number of images per unique robot configuration. Total dataset size = unique configs × this value. |
| `color_mode` | `"color"` | `"color"` or `"grayscale"`. Image only. |
| `model_scalar` | `4.2` | Sigmoid temperature for stochastic labeling (higher = more deterministic) |
| `skew_specs` | (see config) | List of dicts specifying class-balance constraints for training data (e.g., minimum fraction of specific concept values). |
| `concept_missing_mech` | `"none"` | Missingness mechanism: `"none"`, `"mcar"`, or `"mnar"` |
| `intervention_budgets` | `[1, 3]` | Number of concepts to correct per sample |
| `intervention_thresholds` | `[0.2, 0.4]` | Concepts whose predicted probability is within this distance of 0.5 are candidates for intervention |
| `intervention_strategy` | `"kflip"` | `"kflip"` (up to *k* concepts) or `"exact_k"` (exactly *k*) |
| `alignment_constraints` | `{}` | Sign constraints on concept weights (e.g., `{"has_knees": 1}`). Retrains the label predictor and re-evaluates interventions. |
| `difficulty` | `"hard"` | Corpus difficulty (text only) |
| `generic_rate` | `0.7` | Fraction of test set using concept-ambiguous text (text only) |

</details>

```{note}
The `llm` and `clip` regimes call the Gemini API at intervention time. Set your key before running:
`export GEMINI_API_KEY=your_key_here`
```
