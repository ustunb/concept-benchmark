# Robot Classification

This benchmark targets decision-support settings where a human uses the model's concept predictions to improve their own decisions. The task is to predict the species of a fictional robot — **Glorp** or **Drent** — from its body features. Each robot has 9 binary features (mouth type, foot shape, knee presence, etc.). The default labeling rule is: Glorp if mouth is closed, foot is pointy, and robot has knees (all three); Drent otherwise. The labeling function can be deterministic or stochastic (probabilistic), controlled via the `model_type` parameter. Which features matter and which are excluded (via `drop_concepts`) are configurable, mimicking real-world settings where the true relationship between features and labels is unknown. Available as image and text modalities.

```{image} assets/robot_concepts.png
:width: 400px
:align: center
:alt: Robot with annotated concepts
```

## Parameters

All parameters can be passed to `RobotDatasetGenerator()` or as CLI flags to `robot_pipeline.py`:

```python
from concept_benchmark import RobotDatasetGenerator

dataset = RobotDatasetGenerator(
    seed=1014,                # random seed (default: 1014 for image, 1337 for text)
    data_type="image",        # "image" (render PNGs) or "text" (generate descriptions)
    subconcept=True,          # True: 12 fine-grained concepts; False: 7 coarse (default)
    model_type="stochastic",  # "stochastic" (probabilistic) or "deterministic" (threshold)
    model_scalar=4.2,         # sigmoid temperature for stochastic labeling
    size="medium",            # image resolution: "small" (8px), "medium" (32px), "large" (600px)
    color_mode="color",       # "color" or "grayscale" (image only)
    samples_per_instance=4,   # images per unique robot config (total = configs × this)
    concept_missing=0.0,      # fraction of concept labels masked during training
    concept_missing_mech="none",  # missingness mechanism: "none", "mcar", or "mnar"
    # label_formula: scoring rule for class assignment (see config.py for default)
    # drop_concepts: which features to exclude (preset via subconcept flag)
    # skew_specs: class-balance constraints for training data (see config.py)
    # --- text-only ---
    # difficulty="hard"       # corpus difficulty
    # generic_rate=0.7        # fraction of test set using concept-ambiguous text
).generate()
```

Run the full pipeline (interventions, alignment, regime comparisons) from the CLI:

```bash
python scripts/robot_pipeline.py --seed 1014 --subconcept                        # basic run
python scripts/robot_pipeline.py --seed 1014 --subconcept \
    --regimes baseline expert subjective machine                                  # intervention regimes
python scripts/robot_pipeline.py --seed 1014 --subconcept --concept-missing 0.2  # concept noise
python scripts/robot_pipeline.py --seed 1014 --subconcept --stages cbm dnn intervene  # specific stages
# run --help for all flags
```

```{note}
The `llm` and `clip` regimes call the Gemini API at intervention time. Set `GEMINI_API_KEY` before running.
```
