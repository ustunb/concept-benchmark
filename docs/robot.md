# Robot Classification

This benchmark targets decision-support settings where a human uses the model's concept predictions to improve their own decisions. The task is to predict the species of a fictional robot — **Glorp** or **Drent** — from its body features. Each robot has 9 binary features (mouth type, foot shape, knee presence, etc.). The default labeling rule is: Glorp if mouth is closed, foot is pointy, and robot has knees (all three); Drent otherwise. The labeling function can be deterministic or stochastic (probabilistic), controlled via `use_stochastic_labels`. Which features matter and which are excluded (via `excluded_concepts`) are configurable, mimicking real-world settings where the true relationship between features and labels is unknown. Available as image and text modalities.

```{image} assets/robot_concepts.png
:width: 400px
:align: center
:alt: Robot with annotated concepts
```

## Parameters

All parameters can be passed to `DatasetGenerator("robot", ...)` or as CLI flags to `robot_pipeline.py`:

```python
from concept_benchmark import DatasetGenerator

dataset = DatasetGenerator(
    "robot",
    seed=1014,                       # random seed (default: 1014 for image, 1337 for text)
    concept_preset="foot_subtypes",  # "foot_subtypes": 12 fine-grained; "ground_truth": 7 (default)
    use_stochastic_labels=True,      # True (probabilistic) or False (deterministic threshold)
    image_size="medium",             # "small" (8px), "medium" (32px, default), or "large" (600px)
    color_mode="color",              # "color" or "grayscale" (image only)
    renders_per_robot=4,             # images per unique robot config (total = configs × this)
    missing_fraction=0.0,            # fraction of concept labels masked during training
    missing_mechanism="mcar",        # missingness mechanism: "mcar" or "mnar"
    label_formula={                  # scoring rule for class assignment
        "terms": {
            "mouth_type": {"value": "closed", "weight": 5.0},
            "foot_shape": {"value": "pointy", "weight": 8.0},
            "has_knees":  {"value": "true",   "weight": -5.0},
        },
        "intercept": 2.0,
        "temperature": 4.2,
    },
    # excluded_concepts: which features to exclude (preset via concept_preset)
).generate()
```

## Pipeline

To train models and run the full evaluation (interventions, alignment, etc.) without writing Python, use the pipeline script:

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes
```

Run `python scripts/robot_pipeline.py --help` for the full list of options.
