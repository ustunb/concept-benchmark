# Robot Classification

This benchmark targets decision-support settings where a human uses the model's concept predictions to improve their own decisions. The task is to predict the species of a fictional robot — **Glorp** or **Drent** — from its body features. Each robot has 9 binary features (mouth type, foot shape, knee presence, etc.). The default labeling rule is: Glorp if mouth is closed, foot is pointy, and robot has knees (all three); Drent otherwise. The labeling function can be deterministic or stochastic (probabilistic), controlled via `use_stochastic_labels`. Which features matter and which are excluded (via `excluded_concepts`) are configurable, mimicking real-world settings where the true relationship between features and labels is unknown. Available as image and text modalities.

```{image} assets/robot_concepts.png
:width: 400px
:align: center
:alt: Robot with annotated concepts
```

## Parameters

All parameters below can be passed to `DatasetGenerator("robot", ...)`. Common parameters apply to both image and text modalities; scope-specific parameters are ignored when the other modality is selected.

```python
from concept_benchmark import DatasetGenerator

dataset = DatasetGenerator(
    "robot",
    # ── Common (image + text) ──
    seed=1014,                       # random seed (default: 1014 for image, 1337 for text)
    data_type="image",               # "image" (default) or "text"
    use_stochastic_labels=True,      # True (probabilistic) or False (deterministic threshold)
    train_size=3800,                 # number of training samples
    test_size=10000,                 # number of test samples
    label_formula={                  # scoring rule for class assignment
        "terms": {
            "mouth_type": {"value": "closed", "weight": 5.0},
            "foot_shape": {"value": "pointy", "weight": 8.0},
            "has_knees":  {"value": "true",   "weight": -5.0},
        },
        "intercept": 2.0,
        "temperature": 4.2,
    },
    missing_fraction=0.0,            # fraction of concept labels masked during training
    missing_mechanism="mcar",        # missingness mechanism: "mcar" or "mnar"
    concept_preset="foot_subtypes",  # "ground_truth" (7 concepts) or "foot_subtypes" (12)
    renders_per_robot=4,             # samples per unique robot config (image: 4, text: 1)
    sampling_constraints=[           # min-fraction constraints for skewed splits
        {"concepts": {"foot_shape_pointy_4sided": 1}, "min_fraction": 0.49},
        # ...
    ],
    excluded_concepts=None,          # features to exclude (auto-set by concept_preset)
    fine_grained_concepts=["foot_shape_subtype"],  # which features expand into subconcepts
    # ── Image-only (data_type="image") ──
    image_size="medium",             # "small" (8px), "medium" (32px), or "large" (600px)
    color_mode="color",              # "color" or "grayscale"
    render_images=True,              # set False to skip rendering PNGs (faster)
    # ── Text-only (data_type="text") ──
    template_complexity="high",      # template complexity level
).generate()
```

## Pipeline

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes
```

Run `python scripts/robot_pipeline.py --help` for the full list of options.
