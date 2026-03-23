# Robot Classification


```{image} assets/robot_concepts.png
:width: 400px
:align: center
:alt: Robot with annotated concepts
```

## Parameters

All parameters below can be passed to `DatasetGenerator(...)` (imported from `concept_benchmark.robots`). Common parameters apply to both image and text modalities; scope-specific parameters are ignored when the other modality is selected.

```python
from concept_benchmark.robots import DatasetGenerator, LabelFormula

dataset = DatasetGenerator(
    # ── Common (image + text) ──
    seed=1014,                       # random seed (default: 1014 for image, 1337 for text)
    data_type="image",               # "image" (default) or "text"
    concepts={                           # 9 features (default: ROBOT_CONCEPTS)
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": ["round", "edgy"],       # collapsed to binary by default
        "foot_shape": ["flat", "pointy"],      # collapsed to binary by default
        # Subconcepts (use expand_concepts to expose individual subtypes):
        #   hand_shape: round_circle, round_oval, round_oval2,
        #               edgy_triangle, edgy_square, edgy_trapezoid
        #   foot_shape: flat_trapezoid, flat_rounded, flat_square, flat_5sided,
        #               flat_lshaped, pointy_trapezoid, pointy_rounded,
        #               pointy_square, pointy_3sided, pointy_4sided
    },
    label_formula=LabelFormula(       # scoring rule for class assignment
        mouth_type=("closed", 5.0),   #   score = 5·[mouth=closed] + 8·[foot=pointy] - 5·[knees=true] + 2
        foot_shape=("pointy", 8.0),
        has_knees=("true", -5.0),
        intercept=2.0,
        temperature=4.2,              #   P(Glorp) = σ(4.2 × score)
    ),
    concept_preset="foot_subtypes",  # "ground_truth" (7 concepts) or "foot_subtypes" (12)
    renders_per_robot=4,             # samples per unique robot config (image: 4, text: 1)
    expand_concepts=["foot_shape"],                 # which features expand into subconcepts
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
