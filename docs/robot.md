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
    render_space_mode="legacy",      # "legacy", "continuous_light", or "continuous_heavy"
    validate_renders=True,           # used in continuous image modes
    max_render_validation_attempts=8,
    group_split_by_semantic_id=False,# opt-in grouped split for continuous modes
    validation_checks={              # optional per-rule validation control
        "hands_head_clearance": "auto",  # "auto", "on", or "off"
        "hands_body_clearance": "auto",
    },
    render_nuisance={                # optional nuisance range overrides
        "translate_x_frac": [-0.04, 0.04],
        "translate_y_frac": [-0.03, 0.03],
        "arm_angle_offset_deg": [-18.0, 22.0],
        "leg_spread_deg": [-8.0, 10.0],
    },
    # ── Text-only (data_type="text") ──
    template_complexity="high",      # template complexity level
    include_pose_text=False,         # off by default; adds minimal neutral pose text
    pose_text_mode="neutral",
).generate()
```

## Semantic IDs vs. Render IDs

The benchmark now separates:

- `semantic_id`: the finite concept identity. This controls the concept vector and the label.
- `render_id`: nuisance-only variation. This controls pose, scale, translation, mild rotation, stroke jitter, and small palette changes.
- `instance_id`: the full rendered sample identifier.

In `render_space_mode="legacy"`, generation stays on the old paper-style path. In the continuous modes, the semantic space stays finite but the render-instance space becomes effectively unbounded.

`continuous_light` is the safe default for 32x32 renders. `continuous_heavy` pushes the nuisance ranges further and is more useful for larger resolutions or dedicated robustness checks.

## Validation

Continuous image generation uses cheap deterministic validation and rejection sampling. The checks include:

- the overall robot staying inside the frame,
- non-degenerate foreground coverage,
- required part visibility,
- no accidental elbow or knee markers when those concepts are absent,
- no clipping of the head, mouth, feet, or body,
- optional per-pair clearance checks such as `hands_head_clearance` or `hands_body_clearance`.

Each validation rule can be disabled independently through `validation_checks`. For example, you can keep clipping and visibility validation on while disabling only `hands_body_clearance`.

Validation metadata is stored in the robot catalog, including:

- `semantic_id`, `render_id`, `instance_id`
- sampled nuisance values
- `validation_passed`
- `validation_attempts`
- `validation_fail_reason`
- `validation_failed_check`
- `validation_used_fallback`

If repeated continuous samples fail validation, the generator falls back to a canonical accepted state instead of emitting a broken render.

## Optional Pose-Aware Text

Text remains unchanged unless `include_pose_text=True`.

When enabled, the text generator appends only minimal neutral descriptors, for example:

- `arms angled slightly upward`
- `standing with a wider stance`
- `leaning a bit to the left`

This path is deterministic and does not depend on any external API.

## Preview Utility

Use the preview script to inspect nuisance variation without training:

```bash
python scripts/preview_robot_render_space.py --mode continuous_light --output-dir results/robot_preview
```

The script writes:

- `same_semantic.png`: one semantic robot under many nuisance states
- `semantic_variety.png`: several semantic identities in continuous mode
- `legacy_vs_continuous.png`: a direct legacy versus continuous comparison

## Pipeline

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes
```

Run `python scripts/robot_pipeline.py --help` for the full list of options.
