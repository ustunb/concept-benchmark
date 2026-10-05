# Robot Classification


```{image} assets/robot_concepts.png
:width: 400px
:align: center
:alt: Robot with annotated concepts
```

## Parameters

All parameters below can be passed to `DatasetGenerator(...)` (imported from `concept_benchmark.robots`). Common parameters apply to both image and text modalities; scope-specific parameters are ignored when the other modality is selected.

```python
from concept_benchmark.robots import DatasetGenerator, LabelFormula, F

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
    label_rule="balanced",           # "balanced" (default) or "sparse"; see Labeling rules below
    label_formula=None,              # or your own LabelFormula (see below), which replaces the rule's formula
    concept_preset="foot_subtypes",  # "ground_truth" or "foot_subtypes" (expands foot_shape into subtypes)
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

`generate()` returns the unsplit dataset with every concept; `generate_splits()` returns the train/val/test split of the paper (3,800 / validation / 10,000 robots with the preset's concepts).

## Labeling rules

A robot is a Glorp with probability `σ(4.2 × score)`, where the score comes from one of two rules:

| Rule | Score | What it models |
|------|-------|----------------|
| **balanced** (default) | `6·[mouth closed] + 6·[body round] + 6·[head round] + 6·[antennae] + 6·[ears triangle] + 8·[foot pointy] − 3·[knees] − 2·[elbows] − 16.5` | No single concept decides the label and both classes are equally likely. `has_elbows` drives the label but is not a concept, so it is visible in the image and cannot be intervened on. Used for the main results of the paper. |
| **sparse** | `5·[mouth closed] + 8·[foot pointy] − 5·[knees] + 2` | Three concepts decide the label and 87.5% of robots are Glorps. Models a task where the class of interest is rare. |

Each rule also sets which foot subtypes the training split of `generate_splits()` holds: six of the ten subtypes, at 30%/10% shares under the balanced rule and 49%/0.5% under the sparse rule (`concept_benchmark.config.ROBOT_LABEL_RULES`). To use your own rule, pass a `LabelFormula`:

```python
from concept_benchmark.robots import DatasetGenerator, LabelFormula, F

dataset = DatasetGenerator(
    seed=1014,
    label_formula=LabelFormula(
        score=5 * F("mouth_type").closed + 8 * F("foot_shape").pointy - 5 * F("has_knees").true + 2,
        temperature=4.2,              # P(Glorp) = σ(4.2 × score)
        stochastic=True,
    ),
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
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --cbm-family cem
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --cbm-family probcbm

# The rare-class rule
python scripts/robot_pipeline.py --seed 1014 --label-rule sparse

# Add plot to generate figures from results
python scripts/robot_pipeline.py --seed 1014 --stages setup cbm dnn intervene align collect plot
```

| Option | Description |
|--------|-------------|
| `--concept-preset` | `ground_truth` (7 true concepts) or `foot_subtypes` (12 human concepts) |
| `--label-rule` | `balanced` (default) or `sparse` |
| `--cbm-family` | `cbm`, `cem`, `probcbm` or `ecbm` |
| `--concept-sources` | Who annotates the concepts: `ground_truth`, `human_concepts`, `machine_annotation`, `llm_concepts`, `clip_concepts` |
| `--intervention-sources` | Who answers at test time: `perfect`, `expert`, `llm`, or `self` (the model's own predictions) |
| `--budgets` | Intervention budgets, e.g. `1 3 max` |
| `--intervention-encoding` | What a label-free CBM reads after an intervention: `binary` (default), `percentile` or `binary_revealed` |
| `--dump-interventions DIR` | Save what was asked and answered for each budget (input to `plot_concept_report`) |

Run `python scripts/robot_pipeline.py --help` for the full list of options (including training and LLM parameters). See [Interventions](interventions.md) for the concept and intervention sources.

Over the 10 seeds of the paper, the CBM reaches 84.5% with the true concepts and 77.6% with the human concepts before interventions, and 92.0% and 85.7% once every concept is corrected; the DNN reaches 88%. A single run differs from these means by a point or two, and across hardware.

```{image} assets/intervention_curve.png
:width: 500px
:align: center
:alt: Accuracy against the intervention budget for four architectures
```
