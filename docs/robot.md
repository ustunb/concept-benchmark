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
    seed=1014,                       # random seed (default: 1014; the text pipeline uses 1337)
    data_type="image",               # "image" (default) or "text"
    concepts={                           # 9 features (default: ROBOT_CONCEPTS)
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        # the two shapes list their subtypes; each becomes one binary concept (round/edgy, flat/pointy)
        # unless it is named in expand_concepts
        "hand_shape": ["round_circle", "round_oval", "round_oval2",
                       "edgy_triangle", "edgy_square", "edgy_trapezoid"],
        "foot_shape": ["flat_trapezoid", "flat_rounded", "flat_square", "flat_5sided", "flat_lshaped",
                       "pointy_trapezoid", "pointy_rounded", "pointy_square", "pointy_3sided", "pointy_4sided"],
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
    # ── Text-only (data_type="text") ──
    template_complexity="high",      # template complexity level
    corpus_file=None,                # a caption corpus of your own, see below
).generate()
```

The captions come from a corpus of templates: `hard_corpus.jsonl` (`template_complexity="high"`), `templates.txt` (`"medium"`) or `templates_simple.txt` (`"low"`), all under `concept_benchmark/synthetic/robot/static/text_templates/`. `corpus_file` names another file from that folder, or a path of your own, and the folder holds variants in which a concept is not stated in the text: `templates_foot_generic.jsonl` and `templates_simple_foot_generic.jsonl` say that the robot has feet but not their shape, `hard_corpus_foot_generic.jsonl` does the same in the harder style, `hard_corpus_ears_generic.jsonl` leaves the ears vague, `hard_corpus_no_antenna.jsonl` never mentions antennae, and `templates_simple_v2.txt` is a plainer phrasing. A concept that the captions never state cannot be detected from the text, which is the text-modality counterpart of a concept that is not visible in an image.

`generate()` returns the unsplit dataset with every concept; `generate_splits()` returns the train/val/test split of the paper (3,800 / validation / 10,000 robots with the preset's concepts).

## Labeling rules

A robot is a Glorp with probability `σ(4.2 × score)`, where the score comes from one of two rules:

| Rule | Score | What it models |
|------|-------|----------------|
| **balanced** (default) | `6·[mouth closed] + 6·[body round] + 6·[head round] + 6·[antennae] + 6·[ears triangle] + 8·[foot pointy] − 3·[knees] − 2·[elbows] − 16.5` | No single concept decides the label and both classes are equally likely. `has_elbows` drives the label but is not a concept, so it cannot be intervened on; its dot is drawn on every robot below 120 px, so it cannot be read from the 32 px images either. Used for the main results of the paper. |
| **sparse** | `5·[mouth closed] + 8·[foot pointy] − 5·[knees] + 2` | Three concepts decide the label and 87.5% of robots are Glorps. Models a task where the class of interest is rare. |

Each rule also sets which foot subtypes the training split of `generate_splits()` holds: six of the ten subtypes, at 30%/10% shares under the balanced rule and 49%/0.5% under the sparse rule (`concept_benchmark.config.ROBOT_LABEL_RULES`). To use your own rule, pass a `LabelFormula`:

```python
from concept_benchmark.robots import DatasetGenerator, LabelFormula, F

dataset = DatasetGenerator(
    seed=1014,
    label_formula=LabelFormula(
        score=4 * F("mouth_type").closed + 6 * F("body_shape").round - 3 * F("has_knees").true - 2,
        temperature=4.2,              # P(Glorp) = σ(4.2 × score)
        stochastic=True,
    ),
).generate()
```

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
| `--concept-sources` | Who annotates the concepts: `ground_truth`, `human_concepts`, `noisy_human_concepts`, `machine_annotation`, `llm_concepts`, `clip_concepts` |
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
