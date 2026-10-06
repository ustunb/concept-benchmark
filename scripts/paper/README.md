# Scripts behind the paper

Everything in this folder is specific to the paper: it builds the paper's tables, figures and tests from
`results/paper/`, runs the diagnostics behind its findings, or prepares the LLM intervention caches. The benchmark's own
entry points are one level up (`scripts/robot_pipeline.py`, `scripts/sudoku_pipeline.py`,
`scripts/robot_text_pipeline.py`, and the data generators).

Every table, figure and test in the results section is produced by a script from one results folder. The folder is
not part of the repository. Get one in either way:

- **The paper's results.** Download `paper-results.zip` from the
  [releases page](https://github.com/ustunb/concept-benchmark/releases) and unpack it. It holds every file the tables,
  figures and tests read. `paper-results-large.zip` adds the intervention records, datasets, models, LLM answers and
  224 px images that the answer-exchange table, the alignment evaluation and the diagnostics read.
- **Your own runs.** Run the pipelines (see [`EXPERIMENTS.md`](../../EXPERIMENTS.md)), then copy their output into the
  same layout: `python scripts/paper/collect_pipeline_runs.py --runs results --results-root my_results`.

Pass the folder to any script as `--results-root <folder>`, or set `CONCEPT_BENCHMARK_PAPER_RESULTS` once. Without
either, the scripts read `results/paper/`. The "Reads" columns below are paths inside that folder. Run from the repository
root with `PYTHONPATH=.` and the project's Python. `_common.py` holds the paths and helpers the scripts share
(result folders, mean ± SE, table cells, pdflatex) and puts the repository root and `scripts/` on the import path.

## Tables

| Exhibit | Command | Reads |
|---|---|---|
| Robot table (accuracy by concept set, architecture, intervention source, budget) | `python scripts/paper/make_robot_big_table.py --rule balanced --out <paper>/tables/robot_big_table.tex` (`--rule sparse` for the appendix table) | `robot/balanced_rule/`, `robot/grid/` |
| Sudoku table | `python scripts/paper/make_sudoku_table.py --out <paper>/tables/sudoku_px_sweep.tex` | `sudoku/cells/`, `sudoku/selective/` |
| Sudoku sensitivity tables (accuracy target, cost of a check) and their tests | `python scripts/paper/make_sudoku_sensitivity_tables.py --tau-out <paper>/tables/sudoku_tau_sweep.tex --cost-out <paper>/tables/sudoku_cost_models.tex` | `sudoku/cells/` |
| Sudoku confidence tables and error breakdown | `python scripts/paper/make_sudoku_confidence_tables.py --classwise-out <paper>/tables/sudoku_classwise_threshold.tex --confirmed-out <paper>/tables/sudoku_confirmed_confidence.tex` | `sudoku/confidence/`, `sudoku/cells/` |
| Real-dataset table (Derm7pt, CUB-25) | `python experiments/real_data.py --out results/paper/real_datasets --derm-root <Derm7pt> --cub-root <CUB_200_2011>` then `python scripts/paper/make_real_data_table.py --out <paper>/tables/real_datasets.tex` | `real_datasets/` |
| Own-answers table | `python scripts/paper/make_own_answers_table.py --out <paper>/tables/own_answers.tex` | `robot/balanced_rule/` (`isrc-self`, `isrc-perfect`) |
| Answer-exchange table | `python scripts/paper/make_answer_exchange_table.py --out <paper>/tables/answer_exchange.tex` | `robot/balanced_rule/intervention_records/`, CBM models |

## Figures

The `plot_*` scripts write `<out>.tex` and compile `<out>.pdf` with pdflatex.

| Exhibit | Command | Reads |
|---|---|---|
| Misspecified-concepts figure | `python scripts/paper/plot_decision_support_panel.py --out results/paper/figures/fig_decision_support` | `robot/balanced_rule/` |
| Architectures figure | `python scripts/paper/plot_architecture_response.py --rule balanced --concepts human --out results/paper/figures/fig_architecture_response_balanced_human` | `robot/balanced_rule/` |
| Concept-source heatmap | `python scripts/paper/plot_delta_accuracy_pipelines.py --layout E --rule balanced --out results/paper/figures/fig_delta_accuracy_balanced_E` | `robot/balanced_rule/` |
| Alignment figure | `python scripts/paper/plot_alignment.py --out results/paper/figures/fig_alignment` | `robot/alignment/` |
| Example plots of the documentation (`docs/assets/*.png`, drawn with `concept_benchmark.evaluation`) | `python scripts/paper/make_readme_figures.py [--out <dir>]` | `robot/balanced_rule/`, `robot/alignment/`, `sudoku/cells/`, `sudoku/confidence/` |

## Tests

| Exhibit | Command | Reads |
|---|---|---|
| Paired tests quoted in the findings | `python scripts/paper/run_paired_tests.py` | `robot/balanced_rule/` |
| Alignment results (writes `robot/alignment/`) and their paired tests | `python scripts/paper/evaluate_alignment_balanced.py` (`--install` once, to extract the detector outputs from the intervention records) | CBM models, datasets, `robot/balanced_rule/detector_outputs/` |

## Diagnostics

The first two rows read the results folder only. The others load models (and most of them the robot images), so they run
where the models were trained; each docstring gives the checkout and arguments.

| Question | Command | Reads |
|---|---|---|
| How accurate are the interveners on the concepts intervened on? | `python scripts/paper/summarize_intervention_records.py --by-concept` | `robot/balanced_rule/intervention_records/` |
| What is the best achievable accuracy under the balanced rule's stochastic labels? | `python scripts/paper/compute_label_ceiling.py` | `robot/datasets/` |
| Why do `human_concepts` cap the CBM (robots with an unlisted foot subtype)? | `python scripts/paper/check_unlisted_subtypes.py [--images <robot_images>]` | CBM models, datasets |
| How do CEM and ECBM handle robots with an unlisted foot subtype? | `python scripts/paper/diagnose_incomplete_concepts.py --models <dir> --data-root <runs> --images <dir> --out <csv>` | CEM and ECBM models, run datasets, images |
| Does CEM's label predictor use the foot information its inputs carry? | `python scripts/paper/diagnose_cem_foot_usage.py --models <dir> --data-root <runs> --images <dir> --out <csv>` | CEM models, run datasets, images |
| Image-only vs joint label accuracy of one ECBM model | `python scripts/paper/eval_ecbm_image_only.py --model <file> --seed 1015 --data-root <runs> --images <dir>` | one ECBM model, run dataset, images |
| ECBM retrained with its concept terms switched off | `python scripts/paper/train_ecbm_label_only.py --seed 1015` | the seed's dataset (stage `setup`) |
| Does `experiments/baselines/ecbm.py` reproduce the authors' ECBM? | `python scripts/paper/check_ecbm_equivalence.py --original <ECBM checkout>` | the authors' code |
| Why does a partial intervention lower ECBM's accuracy? | `python scripts/paper/measure_ecbm_response.py --root <dir with run_ecbm_s<seed>> --out <csv>` | ECBM models and datasets of the runs |
| Why do architectures differ in their response to interventions? | `python scripts/paper/measure_intervention_response.py --images <robot_images>` | sparse-rule models, datasets, `robot/grid/` |
| Why does ProbCBM ignore single interventions? | `python scripts/paper/measure_probcbm_response.py --images <robot_images> --pipeline-root <root> --out <csv>` | ProbCBM and CBM models, datasets, images |
| Does ProbCBM learn the rule concepts on the training robots? | `python scripts/paper/measure_probcbm_detection.py --pipeline-root <root> --images <robot_images> --out <csv>` | ProbCBM models, datasets, images |
| Why do some ProbCBM runs respond to a full intervention and others not? | `python scripts/paper/measure_probcbm_heads.py --pipeline-root <root> --out <csv>` | ProbCBM models, datasets |
| ProbCBM with concept prototypes held fixed and opposite (causal test) | `PROBCBM_FIXED_PROTOTYPES=1 python scripts/paper/train_probcbm_fixed_prototypes.py --seed 1014 ...` (arguments of `scripts/robot_pipeline.py`) | as the robot pipeline |
| Concept leakage in CBM, CEM, ProbCBM and ECBM (seed 1014) | `python scripts/paper/diagnose_leakage.py [--concept-preset foot_subtypes]` | the pipeline's dataset and models |

## Data preparation for LLM interventions

Run in this order for new seeds; the pipeline then reads the caches with `--llm-cache-only`.

| Step | Command | Reads |
|---|---|---|
| Export each seed's test robots and concept names | `python scripts/paper/export_seed_tests.py --seeds 1018 1019 --out <dir>/seed_tests.json` | `robot/datasets/` |
| Render the test robots that lack a 224 px image | `python scripts/paper/render_robots_224.py --seed-tests <dir>/seed_tests.json` | `images/robot_224px/`, the 32 px dataset images (`--small-images`) |
| Fill the new seeds' caches with answers already given for the same image and questions | `python scripts/paper/assemble_llm_caches.py --seed-tests <dir>/seed_tests.json --out-dir <dir> --seeds 1018 1019` | `robot/llm_caches/` |
| Ask Gemini about the robots still missing (resumable; needs `GEMINI_API_KEY`) | `python scripts/paper/build_llm_caches.py --seed-tests <dir>/seed_tests.json --image-dir <results folder>/images/robot_224px --out-dir <dir> --seeds 1018 1019` | 224 px images, caches in `<dir>` |

## Pipeline options behind the diagnostics

These produce the inputs of the own-answers, answer-exchange and confidence tables and of the intervention-record
summary on any model the pipelines can train or load.

| Question | Option | Output |
|---|---|---|
| How much of an intervention's effect is the mechanism itself? | `python scripts/robot_pipeline.py ... --intervention-sources self` | a result cell in which every selected concept is set to the value the model already predicts (no answer changes) |
| Which concepts were intervened on, and what was answered? | `python scripts/robot_pipeline.py ... --dump-interventions DIR` | one `.npz` per cell and budget: mask, detector output, answers, true values, labels before/after |
| How confident is a sudoku model, and how confident can checks make it? | `python scripts/sudoku_pipeline.py ... --stages diagnose` | `sudoku_<model>_confidence_*.npz`: P(valid) on validation/test, concept probabilities, P(valid) with every concept set to its true value |

## Notes

- Sudoku abstention: the threshold is fitted on validation and applied with one rule (`abstention_threshold` in
  `concept_benchmark.evaluation` keeps predictions exactly at the threshold; see `tests/test_selective_thresholds.py`).
  `classwise_thresholds` fits one threshold per predicted class.
- ProbCBM samples at prediction time, so its confidence files and cells are one draw each.
- Models are saved with `device="cuda"`; scripts that load them on a CPU call `set_cpu` (defined with the
  intervention-response diagnostic) or read saved detector outputs instead.
