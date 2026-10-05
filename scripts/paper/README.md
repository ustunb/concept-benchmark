# Scripts behind the paper

Everything in this folder is specific to the paper: it builds the paper's tables, figures and tests from
`results/paper/`, or installs cluster outputs into that tree. The benchmark's own entry points are one level up
(`scripts/robot_pipeline.py`, `scripts/sudoku_pipeline.py`, `scripts/robot_text_pipeline.py`, and the data generators).

Every table, figure and test in the results section is produced by a script from the files in `results/paper/`
(see `results/paper/README.md` for the layout and `INDEX.csv` for the source of every file). Run from the repository
root with `PYTHONPATH=.` and the project's Python.

## Exhibits

| Exhibit | Command | Reads |
|---|---|---|
| Robot table (accuracy by concept set, architecture, intervention source, budget) | `python scripts/paper/make_robot_big_table.py --rule balanced --out <paper>/tables/robot_big_table.tex` (`--rule sparse` for the appendix table) | `robot/balanced_rule/`, `robot/grid/` |
| Misspecified-concepts figure | `python scripts/paper/plot_decision_support_panel.py --out results/paper/figures/fig_decision_support` | `robot/balanced_rule/` |
| Architectures figure | `python scripts/paper/plot_architecture_response.py --rule balanced --concepts human --out results/paper/figures/fig_architecture_response_balanced_human` | `robot/balanced_rule/` |
| Concept-source heatmap | `python scripts/paper/plot_delta_accuracy_pipelines.py --layout E --rule balanced --out results/paper/figures/fig_delta_accuracy_balanced_E` | `robot/balanced_rule/` |
| Alignment figure and tests | `python scripts/paper/evaluate_alignment_balanced.py` then `python scripts/paper/plot_alignment.py --out results/paper/figures/fig_alignment` | models, datasets, `robot/balanced_rule/detector_outputs/` |
| Sudoku table | `python scripts/paper/make_sudoku_table.py --out <paper>/tables/sudoku_px_sweep.tex` | `sudoku/cells/`, `sudoku/selective/` |
| Real-dataset table (Derm7pt, CUB-25) | `python experiments/real_data_up_to_k.py --policy paper --out results/paper/real_datasets --derm-root <Derm7pt> --cub-root <CUB_200_2011>` then `python scripts/paper/make_real_data_table.py --out <paper>/tables/real_datasets.tex` | `real_datasets/` |
| Sudoku sensitivity tables (accuracy target, cost of a check) and their tests | `python scripts/paper/make_sudoku_sensitivity_tables.py --tau-out <paper>/tables/sudoku_tau_sweep.tex --cost-out <paper>/tables/sudoku_cost_models.tex` | `sudoku/cells/` |
| Own-answers table | `python scripts/paper/make_own_answers_table.py --out <paper>/tables/own_answers.tex` | `robot/balanced_rule/` (`isrc-self`, `isrc-perfect`) |
| Answer-exchange table | `python scripts/paper/make_answer_exchange_table.py --out <paper>/tables/answer_exchange.tex` | `robot/balanced_rule/intervention_records/`, CBM models |
| Sudoku confidence tables and error breakdown | `python scripts/paper/make_sudoku_confidence_tables.py --classwise-out <paper>/tables/sudoku_classwise_threshold.tex --confirmed-out <paper>/tables/sudoku_confirmed_confidence.tex` | `sudoku/confidence/`, `sudoku/cells/` |
| Paired tests quoted in the findings | `python scripts/paper/run_paired_tests.py` | `robot/balanced_rule/` |
| Interveners' accuracy on the concepts intervened on | `python scripts/paper/summarize_intervention_records.py --by-concept` | `robot/balanced_rule/intervention_records/` |

## Pipeline options behind the diagnostics

These produce the inputs of the last rows on any model the pipelines can train or load.

| Question | Option | Output |
|---|---|---|
| How much of an intervention's effect is the mechanism itself? | `python scripts/robot_pipeline.py ... --intervention-sources self` | a result cell in which every selected concept is set to the value the model already predicts (no answer changes) |
| Which concepts were intervened on, and what was answered? | `python scripts/robot_pipeline.py ... --dump-interventions DIR` | one `.npz` per cell and budget: mask, detector output, answers, true values, labels before/after |
| How confident is a sudoku model, and how confident can checks make it? | `python scripts/sudoku_pipeline.py ... --stages diagnose` | `sudoku_<model>_confidence_*.npz`: P(valid) on validation/test, concept probabilities, P(valid) with every concept set to its true value |

`scripts/paper/install_intervention_diagnostics.py` copies such outputs into `results/paper/` under the names the table scripts read.

## Notes

- Sudoku abstention: the threshold is fitted on validation and applied with one rule (`_selective_accuracy_threshold`
  keeps predictions exactly at the threshold; see `tests/test_selective_thresholds.py`). `_classwise_accuracy_thresholds`
  fits one threshold per predicted class.
- ProbCBM samples at prediction time, so its confidence files and cells are one draw each.
- Models are saved with `device="cuda"`; scripts that load them on a CPU call `set_cpu` from
  `scripts/paper/measure_intervention_response.py` or read saved detector outputs instead.
