# Changelog

## 0.4.0

### Changed

- **The default robot label rule is now `balanced`.** No single concept decides the label and both classes are equally likely. The previous rule is `label_rule="sparse"` (CLI: `--label-rule sparse`). Datasets generated with default settings differ from 0.3.x; a config saved by 0.3.x needs `label_rule: sparse`.
- `plot_alignment_comparison` takes a table with one row per run (`concepts`, `model`, `accuracy_before`, optional `accuracy_after`) instead of a dict of gains.
- `plot_intervention_curve` averages several runs, draws a standard-error band and takes `group=` to draw one line per model or concept set.
- ECBM follows the authors' code; the settings `ecbm_emb_size`, `ecbm_inference_steps` and `ecbm_inference_lr` are gone and must be deleted from a config saved by 0.3.x.
- `set_deterministic_seed` also turns on PyTorch's deterministic algorithms (with warnings instead of errors).

### Added

- `DatasetGenerator.generate_splits()`: the train/validation/test split of the paper.
- `ROBOT_LABEL_RULES`, `label_rule`: the two label rules and the training foot subtypes that go with each.
- Plots: `plot_intervention_heatmap`, `plot_automation`, `plot_concept_report`, `plot_answer_reliance`, `plot_confidence`.
- Metrics: `abstention_threshold`, `decision_threshold`, `selective_at` (the paper's selective-classification protocol), `classwise_thresholds`, `intervention_metrics`.
- `experiments.evaluate`: `train_cbm`, `train_dnn`, `intervention_table`, `automation_table`, `coverage_at_target`, `predict_labels`, `predict_proba_positive`; the pipelines and the examples are built from these blocks.
- `plot_intervention_curve` and `plot_automation` draw the paper's two panels.
- Robot pipeline: `--label-rule`, `--intervention-encoding`, `--intervention-sources self`, `--dump-interventions`, the `noisy_human_concepts` concept source, and all four architectures on every concept source.
- Sudoku pipeline: ECBM, the optional `diagnose` stage, and abstention thresholds fitted with the rule that applies them.
- `experiments/real_data.py`: interventions on Derm7pt and CUB.
- `scripts/paper/`: the scripts behind every table, figure and test of the paper. They read one results folder (`--results-root`): the paper's results, a separate download on the releases page, or your own runs copied with `collect_pipeline_runs.py`.
- `LICENSE`, `concept_benchmark.__version__`.

### Fixed

- Sudoku automation follows conceptual safeguards as published: the label probability is Platt-scaled on validation (`PlattScaling`, `calibrate` in the sudoku config, `calibrate=` in `automation_table` and `coverage_at_target`), the model predicts a valid board above 0.5 and abstains on `[t, 1 - t]` with `t` fitted once on validation. The tuned decision cut that the pipeline refit at every budget is gone. Uncalibrated, the propagated probability of a valid board is about 0.3 (27 concepts at 0.96 each), and the safeguard deferred every valid board at 18 px.
- Every sudoku board is written in its own handwriting (the board index seeds the glyphs; before, a digit in a cell was drawn the same way on every board, 729 glyphs in all), and the digit recognizer is trained on the training boards only, so the validation boards (where the thresholds are fitted) and the test boards carry handwriting it has never seen. Detection errors are now real: about 15% of 50 px boards carry a misread digit.
- Sudoku boards at any `cell_px` are the 50 px board resampled: printed starters, candidate bubbles and lines keep their proportions (at 18 px the starters used to overflow their cells and the bubbles covered the digits). The drawing at 50 px is unchanged; the handwriting (next entry) changes every board.
- The sudoku digit recognizer trains until its validation cell accuracy stops improving (`ocr_epochs` = 20, `ocr_patience` = 3 in the sudoku config) and keeps the best epoch. It used to train for two epochs with no check: on one seed in ten it stopped at 97.6% cell accuracy (the others 99.6–99.9%), which put a misread digit on 84% of that seed's boards and halved the coverage of every model on it.
- Sudoku: corruptions that cancelled out left about 1% of the "invalid" boards valid; the generator now redraws until a concept is violated. Datasets regenerated with 0.4.0 differ from 0.3.x in those boards.
- Abstention follows one rule, `min(p, 1 - p) >= t` (`abstention_mask`), in the threshold fit, the measures and the intervention strategies; the two inequalities used before could disagree by one floating-point ulp.
- `decision_threshold` returns a threshold inside a run of tied best thresholds and the accuracy at that threshold.
- A concept probability of exactly 0.5 counts as present everywhere, and a label logit of exactly 0 predicts the negative class, as `LogisticRegression.predict` does.

### Removed

- `concept_benchmark.metrics.compute_selective_metric` and `experiments.metrics` (use `selective_at`, which applies the same abstention band).
- `experiments.utils` keeps only `run_alignment`: `determine_device`, `get_loader_config`, `compute_accuracy` and `patch_macos_dataloader` live in `concept_benchmark.utils`, and `experiments.utils.train_dnn` is replaced by `experiments.evaluate.train_dnn`, which returns the trained model instead of a test accuracy.

- `plot_regime_comparison` (use `plot_intervention_heatmap`).
