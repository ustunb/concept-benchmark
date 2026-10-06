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

### Removed

- `concept_benchmark.metrics.compute_selective_metric` and `experiments.metrics` (use `selective_at`, which applies the same abstention band).
- `experiments.utils` keeps only `run_alignment`: `determine_device`, `get_loader_config`, `compute_accuracy` and `patch_macos_dataloader` live in `concept_benchmark.utils`, and `experiments.utils.train_dnn` is replaced by `experiments.evaluate.train_dnn`, which returns the trained model instead of a test accuracy.

- `plot_regime_comparison` (use `plot_intervention_heatmap`).
