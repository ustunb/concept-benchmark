"""Evaluate the score-based intervention strategy.

Sweeps across dataset configurations and score thresholds, recording the
metrics exposed via `InterventionResult.strat_metrics` into a melted CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tqdm.auto import tqdm

from concept_benchmark.intervention import InterventionConfig, ScoreIntervention
from concept_benchmark.paths import results_dir

import utils as big_demo_utils
from eval_common import (
    BASE_DATASET_CONFIGS,
    INTERVENTION_SPLITS,
    ConceptInterventionRunner,
    MetricRecord,
    apply_missingness,
    build_cbm,
    build_settings,
    default_concept_noise,
    default_missingness_levels,
    default_target_options,
    iter_splits,
    load_dataset,
    write_metrics_csv,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir / "big_demo" / "score_intervention_metrics.csv",
        help="Destination CSV path for the melted metric table.",
    )
    parser.add_argument(
        "--data-names",
        nargs="+",
        choices=tuple(BASE_DATASET_CONFIGS.keys()),
        default=list(BASE_DATASET_CONFIGS.keys()),
        help="Datasets to evaluate (defaults to both).",
    )
    parser.add_argument(
        "--concept-noise",
        type=float,
        nargs="+",
        help="Override the concept noise sweep (default uses utils.CONCEPT_NOISE).",
    )
    parser.add_argument(
        "--target-accuracy-labels",
        choices=tuple(big_demo_utils.DIFFICULTY.keys()),
        nargs="+",
        help="Subset of target accuracy labels (easy/medium/hard).",
    )
    parser.add_argument(
        "--missing-mechanisms",
        nargs="+",
        choices=("none", "mcar", "mnar"),
        default=["none", "mcar", "mnar"],
        help="Concept missingness mechanisms to consider.",
    )
    parser.add_argument(
        "--concept-missing",
        type=float,
        nargs="+",
        help="Override concept missingness levels (defaults to utils.CONCEPT_MISSING).",
    )
    parser.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        default=[0.2],
        help="Score thresholds used to decide whether to intervene.",
    )
    parser.add_argument(
        "--max-concepts-per-instance",
        type=int,
        default=1,
        help="Maximum number of concepts to overwrite per instance.",
    )
    parser.add_argument(
        "--fold-id",
        default="K05N01",
        help="Dataset fold identifier used when splitting the dataset.",
    )
    parser.add_argument(
        "--fold-val",
        type=int,
        default=4,
        help="Validation fold index passed to dataset.split().",
    )
    parser.add_argument(
        "--fold-test",
        type=int,
        default=5,
        help="Test fold index passed to dataset.split().",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress reporting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skip information when assets are missing.",
    )
    return parser.parse_args(argv)


def _estimate_total_configs(
    *,
    data_names: Sequence[str],
    concept_noises: Sequence[float],
    target_labels: Sequence[str],
    mechanisms: Sequence[str],
    missing_levels: Sequence[float],
    thresholds: Sequence[float],
) -> int:
    missing_counts = 0
    for mech in mechanisms:
        if mech == "none":
            missing_counts += 1
        else:
            missing_counts += len(missing_levels)
    return (
        len(data_names)
        * len(concept_noises)
        * len(target_labels)
        * missing_counts
        * len(thresholds)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    thresholds = list(args.score_thresholds)
    for threshold in thresholds:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Score thresholds must lie within [0, 1].")

    if args.max_concepts_per_instance <= 0:
        raise ValueError("--max-concepts-per-instance must be positive.")

    target_pairs = default_target_options(args.target_accuracy_labels)
    concept_noises = default_concept_noise(args.concept_noise)
    missing_levels = default_missingness_levels(args.concept_missing)

    device = big_demo_utils.determine_device()

    total = None if args.no_progress else _estimate_total_configs(
        data_names=args.data_names,
        concept_noises=concept_noises,
        target_labels=[label for label, _ in target_pairs],
        mechanisms=args.missing_mechanisms,
        missing_levels=missing_levels,
        thresholds=thresholds,
    )
    progress = None if total is None else tqdm(total=total, desc="Configs")

    records: List[MetricRecord] = []
    strategy = ScoreIntervention()

    try:
        for data_name in args.data_names:
            for concept_noise in concept_noises:
                for target_label, target_value in target_pairs:
                    for mechanism in args.missing_mechanisms:
                        levels: Iterable[float]
                        if mechanism == "none":
                            levels = (0.0,)
                        else:
                            levels = missing_levels
                        for concept_missing in levels:
                            settings = build_settings(
                                data_name=data_name,
                                concept_noise=concept_noise,
                                target_accuracy=target_value,
                                concept_missing=concept_missing,
                                concept_missing_mech=mechanism,
                            )

                            try:
                                dataset = load_dataset(
                                    settings,
                                    fold_id=args.fold_id,
                                    fold_val=args.fold_val,
                                    fold_test=args.fold_test,
                                )
                            except FileNotFoundError as err:
                                if args.verbose:
                                    print(f"Skipping dataset {settings}: {err}")
                                if progress is not None:
                                    progress.update(len(thresholds))
                                continue

                            apply_missingness(dataset, mechanism, float(concept_missing))

                            cbm = build_cbm(settings, device, propagate=False)
                            if cbm is None:
                                if args.verbose:
                                    print(
                                        "Score intervention CBM missing for "
                                        f"{settings}"
                                    )
                                if progress is not None:
                                    progress.update(len(thresholds))
                                continue

                            runner = ConceptInterventionRunner(model=cbm)

                            for threshold in thresholds:
                                config = InterventionConfig(
                                    score_threshold=threshold,
                                    max_concepts_per_instance=args.max_concepts_per_instance,
                                )
                                for split_name, split_data in iter_splits(dataset):
                                    if split_name not in INTERVENTION_SPLITS:
                                        continue
                                    result = runner.run(
                                        strategy=strategy,
                                        config=config,
                                        dataset=split_data,
                                    )
                                    for metric_name, metric_value in result.strat_metrics.items():
                                        if metric_value is None:
                                            continue
                                        records.append(
                                            MetricRecord(
                                                strategy=strategy.name,
                                                metric=metric_name,
                                                value=float(metric_value),
                                                split=split_name,
                                                data_name=settings["data_name"],
                                                data_type=settings["data_type"],
                                                concept_noise=float(settings["concept_noise"]),
                                                concept_missing=float(settings["concept_missing"]),
                                                concept_missing_mech=settings["concept_missing_mech"],
                                                target_accuracy_label=target_label,
                                                target_accuracy_value=target_value,
                                                params={
                                                    "score_threshold": threshold,
                                                    "max_concepts_per_instance": args.max_concepts_per_instance,
                                                },
                                            )
                                        )
                                if progress is not None:
                                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    write_metrics_csv(records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
