"""Evaluate trained big demo models and export accuracy metrics to CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

import utils as big_demo_utils
from concept_benchmark.ext.fileutils import load
from concept_benchmark.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionConfig,
    RandomInterventionStrategy,
)
from concept_benchmark.models import ConceptBasedModel, ConceptDetector, FrontEndModel
from demo_models import RobotClassifierCNN, SudokuValidatorCNN
from utils import (
    DIFFICULTY,
    CONCEPT_NOISE,
    determine_device,
    get_dataset_file,
    get_model_file,
)


CONCEPT_MISSING_RATE: float = getattr(
    big_demo_utils,
    "CONCEPT_MISSING",
    getattr(big_demo_utils, "CONCET_MISSING", 0.05),
)

DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "sudoku": {
        "data_name": "sudoku",
        "data_type": "tabular",
        "n": 3,
        "max_corrupt": 21,
    },
    "robot": {
        "data_name": "robot",
        "data_type": "image",
        "n": 1,
    },
}

DIFFICULTY_OPTIONS: Tuple[Tuple[str, float], ...] = tuple(DIFFICULTY.items())
CONCEPT_NOISE_OPTIONS: Tuple[Tuple[str, float], ...] = (("off", 0.0), ("on", CONCEPT_NOISE))
MISSING_MECHANISMS: Tuple[str, ...] = ("none", "mcar", "mnar")
DEFAULT_SEED: int = 42

BATCH_SIZE: int = 128


@dataclass(frozen=True)
class MetricRow:
    model_name: str
    data_name: str
    data_type: str
    concept_noise: float
    concept_missing: float
    concept_missing_mech: str
    target_accuracy: str
    target_accuracy_value: float
    split: str
    metric: str
    value: float
    intervention_strategy: str


def _iter_splits(dataset) -> Iterator[Tuple[str, object]]:
    yield "train", dataset.training
    validation = getattr(dataset, "validation", None)
    if validation is not None:
        yield "validation", validation
    test = getattr(dataset, "test", None)
    if test is not None:
        yield "test", test


def _load_dataset(settings: Mapping[str, object]):
    dataset_path = get_dataset_file(**settings)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")
    dataset = load(dataset_path)
    dataset.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)
    return dataset


def _apply_missingness(dataset, mechanism: str, missing_rate: float) -> None:
    if mechanism == "none":
        dataset.sample_concept_missingness(enable=False)
    else:
        dataset.sample_concept_missingness(
            p=missing_rate,
            mechanism=mechanism,
            rng=DEFAULT_SEED,
            enable=True,
        )


def _build_dnn_model(data_name: str) -> torch.nn.Module:
    if data_name == "sudoku":
        return SudokuValidatorCNN()
    if data_name == "robot":
        return RobotClassifierCNN()
    raise ValueError(f"Unsupported data_name for DNN model: {data_name}")


def _evaluate_dnn(
    dataset,
    settings: Mapping[str, object],
    device: torch.device,
) -> Optional[Dict[str, float]]:
    weights_path = get_model_file(model_type="dnn", **settings)
    if not weights_path.is_file():
        return None

    weights = load(weights_path)
    model = _build_dnn_model(settings["data_name"])  # type: ignore[index]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    metrics: Dict[str, float] = {}
    for split_name, split_data in _iter_splits(dataset):
        loader = split_data.loader(
            shuffle=False,
            batch_size=BATCH_SIZE,
            num_workers=0,
            pin_memory=False,
        )
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                X, _, y = batch
                X = X.to(device)
                y = y.to(device)
                outputs = model(X).squeeze()
                preds = (outputs > 0.5).long()
                correct += (preds == y).sum().item()
                total += y.size(0)
        metrics[split_name] = correct / total if total else float("nan")
    return metrics


def _load_concept_detector(path: Path, device: torch.device) -> ConceptDetector:
    detector = load(path)
    if not isinstance(detector, ConceptDetector):
        raise TypeError(f"Expected ConceptDetector at {path}, got {type(detector)!r}")
    detector.to(device)
    return detector


def _load_front_end(path: Path) -> FrontEndModel:
    model = load(path)
    if not isinstance(model, FrontEndModel):
        raise TypeError(f"Expected FrontEndModel at {path}, got {type(model)!r}")
    return model


def _build_cbm(
    settings: Mapping[str, object],
    device: torch.device,
    *,
    propagate: bool,
) -> Optional[ConceptBasedModel]:
    cd_path = get_model_file(model_type="cd", **settings)
    fe_path = get_model_file(model_type="fe", **settings)

    if not cd_path.is_file() or not fe_path.is_file():
        return None

    concept_detector = _load_concept_detector(cd_path, device)
    front_end_model = _load_front_end(fe_path)

    return ConceptBasedModel(
        concept_detector=concept_detector,
        front_end_model=front_end_model,
        propagate=propagate,
    )


def _evaluate_concept_model(
    dataset,
    settings: Mapping[str, object],
    device: torch.device,
    *,
    propagate: bool,
    tau: Optional[float] = None,
    cbm: Optional[ConceptBasedModel] = None,
) -> Optional[Dict[str, float]]:
    if cbm is None:
        cbm = _build_cbm(settings, device, propagate=propagate)
    if cbm is None:
        return None

    metrics: Dict[str, float] = {}
    for split_name, split_data in _iter_splits(dataset):
        probas = cbm.predict_proba(split_data)
        preds = probas.argmax(axis=1)
        y_true = np.asarray(split_data.y)

        if tau is not None:
            if probas.shape[1] == 2:
                label_probs = probas[:, 1]
            else:
                label_probs = np.take_along_axis(
                    probas,
                    preds[:, None],
                    axis=1,
                ).squeeze(axis=1)
            mask = (label_probs <= tau) | (label_probs >= 1.0 - tau)
            if not np.any(mask):
                metrics[split_name] = float("nan")
                continue
            considered_preds = preds[mask]
            considered_true = y_true[mask]
        else:
            considered_preds = preds
            considered_true = y_true

        if considered_true.size == 0:
            metrics[split_name] = float("nan")
        else:
            metrics[split_name] = float((considered_preds == considered_true).mean())
    return metrics


INTERVENTION_SPLITS = {"validation", "test"}


def _evaluate_interventions(
    cbm: ConceptBasedModel,
    dataset,
    *,
    strategy_name: str,
    config: InterventionConfig,
    random_strategy: bool = False,
) -> Dict[str, float]:
    runner = ConceptInterventionRunner(model=cbm)
    if random_strategy:
        strategy = RandomInterventionStrategy()
    elif strategy_name == "conceptual_safeguard":
        strategy = ConceptualSafeguardsStrategy()
    else:
        strategy = RandomInterventionStrategy()

    metrics: Dict[str, float] = {}
    for split_name, split_data in _iter_splits(dataset):
        if split_name not in INTERVENTION_SPLITS:
            continue
        result = runner.run(
            strategy=strategy,
            config=config,
            dataset=split_data,
        )
        y_true = np.asarray(split_data.y)
        metrics[split_name] = float((result.y_pred_after == y_true).mean())
    return metrics


def _append_metrics(
    rows: List[MetricRow],
    model_name: str,
    base_settings: Mapping[str, object],
    metrics: Mapping[str, float],
    *,
    target_label: str,
    target_value: float,
    intervention_strategy: str,
) -> None:
    for split_name, value in metrics.items():
        rows.append(
            MetricRow(
                model_name=model_name,
                data_name=str(base_settings["data_name"]),
                data_type=str(base_settings["data_type"]),
                concept_noise=float(base_settings.get("concept_noise", 0.0)),
                concept_missing=float(base_settings.get("concept_missing", 0.0)),
                concept_missing_mech=str(base_settings.get("concept_missing_mech", "none")),
                target_accuracy=target_label,
                target_accuracy_value=target_value,
                split=split_name,
                metric="accuracy",
                value=float(value),
                intervention_strategy=intervention_strategy,
            )
        )


def _write_csv(rows: Sequence[MetricRow], output_path: Path) -> None:
    if not rows:
        print("No metrics to write; skipping CSV export.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model_name",
            "data_name",
            "data_type",
            "concept_noise",
            "concept_missing",
            "concept_missing_mech",
            "target_accuracy",
            "target_accuracy_value",
            "split",
            "metric",
            "value",
            "intervention_strategy",
        ])
        for row in rows:
            writer.writerow([
                row.model_name,
                row.data_name,
                row.data_type,
                f"{row.concept_noise:.6f}",
                f"{row.concept_missing:.6f}",
                row.concept_missing_mech,
                row.target_accuracy,
                f"{row.target_accuracy_value:.6f}",
                row.split,
                row.metric,
                f"{row.value:.6f}",
                row.intervention_strategy,
            ])
    print(f"Saved metrics to {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained big demo models and export metrics to CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "big_demo" / "metrics.csv",
        help="Path to the CSV file that will store the metrics",
    )
    parser.add_argument(
        "--data-names",
        nargs="+",
        choices=tuple(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Subset of datasets to evaluate",
    )
    parser.add_argument(
        "--skip-dnn",
        action="store_true",
        help="Skip evaluation of standalone DNN models",
    )
    parser.add_argument(
        "--skip-cbm",
        action="store_true",
        help="Skip evaluation of concept bottleneck models (propagate=False)",
    )
    parser.add_argument(
        "--skip-safeguard",
        action="store_true",
        help="Skip evaluation of conceptual safeguard models (propagate=True)",
    )
    parser.add_argument(
        "--safeguard-tau",
        type=float,
        default=0.2,
        help="Confidence threshold tau used when evaluating conceptual safeguards",
    )
    parser.add_argument(
        "--apply-interventions",
        action="store_true",
        help="Evaluate concept interventions in addition to vanilla predictions",
    )
    parser.add_argument(
        "--intervention-budget",
        type=float,
        default=0.2,
        help="Fractional instance budget for random concept interventions",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information during evaluation",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar output",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not (0.0 <= args.safeguard_tau <= 0.5):
        raise ValueError("--safeguard-tau must be between 0 and 0.5")
    if args.intervention_budget is not None and args.intervention_budget < 0:
        raise ValueError("--intervention-budget must be non-negative")
    device = determine_device()

    rows: List[MetricRow] = []

    total_configs = (
        len(args.data_names)
        * len(CONCEPT_NOISE_OPTIONS)
        * len(DIFFICULTY_OPTIONS)
    )
    progress = None if args.no_progress else tqdm(total=total_configs, desc="Configs")

    try:
        for data_name in args.data_names:
            base_cfg = DATASET_CONFIGS[data_name]
            for noise_label, concept_noise in CONCEPT_NOISE_OPTIONS:
                for target_label, target_value in DIFFICULTY_OPTIONS:
                    dataset_settings: Dict[str, object] = dict(base_cfg)
                    dataset_settings.update(
                        {
                            "concept_noise": concept_noise,
                            "target_accuracy": target_value,
                        }
                    )

                    try:
                        dataset_for_dnn = _load_dataset(dataset_settings)
                    except FileNotFoundError as err:
                        if args.verbose:
                            print(
                                f"Skipping dataset ({data_name}, noise={noise_label}, target={target_label}): {err}"
                            )
                        if progress is not None:
                            progress.update(1)
                        continue

                    if not args.skip_dnn:
                        dnn_metrics = _evaluate_dnn(dataset_for_dnn, dataset_settings, device)
                        if dnn_metrics is not None:
                            _append_metrics(
                                rows,
                                model_name="dnn",
                                base_settings=dict(
                                    dataset_settings,
                                    concept_missing=0.0,
                                    concept_missing_mech="none",
                                ),
                                metrics=dnn_metrics,
                                target_label=target_label,
                                target_value=target_value,
                                intervention_strategy="none",
                            )
                        elif args.verbose:
                            print(
                                "DNN weights missing for "
                                f"dataset={data_name}, noise={noise_label}, target={target_label}"
                            )

                    if not (args.skip_cbm and args.skip_safeguard):
                        for mechanism in MISSING_MECHANISMS:
                            concept_missing_value = 0.0 if mechanism == "none" else CONCEPT_MISSING_RATE
                            combo_settings: Dict[str, object] = dict(dataset_settings)
                            combo_settings.update(
                                {
                                    "concept_missing": concept_missing_value,
                                    "concept_missing_mech": mechanism,
                                }
                            )

                            try:
                                dataset = _load_dataset(combo_settings)
                            except FileNotFoundError:
                                if args.verbose:
                                    print(
                                        "Skipping concept combo (dataset=%s, noise=%s, target=%s, mech=%s): dataset missing"
                                        % (data_name, noise_label, target_label, mechanism)
                                    )
                                continue

                            _apply_missingness(dataset, mechanism, concept_missing_value)

                            if not args.skip_cbm:
                                cbm = _build_cbm(combo_settings, device, propagate=False)
                                if cbm is not None:
                                    cbm_metrics = _evaluate_concept_model(
                                        dataset,
                                        combo_settings,
                                        device,
                                        propagate=False,
                                        cbm=cbm,
                                    )
                                    if cbm_metrics is not None:
                                        _append_metrics(
                                            rows,
                                            model_name="concept_bottleneck",
                                            base_settings=combo_settings,
                                            metrics=cbm_metrics,
                                            target_label=target_label,
                                            target_value=target_value,
                                            intervention_strategy="none",
                                        )
                                    if args.apply_interventions:
                                        intervention_metrics = _evaluate_interventions(
                                            cbm,
                                            dataset,
                                            strategy_name="random",
                                            config=InterventionConfig(
                                                instance_budget=args.intervention_budget,
                                            ),
                                            random_strategy=True,
                                        )
                                        if intervention_metrics:
                                            _append_metrics(
                                                rows,
                                                model_name="concept_bottleneck",
                                                base_settings=combo_settings,
                                                metrics=intervention_metrics,
                                                target_label=target_label,
                                                target_value=target_value,
                                                intervention_strategy="random",
                                            )
                                elif args.verbose:
                                    print(
                                        "Concept detector/front-end missing for "
                                        f"dataset={data_name}, noise={noise_label}, target={target_label}, mech={mechanism}"
                                    )

                            if not args.skip_safeguard:
                                safeguard_cbm = _build_cbm(combo_settings, device, propagate=True)
                                if safeguard_cbm is not None:
                                    safeguard_metrics = _evaluate_concept_model(
                                        dataset,
                                        combo_settings,
                                        device,
                                        propagate=True,
                                        tau=args.safeguard_tau,
                                        cbm=safeguard_cbm,
                                    )
                                    if safeguard_metrics is not None:
                                        _append_metrics(
                                            rows,
                                            model_name="conceptual_safeguard",
                                            base_settings=combo_settings,
                                            metrics=safeguard_metrics,
                                            target_label=target_label,
                                            target_value=target_value,
                                            intervention_strategy="none",
                                        )
                                    if args.apply_interventions:
                                        intervention_metrics = _evaluate_interventions(
                                            safeguard_cbm,
                                            dataset,
                                            strategy_name="conceptual_safeguard",
                                            config=InterventionConfig(tau=args.safeguard_tau),
                                        )
                                        if intervention_metrics:
                                            _append_metrics(
                                                rows,
                                                model_name="conceptual_safeguard",
                                                base_settings=combo_settings,
                                                metrics=intervention_metrics,
                                                target_label=target_label,
                                                target_value=target_value,
                                                intervention_strategy="conceptual_safeguard",
                                            )
                                elif args.verbose:
                                    print(
                                        "Conceptual safeguard assets missing for "
                                        f"dataset={data_name}, noise={noise_label}, target={target_label}, mech={mechanism}"
                                    )

                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    _write_csv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
