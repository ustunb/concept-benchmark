"""Alignment experiment under the balanced rule (standard CBM, true and human concepts, seeds 1014-1023).

Runs the pipeline's own alignment and intervention code on the installed models:
  1. `run_alignment` retrains the CBM's label predictor on the true training concepts with the pipeline's constraint
     (`RobotBenchmarkConfig.get_alignment_constraints()`: the weight on HasKnees must be non-negative);
  2. `_test_interventions` runs perfect interventions (k = 1, 3, max; threshold 0.2) on the unconstrained and on the
     constrained label predictor.
The concept detector's outputs on the test robots are the ones of the paper's runs, extracted with `--install` from the
installed intervention records (the detector gives slightly different probabilities on other hardware). As a positive control,
the unconstrained model must reproduce the installed accuracies of the paper's runs at every budget, exactly.

Writes one row per concept set and seed to results/paper/robot/alignment/ and prints the paired tests of the paper.

    PYTHONPATH=. python scripts/paper/evaluate_alignment_balanced.py --install   # once: extract detector outputs from the records
    PYTHONPATH=. python scripts/paper/evaluate_alignment_balanced.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import statistics as st
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

from _common import (
    BALANCED_RULE,
    BALANCED_TAG,
    INTERVENTION_RECORDS,
    PAPER_RESULTS,
    REPO,
    ROBOT_DATASETS,
    read_budget_rows,
)

warnings.filterwarnings("ignore")
from concept_benchmark.config import RobotBenchmarkConfig  # noqa: E402
from concept_benchmark.ext.fileutils import load  # noqa: E402
from experiments.alignment import align_frontend_weights  # noqa: E402
from experiments.utils import run_alignment  # noqa: E402
from robot_pipeline import InterventionSettings, _test_interventions  # noqa: E402

SEEDS = tuple(range(1014, 1024))
CONCEPT_SETS = {"true": "ideal", "human": "subconcept"}  # concept set -> model tag
DETECTOR = BALANCED_RULE / "detector_outputs"
RESULTS = (
    PAPER_RESULTS
    / f"robot/alignment/{BALANCED_TAG}__arch-cbm__isrc-perfect__alignment.csv"
)
BUDGETS = ("1", "3", "max")


def detector_file(concepts: str, seed: int) -> Path:
    return (
        DETECTOR
        / f"{BALANCED_TAG}__concepts-{concepts}__arch-cbm__seed-{seed}__test-concept-probabilities.npz"
    )


def update_index(entries: list[dict]) -> None:
    index = PAPER_RESULTS / "INDEX.csv"
    with index.open() as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, {r["path"]: r for r in reader}
    for e in entries:
        rows[e["path"]] = e
    with index.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["path"]))


def entry(path: Path, role: str, source: str, note: str) -> dict:
    return {
        "path": str(path.relative_to(PAPER_RESULTS)),
        "role": role,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": source,
        "note": note,
    }


def install_detector_outputs() -> None:
    """Extract the CBM detector's test probabilities of the paper's runs from the installed intervention records."""
    DETECTOR.mkdir(parents=True, exist_ok=True)
    entries = []
    for concepts in CONCEPT_SETS:
        for seed in SEEDS:
            src = (
                INTERVENTION_RECORDS
                / f"{BALANCED_TAG}__concepts-{concepts}__arch-cbm__isrc-perfect__budget-1__seed-{seed}__records.npz"
            )
            d = np.load(src)
            test = load(
                ROBOT_DATASETS
                / f"{BALANCED_TAG}__concepts-{concepts}__seed-{seed}__dataset.data"
            ).test
            if not (
                np.array_equal(d["C_true"], np.asarray(test.C).astype(np.int8))
                and np.array_equal(d["y"], np.asarray(test.y).astype(np.int8))
            ):
                raise SystemExit(
                    f"{src}: concepts or labels differ from the installed dataset"
                )
            dest = detector_file(concepts, seed)
            np.savez_compressed(
                dest, C_pred=d["C_pred"], concept_names=d["concept_names"]
            )
            entries.append(
                entry(
                    dest,
                    "diagnostic",
                    str(src.relative_to(REPO / "results")),
                    "CBM concept probabilities on the test robots, from the paper's run on Bridges",
                )
            )
    update_index(entries)
    print(
        f"installed {len(entries)} detector-output files in {DETECTOR.relative_to(REPO)}"
    )


class _SavedDetector:
    """Stands in for the concept detector and returns its saved outputs on the test robots."""

    def __init__(self, proba: np.ndarray):
        self.proba = proba

    def predict_proba(self, dataset):
        return self.proba

    def predict(self, dataset):
        return (self.proba >= 0.5).astype(int)


def installed_accuracies(concepts: str, seed: int) -> list[float]:
    path = (
        BALANCED_RULE
        / f"{BALANCED_TAG}__concepts-{concepts}__arch-cbm__isrc-perfect__strategy-upto__seed-{seed}__results.csv"
    )
    rows = read_budget_rows(path)
    return [float(r["accuracy"]) for r in rows]


def interventions(frontend, proba, test, seed: int, accuracy_k0: float) -> list[float]:
    settings = InterventionSettings(
        seed=seed,
        budgets=[1, 3, test.C.shape[1]],
        intervention_accuracy=1.0,
        intervention_threshold=0.2,
    )
    _, _, results = _test_interventions(
        prob_test=proba, settings=settings, acc_det=accuracy_k0, fe=frontend, test=test
    )
    return [float(v["accuracy"]) for v in results.values()]


def evaluate() -> list[dict]:
    rows = []
    for concepts, model_tag in CONCEPT_SETS.items():
        for seed in SEEDS:
            data = load(
                ROBOT_DATASETS
                / f"{BALANCED_TAG}__concepts-{concepts}__seed-{seed}__dataset.data"
            )
            model = load(
                next(
                    (PAPER_RESULTS / "models/robot/balanced").glob(
                        f"{BALANCED_TAG}__{model_tag}_cbm_seed{seed}__*"
                    )
                )
            )
            model.label_predictor.model.multi_class = (
                "auto"  # attribute expected by newer scikit-learn
            )
            proba = np.load(detector_file(concepts, seed))["C_pred"]
            model.concept_detector = _SavedDetector(proba)
            constraints = RobotBenchmarkConfig(seed=seed).get_alignment_constraints()
            aligned = run_alignment(
                concept_based_model=model,
                train_dataset=data.train,
                test_dataset=data.test,
                monotonicity_constraints=constraints,
                save_path=None,
            )
            names = list(data.test.concepts)
            constrained = align_frontend_weights(
                copy.deepcopy(model.label_predictor), names, aligned["aligned_weights"]
            )
            constrained.model.multi_class = "auto"
            train_c, train_y = data.train.C.astype(np.float32), data.train.y.astype(int)
            before = [
                aligned["original_accuracy"],
                *interventions(
                    model.label_predictor,
                    proba,
                    data.test,
                    seed,
                    aligned["original_accuracy"],
                ),
            ]
            after = [
                aligned["aligned_accuracy"],
                *interventions(
                    constrained, proba, data.test, seed, aligned["aligned_accuracy"]
                ),
            ]
            if (
                max(
                    abs(a - b)
                    for a, b in zip(before, installed_accuracies(concepts, seed))
                )
                > 0
            ):
                raise SystemExit(
                    f"{concepts} seed {seed}: the unconstrained model does not reproduce the installed run: {before}"
                )
            row = {
                "concepts": concepts,
                "seed": seed,
                "w_knees_before": float(
                    model.label_predictor.model.coef_.ravel()[names.index("has_knees")]
                ),
                "w_knees_after": float(aligned["aligned_weights"]["has_knees"]),
                "train_before": 100
                * float(
                    (
                        np.argmax(model.label_predictor.predict_proba(train_c), axis=1)
                        == train_y
                    ).mean()
                ),
                "train_after": 100
                * float(
                    (
                        np.argmax(constrained.predict_proba(train_c), axis=1) == train_y
                    ).mean()
                ),
            }
            for label, b, a in zip(("0", *BUDGETS), before, after):
                row[f"k{label}_before"], row[f"k{label}_after"] = 100 * b, 100 * a
            rows.append(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )
            print(
                f"{concepts} seed {seed}: k=0 {row['k0_before']:.2f} -> {row['k0_after']:.2f} | k=max {row['kmax_before']:.2f} -> {row['kmax_after']:.2f}",
                flush=True,
            )
    return rows


def report(rows: list[dict]) -> None:
    def paired(name: str, a: list[float], b: list[float]) -> None:
        diff = [x - y for x, y in zip(a, b)]
        t, p = stats.ttest_rel(a, b)
        print(
            f"  {name:46s} {st.mean(a):6.2f} vs {st.mean(b):6.2f}  diff {st.mean(diff):+5.2f}  t={t:+6.2f}  p={p:.4f}  ({sum(d > 0 for d in diff)}/{len(diff)} seeds higher)"
        )

    for concepts in CONCEPT_SETS:
        r = [x for x in rows if x["concepts"] == concepts]
        col = lambda key: [float(x[key]) for x in r]  # noqa: E731
        benefit = lambda which: [
            st.mean([a, b, c]) - z
            for a, b, c, z in zip(
                col(f"k1_{which}"),
                col(f"k3_{which}"),
                col(f"kmax_{which}"),
                col(f"k0_{which}"),
            )
        ]  # noqa: E731
        print(f"\n{concepts}_concepts, constrained vs unconstrained (n={len(r)})")
        print(
            f"  weight on HasKnees: {st.mean(col('w_knees_before')):+.2f} -> {st.mean(col('w_knees_after')):+.2f}"
        )
        paired("training accuracy", col("train_after"), col("train_before"))
        paired("accuracy before interventions (k=0)", col("k0_after"), col("k0_before"))
        paired(
            "accuracy after interventions (k=max)",
            col("kmax_after"),
            col("kmax_before"),
        )
        paired(
            "benefit of interventions (mean over k, minus k=0)",
            benefit("after"),
            benefit("before"),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--install",
        action="store_true",
        help="copy the detector outputs from the intervention dumps and exit",
    )
    args = ap.parse_args()
    if args.install:
        install_detector_outputs()
        return
    rows = evaluate()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    update_index(
        [
            entry(
                RESULTS,
                "cited",
                "scripts/paper/evaluate_alignment_balanced.py",
                "alignment under the balanced rule: CBM with and without the HasKnees constraint, perfect interventions",
            )
        ]
    )
    print(
        f"\nwrote {RESULTS.relative_to(REPO)}; the unconstrained model reproduces the installed runs exactly on all {len(rows)} cases"
    )
    report(rows)


if __name__ == "__main__":
    main()
