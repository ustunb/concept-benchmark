"""Why do architectures differ in their response to interventions? Diagnostics on the saved grid models.

For each architecture, concept set (sparse rule) and seed, on the test set and through the same prediction
calls as the pipeline's intervention code:
  * accuracy without interventions (must reproduce the grid's k=0 value) and the share of majority-class
    predictions;
  * per-concept detection accuracy;
  * single-concept interventions: set one concept to its true value on every test robot and record how often
    the prediction changes and how accuracy moves; plus intervening on all concepts at once;
  * ProbCBM only: mean learned uncertainty (log sigma) per concept.
Writes one CSV row per (arch, concepts, seed, concept) to --out and prints a summary.

Run from a code checkout of `grid-seeded-lfcbm` (the models are pickled with that code):
    cd <checkout> && PYTHONPATH=. python <repo>/scripts/paper/measure_intervention_response.py --images <robot_images>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _common import PAPER_RESULTS, ROBOT_DATASETS, SPARSE_RULE, filename_fields
from concept_benchmark.ext.fileutils import load
from experiments.intervention import predict_label_proba_from_concepts

GRID_K0 = {}  # (concepts, arch, seed) -> accuracy at k=0 from the installed grid


def read_grid_k0(grid_dir: Path) -> None:
    for f in grid_dir.glob("*isrc-perfect*.csv"):
        fields = filename_fields(f)
        rows = list(csv.DictReader(f.open()))
        k0 = next(r for r in rows if r["budget"] == "0")
        GRID_K0[(fields["concepts"], fields["arch"], fields["seed"])] = float(
            k0["accuracy"]
        )


def set_cpu(obj, seen=None) -> None:
    """Point every evaluation config inside a pickled model at the CPU.

    Also lets an older scikit-learn read binary LogisticRegression models pickled with 1.8, which dropped
    `multi_class` (binary probabilities are identical either way; the k=0 check against the grid confirms it).
    """
    seen = seen if seen is not None else set()
    if id(obj) in seen or not hasattr(obj, "__dict__"):
        return
    seen.add(id(obj))
    if type(obj).__name__ == "LogisticRegression" and not hasattr(obj, "multi_class"):
        obj.multi_class = "auto"
    for name in ("_eval_config", "eval_config"):
        cfg = getattr(obj, name, None)
        if isinstance(cfg, dict):
            cfg["device"] = "cpu"
    for value in vars(obj).values():
        if hasattr(value, "__dict__") and type(value).__module__.split(".")[0] in (
            "experiments",
            "concept_benchmark",
            "sklearn",
        ):
            set_cpu(value, seen)


def label_proba(
    model,
    arch: str,
    concepts: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Label probabilities for (possibly intervened) concepts, as the pipeline computes them."""
    if arch == "cbm":
        return model.label_predictor.predict_proba((concepts >= 0.5).astype(int))
    rows = np.arange(concepts.shape[0])
    return predict_label_proba_from_concepts(
        model,
        concepts,
        row_indices=rows,
        baseline_concepts=baseline,
        intervention_mask=mask,
    )


def model_and_data_files(
    arch: str, concepts_name: str, seed: int, pipeline_root: Path | None
) -> tuple[Path, Path]:
    """Installed-tree names by default; the pipeline's own names under `pipeline_root/run_s<seed>/results`."""
    if pipeline_root is None:
        data = (
            ROBOT_DATASETS
            / f"robot__rule-sparse__concepts-{concepts_name}__seed-{seed}__dataset.data"
        )
        model = next(
            (PAPER_RESULTS / "models/robot/grid").glob(
                f"robot__rule-sparse__concepts-{concepts_name}__arch-{arch}__*seed-{seed}__model.pt"
            )
        )
        return model, data
    preset = "ideal" if concepts_name == "true" else "subconcept"
    results = pipeline_root / f"run_s{seed}" / "results"
    return (
        results / f"robot_image_stochastic_4_{preset}_{arch}_seed{seed}.model",
        results / f"robot_image_4_{preset}_seed{seed}.data",
    )


def diagnose(
    arch: str, concepts_name: str, seed: int, images: Path, pipeline_root: Path | None
) -> list[dict]:
    model_file, data_file = model_and_data_files(
        arch, concepts_name, seed, pipeline_root
    )
    data = load(data_file)
    test = data.test
    test.base_dir = Path(images)
    model = load(model_file)
    set_cpu(model)

    probs = model.concept_detector.predict_proba(test)
    truth = np.asarray(test.C, dtype=float)
    y = np.asarray(test.y).astype(int)
    majority = int(np.round(y.mean()))
    names = list(test.concepts)

    base = label_proba(model, arch, probs, probs, None)
    pred0 = base.argmax(axis=1)
    acc0 = float((pred0 == y).mean())
    grid_k0 = GRID_K0[(concepts_name, arch, str(seed))]
    logsigma = None
    if arch == "probcbm":
        cache = getattr(model, "_prediction_cache", None)
        if (
            cache is not None
            and getattr(cache, "probcbm_pred_logsigma", None) is not None
        ):
            values = (
                cache.probcbm_pred_logsigma.numpy()
            )  # (rows, concepts, ...embedding dims)
            logsigma = values.reshape(values.shape[0], values.shape[1], -1).mean(
                axis=(0, 2)
            )

    common = {
        "arch": arch,
        "concepts": concepts_name,
        "seed": seed,
        "acc_k0": round(acc0, 4),
        "grid_acc_k0": grid_k0,
        "majority_share": round(float((pred0 == majority).mean()), 4),
    }
    rows = []
    for j, name in enumerate([*names, "ALL"]):
        mask = np.zeros_like(probs, dtype=bool)
        if name == "ALL":
            mask[:] = True
        else:
            mask[:, j] = True
        intervened = np.where(mask, truth, probs)
        pred = label_proba(model, arch, intervened, probs, mask).argmax(axis=1)
        rows.append(
            {
                **common,
                "concept": name,
                "detection_acc": round(
                    float(((probs[:, j] >= 0.5) == (truth[:, j] >= 0.5)).mean()), 4
                )
                if name != "ALL"
                else "",
                "changed_share": round(float((pred != pred0).mean()), 4),
                "acc_after": round(float((pred == y).mean()), 4),
                "logsigma": round(float(logsigma[j]), 3)
                if logsigma is not None and name != "ALL"
                else "",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images", type=Path, required=True, help="Folder with the 32px robot images."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PAPER_RESULTS / "robot/diagnostics/intervention_response.csv",
    )
    ap.add_argument("--seeds", default="1014,1015,1016,1017")
    ap.add_argument("--archs", default="cbm,cem,probcbm,ecbm")
    ap.add_argument(
        "--pipeline-root",
        type=Path,
        default=None,
        help="Read models/datasets with pipeline names from <root>/run_s<seed>/results instead of results/paper.",
    )
    ap.add_argument(
        "--grid-dir",
        type=Path,
        default=SPARSE_RULE,
        help="Installed grid CSVs (k=0 check).",
    )
    args = ap.parse_args()
    read_grid_k0(args.grid_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    writer = None
    fh = args.out.open("w", newline="")
    for arch in args.archs.split(","):
        for concepts_name in ("true", "human"):
            for seed in map(int, args.seeds.split(",")):
                rows = diagnose(
                    arch, concepts_name, seed, args.images, args.pipeline_root
                )
                all_rows += rows
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                fh.flush()
                r = rows[-1]
                print(
                    f"{arch:8} {concepts_name:5} {seed}: acc k0 {r['acc_k0']:.4f} (grid {r['grid_acc_k0']:.4f}), "
                    f"majority share {r['majority_share']:.2f}, all-concepts intervention acc {r['acc_after']:.4f}",
                    flush=True,
                )
    fh.close()
    print(f"wrote {args.out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
