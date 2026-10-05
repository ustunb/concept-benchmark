"""Interventions on real datasets with concept annotations (Derm7pt, CUB-25).

For each dataset we train a CBM on (a) the annotated concepts and (b) label-free concepts (LF-CBM, CLIP scores), and
correct the predicted concepts with their true values using the intervention policy of scripts/robot_pipeline.py
(``KFlipInterventionStrategy``): up to k concepts of the images whose prediction is likely to change, and every
concept of those images at k = max. Label-free interventions set a concept's score to the 5th or 95th percentile of
its training scores; the other scores stay continuous.

    python experiments/real_data.py --out results/paper/real_datasets \
        --derm-root <Derm7pt> --cub-root <CUB_200_2011>

Derm7pt: https://derm.cs.sfu.ca (folder with ``meta/`` and ``images/``).
CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011 (folder with ``images/`` and ``attributes/``).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from concept_benchmark.utils import determine_device
from experiments.intervention import InterventionBatch, InterventionConfig
from experiments.kflip import KFlipInterventionStrategy
from experiments.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig, _CLIPEncoder
from experiments.models import ConceptBasedModel, FrontEndModel

SCORE_THRESHOLD = 0.2

DERM7PT_CONCEPTS = {
    "pigment_network": "an atypical pigment network",
    "streaks": "irregular streaks",
    "pigmentation": "irregular pigmentation",
    "regression_structures": "regression structures",
    "dots_and_globules": "irregular dots and globules",
    "blue_whitish_veil": "a blue-whitish veil",
    "vascular_structures": "atypical vascular structures",
}
DERM7PT_CLIP = ("ViT-B-32-quickgelu", "openai")
DERM7PT_BUDGETS = [1, 2, 5, "max"]

CUB_N_CLASSES = 25
CUB_CLIP = ("ViT-B-32", "laion2b_s34b_b79k")
CUB_BUDGETS = [1, 10, "max"]
CUB_SEED = 0
CUB_LABEL_PREDICTOR = dict(
    hidden_layer_sizes=(128,), max_iter=800, random_state=CUB_SEED
)


class _RowPredictor(FrontEndModel):
    """Label predictor for one image, as the intervention policy sees it."""

    supports_aligned_concept_replay = True
    _kflip_fast_path = False

    def __init__(self, predict, base_row, value_one, value_zero) -> None:
        super().__init__()
        self.predict, self.base_row = predict, base_row
        self.value_one, self.value_zero = value_one, value_zero

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        raise AssertionError("KFlip should use aligned replay for this predictor.")

    def predict_proba_from_concepts(
        self,
        concepts,
        *,
        row_indices=None,
        baseline_concepts=None,
        intervention_mask=None,
    ):
        full = np.repeat(self.base_row[None, :], len(concepts), axis=0)
        if intervention_mask is not None:
            values = np.where(
                np.asarray(concepts) >= 0.5, self.value_one, self.value_zero
            )
            full = np.where(intervention_mask, values, full)
        return self.predict(full)


def select_concepts(predict, base, probs, budget, value_one, value_zero) -> np.ndarray:
    """Concepts the policy asks about for each image (rows of `base` are what the label predictor reads)."""
    n_images, n_concepts = base.shape
    mask = np.zeros((n_images, n_concepts), dtype=bool)
    # as in scripts/robot_pipeline.py: k = max corrects every concept of the images selected
    strategy = KFlipInterventionStrategy(use_exact_k=budget >= n_concepts)
    config = InterventionConfig(
        per_instance_budget=int(min(budget, n_concepts)),
        score_threshold=SCORE_THRESHOLD,
        random_state=0,
    )
    for i in range(n_images):
        model = ConceptBasedModel(
            label_predictor=_RowPredictor(
                predict, base[i].astype(float), value_one, value_zero
            )
        )
        batch = InterventionBatch(
            C_pred=probs[i][None, :].astype(np.float32),
            C_true=np.zeros((1, n_concepts), dtype=np.float32),
        )
        mask[i] = strategy.propose(model, batch, config).mask[0]
    return mask


def measure_accuracy_per_budget(
    predict, base, probs, truth, y, budgets, value_one, value_zero
) -> list[dict]:
    """Accuracy before interventions and at each budget; `truth` holds the 0/1 true concepts."""
    rows = [
        {
            "budget": 0,
            "accuracy": float((predict(base).argmax(1) == y).mean()),
            "intervened": 0,
            "concepts_per_image": 0.0,
            "largest_subset": 0,
        }
    ]
    revealed = np.where(truth >= 0.5, value_one[None, :], value_zero[None, :])
    for budget in budgets:
        k = base.shape[1] if budget == "max" else int(budget)
        mask = select_concepts(predict, base, probs, k, value_one, value_zero)
        after = predict(np.where(mask, revealed, base)).argmax(1)
        is_selected = mask.any(axis=1)
        rows.append(
            {
                "budget": budget,
                "accuracy": float((after == y).mean()),
                "intervened": int(is_selected.sum()),
                "concepts_per_image": float(mask.sum() / max(1, is_selected.sum())),
                "largest_subset": int(mask.sum(axis=1).max()),
            }
        )
        print(
            f"    k={budget}: accuracy {rows[-1]['accuracy']:.3f}, images {rows[-1]['intervened']}",
            flush=True,
        )
    return rows


def fit_concept_probes(features_train, concepts_train, features, seed) -> np.ndarray:
    """Probability of each concept from a logistic probe on CLIP features (a constant concept keeps its value)."""
    probabilities = np.zeros((features.shape[0], concepts_train.shape[1]))
    for j in range(concepts_train.shape[1]):
        column = concepts_train[:, j]
        if column.min() == column.max():
            probabilities[:, j] = column[0]
        else:
            probe = LogisticRegression(max_iter=1000, random_state=seed).fit(
                features_train, column
            )
            probabilities[:, j] = probe.predict_proba(features)[:, 1]
    return probabilities


def encode_images(paths: list[str], clip: tuple[str, str], cache: Path) -> np.ndarray:
    """CLIP image features, cached on disk for these paths."""
    if cache.exists():
        saved = np.load(cache, allow_pickle=True)
        if list(saved["paths"]) == list(paths):
            return saved["features"]
    features = _CLIPEncoder(*clip, str(determine_device())).encode_images(
        paths, batch_size=64
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=features, paths=np.array(paths, dtype=object))
    return features


def _encode_derm7pt_concept(name: str, value: str) -> int:
    """1 for the value that indicates melanoma on the 7-point checklist (atypical or irregular), not mere presence."""
    value = value.strip().lower()
    if name == "pigment_network":
        return int(value == "atypical")
    if name in ("streaks", "dots_and_globules"):
        return int(value == "irregular")
    if name == "pigmentation":
        return int("irregular" in value)
    if name == "regression_structures":
        return int(value != "absent")
    if name == "blue_whitish_veil":
        return int(value == "present")
    if name == "vascular_structures":
        return int(value in {"dotted", "linear irregular"})
    raise ValueError(name)


def load_derm7pt(root: Path):
    """Melanoma (1) vs nevus (0) with the seven checklist concepts and the dataset's own split."""
    with open(root / "meta" / "meta.csv") as fh:
        records = list(csv.DictReader(fh))
    kept_rows, y, concepts, paths = [], [], [], []
    for i, record in enumerate(records):
        diagnosis = record["diagnosis"].lower()
        if "melanoma" not in diagnosis and "nevus" not in diagnosis:
            continue
        kept_rows.append(i)
        y.append(int("melanoma" in diagnosis))
        concepts.append(
            [_encode_derm7pt_concept(name, record[name]) for name in DERM7PT_CONCEPTS]
        )
        paths.append(str(root / "images" / record["derm"]))
    position = {row: k for k, row in enumerate(kept_rows)}

    def read_split(name: str) -> np.ndarray:
        with open(root / "meta" / name) as fh:
            rows = [int(r["indexes"]) for r in csv.DictReader(fh)]
        return np.array([position[row] for row in rows if row in position])

    return (
        np.array(y),
        np.array(concepts, dtype=int),
        paths,
        read_split("train_indexes.csv"),
        read_split("valid_indexes.csv"),
        read_split("test_indexes.csv"),
    )


def load_cub(root: Path):
    """The first `CUB_N_CLASSES` species with the per-image annotations of all 312 attributes."""

    def read_columns(path: Path, n: int) -> list[list[str]]:
        with open(path) as fh:
            return [line.split()[:n] for line in fh]

    images = {int(i): p for i, p in read_columns(root / "images.txt", 2)}
    species = {
        int(i): int(c) for i, c in read_columns(root / "image_class_labels.txt", 2)
    }
    is_train_image = {
        int(i): int(s) for i, s in read_columns(root / "train_test_split.txt", 2)
    }
    raw = np.loadtxt(
        root / "attributes" / "image_attribute_labels.txt",
        usecols=(0, 1, 2),
        dtype=int,
        ndmin=2,
    )
    attributes = np.zeros((raw[:, 0].max() + 1, raw[:, 1].max()), dtype=np.int8)
    attributes[raw[:, 0], raw[:, 1] - 1] = raw[:, 2]
    names_file = root / "attributes.txt"
    if not names_file.exists():  # ships one level up in some copies of the dataset
        names_file = root.parent / "attributes.txt"
    texts = [
        "a photo of a bird with "
        + name.replace("has_", "").replace("::", " ").replace("_", " ")
        for _, name in read_columns(names_file, 2)
    ]
    ids = [i for i in images if species[i] <= CUB_N_CLASSES]
    return (
        [str(root / "images" / images[i]) for i in ids],
        np.array([species[i] - 1 for i in ids]),
        attributes[ids],
        np.array([is_train_image[i] for i in ids], dtype=bool),
        texts,
    )


def run_derm7pt(root: Path, out: Path) -> list[dict]:
    y, concepts, paths, train, valid, test = load_derm7pt(root)
    trainval = np.concatenate([train, valid])
    features = encode_images(paths, DERM7PT_CLIP, out / "cache/derm7pt_features.npz")
    n_concepts = concepts.shape[1]
    results = []
    for seed in range(5):  # annotated concepts, bootstrap over the training set
        rng = np.random.default_rng(seed)
        sample = (
            rng.choice(trainval, size=len(trainval), replace=True)
            if seed > 0
            else trainval
        )
        probs = fit_concept_probes(features[sample], concepts[sample], features, seed)[
            test
        ]
        label_predictor = LogisticRegression(max_iter=1000, random_state=seed).fit(
            concepts[sample], y[sample]
        )
        rows = measure_accuracy_per_budget(
            label_predictor.predict_proba,
            (probs >= 0.5).astype(float),
            probs,
            concepts[test],
            y[test],
            DERM7PT_BUDGETS,
            np.ones(n_concepts),
            np.zeros(n_concepts),
        )
        results += [
            {"dataset": "derm7pt", "concepts": "clinician", "seed": seed, **row}
            for row in rows
        ]
    texts = list(DERM7PT_CONCEPTS.values())
    concept_set = LFConceptSet(keys=[f"c{i}" for i in range(len(texts))], texts=texts)
    for seed in range(3):  # label-free concepts; each run has its own CLIP cache folder
        rng = np.random.default_rng(seed)
        sample = (
            rng.choice(trainval, size=len(trainval), replace=True)
            if seed > 0
            else trainval
        )
        cut = int(0.8 * len(sample))
        fit, held_out = sample[:cut], sample[cut:]
        lf = LabelFreeCBM(
            LFTrainingConfig(
                device=str(determine_device()),
                seed=seed,
                cache_dir=out / f"cache/derm7pt_seed{seed}",
            )
        )
        lf.fit(
            train_X=[paths[i] for i in fit],
            train_y=y[fit],
            valid_X=[paths[i] for i in held_out],
            valid_y=y[held_out],
            concept_set=concept_set,
        )
        is_kept = np.asarray(lf.keep_mask, dtype=bool)
        scores_train = lf.concept_proba([paths[i] for i in trainval])
        scores_test = lf.concept_proba([paths[i] for i in test])
        rows = measure_accuracy_per_budget(
            lf.predict_from_probs,
            scores_test,
            scores_test,
            concepts[test][:, is_kept],
            y[test],
            DERM7PT_BUDGETS,
            np.percentile(scores_train, 95, axis=0),
            np.percentile(scores_train, 5, axis=0),
        )
        results += [
            {"dataset": "derm7pt", "concepts": "label_free", "seed": seed, **row}
            for row in rows
        ]
    return results


def run_cub(root: Path, out: Path) -> list[dict]:
    paths, y, concepts, is_train, texts = load_cub(root)
    rng = np.random.default_rng(CUB_SEED)
    shuffled = rng.permutation(np.where(is_train)[0])
    cut = int(0.8 * len(shuffled))
    fit, held_out = shuffled[:cut], shuffled[cut:]
    trainval, test = np.r_[fit, held_out], np.where(~is_train)[0]
    features = encode_images(paths, CUB_CLIP, out / "cache/cub_features.npz")
    n_concepts = concepts.shape[1]
    name = f"cub{CUB_N_CLASSES}"
    results = []
    # annotated attributes: probes on CLIP features, label predictor trained on the true attributes
    probs = fit_concept_probes(
        features[trainval], concepts[trainval], features, CUB_SEED
    )[test]
    label_predictor = MLPClassifier(**CUB_LABEL_PREDICTOR).fit(
        concepts[trainval].astype(float), y[trainval]
    )
    rows = measure_accuracy_per_budget(
        label_predictor.predict_proba,
        (probs >= 0.5).astype(float),
        probs,
        concepts[test],
        y[test],
        CUB_BUDGETS,
        np.ones(n_concepts),
        np.zeros(n_concepts),
    )
    results += [
        {"dataset": name, "concepts": "ground_truth", "seed": CUB_SEED, **row}
        for row in rows
    ]
    # label-free: LF-CBM concept scores, label predictor trained on the scores
    concept_set = LFConceptSet(keys=[f"a{i}" for i in range(len(texts))], texts=texts)
    lf = LabelFreeCBM(
        LFTrainingConfig(
            device=str(determine_device()), seed=CUB_SEED, cache_dir=out / "cache/cub"
        )
    )
    lf.fit(
        train_X=[paths[i] for i in fit],
        train_y=y[fit],
        valid_X=[paths[i] for i in held_out],
        valid_y=y[held_out],
        concept_set=concept_set,
    )
    is_kept = np.asarray(lf.keep_mask, dtype=bool)
    scores = lf.concept_proba(paths)
    label_predictor = MLPClassifier(**CUB_LABEL_PREDICTOR).fit(
        scores[trainval], y[trainval]
    )
    rows = measure_accuracy_per_budget(
        label_predictor.predict_proba,
        scores[test],
        scores[test],
        concepts[test][:, is_kept],
        y[test],
        CUB_BUDGETS,
        np.percentile(scores[trainval], 95, axis=0),
        np.percentile(scores[trainval], 5, axis=0),
    )
    results += [
        {"dataset": name, "concepts": "label_free", "seed": CUB_SEED, **row}
        for row in rows
    ]
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--derm-root",
        type=Path,
        default=None,
        help="Derm7pt folder holding meta/ and images/",
    )
    ap.add_argument(
        "--cub-root",
        type=Path,
        default=None,
        help="CUB_200_2011 folder holding images/ and attributes/",
    )
    args = ap.parse_args()
    if args.derm_root is None and args.cub_root is None:
        ap.error("give --derm-root, --cub-root or both")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {"derm7pt": (run_derm7pt, args.derm_root), "cub": (run_cub, args.cub_root)}
    for name, (run, root) in runs.items():
        if root is None:
            continue
        rows = run(root, args.out)
        with (args.out / f"{name}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out / f'{name}.csv'}", flush=True)


if __name__ == "__main__":
    main()
