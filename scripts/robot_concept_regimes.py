# scripts/robot_regimes.py
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision import transforms

from concept_benchmark.synthetic.robot import create_synthetic_dataset
from concept_benchmark.models import ConceptDetector, FrontEndModel, RobotConceptClassifier
from scripts.dataset_skewing import create_skewed_splits, filter_training_by_string
from scripts.robot_interventions import test_interventions
from scripts.robot_invariance_test import test_concept_detector_invariance
from scripts.robot_utils import (
    _apply_missing,
    _apply_label_noise,
    _rate_tag,
    _get_concept_accuracies,
    _get_confusion_matrix,
    _get_accuracies_per_subconcept,
)


# -------------------- seeds --------------------
def set_seed(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# -------------------- small helpers --------------------
def _device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _len(dset) -> int:
    try:
        return int(dset.n)
    except Exception:
        return int(len(dset))


def _build_groups(concept_names: List[str], spec: Dict) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    used = set()

    for base in list(spec.keys()):
        idxs = [i for i, c in enumerate(concept_names) if c == base or c.startswith(f"{base}_")]
        if idxs:
            groups[base] = idxs
            used.update(idxs)

    for i, c in enumerate(concept_names):
        if i not in used:
            groups[c] = [i]

    return groups


def _flip_onehot_row(row: np.ndarray, idxs: List[int], rng: np.random.Generator) -> None:
    vals = row[idxs]
    if len(idxs) == 1:
        row[idxs[0]] = 1.0 - row[idxs[0]]
        return
    s = int(vals.sum())
    if s != 1:
        return
    active = int(np.argmax(vals))
    choices = [j for j in range(len(idxs)) if j != active]
    if not choices:
        return
    new_j = int(rng.choice(choices))
    row[idxs] = 0.0
    row[idxs[new_j]] = 1.0


def _apply_concept_noise(
    C_in: np.ndarray, concept_names: List[str], spec: Dict, rate: float, rng: np.random.Generator
) -> np.ndarray:
    C = C_in.astype(np.float32).copy()
    groups = _build_groups(concept_names, spec)
    for r in range(C.shape[0]):
        for _, idxs in groups.items():
            if rng.random() < float(rate):
                _flip_onehot_row(C[r], idxs, rng)
    return C


def _clone_with_C(dset, C_new: np.ndarray):
    return dset.__class__(
        parent=dset.parent,
        X=dset.X,
        C=C_new.astype(np.float32),
        y=dset.y,
        meta=dset.meta,
        transform=dset.transform,
        concept_transform=dset.concept_transform,
        target_transform=dset.target_transform,
        base_dir=getattr(dset, "base_dir", None),
    )


def _extract_fe_weights(fe: FrontEndModel, concept_names: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        model = getattr(fe, "model", fe)
        coef = np.asarray(model.coef_)[0]
        bias = float(np.asarray(model.intercept_)[0])
        for i, name in enumerate(concept_names):
            out[name] = float(round(coef[i], 6))
        out["bias"] = float(round(bias, 6))
        return out
    except Exception:
        # fallback: no weights available
        for name in concept_names:
            out[name] = None
        out["bias"] = None
        return out


def _build_slug(S: Dict, miss_tag: str, filter_tag: str, label_noise_tag: str, skew_tag: str, int_acc_tag: str) -> str:
    impute_tag = f"impute{int(S.get('impute_missing', 0))}"
    slug = f"robots_image_{S['model_type']}_{miss_tag}{filter_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{impute_tag}"
    return slug


# -------------------- dataset build to mirror main --------------------
def _define_train_valid_test(
    settings: Dict,
    concept_dataset,
    missingness: str,
    params: Dict,
    rate: float,
    rng: np.random.Generator,
    tf,
):
    if settings.get("skew_concept"):
        train, valid, test = create_skewed_splits(
            concept_dataset,
            skew_specs=settings["skew_concept"],
            test_size=settings.get("test_size", 10000),
            train_skew_size=settings.get("train_size", None),
            rng=rng,
            drop_concepts=settings.get("drop_concepts", []),
        )
    elif settings.get("dataset_characterization", ""):
        train, valid, test = filter_training_by_string(
            concept_dataset, string=settings["dataset_characterization"], rng=rng
        )
    else:
        concept_dataset.split("K05N01", fold_num_validation=4, fold_num_test=5)
        train = concept_dataset.training
        valid = concept_dataset.validation
        test = concept_dataset.test

    standard_size = 108 * int(settings["samples_per_instance"])
    test_params = dict(params)
    test_params["samples_per_instance"] = int(params["test_set_size"] / standard_size) + 1
    test_data = create_synthetic_dataset(**test_params)
    test_data.drop_concepts(settings.get("drop_concepts", []))
    test_data.transform = tf
    test_data.generate_cvindices(seed=int(settings["seed"]))
    rng_test = np.random.default_rng(int(settings["seed"]) + 1234)
    test_indices = rng_test.choice(len(test_data), size=int(params["test_set_size"]), replace=False)
    test = test_data._full.filter(np.isin(np.arange(len(test_data)), test_indices))

    if settings.get("label_noise_rate", 0.0) and float(settings["label_noise_rate"]) > 0.0:
        sd = int(settings["seed"])
        train = _apply_label_noise(train, settings["label_noise_rate"], seed=sd)
        valid = _apply_label_noise(valid, settings["label_noise_rate"], seed=sd)
        test = _apply_label_noise(test, settings["label_noise_rate"], seed=sd)

    if missingness != "complete" and rate > 0:
        Ctr = _apply_missing(train.C, missingness, rate, rng, y=train.y.astype(int))
        train = train.__class__(
            parent=train.parent,
            X=train.X,
            C=Ctr,
            y=train.y,
            meta=train.meta,
            transform=train.transform,
            concept_transform=train.concept_transform,
            target_transform=train.target_transform,
            base_dir=getattr(train, "base_dir", None),
        )

    return test, train, valid


# -------------------- model training wrappers --------------------
def _train_concept_detector(
    settings: Dict,
    config: Dict,
    device: str,
    int_acc_tag: str,
    label_noise_tag: str,
    miss_tag: str,
    model_type_tag: str,
    run_dir: Path,
    seed_tag: str,
    skew_tag: str,
    train,
    valid,
    test,
) -> Tuple[ConceptDetector, Path]:
    n_c = train.n_concepts
    img_size = str(settings.get("image_size", "small"))
    input_size = 600 if img_size == "large" else 32 if img_size == "medium" else 8

    cd = ConceptDetector(model=RobotConceptClassifier(num_concepts=n_c, input_size=input_size))
    det_name = f"detector_dnn_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"

    if settings.get("load_detector"):
        mini_train = train.filter(np.array([True] + [False] * (len(train.C) - 1)))
        mini_valid = valid.filter(np.array([True] + [False] * (len(valid.C) - 1)))
        cd.fit(mini_train, mini_valid, freeze=True, embed_params={"device": device}, fit_params={"epochs": 1, "device": "cpu"})
        state = torch.load(settings["load_detector"], weights_only=False, map_location="cpu")
        cd.load_state_dict(state)
        det_path = Path(settings["load_detector"])
    else:
        cd.fit(train, valid, embed_params={"shuffle": False, **config}, fit_params={"epochs": 50, "lr": 1e-3, "patience": 10, **config})
        det_path = run_dir / det_name
        torch.save(cd.state_dict(), det_path)

    subtype_concepts = [c for c in test.concepts if c.startswith("foot_shape_")]
    for concept in subtype_concepts:
        _ = test_concept_detector_invariance(cd, concept, train.concepts, test, device, num_tests=10)

    return cd, det_path


def _train_frontend(
    H_te: np.ndarray,
    h_train: np.ndarray,
    prob_train: np.ndarray,
    sttngs: Dict,
    int_acc_tag: str,
    label_noise_tag: str,
    miss_tag: str,
    model_type_tag: str,
    run_dir: Path,
    seed_tag: str,
    skew_tag: str,
    test,
    train,
):
    fe = FrontEndModel()
    fe_name = f"frontend_logreg_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pkl"

    if sttngs.get("load_frontend"):
        import pickle

        with open(sttngs["load_frontend"], "rb") as f:
            fe = pickle.load(f)
        fe_path = Path(sttngs["load_frontend"])
    else:
        Ctr = train.C.astype(np.float32)
        if int(sttngs.get("impute_missing", 0)) and np.any(Ctr < 0):
            Cin = Ctr.copy()
            mask = Cin < 0
            Cin[mask] = prob_train[mask]
            fe.fit(Cin, train.y.astype(int))
        else:
            if sttngs.get("CBM_type", "separate") == "sequential":
                keep = np.all(Ctr >= 0, axis=1)
                fe.fit(h_train[keep], train.y[keep].astype(int))
            else:
                fe.fit(Ctr, train.y.astype(int))

        import pickle

        fe_path = run_dir / fe_name
        with open(fe_path, "wb") as f:
            pickle.dump(fe, f)

    y_pred_det = fe.predict_proba(H_te)
    y_pred_gt = fe.predict_proba(test.C.astype(np.float32))
    acc_det = float((y_pred_det.argmax(1) == test.y.astype(int)).mean())
    acc_gt = float((y_pred_gt.argmax(1) == test.y.astype(int)).mean())
    concept_acc_mean = float((H_te == test.C).mean())

    return acc_det, acc_gt, concept_acc_mean, fe, fe_path, y_pred_det


# -------------------- public entrypoint --------------------
def run_regimes(settings: Dict) -> Dict:
    set_seed(settings["seed"])

    S = dict(settings)
    rng = np.random.default_rng(int(S["seed"]))
    base_out = Path(S["out_dir"])
    base_out.mkdir(parents=True, exist_ok=True)

    miss = str(S["missingness"]).lower()
    rate = float(S["missing_rate"])

    int_acc_tag = f"int-acc{int(round(float(S['intervention_accuracy']) * 100))}"
    miss_tag = "complete" if miss == "complete" or rate <= 0 else f"{miss}{_rate_tag(rate)}"
    skew_tag = "_skew" if S.get("skew_concept", []) else ""
    filter_tag = "_filter" if S.get("dataset_characterization", "") else ""
    label_noise_tag = "_label-noise_" if float(S.get("label_noise_rate", 0.0)) else "_"
    model_type_tag = f"_{S['model_type']}_{miss_tag}{filter_tag}_{S.get('image_size','')}"

    params = {
        "samples_per_instance": S["samples_per_instance"],
        "draw": S["draw"],
        "output_directory": S["image_dir"],
        "concepts": S["concepts"],
        "subconcepts": S.get("subconcepts", ["foot_shape_subtype", "hand_shape_subtype"]),
        "spurious_features": S.get("spurious_features", []),
        "drop_concepts": S.get("drop_concepts", []),
        "color_mode": S["color_mode"],
        "model": S["model"],
        "model_type": S["model_type"],
        "size": S["image_size"],
        "scalar": float(S.get("logit_scalar", 1.0)),
        "intercept": float(S.get("logit_intercept", 0.0)),
        "weights": S.get("logit_weights", {}),
        "test_set_size": 10000,
        "train_concept_detector": True,
        "verbose": True,
        "rng_seed": S["seed"],
    }

    data = create_synthetic_dataset(**params)
    tf = transforms.Compose([transforms.ToTensor()])
    data.transform = tf
    data.generate_cvindices(seed=int(S["seed"]))
    test, train, valid = _define_train_valid_test(S, data, miss, params, rate, rng, tf)

    device = _device()
    config = {
        "device": device,
        "batch_size": 32,
        "num_workers": 0 if device == "mps" else 12,
        "pin_memory": False if device == "mps" else True,
    }

    regimes = [str(r).lower() for r in S.get("regimes", [])]
    subjective_grid = S.get("subjective_grid", [0.2])
    expert_acc = float(S.get("human_annotation_accuracy", 0.8))

    # Rollup and per-regime writes
    results: Dict = {}
    results_path = base_out / "robots" / f"{S['run_name']}__regime_results.json"
    (base_out / "robots").mkdir(parents=True, exist_ok=True)

    for regime in regimes:
        run_dir = base_out / "robots" / f"{S['run_name']}__regime-{regime}"
        run_dir.mkdir(parents=True, exist_ok=True)

        slug = _build_slug(S, miss_tag, filter_tag, label_noise_tag, skew_tag, int_acc_tag)
        seed_tag = f"seed{int(S['seed'])}"

        if regime == "perfect":
            cd, det_path = _train_concept_detector(
                S, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir, seed_tag, skew_tag, train, valid, test
            )
            P_tr = cd.predict(train, embed_params={"device": device})
            P_te = cd.predict(test, embed_params={"device": device})
            H_tr = (P_tr > 0.5).astype(np.float32)
            H_te = (P_te > 0.5).astype(np.float32)

            per_concept_acc, train_per_concept_acc = _get_concept_accuracies(H_te, H_tr, test, train)
            acc_det, acc_gt, concept_acc_mean, fe, fe_path, _ = _train_frontend(
                H_te, H_tr, P_tr, S, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir, seed_tag, skew_tag, test, train
            )
            _, _, intervention_results = test_interventions(
                P_te, {**S, "intervention_accuracy": 1.0}, acc_det, fe, rng, test
            )

            subtype_concepts = [c for c in test.concepts if c.startswith("foot_shape_")]
            missing_concepts = [c for c in S.get("drop_concepts", []) if c.startswith("foot_shape_")]
            all_preds, confusion_df = _get_confusion_matrix(subtype_concepts, missing_concepts, fe, H_te, P_te, test)
            per_sub_acc = _get_accuracies_per_subconcept(all_preds, missing_concepts, subtype_concepts)

            confusion_path = run_dir / "confusion.csv"
            confusion_df.to_csv(confusion_path, index=False)
            catalog_csv_path = run_dir / "catalog.csv"
            data.meta["catalog_df"].to_csv(catalog_csv_path, index=False)

            meta_extra = {}
            if "catalog_df_spurious" in data.meta:
                catalog_spu_path = run_dir / "catalog_with_spurious.csv"
                data.meta["catalog_df_spurious"].to_csv(catalog_spu_path, index=False)
                meta_extra["catalog_csv_spurious"] = str(catalog_spu_path)

            meta = {
                "settings": S,
                "run_dir": str(run_dir),
                "artifacts": {"detector": str(det_path), "frontend": str(fe_path)},
                "splits": {"n_train": _len(train), "n_valid": _len(valid), "n_test": _len(test)},
                "concepts": list(data.concepts),
                "intervention_budgets": S.get("budget", []),
                "intervention_acc": 1.0,
                "logit_weights": params.get("weights", {}),
                "naming_slug": slug,
                "catalog_csv": str(catalog_csv_path),
                "df_indices": {
                    "train": list(map(int, train.meta.get("df_indices", []))),
                    "valid": list(map(int, valid.meta.get("df_indices", []))),
                    "test": list(map(int, test.meta.get("df_indices", []))),
                },
                "robot_ids": {
                    "train": list(map(int, train.meta.get("robot_ids", []))),
                    "valid": list(map(int, valid.meta.get("robot_ids", []))),
                    "test": list(map(int, test.meta.get("robot_ids", []))),
                },
            }
            meta.update(meta_extra)
            feweights = _extract_fe_weights(fe, list(test.concepts))
            metrics = {
                "cbm_acc_detected": float(acc_det),
                "cbm_acc_oracle": float(acc_gt),
                "concept_det_acc_mean": float(concept_acc_mean),
                "interventions": intervention_results,
                "frontend_weights": feweights,
                "concept_accuracies": per_concept_acc,
                "model_accuracies_per_concept": per_sub_acc,
                "train_concept_accuracies": train_per_concept_acc,
            }

            meta_path = run_dir / f"meta_cbm_detected_{slug}_{seed_tag}.json"
            metrics_path = run_dir / f"metrics_cbm_detected_{slug}_{seed_tag}.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            results[(regime)] = {"acc_det": acc_det, "acc_gt": acc_gt, "interventions": intervention_results}

        elif regime == "expert":
            cd, det_path = _train_concept_detector(
                S, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir, seed_tag, skew_tag, train, valid, test
            )
            P_tr = cd.predict(train, embed_params={"device": device})
            P_te = cd.predict(test, embed_params={"device": device})
            H_tr = (P_tr > 0.5).astype(np.float32)
            H_te = (P_te > 0.5).astype(np.float32)

            per_concept_acc, train_per_concept_acc = _get_concept_accuracies(H_te, H_tr, test, train)
            acc_det, acc_gt, concept_acc_mean, fe, fe_path, _ = _train_frontend(
                H_te, H_tr, P_tr, S, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir, seed_tag, skew_tag, test, train
            )
            _, _, intervention_results = test_interventions(
                P_te, {**S, "intervention_accuracy": float(S.get("human_annotation_accuracy", 0.8))}, acc_det, fe, rng, test
            )

            subtype_concepts = [c for c in test.concepts if c.startswith("foot_shape_")]
            missing_concepts = [c for c in S.get("drop_concepts", []) if c.startswith("foot_shape_")]
            all_preds, confusion_df = _get_confusion_matrix(subtype_concepts, missing_concepts, fe, H_te, P_te, test)
            per_sub_acc = _get_accuracies_per_subconcept(all_preds, missing_concepts, subtype_concepts)

            confusion_path = run_dir / "confusion.csv"
            confusion_df.to_csv(confusion_path, index=False)
            catalog_csv_path = run_dir / "catalog.csv"
            data.meta["catalog_df"].to_csv(catalog_csv_path, index=False)

            meta_extra = {}
            if "catalog_df_spurious" in data.meta:
                catalog_spu_path = run_dir / "catalog_with_spurious.csv"
                data.meta["catalog_df_spurious"].to_csv(catalog_spu_path, index=False)
                meta_extra["catalog_csv_spurious"] = str(catalog_spu_path)

            meta = {
                "settings": S,
                "run_dir": str(run_dir),
                "artifacts": {"detector": str(det_path), "frontend": str(fe_path)},
                "splits": {"n_train": _len(train), "n_valid": _len(valid), "n_test": _len(test)},
                "concepts": list(data.concepts),
                "intervention_budgets": S.get("budget", []),
                "intervention_acc": float(S.get("human_annotation_accuracy", 0.8)),
                "logit_weights": params.get("weights", {}),
                "naming_slug": slug,
                "catalog_csv": str(catalog_csv_path),
                "df_indices": {
                    "train": list(map(int, train.meta.get("df_indices", []))),
                    "valid": list(map(int, valid.meta.get("df_indices", []))),
                    "test": list(map(int, test.meta.get("df_indices", []))),
                },
                "robot_ids": {
                    "train": list(map(int, train.meta.get("robot_ids", []))),
                    "valid": list(map(int, valid.meta.get("robot_ids", []))),
                    "test": list(map(int, test.meta.get("robot_ids", []))),
                },
            }
            meta.update(meta_extra)
            feweights = _extract_fe_weights(fe, list(test.concepts))
            metrics = {
                "cbm_acc_detected": float(acc_det),
                "cbm_acc_oracle": float(acc_gt),
                "concept_det_acc_mean": float(concept_acc_mean),
                "interventions": intervention_results,
                "frontend_weights": feweights,
                "concept_accuracies": per_concept_acc,
                "model_accuracies_per_concept": per_sub_acc,
                "train_concept_accuracies": train_per_concept_acc,
            }

            meta_path = run_dir / f"meta_cbm_detected_{slug}_{seed_tag}.json"
            metrics_path = run_dir / f"metrics_cbm_detected_{slug}_{seed_tag}.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            results[(regime)] = {"acc_det": acc_det, "acc_gt": acc_gt, "interventions": intervention_results}

        elif regime == "subjective":
            for rate_subj in subjective_grid:
                rate_dir = run_dir / f"rate{int(rate_subj * 100):02d}"
                rate_dir.mkdir(parents=True, exist_ok=True)

                Ctr_noisy = _apply_concept_noise(train.C, train.concepts, S.get("concepts", {}), float(rate_subj), rng)
                Cva_noisy = _apply_concept_noise(valid.C, valid.concepts, S.get("concepts", {}), float(rate_subj), rng)
                tr_noisy = _clone_with_C(train, Ctr_noisy)
                va_noisy = _clone_with_C(valid, Cva_noisy)

                cd, det_path = _train_concept_detector(
                    S, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, rate_dir, seed_tag, skew_tag, tr_noisy, va_noisy, test
                )
                P_tr = cd.predict(tr_noisy, embed_params={"device": device})
                P_te = cd.predict(test, embed_params={"device": device})
                H_tr = (P_tr > 0.5).astype(np.float32)
                H_te = (P_te > 0.5).astype(np.float32)

                per_concept_acc, train_per_concept_acc = _get_concept_accuracies(H_te, H_tr, test, tr_noisy)
                acc_det, acc_gt, concept_acc_mean, fe, fe_path, _ = _train_frontend(
                    H_te, H_tr, P_tr, S, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, rate_dir, seed_tag, skew_tag, test, tr_noisy
                )
                _, _, intervention_results = test_interventions(P_te, S, acc_det, fe, rng, test)

                subtype_concepts = [c for c in test.concepts if c.startswith("foot_shape_")]
                missing_concepts = [c for c in S.get("drop_concepts", []) if c.startswith("foot_shape_")]
                all_preds, confusion_df = _get_confusion_matrix(subtype_concepts, missing_concepts, fe, H_te, P_te, test)
                per_sub_acc = _get_accuracies_per_subconcept(all_preds, missing_concepts, subtype_concepts)

                confusion_path = rate_dir / "confusion.csv"
                confusion_df.to_csv(confusion_path, index=False)
                catalog_csv_path = rate_dir / "catalog.csv"
                data.meta["catalog_df"].to_csv(catalog_csv_path, index=False)

                meta_extra = {}
                if "catalog_df_spurious" in data.meta:
                    catalog_spu_path = rate_dir / "catalog_with_spurious.csv"
                    data.meta["catalog_df_spurious"].to_csv(catalog_spu_path, index=False)
                    meta_extra["catalog_csv_spurious"] = str(catalog_spu_path)

                meta = {
                    "settings": {**S, "regime_subjective_rate": float(rate_subj)},
                    "run_dir": str(rate_dir),
                    "artifacts": {"detector": str(det_path), "frontend": str(fe_path)},
                    "splits": {"n_train": _len(tr_noisy), "n_valid": _len(va_noisy), "n_test": _len(test)},
                    "concepts": list(data.concepts),
                    "intervention_budgets": S.get("budget", []),
                    "intervention_acc": float(S.get("intervention_accuracy", 0.9)),
                    "logit_weights": params.get("weights", {}),
                    "naming_slug": slug,
                    "catalog_csv": str(catalog_csv_path),
                    "df_indices": {
                        "train": list(map(int, tr_noisy.meta.get("df_indices", []))),
                        "valid": list(map(int, va_noisy.meta.get("df_indices", []))),
                        "test": list(map(int, test.meta.get("df_indices", []))),
                    },
                    "robot_ids": {
                        "train": list(map(int, tr_noisy.meta.get("robot_ids", []))),
                        "valid": list(map(int, va_noisy.meta.get("robot_ids", []))),
                        "test": list(map(int, test.meta.get("robot_ids", []))),
                    },
                }
                meta.update(meta_extra)
                feweights = _extract_fe_weights(fe, list(test.concepts))
                metrics = {
                    "cbm_acc_detected": float(acc_det),
                    "cbm_acc_oracle": float(acc_gt),
                    "concept_det_acc_mean": float(concept_acc_mean),
                    "interventions": intervention_results,
                    "frontend_weights": feweights,
                    "concept_accuracies": per_concept_acc,
                    "model_accuracies_per_concept": per_sub_acc,
                    "train_concept_accuracies": train_per_concept_acc,
                }

                meta_path = rate_dir / f"meta_cbm_detected_{slug}_{seed_tag}.json"
                metrics_path = rate_dir / f"metrics_cbm_detected_{slug}_{seed_tag}.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

                results[(regime, float(rate_subj))] = {
                    "acc_det": acc_det,
                    "acc_gt": acc_gt,
                    "interventions": intervention_results,
                }

        elif regime == "machine":
            results[(regime)] = {"note": "label-free CBM not implemented"}

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    return results
