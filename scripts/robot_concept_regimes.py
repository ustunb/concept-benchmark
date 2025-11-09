# scripts/robot_concept_regimes.py
from __future__ import annotations

import os
import json
import random
import hashlib
from pathlib import Path

# join base/item; if not found, try parent-of-base (handles ".../test_images" + "test_images/..")
def _resolve_items(items, base_dir):
    base = Path(base_dir)
    out = []
    for s in items:
        p = Path(str(s))
        q = p if p.is_absolute() else base / p
        if not q.exists():
            alt = base.parent / p if not p.is_absolute() else q
            if alt.exists():
                q = alt
        out.append(str(q.resolve()))
    return out
from typing import Dict, List, Tuple, Optional
from concept_benchmark.paths import data_dir, results_dir, pkg_dir

import numpy as np
import torch
from torchvision import transforms

from concept_benchmark.synthetic.robot import create_synthetic_dataset
from concept_benchmark.models import ConceptDetector, FrontEndModel, RobotConceptClassifier, ConceptBasedModel
from concept_benchmark.lfcbm import LabelFreeCBM, LFTrainingConfig, LFConceptSet
from scripts.dataset_skewing import create_skewed_splits, filter_training_by_string
from concept_benchmark.intervention import InterventionConfig, ConceptInterventionRunner
from concept_benchmark.kflip import KFlipInterventionStrategy as ScoreIntervention
from scripts.robot_invariance_test import test_concept_detector_invariance
from scripts.robot_utils import (
    _apply_missing,
    _apply_label_noise,
    _rate_tag,
    _get_concept_accuracies,
    _get_confusion_matrix,
    _get_accuracies_per_subconcept,
)

def test_interventions(prob_test, sttngs, acc_det, fe, test):
    intervention_results = {}
    rng = np.random.default_rng(int(sttngs["seed"]))
    budgets = sttngs.get('budget', [1])
    human_acc = sttngs.get("intervention_accuracy", 0.9)
    err_prob = 1.0 - human_acc

    cbm = ConceptBasedModel(concept_detector=None, front_end_model=fe)
    runner = ConceptInterventionRunner(cbm)

    for budget in budgets:
        config = InterventionConfig(
            max_concepts_per_instance=budget,
            random_state=int(sttngs["seed"]),
            score_threshold=sttngs.get("intervention_threshold", 1.0),
            noise=1.0 - human_acc,
            select_only_abstained=bool(sttngs.get("select_only_abstained", False)),
            tau=sttngs.get("tau", None),
        )

        strategy = ScoreIntervention()

        result = runner.run(
            strategy=strategy,
            config=config,
            dataset=test,
            concept_proba=prob_test,
            labels=test.y.astype(int)
        )

        # Apply human error on masked entries
        mask = result.mask
        C_gt = test.C.astype(np.float32)
        C_after = result.C_intervened.copy()
        mistake_draw = rng.random(C_after.shape) < err_prob
        mistakes = mask & mistake_draw
        C_after[mistakes] = 1.0 - C_gt[mistakes]
        result.C_intervened = C_after

        # Recompute downstream prediction after error injection
        C_final_binary = (result.C_intervened >= 0.5).astype(int)
        result.y_prob_after = fe.predict_proba(C_final_binary)
        result.y_pred_after = np.argmax(result.y_prob_after, axis=1)

        acc_intervened = float((result.y_pred_after == test.y.astype(int)).mean())

        n_intervened = int(np.sum(result.mask))
        n_samples = prob_test.shape[0]

        intervened_concepts = np.any(result.mask, axis=0)
        C_pred_binary = (result.C_pred >= 0.5).astype(int)
        actual_edits_mask = (C_pred_binary != C_final_binary)
        prediction_num_concepts_intervened_on = {int(i): int(np.sum(actual_edits_mask[i])) for i in range(n_samples)}

        concept_intervention_counts = {
            c: f"{int(np.sum(result.mask[:, i]))} ({int(np.sum(actual_edits_mask[:, i]))})"
            for i, c in enumerate(test.concepts) if intervened_concepts[i]
        }

        key = f"top_{budget}_human_acc_{int(human_acc * 100)}"
        intervention_results[key] = {
            "accuracy": acc_intervened,
            "accuracy_gain": acc_intervened - acc_det,
            "predictions_intervened_on": int(np.sum(np.any(result.mask, axis=1))),
            "interventions_rate": float(np.sum(np.any(result.mask, axis=1)) / n_samples),
            "intervention_rate": float(np.sum(np.any(result.mask, axis=1)) / n_samples),
            "avg_edits_per_intervention": float(sum(prediction_num_concepts_intervened_on.values())) / n_samples,
            "total_concept_checks": n_intervened,
            "total_concept_edits_made": int(sum(prediction_num_concepts_intervened_on.values())),
            "concept_interventions": concept_intervention_counts,
            "human_accuracy": human_acc
        }

    return budgets, human_acc, intervention_results

class FEOnProbs(FrontEndModel):
    def __init__(self, clf):
        super().__init__()
        self.model = clf
    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        P = np.clip(P, 1e-6, 1 - 1e-6)
        Z = np.log(P / (1.0 - P))
        return self.model.predict_proba(Z)


# seeds
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

# helpers
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
        for name in concept_names:
            out[name] = None
        out["bias"] = None
        return out

def _build_slug(S: Dict, miss_tag: str, filter_tag: str, label_noise_tag: str, skew_tag: str, int_acc_tag: str) -> str:
    impute_tag = f"impute{int(S.get('impute_missing', 0))}"
    return f"robots_image_{S['model_type']}_{miss_tag}{filter_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{impute_tag}"

def _merge_save_npz(path: Path, new_arrays: Dict[str, np.ndarray]) -> None:
    merged = {}
    if path.exists():
        old = np.load(path)
        for k in old.files:
            merged[k] = old[k]
    merged.update(new_arrays)
    np.savez_compressed(path, **merged)

def _training_critical_subset(S: Dict) -> Dict:
    # exclude intervention-only knobs; include data, split, and model training knobs
    keep = {
        "concepts", "subconcepts", "additional_features", "spurious_features", "drop_concepts",
        "model", "model_type", "image_size", "color_mode",
        "logit_scalar", "logit_intercept", "logit_weights",
        "knows_concepts", "impute_missing", "CBM_type",
        "missingness", "missing_rate", "label_noise_rate",
        "skew_concept", "dataset_characterization",
        "train_size", "test_size", "samples_per_instance", "seed",
    }
    return {k: S[k] for k in sorted(keep) if k in S}

def _anchor_key(S: Dict, *, concept_noise_rate: float = 0.0) -> str:
    crit = _training_critical_subset(S)
    crit["concept_noise_rate"] = float(concept_noise_rate)
    payload = json.dumps(crit, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

def _anchor_paths(base_out: Path, run_name: str, key: str) -> Dict[str, Path]:
    anchor_dir = base_out / "anchors" / f"{run_name}__{key}"
    return {
        "dir": anchor_dir,
        "meta": anchor_dir / "anchor_meta.json",
        "detector": anchor_dir / "detector.pt",
        "frontend": anchor_dir / "frontend.pkl",
        "probs": anchor_dir / "probs.npz",
    }

def _resolve_anchor(S: Dict, base_out: Path, run_name: str, *, concept_noise_rate: float = 0.0) -> Tuple[Dict, Dict[str, Path]]:
    key = _anchor_key(S, concept_noise_rate=concept_noise_rate)
    paths = _anchor_paths(base_out, run_name, key)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    # Prefer existing anchor artifacts if present
    if paths["detector"].exists():
        S["load_detector"] = str(paths["detector"])
    if paths["frontend"].exists():
        S["load_frontend"] = str(paths["frontend"])
    S["anchor_dir"] = str(paths["dir"])
    return S, paths

def _load_artifacts_and_cache(
    S: Dict, run_dir: Path, slug: str, seed_tag: str, force: bool
) -> Tuple[Dict, Optional[np.lib.npyio.NpzFile], Path, Path, Path]:
    meta_path = run_dir / f"meta_cbm_detected_{slug}_{seed_tag}.json"
    metrics_path = run_dir / f"metrics_cbm_detected_{slug}_{seed_tag}.json"
    prob_npz_path = run_dir / f"probs_{slug}_{seed_tag}.npz"
    prob_cache = None
    if not force:
        if not meta_path.exists():
            cands = list(run_dir.glob(f"**/meta_cbm_detected_{slug}_{seed_tag}.json"))
            if cands:
                meta_path = cands[0]
        if meta_path.exists():
            with open(meta_path, "r") as f:
                _meta_prev = json.load(f)
            arts = _meta_prev.get("artifacts", {})
            if "load_detector" not in S and arts.get("detector") and os.path.exists(arts["detector"]):
                S["load_detector"] = arts["detector"]
            if "load_frontend" not in S and arts.get("frontend") and os.path.exists(arts["frontend"]):
                S["load_frontend"] = arts["frontend"]
        if not prob_npz_path.exists():
            nc = list(run_dir.glob(f"**/probs_{slug}_{seed_tag}.npz"))
            if nc:
                prob_npz_path = nc[0]
        if prob_npz_path.exists():
            prob_cache = np.load(prob_npz_path)
    return S, prob_cache, prob_npz_path, meta_path, metrics_path

def _write_catalogs(out_dir: Path, data_meta: Dict) -> Tuple[str, Dict]:
    catalog_csv_path = out_dir / "catalog.csv"
    data_meta["catalog_df"].to_csv(catalog_csv_path, index=False)
    meta_extra = {}
    if "catalog_df_spurious" in data_meta:
        catalog_spu_path = out_dir / "catalog_with_spurious.csv"
        data_meta["catalog_df_spurious"].to_csv(catalog_spu_path, index=False)
        meta_extra["catalog_csv_spurious"] = str(catalog_spu_path)
    return str(catalog_csv_path), meta_extra

def _emit_confusion(out_dir: Path, fe: FrontEndModel, H_te: np.ndarray, P_te: np.ndarray, test, drop_list: List[str]) -> Dict:
    subtype = [c for c in test.concepts if c.startswith("foot_shape_")]
    missing = [c for c in drop_list if c.startswith("foot_shape_")]
    all_preds, confusion_df = _get_confusion_matrix(subtype, missing, fe, H_te, P_te, test)
    (out_dir / "confusion.csv").write_text(confusion_df.to_csv(index=False))
    return _get_accuracies_per_subconcept(all_preds, missing, subtype)

# dataset build to mirror main
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

    if float(settings.get("label_noise_rate", 0.0)) > 0.0:
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
    # Normalize dataset paths in-place and force numpy arrays for boolean indexing
    def _norm_inplace(ds):
        if hasattr(ds, "X"):
            Xn = []
            for x in ds.X:
                p = Path(str(x))
                if "robot_images" in p.parts:
                    i = p.parts.index("robot_images")
                    p = Path(*p.parts[i+1:])
                Xn.append(p.as_posix())
            ds.X = np.array(Xn, dtype=object)
        return ds
    _norm_inplace(train); _norm_inplace(valid); _norm_inplace(test)
    return test, train, valid

# model training wrappers
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

    # Save to anchor if provided, else to run_dir with full name
    save_dir = Path(settings.get("anchor_dir", run_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    det_name = "detector.pt" if settings.get("anchor_dir") else f"detector_dnn_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"

    if settings.get("load_detector"):
        mini_train = train.filter(np.array([True] + [False] * (len(train.C) - 1)))
        mini_valid = valid.filter(np.array([True] + [False] * (len(valid.C) - 1)))
        cd.fit(mini_train, mini_valid, freeze=True, embed_params={"device": device}, fit_params={"epochs": 1, "device": "cpu"})
        state = torch.load(settings["load_detector"], weights_only=False, map_location="cpu")
        cd.load_state_dict(state)
        det_path = Path(settings["load_detector"])
    else:
        cd.fit(train, valid, embed_params={"shuffle": False, **config}, fit_params={"epochs": 50, "lr": 1e-3, "patience": 10, **config})
        det_path = save_dir / det_name
        torch.save(cd.state_dict(), det_path)

    # for concept in [c for c in test.concepts if c.startswith("foot_shape_")]:
    #     _ = test_concept_detector_invariance(cd, concept, train.concepts, test, device, num_tests=10)
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
    # Save to anchor if provided, else to run_dir with full name
    save_dir = Path(sttngs.get("anchor_dir", run_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    fe_name = "frontend.pkl" if sttngs.get("anchor_dir") else f"frontend_logreg_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pkl"

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
        fe_path = save_dir / fe_name
        with open(fe_path, "wb") as f:
            pickle.dump(fe, f)

    y_pred_det = fe.predict_proba(H_te)
    y_pred_gt = fe.predict_proba(test.C.astype(np.float32))
    acc_det = float((y_pred_det.argmax(1) == test.y.astype(int)).mean())
    acc_gt = float((y_pred_gt.argmax(1) == test.y.astype(int)).mean())
    concept_acc_mean = float((H_te == test.C).mean())
    return acc_det, acc_gt, concept_acc_mean, fe, fe_path, y_pred_det

# regime runners
def _run_fixed_regime(
    regime: str,
    S: Dict,
    config: Dict,
    device: str,
    int_acc_tag: str,
    label_noise_tag: str,
    miss_tag: str,
    model_type_tag: str,
    run_dir: Path,
    seed_tag: str,
    skew_tag: str,
    slug: str,
    train,
    valid,
    test,
    data,
    prob_npz_path: Path,
    prob_cache: Optional[np.lib.npyio.NpzFile],
    force: bool,
    rng,
    params: Dict,
):
    cd, det_path = _train_concept_detector(
        S, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir, seed_tag, skew_tag, train, valid, test
    )

    if (not force) and prob_cache is not None and {"P_tr", "P_te"}.issubset(set(prob_cache.files)):
        P_tr = prob_cache["P_tr"]
        P_te = prob_cache["P_te"]
    else:
        P_tr = cd.predict(train, embed_params={"device": device})
        P_te = cd.predict(test, embed_params={"device": device})
        _merge_save_npz(
            prob_npz_path,
            {
                "P_tr": P_tr,
                "P_te": P_te,
                "C_tr": train.C.astype(np.float32),
                "C_te": test.C.astype(np.float32),
                "y_tr": train.y.astype(int),
                "y_te": test.y.astype(int),
                "concepts": np.array(list(test.concepts)),
            },
        )

    H_tr = (P_tr > 0.5).astype(np.float32)
    H_te = (P_te > 0.5).astype(np.float32)
    per_concept_acc, train_per_concept_acc = _get_concept_accuracies(H_te, H_tr, test, train)
    acc_det, acc_gt, concept_acc_mean, fe, fe_path, _ = _train_frontend(
        H_te, H_tr, P_tr, S, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir, seed_tag, skew_tag, test, train
    )

    ia_val = 1.0 if regime == "perfect" else float(S.get("human_annotation_accuracy", 0.8))
    _, _, intervention_results = test_interventions(P_te, {**S, "intervention_accuracy": ia_val}, acc_det, fe, test)

    per_sub_acc = _emit_confusion(run_dir, fe, H_te, P_te, test, S.get("drop_concepts", []))
    catalog_csv_path, meta_extra = _write_catalogs(run_dir, data.meta)

    meta = {
        "settings": S,
        "run_dir": str(run_dir),
        "artifacts": {"detector": str(det_path), "frontend": str(fe_path)},
        "splits": {"n_train": _len(train), "n_valid": _len(valid), "n_test": _len(test)},
        "concepts": list(data.concepts),
        "intervention_budgets": S.get("budget", []),
        "intervention_acc": ia_val,
        "logit_weights": params.get("weights", {}),
        "naming_slug": slug,
        "catalog_csv": catalog_csv_path,
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
        "prob_test_npz": str(prob_npz_path),
        "concept_names": list(test.concepts),
    }
    return meta, metrics, {"acc_det": acc_det, "acc_gt": acc_gt, "interventions": intervention_results}

def _run_subjective_rate(
    rate_subj: float,
    S: Dict,
    config: Dict,
    device: str,
    int_acc_tag: str,
    label_noise_tag: str,
    miss_tag: str,
    model_type_tag: str,
    rate_dir: Path,
    seed_tag: str,
    skew_tag: str,
    slug: str,
    train,
    valid,
    test,
    data,
    prob_npz_path: Path,
    prob_cache: Optional[np.lib.npyio.NpzFile],
    force: bool,
    rng,
    params: Dict,
):
    Ctr_noisy = _apply_concept_noise(train.C, train.concepts, S.get("concepts", {}), float(rate_subj), rng)
    Cva_noisy = _apply_concept_noise(valid.C, valid.concepts, S.get("concepts", {}), float(rate_subj), rng)
    tr_noisy = _clone_with_C(train, Ctr_noisy)
    va_noisy = _clone_with_C(valid, Cva_noisy)

    if (not force) and prob_cache is not None and {"P_tr", "P_te"}.issubset(set(prob_cache.files)):
        P_tr = prob_cache["P_tr"]
        P_te = prob_cache["P_te"]
        cd = None
        det_path = Path(S.get("load_detector", "")) if S.get("load_detector") else Path("")
    else:
        cd, det_path = _train_concept_detector(
            S, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, rate_dir, seed_tag, skew_tag, tr_noisy, va_noisy, test
        )
        P_tr = cd.predict(tr_noisy, embed_params={"device": device})
        P_te = cd.predict(test, embed_params={"device": device})
        _merge_save_npz(
            prob_npz_path,
            {
                "P_tr": P_tr,
                "P_te": P_te,
                "C_tr": tr_noisy.C.astype(np.float32),
                "C_te": test.C.astype(np.float32),
                "y_tr": tr_noisy.y.astype(int),
                "y_te": test.y.astype(int),
                "concepts": np.array(list(test.concepts)),
            },
        )

    H_tr = (P_tr > 0.5).astype(np.float32)
    H_te = (P_te > 0.5).astype(np.float32)

    per_concept_acc, train_per_concept_acc = _get_concept_accuracies(H_te, H_tr, test, tr_noisy)
    acc_det, acc_gt, concept_acc_mean, fe, fe_path, _ = _train_frontend(
        H_te, H_tr, P_tr, S, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, rate_dir, seed_tag, skew_tag, test, tr_noisy
    )
    _, _, intervention_results = test_interventions(P_te, S, acc_det, fe, test)

    per_sub_acc = _emit_confusion(rate_dir, fe, H_te, P_te, test, S.get("drop_concepts", []))
    catalog_csv_path, meta_extra = _write_catalogs(rate_dir, data.meta)

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
        "catalog_csv": catalog_csv_path,
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
        "prob_test_npz": str(prob_npz_path),
        "concept_names": list(test.concepts),
    }
    return meta, metrics, {"acc_det": acc_det, "acc_gt": acc_gt, "interventions": intervention_results}

# entrypoint
def run_regimes(settings: Dict) -> Dict:
    set_seed(settings["seed"])

    S = dict(settings)
    if S.get("draw_only", 0):
        S["draw"] = 1
    force = bool(S.get("force", False))
    if force:
        S.pop("load_detector", None)
        S.pop("load_frontend", None)
    rng = np.random.default_rng(int(S["seed"]))
    base_root = Path(S["out_dir"])
    base_out = base_root
    base_out.mkdir(parents=True, exist_ok=True)

    miss = str(S["missingness"]).lower()
    rate = float(S["missing_rate"])

    int_acc_tag = f"int-acc{int(round(float(S['intervention_accuracy']) * 100))}"
    miss_tag = "complete" if miss == "complete" or rate <= 0 else f"{miss}{_rate_tag(rate)}"
    skew_tag = "_skew" if S.get("skew_concept", []) else ""
    filter_tag = "_filter" if S.get("dataset_characterization", "") else ""
    label_noise_tag = "_label-noise_" if float(S.get("label_noise_rate", 0.0)) else "_"
    model_type_tag = f"{S['model_type']}_"  # filenames

    params = {
        "samples_per_instance": S["samples_per_instance"],
        "draw": S["draw"],
        "output_directory": S.get("image_dir", base_root / "images"),
        "concepts": S["concepts"],
        "additional_features": [] if S.get("knows_concepts", True) else S.get("subconcepts", ["foot_shape_subtype", "hand_shape_subtype"]),
        "spurious_features": S.get("spurious_features", []),
        "drop_concepts": S.get("drop_concepts", []),
        "color_mode": S["color_mode"],
        "model": S["model"],
        "model_type": S["model_type"],
        "size": S["image_size"],
        "scalar": float(S.get("logit_scalar", 1.0)),
        "intercept": float(S.get("logit_intercept", 0.0)),
        "weights": S.get("logit_weights", {}),
        "test_set_size": int(S.get("test_size", 10000)),
        "train_concept_detector": True,
        "verbose": True,
        "rng_seed": S["seed"],
    }

    if Path(params["output_directory"]).exists() and not bool(S.get("draw", False)):
        params["draw"] = False

    data = create_synthetic_dataset(**params)
    tf = transforms.Compose([transforms.ToTensor()])
    data.transform = tf
    data.generate_cvindices(seed=int(S["seed"]))

    if S.get("draw_only", 0):
        draw_dir = base_out / f"{S['run_name']}__draw_only"
        draw_dir.mkdir(parents=True, exist_ok=True)
        catalog_csv_path, _ = _write_catalogs(draw_dir, data.meta)
        return {"draw_only": {"catalog_csv": catalog_csv_path, "n_images": int(data.meta["catalog_df"].shape[0])}}

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

    results: Dict = {}
    results_file = base_out / f"{S['run_name']}__regime_results.json"

    for regime in regimes:
        run_dir = base_out / f"{S['run_name']}__regime-{regime}"
        run_dir.mkdir(parents=True, exist_ok=True)

        slug = _build_slug(S, miss_tag, filter_tag, label_noise_tag, skew_tag, int_acc_tag)
        seed_tag = f"seed{int(S['seed'])}"

        if regime in {"perfect", "expert"}:
            # Anchor = noise-free training; reuse across different intervention settings
            S_anchor, _ = _resolve_anchor(dict(S), base_out, S["run_name"], concept_noise_rate=0.0)
            # Prefer anchor artifacts over per-regime meta
            S_anchor, prob_cache, prob_npz_path, meta_path, metrics_path = _load_artifacts_and_cache(
                S_anchor, run_dir, slug, seed_tag, force
            )
            meta, metrics, res = _run_fixed_regime(
                regime, S_anchor, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag,
                run_dir, seed_tag, skew_tag, slug, train, valid, test, data, prob_npz_path, prob_cache, force, rng,
                params
            )
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            results[regime] = res


        elif regime == "subjective":
            for rate_subj in subjective_grid:
                rate_dir = run_dir / f"rate{int(float(rate_subj) * 100):02d}"
                rate_dir.mkdir(parents=True, exist_ok=True)

                # Anchor keyed by concept noise rate; independent of intervention accuracy
                S_rate = dict(S)
                S_rate, _anchor_paths_dict = _resolve_anchor(S_rate, base_out, S_rate["run_name"],
                                                             concept_noise_rate=float(rate_subj))
                S_rate, prob_cache, prob_npz_path, meta_path_unused, metrics_path_unused = _load_artifacts_and_cache(
                    S_rate, rate_dir, slug, seed_tag, force
                )
                meta, metrics, res = _run_subjective_rate(
                    float(rate_subj), S_rate, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag,
                    rate_dir, seed_tag, skew_tag, slug, train, valid, test, data, prob_npz_path, prob_cache, force, rng,
                    params

                )
                with open(rate_dir / f"meta_cbm_detected_{slug}_{seed_tag}.json", "w") as f:
                    json.dump(meta, f, indent=2)

                with open(rate_dir / f"metrics_cbm_detected_{slug}_{seed_tag}.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                results[(regime, float(rate_subj))] = res


        elif regime in {"machine", "machine_annotation"}:

            # ---- Label-free CBM regime (machine annotation) ----

            # Requirements:

            #   - settings["concepts_file"] points to a CSV/JSON/JSONL/TXT concept list

            #   - concept keys align with dataset concepts (or are aligned by LFConceptSet)

            concepts_file = S.get("concepts_file", "")
            # Accept either a concept file or the in-memory concept list used elsewhere.
            if concepts_file:
                concept_set = LFConceptSet.from_file(concepts_file, dataset_keys=list(test.concepts))
            else:
                ds_keys = list(test.concepts)
                concept_set = LFConceptSet(keys=ds_keys, texts=[LFConceptSet._normalize_key(k) for k in ds_keys])

            # Config: respect user overrides under S["lfcbm"], default to paper settings
            lf_cfg = S.get("lfcbm", {})
            cfg = LFTrainingConfig(
                clip_model=str(lf_cfg.get("clip_model", "ViT-B-32")),
                clip_pretrained=str(lf_cfg.get("clip_pretrained", "laion2b_s34b_b79k")),
                device=device,
                lr=float(lf_cfg.get("lr", 1e-2)),
                weight_decay=float(lf_cfg.get("weight_decay", 1e-4)),
                max_epochs=int(lf_cfg.get("max_epochs", 200)),
                patience=int(lf_cfg.get("patience", 10)),
                seed=int(S["seed"]),
                l1_ratio=float(lf_cfg.get("l1_ratio", 0.99)),
                C_grid=tuple(map(float, lf_cfg.get("C_grid", (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)))),
                target_nonzero_per_class=tuple(map(int, lf_cfg.get("target_nonzero_per_class", (25, 35)))),
                cache_dir=run_dir / "lfcbm_cache",
                batch_size=int(lf_cfg.get("batch_size", 256)),
            )
            lf = LabelFreeCBM(cfg)

            # Fit label-free CBM on current split
            stats = lf.fit(
                train_X=_resolve_items(train.X, getattr(train, "base_dir", Path("."))),
                train_y=train.y.astype(int),
                valid_X=_resolve_items(valid.X, getattr(valid, "base_dir", Path("."))),
                valid_y=valid.y.astype(int),
                concept_set=concept_set,
                cache_dir=cfg.cache_dir,
            )

            resolved_te = _resolve_items(test.X, getattr(test, "base_dir", Path(".")))
            missing = [p for p in resolved_te if not Path(p).exists()]
            print("missing test files:", len(missing), missing[:3])

            # Compute concept probabilities on train/test
            P_tr = lf.concept_proba(
                [str((getattr(train, "base_dir", Path(".")) / Path(p)).resolve()) for p in train.X])  # (Ntr, Mk)
            P_te = lf.concept_proba(
                [str((getattr(test, "base_dir", Path(".")) / Path(p)).resolve()) for p in test.X])  # (Nte, Mk)
            H_tr = (P_tr > 0.5).astype(np.float32)
            H_te = (P_te > 0.5).astype(np.float32)

            # Save probabilities to match the rest of the pipeline convention
            prob_npz_path = run_dir / f"probs_{slug}_{seed_tag}.npz"
            _merge_save_npz(
                prob_npz_path,
                {
                    "P_tr": P_tr,
                    "P_te": P_te,
                    "C_tr": train.C.astype(np.float32),
                    "C_te": test.C.astype(np.float32),
                    "y_tr": train.y.astype(int),
                    "y_te": test.y.astype(int),
                    "concepts": np.array(list(test.concepts)),
                },
            )

            # Build a lightweight front-end adapter that expects probabilities P and internally uses logit(P).
            fe = FEOnProbs(lf.classifier)
            import pickle
            fe_name = f"frontend_{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pkl"
            fe_path = run_dir / fe_name
            with open(fe_path, "wb") as f:
                pickle.dump(fe, f)

            # Accuracies
            y_pred_det = fe.predict_proba(P_te)
            acc_det = float((y_pred_det.argmax(1) == test.y.astype(int)).mean())

            # Oracle (use ground-truth concepts; clamp to avoid inf logit)
            C_oracle = np.clip(test.C.astype(np.float32), 1e-6, 1 - 1e-6)
            y_pred_gt = fe.predict_proba(C_oracle)
            acc_gt = float((y_pred_gt.argmax(1) == test.y.astype(int)).mean())

            # Per-concept accuracies vs. ground truth
            per_concept_acc, train_per_concept_acc = _get_concept_accuracies(H_te, H_tr, test, train)

            # Interventions use probabilities
            ia_val = float(S.get("human_annotation_accuracy", 0.8))
            _, _, intervention_results = test_interventions(P_te, {**S, "intervention_accuracy": ia_val}, acc_det, fe,
                                                            test)
            # Confusion and catalogs
            per_sub_acc = _emit_confusion(run_dir, fe, H_te, P_te, test, S.get("drop_concepts", []))
            catalog_csv_path, meta_extra = _write_catalogs(run_dir, data.meta)

            # Persist LF-CBM artefacts
            lf_paths = lf.save(run_dir / "lfcbm_artifacts")
            meta = {
                "settings": S,
                "run_dir": str(run_dir),
                "artifacts": {**lf_paths, "detector": "", "frontend": str(fe_path), "probs_npz": str(prob_npz_path)},
                "splits": {"n_train": _len(train), "n_valid": _len(valid), "n_test": _len(test)},
                "concepts": list(test.concepts),
                "intervention_budgets": S.get("budget", []),
                "intervention_acc": ia_val,
                "naming_slug": slug,
                "catalog_csv": catalog_csv_path,
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
            metrics = {
                "cbm_acc_detected": float(acc_det),
                "cbm_acc_oracle": float(acc_gt),
                "concept_det_acc_mean": float((H_te == test.C).mean()),
                "interventions": intervention_results,
                "concept_accuracies": per_concept_acc,
                "model_accuracies_per_concept": per_sub_acc,
                "train_concept_accuracies": train_per_concept_acc,
                "prob_test_npz": str(prob_npz_path),
                "concept_names": list(test.concepts),
                "lfcbm_train_stats": stats,
            }

            with open(run_dir / f"meta_cbm_detected_{slug}_{seed_tag}.json", "w") as f:
                json.dump(meta, f, indent=2)

            with open(run_dir / f"metrics_cbm_detected_{slug}_{seed_tag}.json", "w") as f:
                json.dump(metrics, f, indent=2)

            results[regime] = {
                "acc_detected": acc_det,
                "acc_ground_truth": acc_gt,
                "concept_acc_mean": float((H_te == test.C).mean()),
            }

    def _jsonify(o):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                k = "/".join(map(str, k)) if isinstance(k, tuple) else str(k)
                out[k] = _jsonify(v)
            return out
        if isinstance(o, (list, tuple)):
            return [_jsonify(x) for x in o]
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if hasattr(o, "tolist"):
            try:
                return o.tolist()
            except Exception:
                pass
        if isinstance(o, Path):
            return str(o)
        return o

    with open(results_file, "w") as f:
        json.dump(_jsonify(results), f, indent=2, sort_keys=True)
    return results


# optional example settings block kept for parity with training (no __main__ guard)
from concept_benchmark.paths import results_dir

settings = {
    "samples_per_instance": 4,
    "draw": 0,
    "draw_only": 1,
    "CBM_type": "separate",
    "image_dir": str(results_dir.parent / "data" / "robot_images"),
    "image_size": "large",
    "color_mode": "color",
    "seed": 1002,
    "model": "'glorp' if (int(row['mouth_type']=='closed') + int(row['foot_shape']=='pointy'))>= 3 else 'drent'",
    "dataset_characterization": "",
    "test_size": 10000,
    "train_size": 3800,
    "knows_concepts": False,
    "concepts": {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": [
            "round_circle","round_oval","round_oval2",
            "edgy_triangle","edgy_square","edgy_trapezoid",
        ],
        "foot_shape": [
            "flat_trapezoid","flat_rounded","flat_square","flat_5sided","flat_lshaped",
            "pointy_trapezoid","pointy_rounded","pointy_square","pointy_3sided","pointy_4sided",
        ],
    },
    "subconcepts": ["foot_shape_subtype"],
    "spurious_features": ["has_elbows", "hand_shape"],
    "drop_concepts": [
        "foot_shape_flat_rounded","foot_shape_pointy_trapezoid",
        "foot_shape_pointy_3sided","foot_shape_flat_lshaped","foot_shape"
    ],
    "human_alignment": {
        "foot_shape_pointy_4sided": 5,
        "foot_shape_pointy_rounded": 3,
        "foot_shape_pointy_square": 1,
        "foot_shape_flat_5sided": -5,
        "foot_shape_flat_square": -1,
        "foot_shape_flat_trapezoid": 0,
        "mouth_type": -5,
        "bias": 4
    },
    "model_type": "stochastic",
    "logit_scalar": 1.0,
    "logit_intercept": 3,
    "logit_weights": {"mouth_type": 5, "foot_shape": 10},
    "label_noise_rate": 0,
    "missingness": "complete",
    "missing_rate": 1.0,
    "impute_missing": 0,
    "skew_concept": [
        {"concepts": {"foot_shape_pointy_square": 1}, "min_fraction": 0.005},
        {"concepts": {"foot_shape_pointy_rounded": 1}, "min_fraction": 0.005},
        {"concepts": {"foot_shape_pointy_4sided": 1}, "min_fraction": 0.49},
        {"concepts": {"foot_shape_flat_square": 1}, "min_fraction": 0.005},
        {"concepts": {"foot_shape_flat_trapezoid": 1}, "min_fraction": 0.005},
        {"concepts": {"foot_shape_flat_5sided": 1}, "min_fraction": 0.49},
    ],

    "budget": [3],

    # Keep 0.9 here to match your existing NPZ filename slug (int-acc90),
    # but "perfect" will still run with 1.0 human accuracy internally.
    "intervention_accuracy": 0.8,
    "human_annotation_accuracy": 0.8,

    "intervention_threshold": 0.2,

    "out_dir": str(results_dir / "robots"),
    "run_name": "cbm_run_1002_subconcepts",

    # Let the script pick up anchors/NPZs; do NOT force recompute.
    "load_detector": "",
    "load_frontend": "",
    "force": 0,

    # Only run perfect to regenerate interventions/metrics
    "subjective_grid": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "regimes": ["subjective"],

    "concepts_file": None,
    "lfcbm": {
      "clip_model": "ViT-B-32",
      "clip_pretrained": "laion2b_s34b_b79k",
      "l1_ratio": 0.99,
      "C_grid": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
      "target_nonzero_per_class": [25, 35],
      "max_epochs": 200,
      "patience": 10
    }
}

run_regimes(settings)
