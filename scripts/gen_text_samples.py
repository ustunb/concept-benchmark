from __future__ import annotations
import csv
import hashlib
import re
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from concept_benchmark.synthetic.helper.textgen import create_synthetic_dataset
from concept_benchmark.paths import pkg_dir, results_dir, data_dir
from concept_benchmark.synthetic.helper.text_concept_detector import TextConceptDetector
from concept_benchmark.models import ConceptBasedModel, FrontEndModel
from concept_benchmark.ext.fileutils import save as save_obj
from concept_benchmark.metrics import calc_metric
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.ext.fileutils import load as load_obj
from concept_benchmark.synthetic.helper.utils import apply_subjective_noise, apply_machine_noise
from types import SimpleNamespace
import argparse, psutil
from scripts.lfcbm_text import LabelFreeDetector
from itertools import product
import builtins, functools

print = functools.partial(builtins.print, end="\n\n")

ROBOT_RUN_DIR = results_dir / "robot_text"
ROBOT_DATA_DIR = data_dir / "robot_text"
ROBOT_RUN_DIR.mkdir(parents=True, exist_ok=True)
ROBOT_DATA_DIR.mkdir(parents=True, exist_ok=True)

settings = {
    "variant": "perfect",
    "variants_per_row": 3,
    "imperfect_strategy": "missing_concepts",
    "heldout_concepts": [],
    "mask_p": 0.0,
    "mask_mode": "mask",
    "mask_rate": 0.0,
    "test_label_prior": "",
    "seed": 1337,
    "concept_mode": "hard",
    "train_on_detected": False,
    "templates_file": "",
    "label_model_expr": "",
    "corr_pair": "",
    "train_corr": 1.0,
    "test_break": 1.0,
    "test_corr": -1.0,
    "budgets": "0",
    "target_acc_grid": "raw",
    "target_acc_concepts": "",
    "intervene_allow": "",
    "human_acc": 1.0,
    "human_acc_concepts": "",
    "skew_concept": "",
    "make_plots": 0,
    "policy": "uncertainty",
    "concept_include": "",
    "concept_exclude": "",
    "blackbox_metrics": str(
        results_dir / "robot_baseline" / "text" / "baseline_text_distilbert_seed1337" / "baseline_dnn_robots_text_distilbert-base-uncased_seed1337_metrics.json"),
    "concept_source": "none",
    "machine_method": "means",
    "concepts_csv": "data/robot_text/concepts/concepts.csv",
    "lf_alpha": 0.5,
    "lf_threshold": 0.5,
    "lf_mode": "soft",
    "lf_ridge": False,
    "lf_ridge_alpha": 1.0,
    "lf_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "lf_device": "cuda" if (
        lambda: hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available())() else "cpu",
    "lf_batch_size": 64,
    "machine_k": 16,
    "machine_soft": 1,
    "machine_seed": 0,
    "machine_upper_bound": 0,
    "run_name": "cbm_text_complete_trainDetected_inferDetected_seed1337",
    "force_rerun": 0,
    "template_difficulty": "hard",
    "test_label_flip": 0.0,
    "intervention_error_mode": "miss",
}


def _csv_list(s: str) -> list[str]:
    s = str(s).strip()
    return [t.strip() for t in s.split(",")] if s else []


def _csv_kv_float(s: str) -> dict:
    out = {}
    if not s: return out
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part: continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except:
            pass
    return out


# --- Hard corpus (LLM-free) helpers ---

def _signals_from_row(row: dict) -> dict:
    head = str(row["head_shape"])
    body = str(row["body_shape"])
    return {
        "head_body_same": (head == body),
        "has_antennae_bool": (str(row["has_antennae"]).lower() == "true"),
        "corners_head": (head == "square"),
        "corners_body": (body == "square"),
        "rounded_head": (head == "round"),
        "rounded_body": (body == "round"),
        "ears_shape": str(row["ears_shape"]),
        "mouth_type": str(row["mouth_type"]),
    }


def _line_matches(sig: dict, cond: dict) -> bool:
    for k, v in cond.items():
        if k == "any":  # wildcard
            continue
        if k not in sig:
            return False
        if isinstance(v, bool):
            if bool(sig[k]) != v:
                return False
        else:
            if str(sig[k]) != str(v):
                return False
    return True


def _load_jsonl(p: Path) -> list[dict]:
    # Accept UTF-8 BOM and either JSONL, a JSON array file, or plain-text corpus
    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"HardCorpus file is empty: {p}")

    # Case 1: whole file is a JSON array
    if text.startswith("["):
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError("Top-level JSON is not a list")
        return arr

    items, plain_lines = [], []

    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        # skip blanks, comments, markdown fences
        if not s or s.startswith("#") or s.startswith("//") or s.startswith("```"):
            continue
        # Try JSONL first
        try:
            items.append(json.loads(s))
        except json.JSONDecodeError:
            # Treat as plain text sentence; wrap later
            plain_lines.append(s)

    if items:
        return items

    if plain_lines:
        # Wrap each plain-text line into a minimal JSON object
        return [{"id": f"pt_{i:04d}", "when": {"any": True}, "text": s}
                for i, s in enumerate(plain_lines, 1)]

    raise ValueError(f"No valid JSON or plain-text lines found in {p}.")


def _nat_from_tokens(row: dict, seed: int) -> dict:
    fp = (
        f"{row['head_shape']}|{row['body_shape']}|{row['foot_shape']}|"
        f"{row['ears_shape']}|{row['mouth_type']}|{row['hand_shape']}|"
        f"{row['has_antennae']}|{row['has_knees']}|{row['has_elbows']}"
    )

    def pick(opts, key):
        h = hashlib.sha256(f"{seed}:{fp}:{key}".encode()).hexdigest()
        return opts[int(h, 16) % len(opts)]

    head_map = {
        "square": ["square", "square-shaped", "boxy", "right-angled", "angular"],
        "round": ["round", "rounded", "dome-shaped", "curved", "circular"],
    }
    body_map = {
        "square": ["square", "square-bodied", "boxy", "right-angled", "angular"],
        "round": ["rounded", "barrel-shaped", "curved", "tubular", "cylindrical"],
    }
    ears_map = {
        "square": ["square", "square-cut", "right-angled", "box-form", "angular"],  # no "boxy"
        "triangle": ["triangular", "three-angled", "pointed", "tri-corner", "tapered"],
    }
    mouth_map = {"closed": ["closed"], "open": ["open"]}
    hands_map = {
        "round_circle": ["round mitts", "round hands", "circular mitts", "circular hands", "rounded mitts"],
        "wide_oval": ["wide ovals", "broad ovals", "oval hands", "broad-oval hands", "wide-oval hands"],
        "tall_oval": ["tall ovals", "long ovals", "elongated ovals", "oval grips", "oval mitts"],
        "edgy_square": ["square-edged grippers", "square claws", "right-angled grippers", "angular grippers",
                        "square clamps"],
        "edgy_triangle": ["triangular grippers", "pointed grippers", "three-angled grippers", "tri-point grippers",
                          "tapered grippers"],
        "edgy_trapezoid": ["trapezoid grippers", "trapezoidal grippers", "trapezoid claws", "angled trapezoids",
                           "trapezoid clamps"],
    }
    feet_map = {
        "flat_4sided": ["flat four-sided pads", "flat four-sided feet", "flat quad pads", "flat quad feet",
                        "flat square pads"],
        "flat_5sided": ["flat five-sided pads", "flat pentagonal pads", "flat five-sided feet", "flat pentagon pads",
                        "flat pentagon feet"],
        "flat_lshaped": ["L-shaped feet", "L-shaped pads", "ell-shaped feet", "ell-shaped pads", "right-angle feet"],
        "pointy_3sided": ["three-point feet", "triangular points", "tri-point feet", "three-tipped feet",
                          "tri-tipped feet"],
        "pointy_4sided": ["four-point feet", "quad-point feet", "four-tipped feet", "quad-tipped feet",
                          "pointed four-sided feet"],
        "pointy_6sided": ["six-point feet", "hex-point feet", "six-tipped feet", "hex-tipped feet", "pointed hex feet"],
    }

    head_nat = pick(head_map[str(row["head_shape"])], "HEAD")
    body_nat = pick(body_map[str(row["body_shape"])], "BODY")
    ears_nat = pick(ears_map[str(row["ears_shape"])], "EARS")
    mouth_nat = pick(mouth_map[str(row["mouth_type"])], "MOUTH")
    hands_nat = pick(hands_map[str(row["hand_shape"])], "HANDS")
    feet_nat = pick(feet_map[str(row["foot_shape"])], "FEET")

    ant_nat = "has antennae" if str(row["has_antennae"]).lower() == "true" else "no antennae"
    knees_nat = "has knees" if str(row["has_knees"]).lower() == "true" else "no knees"
    elbows_nat = "has elbows" if str(row["has_elbows"]).lower() == "true" else "no elbows"

    return {
        "HEAD_NAT": head_nat,
        "BODY_NAT": body_nat,
        "EARS_NAT": ears_nat,
        "MOUTH_NAT": mouth_nat,
        "HANDS_NAT": hands_nat,
        "FEET_NAT": feet_nat,
        "ANT_NAT": ant_nat,
        "KNEES_NAT": knees_nat,
        "ELBOWS_NAT": elbows_nat,
    }


def _core_concept_names() -> list[str]:
    return [
        "head_is_square",
        "body_is_square",
        "has_knees",
        "has_elbows",
        "foot_is_pointy",
        "has_antennae",
        "ears_is_triangle",
        "mouth_is_open",
        "hands_are_pointy",
    ]


def _core_vector_from_row(row: dict) -> np.ndarray:
    def b(x): return str(x).lower()

    return np.array([
        1.0 if str(row["head_shape"]) == "square" else 0.0,
        1.0 if str(row["body_shape"]) == "square" else 0.0,
        1.0 if b(row["has_knees"]) == "true" else 0.0,
        1.0 if b(row["has_elbows"]) == "true" else 0.0,
        1.0 if str(row["foot_shape"]).startswith("pointy_") else 0.0,
        1.0 if b(row["has_antennae"]) == "true" else 0.0,
        1.0 if str(row["ears_shape"]) == "triangle" else 0.0,
        1.0 if str(row["mouth_type"]) == "open" else 0.0,
        1.0 if str(row["hand_shape"]).startswith("edgy_") else 0.0,
    ], dtype=np.float32)


def _build_ds_from_corpus(catalog_df: pd.DataFrame, params, corpus_path: Path, variants_per_row: int, seed: int,
                          row_variants: list[int] | None = None, generic_path: Path | None = None,
                          generic_rate: float = 0.5):
    corpus_spec = _load_jsonl(corpus_path)
    corpus_gen = _load_jsonl(generic_path) if (generic_path is not None and Path(generic_path).is_file()) else []
    names = _core_concept_names()
    classes = [0, 1]
    X, C, y, row_index, ears_generic = [], [], [], [], []
    for i, sr in catalog_df.iterrows():
        row = {k: sr[k] for k in params["concepts"].keys()}
        repeats = int(row_variants[i]) if row_variants is not None else int(variants_per_row)
        for v in range(repeats):
            key = f"{seed}:{i}:{v}:ears_generic"
            h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
            use_gen = (len(corpus_gen) > 0) and ((h % 1000000) < int(max(0.0, min(1.0, float(generic_rate))) * 1000000))
            corpus = corpus_gen if use_gen else corpus_spec
            text = _render_from_corpus(row, corpus, seed + v)
            X.append(text)
            C.append(_core_vector_from_row(row))
            y.append(1 if str(sr["label"]) == "glorp" else 0)
            row_index.append(i)
            ears_generic.append(bool(use_gen))
    X = [str(t) for t in X]
    C = np.stack(C, axis=0).astype(np.float32)
    y = np.asarray(y, dtype=int)
    ds = ConceptDatasetSample(
        X=X, C=C, y=y, meta={"concepts": tuple(names), "classes": (0, 1), "data_type": "text"}
    )
    setattr(ds, "_full", type("Full", (), {"meta": {"row_index": np.asarray(row_index, dtype=int)}}))
    ds.ears_generic_mask = np.asarray(ears_generic, dtype=bool)
    return ds


def _render_from_corpus(row: dict, corpus: list[dict], seed: int) -> str:
    try:
        sig = _signals_from_row(row)
        cand = [it for it in corpus if _line_matches(sig, it.get("when", {}))]
        if not cand:
            cand = corpus
    except Exception:
        cand = corpus
    key = f'{seed}:{row["head_shape"]}:{row["body_shape"]}:{row["foot_shape"]}:{row["ears_shape"]}:{row["mouth_type"]}:{row["hand_shape"]}:{row["has_antennae"]}:{row["has_knees"]}:{row["has_elbows"]}'
    idx = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(cand)
    txt = str(cand[idx].get("text", ""))
    nat = _nat_from_tokens(row, seed)
    for k, v in nat.items():
        ph = "{" + k + "}"
        if ph in txt:
            txt = txt.replace(ph, v)
    raw_map = {
        "head_shape": str(row["head_shape"]),
        "body_shape": str(row["body_shape"]),
        "ears_shape": str(row["ears_shape"]),
        "mouth_type": str(row["mouth_type"]),
        "hand_shape": str(row["hand_shape"]),
        "foot_shape": str(row["foot_shape"]),
        "has_antennae": str(row["has_antennae"]),
        "has_knees": str(row["has_knees"]),
        "has_elbows": str(row["has_elbows"]),
    }
    for k, v in raw_map.items():
        ph = "{" + k + "}"
        if ph in txt:
            txt = txt.replace(ph, v)
    return txt


if psutil.Process(psutil.Process().ppid()).name().lower().startswith("pycharm"):
    args_obj = SimpleNamespace(**settings)
else:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--variant", choices=["perfect", "imperfect"], default=settings["variant"])
    ap.add_argument("--variants-per-row", type=int)
    ap.add_argument("--variants-per-row-minority", type=int, default=0)
    ap.add_argument("--variants-per-row-majority", type=int, default=0)
    ap.add_argument("--minority_mult", type=float, default=1.0)
    ap.add_argument("--train-balance-enable", type=int, default=0)
    ap.add_argument("--train-target-pos-frac", type=float, default=0.5)
    ap.add_argument("--train-target-generic-frac", type=float, default=0.5)
    ap.add_argument("--train-balance-within-label", type=int, default=1)
    ap.add_argument("--val-balance-enable", type=int, default=0)
    ap.add_argument("--val-target-generic-frac", type=float, default=0.5)
    ap.add_argument("--val-balance-within-label", type=int, default=1)
    ap.add_argument("--test-balance-enable", type=int, default=0)
    ap.add_argument("--test-target-generic-frac", type=float, default=0.5)
    ap.add_argument("--test-balance-within-label", type=int, default=1)
    ap.add_argument("--imperfect-strategy", choices=["missing_concepts", "label_prior_shift"],
                    dest="imperfect_strategy", default=settings["imperfect_strategy"])
    ap.add_argument("--heldout-concepts", type=_csv_list, default=settings["heldout_concepts"])
    ap.add_argument("--mask-p", type=float, default=settings["mask_p"])
    ap.add_argument("--test-label-prior", type=str, default=settings["test_label_prior"])
    ap.add_argument("--seed", type=int, default=settings["seed"])
    ap.add_argument("--concept-mode", choices=["hard", "soft"], default=settings["concept_mode"])
    ap.add_argument("--train-on-detected", action="store_true", default=settings["train_on_detected"])
    ap.add_argument("--templates-file", type=str, default=settings["templates_file"])
    ap.add_argument("--label-model-expr", type=str, default=settings["label_model_expr"])
    ap.add_argument("--corr-pair", type=str, default=settings["corr_pair"])
    ap.add_argument("--train-corr", type=float, default=settings["train_corr"])
    ap.add_argument("--test-break", type=float, default=settings["test_break"])
    ap.add_argument("--test-corr", type=float, default=settings["test_corr"])
    ap.add_argument("--budgets", type=str, default=settings["budgets"])
    ap.add_argument("--target-acc-grid", type=str, default=settings["target_acc_grid"])
    ap.add_argument("--target-acc-concepts", type=str, default=settings["target_acc_concepts"])
    ap.add_argument("--intervene-allow", type=str, default=settings["intervene_allow"])
    ap.add_argument("--human-acc", type=float, default=settings["human_acc"])
    ap.add_argument("--human-acc-concepts", type=str, default=settings["human_acc_concepts"])
    ap.add_argument("--skew-concept", type=str, default=settings["skew_concept"])
    ap.add_argument("--make-plots", type=int, default=settings["make_plots"])
    ap.add_argument("--policy", choices=["uncertainty", "oracle"], default=settings["policy"])
    ap.add_argument("--concept-include", type=str, default=settings["concept_include"])
    ap.add_argument("--concept-exclude", type=str, default=settings["concept_exclude"])
    ap.add_argument("--blackbox_metrics", type=str, default=settings["blackbox_metrics"])
    ap.add_argument("--human-alone", type=float, default=0.75)
    ap.add_argument("--concept-source", type=str, choices=["gt", "detected", "machine", "human", "none"],
                    default=settings["concept_source"])
    ap.add_argument("--machine-method", type=str, choices=["means", "lfcbm"],
                    default=settings["machine_method"])
    ap.add_argument("--concepts-csv", type=str, default=settings.get("concepts_csv", ""))
    ap.add_argument("--lf-alpha", type=float, default=settings.get("lf_alpha", 0.0))
    ap.add_argument("--lf-threshold", type=float, default=settings.get("lf_threshold", 0.0))
    ap.add_argument("--lf-mode", type=str, choices=["hard", "soft"], default=settings.get("lf_mode", "hard"))
    ap.add_argument("--lf-ridge", action="store_true", default=settings.get("lf_ridge", False))
    ap.add_argument("--lf-ridge-alpha", type=float, default=settings.get("lf_ridge_alpha", 0.0))
    ap.add_argument("--lf-encoder", type=str, default=settings.get("lf_encoder", ""))
    ap.add_argument("--lf-device", type=str, default=settings.get("lf_device", "cpu"))
    ap.add_argument("--lf-batch-size", type=int, default=settings.get("lf_batch_size", 32))
    ap.add_argument("--machine-k", type=int, default=settings["machine_k"])
    ap.add_argument("--machine-soft", type=int, default=settings["machine_soft"])
    ap.add_argument("--machine-seed", type=int, default=settings["machine_seed"])
    ap.add_argument("--machine-upper-bound", type=int, default=settings["machine_upper_bound"])
    ap.add_argument("--mask-mode", choices=["rowdrop", "mask"], default=settings["mask_mode"])
    ap.add_argument("--mask-rate", type=float, default=settings["mask_rate"])
    ap.add_argument("--run-name", type=str, default=settings["run_name"])
    ap.add_argument("--force-rerun", type=int, default=settings["force_rerun"])
    ap.add_argument("--template-difficulty", choices=["easy", "medium", "hard"],
                    default=settings["template_difficulty"])
    ap.add_argument("--test-miss", type=str, default="")
    ap.add_argument("--test-miss-rate", type=float, default=0.0)
    ap.add_argument("--test-miss-mode", choices=["drop_cols", "soft0.5", "hard0"], default="soft0.5")
    ap.add_argument("--intervention-error-mode", choices=["miss", "flip", "both"],
                    default=settings["intervention_error_mode"])
    ap.add_argument("--reuse-detector", type=int, default=0)
    ap.add_argument("--detector-model", type=str, default="")
    ap.add_argument("--skip-fit", type=int, default=0)
    ap.add_argument("--test-label-flip", type=float, default=0.0)
    ap.add_argument("--label-model-type", choices=["deterministic", "stochastic"], default="deterministic")
    ap.add_argument("--label-model-alpha", type=float, default=10.0)
    ap.add_argument("--label-model-bias", type=float, default=1.0)
    ap.add_argument("--concept-label-noise-mode", choices=["none", "subjective", "machine"], default="none")
    ap.add_argument("--concept-label-noise-rate", type=float, default=0.20)
    ap.add_argument("--concept-label-noise-confusion", type=str, default="")
    ap.add_argument("--redact-concepts", type=str, default="")
    ap.add_argument("--redact-splits", type=str, default="")
    ap.add_argument("--generic-rate", type=float, default=0.5)
    ap.add_argument("--generic-tol", type=float, default=0.02)
    ap.add_argument("--generic-enable", type=int, default=0)

    known, _ = ap.parse_known_args()
    merged = dict(settings)
    merged.update({
        "variant": known.variant,
        "variants_per_row": int(known.variants_per_row),
        "imperfect_strategy": known.imperfect_strategy,
        "heldout_concepts": known.heldout_concepts if isinstance(known.heldout_concepts, list) else _csv_list(
            known.heldout_concepts),
        "mask_p": known.mask_p,
        "test_label_prior": known.test_label_prior,
        "seed": known.seed,
        "concept_mode": known.concept_mode,
        "train_on_detected": bool(known.train_on_detected),
        "templates_file": known.templates_file or "",
        "label_model_expr": known.label_model_expr or "",
        "corr_pair": known.corr_pair or "",
        "train_corr": float(known.train_corr),
        "test_break": float(known.test_break),
        "test_corr": float(known.test_corr),
        "budgets": known.budgets,
        "target_acc_grid": known.target_acc_grid,
        "target_acc_concepts": known.target_acc_concepts,
        "intervene_allow": known.intervene_allow,
        "human_acc": float(known.human_acc),
        "human_acc_concepts": known.human_acc_concepts,
        "skew_concept": known.skew_concept,
        "make_plots": int(known.make_plots),
        "policy": known.policy,
        "concept_include": known.concept_include,
        "concept_exclude": known.concept_exclude,
        "blackbox_metrics": known.blackbox_metrics,
        "human_alone": float(getattr(known, "human_alone", 0.75)),
        "concept_source": known.concept_source,
        "machine_method": known.machine_method,
        "machine_k": int(known.machine_k),
        "machine_soft": int(known.machine_soft),
        "machine_seed": int(known.machine_seed),
        "machine_upper_bound": int(known.machine_upper_bound),
        "mask_mode": known.mask_mode,
        "mask_rate": float(known.mask_rate),
        "run_name": (known.run_name or "").strip(),
        "force_rerun": int(known.force_rerun),
        "template_difficulty": known.template_difficulty,
        "test_miss": known.test_miss,
        "test_miss_rate": float(known.test_miss_rate),
        "test_miss_mode": known.test_miss_mode,
        "reuse_detector": int(known.reuse_detector),
        "detector_model": known.detector_model or "",
        "label_model_type": known.label_model_type,
        "label_model_alpha": float(known.label_model_alpha),
        "label_model_bias": float(known.label_model_bias),
        "concept_label_noise_mode": known.concept_label_noise_mode,
        "concept_label_noise_rate": float(known.concept_label_noise_rate),
        "concept_label_noise_confusion": known.concept_label_noise_confusion or "",
        "variants_per_row_minority": int(getattr(known, "variants_per_row_minority", 0)),
        "variants_per_row_majority": int(getattr(known, "variants_per_row_majority", 0)),
        "minority_mult": float(getattr(known, "minority_mult", 1.0)),
        "redact_concepts": known.redact_concepts or "",
        "redact_splits": known.redact_splits or "",
        "generic_enable": int(getattr(known, "generic_enable", 0)),
        "generic_rate": float(getattr(known, "generic_rate", 0.5)),
        "generic_tol": float(getattr(known, "generic_tol", 0.02)),
    })
    if merged["test_corr"] is not None and merged["test_corr"] >= 0:
        merged["test_break"] = max(0.0, min(1.0, 1.0 - float(merged["test_corr"])))
    args_obj = SimpleNamespace(**merged)

template_file_name = "Templates.txt" if args_obj.template_difficulty == "medium" else "Templates_simple.txt"
tpl_path = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / template_file_name
with open(tpl_path, "r", encoding="utf-8-sig") as f:
    templates = [ln.strip() for ln in f if ln.strip()]

if args_obj.templates_file:
    with open(Path(args_obj.templates_file), "r", encoding="utf-8-sig") as f:
        templates = [ln.strip() for ln in f if ln.strip()]

VARIANT = args_obj.variant
IMPERFECT_STRATEGY = args_obj.imperfect_strategy
HELDOUT_CONCEPTS = args_obj.heldout_concepts
MASK_P = float(args_obj.mask_p)
TEST_LABEL_PRIOR = args_obj.test_label_prior
SEED = int(args_obj.seed)
CONCEPT_MODE = args_obj.concept_mode
train_on_detected = bool(args_obj.train_on_detected)

params = {
    "concepts": {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "foot_shape": ["flat_4sided", "flat_5sided", "flat_lshaped", "pointy_3sided", "pointy_4sided", "pointy_6sided"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": ["round_circle", "wide_oval", "tall_oval", "edgy_square", "edgy_triangle", "edgy_trapezoid"],
    },
    "model": "'glorp' if (min(int(row[\"ears_shape\"]==\"square\"), int(row[\"body_shape\"]==\"square\")) >= 1) else 'drent'",
}

if args_obj.label_model_expr:
    params["model"] = args_obj.label_model_expr

if not args_obj.label_model_expr:
    _mdl = str(params.get("model", ""))
    if ("foot_shape" in _mdl) and ("has_antennae" not in _mdl):
        print(
            "WARNING: label_model_expr not provided; using DEFAULT rule that includes foot_shape (and not antennae). "
            "Pass --label-model-expr to avoid unintended accuracy.")


def enumerate_concepts(params, shuffle=True, seed=0):
    cols = list(params["concepts"].keys())
    grids = [params["concepts"][c] for c in cols]
    combos = list(product(*grids))
    df = pd.DataFrame(combos, columns=cols)
    if shuffle:
        rng = np.random.default_rng(seed)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df


def sample_concepts(params, n=50, seed=0):
    rng = np.random.default_rng(seed)
    cols = list(params["concepts"].keys())
    rows = []
    for _ in range(n):
        r = {}
        for c in cols:
            r[c] = rng.choice(params["concepts"][c])
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)


def _indices_for_names(names, specs):
    idx = []
    for tok in [t.strip() for t in str(specs).split(",") if t.strip()]:
        idx.extend(_indices_for(names, tok))
    return sorted(set(idx))


def _apply_test_missing(H, names, specs, rate, mode, seed):
    if not specs or float(rate) <= 0:
        return H, names, None
    rng = np.random.default_rng(int(seed) + 777)
    idx = _indices_for_names(names, specs) if specs != "*" else list(range(H.shape[1]))

    if mode == "drop_cols":
        keep = [j for j in range(H.shape[1]) if j not in idx or rng.random() >= rate]
        if not keep:
            keep = list(range(H.shape[1]))
        H2 = H[:, keep]
        names2 = [names[j] for j in keep]
        meta = {
            "mode": "drop_cols",
            "specs": str(specs),
            "rate": float(rate),
            "kept_cols": names2,
            "dropped_cols": [names[j] for j in range(H.shape[1]) if j not in keep],
        }
        return H2, names2, meta

    m = rng.random((H.shape[0], len(idx))) < rate
    for t, j in enumerate(idx):
        if mode == "soft0.5":
            H[:, j] = np.where(m[:, t], 0.5, H[:, j])
        else:
            H[:, j] = np.where(m[:, t], 0, H[:, j])
    realized = {names[j]: float(m[:, t].mean()) for t, j in enumerate(idx)}
    meta = {
        "mode": str(mode),
        "specs": str(specs),
        "rate": float(rate),
        "masked_cols": [names[j] for j in idx],
        "realized": realized,
    }
    return H, names, meta


def compute_label(df: pd.DataFrame, model_expr: str,
                  label_model_type: str = "deterministic",
                  alpha: float = 1.0, bias: float = 0.0, seed: int = 0,
                  noise_rate: float = 0.0, flip_probs: tuple[float, float] | None = None) -> pd.Series:
    SAFE_GLOBALS = {"__builtins__": None, "int": int, "str": str, "float": float, "bool": bool,
                    "any": any, "all": all, "np": np, "min": min, "max": max}
    rng = np.random.default_rng(int(seed))

    def _cond_to_score(expr: str) -> str | None:
        m = re.search(r"\bif\s+(?P<cond>.+?)\s+else\b", expr)
        cond = m.group("cond").strip() if m else expr.strip()
        m2 = re.search(r"^(?P<lhs>.+?)(?:\s*(?:>=|<=|>|<)\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*$",
                       cond, flags=re.IGNORECASE)
        lhs = m2.group("lhs").strip() if m2 else cond
        while lhs.startswith("(") and lhs.endswith(")"):
            lvl = 0;
            ok = True
            for ch in lhs:
                if ch == "(":
                    lvl += 1
                elif ch == ")":
                    lvl -= 1
                    if lvl < 0: ok = False; break
            if ok and lvl == 0:
                lhs = lhs[1:-1].strip()
            else:
                break
        return lhs or None

    score_expr = _cond_to_score(model_expr) if label_model_type == "stochastic" else None

    def eval_one(sr):
        row = sr.to_dict()
        if label_model_type is None or label_model_type == "deterministic":
            return eval(model_expr, SAFE_GLOBALS, {"row": row})
        score = None
        if score_expr:
            try:
                score = float(eval(score_expr, SAFE_GLOBALS, {"row": row}))
            except Exception:
                score = None
        if score is None:
            try:
                hard = eval(model_expr, SAFE_GLOBALS, {"row": row})
                score = 1.0 if str(hard).strip().lower() == "glorp" else 0.0
            except Exception:
                score = 0.0
        p = 1.0 / (1.0 + float(np.exp(-float(alpha) * (float(score) - float(bias)))))
        return "glorp" if rng.random() < p else "drent"

    labels = df.apply(eval_one, axis=1).astype(str)

    if flip_probs is not None:
        a, b = float(flip_probs[0]), float(flip_probs[1])
        out = []
        for lab in labels:
            if lab == "glorp":
                out.append("drent" if (rng.random() < a) else "glorp")
            else:
                out.append("glorp" if (rng.random() < b) else ("drent" if lab == "drent" else lab))
        return pd.Series(out, index=labels.index)

    if noise_rate and float(noise_rate) > 0.0:
        q = float(noise_rate)
        flips = rng.random(labels.shape[0]) < q
        flipped = labels.copy()
        flipped.iloc[flips] = flipped.iloc[flips].map(lambda L: "drent" if L == "glorp" else "glorp")
        return flipped

    return labels


def _load_detector_from_path(p):
    p = Path(p)
    if p.is_dir():
        cands = sorted([q for q in p.glob("cbm_fe_*_robots_text_*.pkl")])
        if not cands:
            raise FileNotFoundError(f"No cbm_fe_*.pkl under {p}")
        p = cands[0]
    obj = load_obj(str(p))
    if not isinstance(obj, dict) or "detector" not in obj:
        raise ValueError(f"File does not contain a detector: {p}")
    return obj["detector"]


def _subset_sample(sample: ConceptDatasetSample, keep_idx: np.ndarray, concepts, classes) -> ConceptDatasetSample:
    keep_idx = np.asarray(keep_idx, dtype=int)
    X = [str(x) for x in np.array(sample.X, dtype=object)[keep_idx]]
    C = sample.C[keep_idx]
    y = sample.y[keep_idx]
    return ConceptDatasetSample(X=X, C=C, y=y, meta={"concepts": concepts, "classes": classes, "data_type": "text"})


def _apply_missing_concepts(train_sample: ConceptDatasetSample, concepts: list[str], heldout: list[str], mask_p: float,
                            seed: int) -> ConceptDatasetSample:
    if not heldout:
        return train_sample
    name_to_idx = {n: i for i, n in enumerate(concepts)}
    cols = []
    for spec in heldout:
        if spec in name_to_idx:
            cols.append(name_to_idx[spec]);
            continue
        key = spec.split("=", 1)[0].strip()
        cols.extend([i for i, n in enumerate(concepts) if n.startswith(key)])
    cols = sorted(set(cols))
    if not cols:
        return train_sample
    C = train_sample.C.astype(np.float32)
    active = (C[:, cols] > 0.5).any(axis=1)
    if mask_p >= 1.0:
        keep = ~active
    else:
        rng = np.random.default_rng(seed)
        drop = active & (rng.random(active.shape[0]) < mask_p)
        keep = ~drop
    keep_idx = np.where(keep)[0]
    if keep_idx.size == 0:
        keep_idx = np.where(~active)[0]
    return _subset_sample(train_sample, keep_idx, concepts, train_sample.meta.get("classes", []))


def _parse_label_prior(spec: str, classes) -> dict:
    if not spec:
        return {}
    out = {}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        k, v = p.split(":")
        k = k.strip();
        v = float(v.strip())
        if k.isdigit():
            k = int(k)
        else:
            if k in classes:
                k = int(np.where(np.array(classes, dtype=object) == k)[0][0])
            else:
                continue
        out[k] = v
    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] /= s
    return out


def _apply_label_prior_shift(val_sample: ConceptDatasetSample, prior: dict, seed: int) -> ConceptDatasetSample:
    if not prior:
        return val_sample
    rng = np.random.default_rng(seed)
    y = val_sample.y.astype(int)
    classes = sorted(np.unique(y).tolist())
    n = len(y)
    target_counts = {c: int(round(prior.get(c, (y == c).mean()) * n)) for c in classes}
    chosen_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        k = target_counts[c]
        if idx.size == 0:
            continue
        if k <= idx.size:
            sel = rng.choice(idx, size=k, replace=False)
        else:
            sel = rng.choice(idx, size=k, replace=True)
        chosen_idx.append(sel)
    if not chosen_idx:
        return val_sample
    keep_idx = np.concatenate(chosen_idx)
    rng.shuffle(keep_idx)
    return _subset_sample(val_sample, keep_idx, val_sample.meta.get("concepts", []), val_sample.meta.get("classes", []))


def _group_indices(names, key):
    return [i for i, n in enumerate(names) if n.startswith(key + "=")]


def _corr_equal_mask(sample: ConceptDatasetSample, a: str, b: str) -> np.ndarray:
    names = list(sample.concepts)
    ai = _group_indices(names, a)
    bi = _group_indices(names, b)
    T = sample.C.astype(int)
    aa = T[:, ai].argmax(1)
    bb = T[:, bi].argmax(1)
    return aa == bb


def _enforce_corr(sample: ConceptDatasetSample, pair: str, frac_corr: float, seed: int) -> ConceptDatasetSample:
    if not pair:
        return sample
    a, b = [t.strip() for t in pair.split(",")]
    m = _corr_equal_mask(sample, a, b)
    idx_corr = np.where(m)[0]
    idx_brk = np.where(~m)[0]
    n = sample.n
    n_corr = min(idx_corr.size, int(round(frac_corr * n)))
    n_brk = min(idx_brk.size, n - n_corr)
    rng = np.random.default_rng(seed)
    sel_corr = rng.choice(idx_corr, size=n_corr, replace=False) if n_corr > 0 else np.array([], dtype=int)
    sel_brk = rng.choice(idx_brk, size=n_brk, replace=False) if n_brk > 0 else np.array([], dtype=int)
    keep = np.concatenate([sel_corr, sel_brk])
    rng.shuffle(keep)
    return _subset_sample(sample, keep, list(sample.concepts), sample.meta.get("classes", []))


def _parse_target_acc_grid(s: str):
    out = []
    for tok in _csv_list(s):
        t = tok.strip().lower()
        if t in ("true", "raw", "none", "as_is"):
            out.append("raw")
            continue
        try:
            out.append(float(t))
        except:
            pass
    return out


def _degrade_to_acc(H: np.ndarray, T: np.ndarray, target: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = H.copy()
    ok = (H == T).reshape(-1)
    n = ok.size
    cur = int(ok.sum())
    want = int(round(float(target) * n))
    if want == cur:
        return H
    if want < cur:
        k = cur - want
        idx = np.where(ok)[0]
        if k > idx.size: k = idx.size
        if k > 0:
            sel = rng.choice(idx, size=k, replace=False)
            flat = H.reshape(-1)
            flat[sel] = 1 - flat[sel]
            H = flat.reshape(H.shape)
        return H
    k = want - cur
    idx = np.where(~ok)[0]
    if k > idx.size: k = idx.size
    if k > 0:
        flatH = H.reshape(-1)
        flatT = T.reshape(-1)
        sel = rng.choice(idx, size=k, replace=False)
        flatH[sel] = flatT[sel]
        H = flatH.reshape(H.shape)
    return H


def _indices_for(names, spec):
    out = []
    spec = str(spec).strip()
    if "=" in spec:
        if spec in names:
            out.append(names.index(spec))
    else:
        out.extend([i for i, n in enumerate(names)
                    if (n == spec) or n.startswith(spec) or n.startswith(spec + "=")])
    return sorted(set(out))


def _apply_per_concept_degrade(H: np.ndarray, T: np.ndarray, names: list[str], mapping: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = H.copy()
    for k, v in mapping.items():
        idxs = _indices_for(names, k)
        if not idxs: continue
        for j in idxs:
            hj = out[:, j]
            tj = T[:, j]
            ok = (hj == tj)
            n = ok.size
            cur = int(ok.sum())
            want = int(round(float(v) * n))
            if want == cur:
                continue
            if want < cur:
                flip = cur - want
                idx = np.where(ok)[0]
                if flip > idx.size: flip = idx.size
                if flip > 0:
                    sel = rng.choice(idx, size=flip, replace=False)
                    hj2 = hj.copy()
                    hj2[sel] = 1 - hj2[sel]
                    out[:, j] = hj2
            else:
                fix = want - cur
                idx = np.where(~ok)[0]
                if fix > idx.size: fix = idx.size
                if fix > 0:
                    hj2 = hj.copy()
                    sel = rng.choice(idx, size=fix, replace=False)
                    hj2[sel] = tj[sel]
                    out[:, j] = hj2
    return out


def _apply_human_edit(p_row, truth_row, sel_idxs, names, acc_default, acc_map, rng, mode="miss"):
    for j in sel_idxs:
        name = names[j]
        base = name.split("=", 1)[0]
        acc = acc_map.get(name, acc_map.get(base, acc_default))
        u = rng.random()
        if u < acc:
            p_row[j] = truth_row[j]
        else:
            if mode == "flip":
                v = int(truth_row[j])
                p_row[j] = 0 if v == 1 else 1
    return p_row


def _allowed_indices(names, allow_spec):
    if not allow_spec: return np.arange(len(names), dtype=int)
    out = []
    for tok in _csv_list(allow_spec):
        out.extend(_indices_for(names, tok))
    return np.array(sorted(set(out)), dtype=int)


def _apply_skew(sample: ConceptDatasetSample, spec: str, seed: int) -> ConceptDatasetSample:
    if not spec: return sample
    parts = [t.strip() for t in spec.split(",")]
    if len(parts) != 3: return sample
    key, val, p = parts[0], parts[1], float(parts[2])
    names = list(sample.concepts)
    idxs = _indices_for(names, f"{key}={val}")
    if not idxs: return sample
    j = idxs[0]
    T = sample.C.astype(int)
    idx_pos = np.where(T[:, j] == 1)[0]
    idx_neg = np.where(T[:, j] == 0)[0]
    n = sample.n
    n_pos = min(idx_pos.size, int(round(p * n)))
    n_neg = min(idx_neg.size, n - n_pos)
    rng = np.random.default_rng(seed)
    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    keep = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(keep)
    return _subset_sample(sample, keep, names, sample.meta.get("classes", []))


def _select_concept_columns(sample: ConceptDatasetSample, keep_idx: np.ndarray) -> ConceptDatasetSample:
    keep_idx = np.asarray(keep_idx, dtype=int)
    X = sample.X
    C = sample.C[:, keep_idx]
    y = sample.y
    names = list(sample.concepts)
    keep_names = [names[i] for i in keep_idx]
    return ConceptDatasetSample(X=X, C=C, y=y, meta={"concepts": keep_names, "classes": sample.meta.get("classes", []),
                                                     "data_type": "text"})


def _tfidf_fit(texts, seed):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, dtype=np.float32)
    X = vec.fit_transform([str(t) for t in texts])
    return vec, X


def _kmeans_fit(X, k, seed):
    km = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed))
    km.fit(X)
    return km


def _kmeans_soft(X, km):
    D = km.transform(X)
    S = np.exp(-D)
    S_sum = S.sum(axis=1, keepdims=True)
    S_sum[S_sum == 0] = 1.0
    return S / S_sum


def _machine_truth_map(H_train_hard, C_train_true):
    J = H_train_hard.shape[1]
    mapping = []
    for j in range(J):
        col = H_train_hard[:, j].astype(int)
        best = 0
        best_idx = 0
        for t in range(C_train_true.shape[1]):
            c = C_train_true[:, t].astype(int)
            agree = int(((col == 1) & (c == 1)).sum() + ((col == 0) & (c == 0)).sum())
            if agree > best:
                best = agree;
                best_idx = t
        mapping.append(best_idx)
    return np.array(mapping, dtype=int)


cols = list(params["concepts"].keys())
catalog_df = pd.DataFrame([dict(zip(cols, vals)) for vals in product(*[params["concepts"][c] for c in cols])],
                          columns=cols)
catalog_df["label"] = compute_label(
    catalog_df,
    params["model"],
    label_model_type=merged.get("label_model_type", "deterministic"),
    alpha=float(merged.get("label_model_alpha", 1.0)),
    bias=float(merged.get("label_model_bias", 0.0)),
    seed=int(merged.get("seed", 0)),
)

_lbl = catalog_df["label"].astype(str)
print("Label distribution (catalog_df):", {
    "glorp": int((_lbl == "glorp").sum()),
    "drent": int((_lbl == "drent").sum()),
    "total": int(len(_lbl)),
    "pos_frac": round((_lbl == "glorp").mean(), 4),
})

concept_cols = list(params["concepts"].keys())
ds = None

if args_obj.templates_file and str(args_obj.templates_file).lower().endswith(".jsonl"):
    _lbl = catalog_df["label"].astype(str)
    n_pos = int((_lbl == "glorp").sum());
    n_neg = int((_lbl == "drent").sum())
    minority_label = "glorp" if n_pos < n_neg else ("drent" if n_neg < n_pos else "glorp")
    base_vpr = int(args_obj.variants_per_row)
    vpr_min = int(args_obj.variants_per_row_minority) if int(args_obj.variants_per_row_minority) > 0 else max(1,
                                                                                                              int(round(
                                                                                                                  base_vpr * float(
                                                                                                                      args_obj.minority_mult))))
    vpr_maj = int(args_obj.variants_per_row_majority) if int(args_obj.variants_per_row_majority) > 0 else base_vpr
    row_variants = [vpr_min if lab == minority_label else vpr_maj for lab in _lbl]
    base_jsonl = Path(args_obj.templates_file)
    gen_jsonl = base_jsonl.with_name("HardCorpus_EarsGeneric.jsonl")
    ds = _build_ds_from_corpus(catalog_df, params, base_jsonl, base_vpr, SEED, row_variants=row_variants, generic_path=(
        gen_jsonl if (gen_jsonl.is_file() and int(getattr(args_obj, "generic_enable", 0)) == 1) else None),
                               generic_rate=(float(getattr(args_obj, "generic_rate", 0.5)) if int(
                                   getattr(args_obj, "generic_enable", 0)) == 1 else 0.0))
    setattr(ds, "cvindices", None)

elif args_obj.template_difficulty == "hard":
    default_jsonl = pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "HardCorpus.jsonl"
    print(f"Using hard corpus: {default_jsonl}")
    if default_jsonl.is_file():
        _lbl = catalog_df["label"].astype(str)
        n_pos = int((_lbl == "glorp").sum());
        n_neg = int((_lbl == "drent").sum())
        minority_label = "glorp" if n_pos < n_neg else ("drent" if n_neg < n_pos else "glorp")
        base_vpr = int(args_obj.variants_per_row)
        vpr_min = int(args_obj.variants_per_row_minority) if int(args_obj.variants_per_row_minority) > 0 else max(1,
                                                                                                                  int(round(
                                                                                                                      base_vpr * float(
                                                                                                                          args_obj.minority_mult))))
        vpr_maj = int(args_obj.variants_per_row_majority) if int(args_obj.variants_per_row_majority) > 0 else base_vpr
        row_variants = [vpr_min if lab == minority_label else vpr_maj for lab in _lbl]
        gen_jsonl = default_jsonl.with_name("HardCorpus_EarsGeneric.jsonl")
        ds = _build_ds_from_corpus(catalog_df, params, default_jsonl, base_vpr, SEED, row_variants=row_variants,
                                   generic_path=(gen_jsonl if (gen_jsonl.is_file() and int(
                                       getattr(args_obj, "generic_enable", 0)) == 1) else None), generic_rate=(
                float(getattr(args_obj, "generic_rate", 0.5)) if int(
                    getattr(args_obj, "generic_enable", 0)) == 1 else 0.0))
        setattr(ds, "cvindices", None)

    # Fallback to legacy template mechanism if not using hard-corpus
if ds is None:
    llm_user_prompt = "Using the provided attributes, write a natural spoken description (1–3 sentences) that sounds like a person describing an image they saw. Do not invent locations or scenarios; focus only on what the attributes imply."
    ds = create_synthetic_dataset(
        source=catalog_df,
        templates=templates,
        variants_per_row=int(args_obj.variants_per_row),
        include_color=False,
        rng_seed=SEED,
        concept_cols=concept_cols,
        label_col="label",
        label_map={"drent": 0, "glorp": 1},
        text_mode="semi",
        llm_provider="gemini",
        llm_model="gemini-1.5-flash",
        llm_user_prompt=llm_user_prompt,
    )

ds_path = ROBOT_DATA_DIR / "robot_text_dataset.pkl"
save_obj(ds, ds_path, overwrite=True)

print("SAMPLE CAPTIONS:")
for x in ds.X[:6]:
    print("-", x)

print("\nCONCEPT NAMES:", ds.concepts)
print("CLASSES:", ds.classes)
print("N samples:", len(ds))

_y_all = np.asarray(ds.y, dtype=int)
print("Label distribution (dataset):", {
    "glorp": int((_y_all == 1).sum()),
    "drent": int((_y_all == 0).sum()),
    "total": int(_y_all.size),
    "pos_frac": round((_y_all == 1).mean() if _y_all.size else 0.0, 4),
})

seed_tag = f"seed{SEED}"
if VARIANT == "imperfect" and IMPERFECT_STRATEGY == "missing_concepts":
    if args_obj.mask_mode == "mask":
        miss_tag = f"mcar{int(round(float(args_obj.mask_rate) * 100)):03d}"
    else:
        miss_tag = f"mnar{int(round(float(MASK_P) * 100)):03d}"
else:
    miss_tag = "complete"
fe_tag = "detected" if train_on_detected else "gt"
run_slug = f"robots_text_{miss_tag}_{seed_tag}_{fe_tag}"

run_name = (args_obj.run_name or "").strip()
if run_name:
    run_dir = ROBOT_RUN_DIR / run_name
else:
    run_dir = ROBOT_RUN_DIR / f"{run_slug}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)

fe_src_tag = "fe_detected" if train_on_detected else "fe_gt"
model_path = run_dir / f"cbm_{fe_src_tag}_robots_text_{miss_tag}_{seed_tag}.pkl"
metrics_path = run_dir / f"metrics_cbm_{fe_src_tag}_robots_text_{miss_tag}_{seed_tag}.json"
meta_path = run_dir / f"meta_cbm_{fe_src_tag}_robots_text_{miss_tag}_{seed_tag}.json"

if (not int(args_obj.reuse_detector)) and Path(model_path).is_file() and Path(metrics_path).is_file() and Path(
        meta_path).is_file() and int(getattr(args_obj, "force_rerun", 0)) == 0:
    print("Using cached run")
    print("Saved dataset:", ds_path)
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    import sys;

    sys.exit(0)
out_csv = run_dir / f"text_samples_{miss_tag}_{seed_tag}.csv"
row_index = getattr(getattr(ds, "_full", ds), "meta", {}).get("row_index", np.arange(len(ds)))
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    cols = ["text"] + concept_cols + ["label"]
    w.writerow(cols)
    for i, x in enumerate(ds.X):
        src_idx = int(row_index[i])
        row_vals = catalog_df.loc[src_idx, concept_cols].tolist()
        w.writerow([x] + row_vals + [catalog_df.loc[src_idx, "label"]])
cv = getattr(ds, "cvindices", None)
split_fold = cv["by_robot"] if isinstance(cv, dict) and "by_robot" in cv else None
if isinstance(split_fold, np.ndarray):
    for name, code in [("train", 2), ("val", 0), ("test", 1)]:
        idx = np.where(split_fold == code)[0]
        p = run_dir / f"text_samples_{name}_{miss_tag}_{seed_tag}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f);
            w.writerow(["text"] + concept_cols + ["label"])
            for i in idx:
                src_idx = int(row_index[i])
                row_vals = catalog_df.loc[src_idx, concept_cols].tolist()
                w.writerow([str(ds.X[i])] + row_vals + [catalog_df.loc[src_idx, "label"]])
print(f"\nWrote {len(ds)} rows to {out_csv}")

# --- Split that works for both dataset classes (with/without .split) ---
row_index = getattr(getattr(ds, "_full", ds), "meta", {}).get("row_index", np.arange(len(ds)))


def _manual_by_robot_split(ds_obj, row_index_arr, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    base_ids = np.unique(row_index_arr)
    rng.shuffle(base_ids)

    n_ids = len(base_ids)
    n_val = int(np.floor(0.15 * n_ids))
    n_te = int(np.floor(0.15 * n_ids))
    val_ids = set(base_ids[:n_val])
    te_ids = set(base_ids[n_val:n_val + n_te])

    lab_by_id = {}
    y_all = np.asarray(ds_obj.y, dtype=int)
    for i, rid in enumerate(row_index_arr):
        r = int(rid)
        if r not in lab_by_id:
            lab_by_id[r] = int(y_all[i])

    tr_ids = set(base_ids) - val_ids - te_ids
    cls_tr = {lab_by_id[r] for r in tr_ids}
    if len(cls_tr) < 2:
        want = 1 - list(cls_tr)[0] if len(cls_tr) == 1 else 1
        pool = [r for r in list(val_ids) + list(te_ids) if lab_by_id[r] == want]
        if pool:
            add = pool[0]
            rem = next(r for r in tr_ids if lab_by_id[r] != want)
            if add in val_ids:
                val_ids.remove(add);
                val_ids.add(rem)
            else:
                te_ids.remove(add);
                te_ids.add(rem)
            tr_ids.remove(rem);
            tr_ids.add(add)

    fold_arr = np.empty(len(row_index_arr), dtype=int)
    for i, rid in enumerate(row_index_arr):
        r = int(rid)
        fold_arr[i] = 0 if r in val_ids else (1 if r in te_ids else 2)

    def _subset_mask(mask):
        idx = np.where(mask)[0]
        X = [ds_obj.X[i] for i in idx]
        C = ds_obj.C[mask]
        y = ds_obj.y[mask]
        sub = ConceptDatasetSample(
            X=X, C=C, y=y,
            meta={"concepts": ds_obj.concepts, "classes": ds_obj.classes, "data_type": "text"}
        )
        gm = getattr(ds_obj, "ears_generic_mask", None)
        if gm is not None:
            setattr(sub, "ears_generic_mask", np.asarray(gm)[idx])
        return sub

    val_mask = (fold_arr == 0)
    test_mask = (fold_arr == 1)
    train_mask = ~(val_mask | test_mask)

    ds_obj.training = _subset_mask(train_mask)
    ds_obj.validation = _subset_mask(val_mask)
    ds_obj.test = _subset_mask(test_mask)

    try:
        ds_obj.cvindices = {"by_robot": fold_arr}
    except Exception:
        pass


need_split = (
        getattr(ds, "cvindices", None) is None
        or getattr(getattr(ds, "validation", SimpleNamespace(n=0)), "n", 0) == 0
        or getattr(ds, "test", None) is None
)

if hasattr(ds, "split") and need_split:
    rng = np.random.default_rng(0)
    base_ids = np.unique(row_index)
    rng.shuffle(base_ids)
    n = len(base_ids)
    n_val = int(np.floor(0.15 * n))
    n_test = int(np.floor(0.15 * n))
    val_ids = set(base_ids[:n_val])
    test_ids = set(base_ids[n_val:n_val + n_test])

    lab_by_id = {}
    y_all = np.asarray(ds.y, dtype=int)
    for i, rid in enumerate(row_index):
        r = int(rid)
        if r not in lab_by_id:
            lab_by_id[r] = int(y_all[i])

    train_ids = set(base_ids) - val_ids - test_ids
    cls_tr = {lab_by_id[r] for r in train_ids}
    if len(cls_tr) < 2:
        want = 1 - list(cls_tr)[0] if len(cls_tr) == 1 else 1
        pool = [r for r in list(val_ids) + list(test_ids) if lab_by_id[r] == want]
        if pool:
            add = pool[0]
            rem = next(r for r in train_ids if lab_by_id[r] != want)
            if add in val_ids:
                val_ids.remove(add);
                val_ids.add(rem)
            else:
                test_ids.remove(add);
                test_ids.add(rem)
            train_ids.remove(rem);
            train_ids.add(add)

    fold_arr = np.empty(len(row_index), dtype=int)
    for i, rid in enumerate(row_index):
        r = int(rid)
        fold_arr[i] = 0 if r in val_ids else (1 if r in test_ids else 2)

    if getattr(ds, "cvindices", None) is None:
        ds.cvindices = {}
    ds.cvindices["by_robot"] = fold_arr
    ds.split(fold_id="by_robot", fold_num_validation=0, fold_num_test=1)
    gm = getattr(ds, "ears_generic_mask", None)
    if gm is not None:
        mtr = (fold_arr == 2);
        mva = (fold_arr == 0);
        mte = (fold_arr == 1)
        setattr(ds.training, "ears_generic_mask", np.asarray(gm)[mtr])
        setattr(ds.validation, "ears_generic_mask", np.asarray(gm)[mva])
        setattr(ds.test, "ears_generic_mask", np.asarray(gm)[mte])
    print(f"Split sizes → train: {ds.training.n}, val: {ds.validation.n}, test: {ds.test.n}")

    yt = np.asarray(ds.training.y, dtype=int)
    yv = np.asarray(ds.validation.y, dtype=int)
    yte = np.asarray(ds.test.y, dtype=int)
    print("Label distribution (train):", {
        "glorp": int((yt == 1).sum()),
        "drent": int((yt == 0).sum()),
        "total": int(yt.size),
        "pos_frac": round((yt == 1).mean() if yt.size else 0.0, 4),
    })
    print("Label distribution (val):", {
        "glorp": int((yv == 1).sum()),
        "drent": int((yv == 0).sum()),
        "total": int(yv.size),
        "pos_frac": round((yv == 1).mean() if yv.size else 0.0, 4),
    })
    print("Label distribution (test):", {
        "glorp": int((yte == 1).sum()),
        "drent": int((yte == 0).sum()),
        "total": int(yte.size),
        "pos_frac": round((yte == 1).mean() if yte.size else 0.0, 4),
    })


elif need_split:
    _manual_by_robot_split(ds, row_index, n_folds=5, seed=0)
    print(f"Split sizes → train: {ds.training.n}, val: {ds.validation.n}, test: {ds.test.n}")

    yt = np.asarray(ds.training.y, dtype=int)
    yv = np.asarray(ds.validation.y, dtype=int)
    yte = np.asarray(ds.test.y, dtype=int)
    print("Label distribution (train):", {
        "glorp": int((yt == 1).sum()),
        "drent": int((yt == 0).sum()),
        "total": int(yt.size),
        "pos_frac": round((yt == 1).mean() if yt.size else 0.0, 4),
    })
    print("Label distribution (val):", {
        "glorp": int((yv == 1).sum()),
        "drent": int((yv == 0).sum()),
        "total": int(yv.size),
        "pos_frac": round((yv == 1).mean() if yv.size else 0.0, 4),
    })
    print("Label distribution (test):", {
        "glorp": int((yte == 1).sum()),
        "drent": int((yte == 0).sum()),
        "total": int(yte.size),
        "pos_frac": round((yte == 1).mean() if yte.size else 0.0, 4),
    })

if int(getattr(args_obj, "train_balance_enable", 0)) == 1 and hasattr(ds.training, "ears_generic_mask"):
    ytr0 = np.asarray(ds.training.y, dtype=int)
    gtr0 = np.asarray(ds.training.ears_generic_mask, dtype=bool)
    idx0 = np.arange(ytr0.shape[0])
    f_pos = float(getattr(args_obj, "train_target_pos_frac", 0.5))
    f_gen = float(getattr(args_obj, "train_target_generic_frac", 0.5))
    within = int(getattr(args_obj, "train_balance_within_label", 1)) == 1
    rng = np.random.default_rng(int(SEED) + 907)
    if within:
        t = {
            (1, 1): f_pos * f_gen,
            (1, 0): f_pos * (1.0 - f_gen),
            (0, 1): (1.0 - f_pos) * f_gen,
            (0, 0): (1.0 - f_pos) * (1.0 - f_gen),
        }
        avail = {k: idx0[(ytr0 == k[0]) & (gtr0 == (k[1] == 1))] for k in t}
        caps = [avail[k].size / v for k, v in t.items() if v > 0]
        N = int(np.floor(min(caps))) if caps else 0
        take = []
        for k, v in t.items():
            n = int(np.floor(v * N)) if v > 0 else 0
            n = min(n, avail[k].size)
            if n > 0:
                take.append(rng.choice(avail[k], size=n, replace=False))
        take = np.sort(np.concatenate(take)) if take else np.array([], dtype=int)
    else:
        pos_idx = idx0[ytr0 == 1]
        neg_idx = idx0[ytr0 == 0]
        N = min(pos_idx.size, neg_idx.size) * 2
        n_pos = N // 2
        n_neg = N - n_pos
        take = np.sort(np.concatenate([
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ])) if N > 0 else np.array([], dtype=int)
    if take.size > 0:
        X = [ds.training.X[i] for i in take]
        C = ds.training.C[take]
        y = ds.training.y[take]
        tr_ds = ConceptDatasetSample(X=X, C=C, y=y, meta=ds.training.meta)
        setattr(tr_ds, "ears_generic_mask", ds.training.ears_generic_mask[take])
        ds.training = tr_ds

if int(getattr(args_obj, "val_balance_enable", 0)) == 1 and hasattr(ds.validation, "ears_generic_mask"):
    yv0 = np.asarray(ds.validation.y, dtype=int)
    gv0 = np.asarray(ds.validation.ears_generic_mask, dtype=bool)
    idxv = np.arange(yv0.shape[0])
    f_gen_v = float(getattr(args_obj, "val_target_generic_frac", 0.5))
    rngv = np.random.default_rng(int(SEED) + 908)
    take_v = []
    for lab in (0, 1):
        lab_idx = idxv[yv0 == lab]
        if lab_idx.size == 0:
            continue
        lab_gen = lab_idx[gv0[lab_idx]]
        lab_spec = lab_idx[~gv0[lab_idx]]
        want_gen = int(round(f_gen_v * lab_idx.size))
        want_spec = lab_idx.size - want_gen
        sel_gen = rngv.choice(lab_gen, size=min(want_gen, lab_gen.size), replace=False) if lab_gen.size else np.array(
            [], dtype=int)
        sel_spec = rngv.choice(lab_spec, size=min(want_spec, lab_spec.size),
                               replace=False) if lab_spec.size else np.array([], dtype=int)
        short_gen = want_gen - sel_gen.size
        if short_gen > 0 and (lab_spec.size - sel_spec.size) > 0:
            add = rngv.choice(np.setdiff1d(lab_spec, sel_spec), size=min(short_gen, lab_spec.size - sel_spec.size),
                              replace=False)
            sel_spec = np.concatenate([sel_spec, add])
        short_spec = want_spec - sel_spec.size
        if short_spec > 0 and (lab_gen.size - sel_gen.size) > 0:
            add = rngv.choice(np.setdiff1d(lab_gen, sel_gen), size=min(short_spec, lab_gen.size - sel_gen.size),
                              replace=False)
            sel_gen = np.concatenate([sel_gen, add])
        take_v.append(np.concatenate([sel_gen, sel_spec]))
    if take_v:
        take_v = np.sort(np.concatenate(take_v))
        X = [ds.validation.X[i] for i in take_v]
        C = ds.validation.C[take_v]
        y = ds.validation.y[take_v]
        va_ds = ConceptDatasetSample(X=X, C=C, y=y, meta=ds.validation.meta)
        setattr(va_ds, "ears_generic_mask", ds.validation.ears_generic_mask[take_v])
        ds.validation = va_ds

if int(getattr(args_obj, "test_balance_enable", 0)) == 1 and hasattr(ds.test, "ears_generic_mask"):
    yte0 = np.asarray(ds.test.y, dtype=int)
    gte0 = np.asarray(ds.test.ears_generic_mask, dtype=bool)
    idxt = np.arange(yte0.shape[0])
    f_gen_t = float(getattr(args_obj, "test_target_generic_frac", 0.5))
    rngt = np.random.default_rng(int(SEED) + 909)
    take_t = []
    for lab in (0, 1):
        lab_idx = idxt[yte0 == lab]
        if lab_idx.size == 0:
            continue
        lab_gen = lab_idx[gte0[lab_idx]]
        lab_spec = lab_idx[~gte0[lab_idx]]
        want_gen = int(round(f_gen_t * lab_idx.size))
        want_spec = lab_idx.size - want_gen
        sel_gen = rngt.choice(lab_gen, size=min(want_gen, lab_gen.size), replace=False) if lab_gen.size else np.array(
            [], dtype=int)
        sel_spec = rngt.choice(lab_spec, size=min(want_spec, lab_spec.size),
                               replace=False) if lab_spec.size else np.array([], dtype=int)
        short_gen = want_gen - sel_gen.size
        if short_gen > 0 and (lab_spec.size - sel_spec.size) > 0:
            add = rngt.choice(np.setdiff1d(lab_spec, sel_spec), size=min(short_gen, lab_spec.size - sel_spec.size),
                              replace=False)
            sel_spec = np.concatenate([sel_spec, add])
        short_spec = want_spec - sel_spec.size
        if short_spec > 0 and (lab_gen.size - sel_gen.size) > 0:
            add = rngt.choice(np.setdiff1d(lab_gen, sel_gen), size=min(short_spec, lab_gen.size - sel_gen.size),
                              replace=False)
            sel_gen = np.concatenate([sel_gen, add])
        take_t.append(np.concatenate([sel_gen, sel_spec]))
    if take_t:
        take_t = np.sort(np.concatenate(take_t))
        X = [ds.test.X[i] for i in take_t]
        C = ds.test.C[take_t]
        y = ds.test.y[take_t]
        te_ds = ConceptDatasetSample(X=X, C=C, y=y, meta=ds.test.meta)
        setattr(te_ds, "ears_generic_mask", ds.test.ears_generic_mask[take_t])
        ds.test = te_ds

train_ds = ds.training
val_ds = ds.validation
test_ds = ds.test

pat_shape = re.compile(
    r"(?i)\b(square|boxy|box|angular|cornered|right-angled|rectilinear|90-degree|triangle|triangular|tri-corner|three-angled|three-point|pointy|pointed|tapered|wedge|spearhead|spear-tip)\b")


def _leak_sentence_scoped(t):
    sents = re.split(r"[.!?;:]\s+", str(t).lower())
    for s in sents:
        if ("ear" in s) and pat_shape.search(s):
            return True
    return False


def _rates(part):
    gm = getattr(part, "ears_generic_mask", None)
    if gm is None:
        return {"overall": "na", "y1": "na", "y0": "na"}, {"generic_near_ears_shape": "na"}
    yv = np.asarray(part.y, dtype=int)
    overall = float(gm.mean()) if gm.size else float("nan")
    y1 = float(gm[yv == 1].mean()) if (yv == 1).any() else float("nan")
    y0 = float(gm[yv == 0].mean()) if (yv == 0).any() else float("nan")
    leak = int(sum(_leak_sentence_scoped(t) for t, g in zip(part.X, gm) if g))
    return {"overall": overall, "y1": y1, "y0": y0}, {"generic_near_ears_shape": leak}


dist = {};
leak = {}
for name, part in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    d, l = _rates(part)
    dist[name] = d
    leak[name] = l

t_train = float(getattr(args_obj, "train_target_generic_frac", getattr(args_obj, "generic_rate", 0.5))) if int(
    getattr(args_obj, "train_balance_enable", 0)) == 1 else float(getattr(args_obj, "generic_rate", 0.5))
t_val = float(getattr(args_obj, "generic_rate", 0.5))
t_test = float(getattr(args_obj, "generic_rate", 0.5))
tol = float(getattr(args_obj, "generic_tol", 0.02))
if int(getattr(args_obj, "generic_enable", 0)) == 1:
    print(json.dumps({
        "ears_leak_counts_generic": leak,
        "ears_generic_rates": dist,
        "targets": {"train": t_train, "val": t_val, "test": t_test, "tol": tol}
    }, indent=2))

    if any(v.get("generic_near_ears_shape", 0) not in ("na", 0) for v in leak.values()):
        raise SystemExit(3)
    for name, vals in dist.items():
        if vals["overall"] != "na" and np.isfinite(vals["overall"]):
            if abs(vals["overall"] - (t_train if name == "train" else t_val if name == "val" else t_test)) > tol:
                raise SystemExit(4)
        for k in ("y1", "y0"):
            if vals[k] != "na" and np.isfinite(vals[k]):
                if abs(vals[k] - (t_train if name == "train" else t_val if name == "val" else t_test)) > tol:
                    raise SystemExit(4)

# Test-only corpus swap (no-antennae) if requested
_rc = set(t.strip() for t in str(getattr(args_obj, "redact_concepts", "")).split(",") if t.strip())
_rs = set(t.strip().lower() for t in str(getattr(args_obj, "redact_splits", "")).split(",") if t.strip())
if ("has_antennae" in _rc) and ("test" in _rs):
    print("Redacting 'has_antennae' from test set by swapping in no-antennae corpus")
    base_jsonl = Path(args_obj.templates_file) if (
                args_obj.templates_file and str(args_obj.templates_file).lower().endswith(".jsonl")) else (
                pkg_dir / "synthetic" / "helper" / "static" / "text_templates" / "HardCorpus.jsonl")
    cand = [
        base_jsonl.with_name(base_jsonl.stem + "_noANT" + base_jsonl.suffix),
        base_jsonl.parent / "HardCorpus_noANT.jsonl",
        base_jsonl.parent / "HardCorpus_No_Ant.jsonl",
    ]
    alts = [c for c in cand if c.is_file()]
    if not alts:
        pat = re.compile(r"(?i)hardcorpus.*no[_-]?ant.*\.jsonl$")
        for q in base_jsonl.parent.glob("*.jsonl"):
            if pat.search(q.name):
                alts.append(q)
    alt = alts[0] if alts else None
    if alt is not None:
        cv = getattr(ds, "cvindices", None)
        fold = cv["by_robot"] if isinstance(cv, dict) and "by_robot" in cv else None
        if isinstance(fold, np.ndarray):
            row_index_full = getattr(getattr(ds, "_full", ds), "meta", {}).get("row_index", np.arange(len(ds)))
            te_idx = np.where(fold == 1)[0]
            corpus_noant = _load_jsonl(alt)
            newX = []
            for j, i_abs in enumerate(te_idx):
                rid = int(row_index_full[i_abs])
                row = {k: catalog_df.loc[rid, k] for k in params["concepts"].keys()}
                txt = _render_from_corpus(row, corpus_noant, int(SEED) + j)
                newX.append(txt)
            test_ds = ConceptDatasetSample(X=newX, C=test_ds.C, y=test_ds.y, meta=test_ds.meta)

# Leak check
import re as _re_chk

_pat_ant = _re_chk.compile(r"(?i)\bantenna(?:e|s)?\b")


def _count_mentions(sample):
    return int(sum(1 for x in sample.X if _pat_ant.search(str(x))))


print("Leak check — 'antenna' mentions:", {
    "train": _count_mentions(train_ds),
    "val": _count_mentions(val_ds),
    "test": _count_mentions(test_ds),
})

if getattr(args_obj, "redact_concepts", ""):
    _rc = set(t.strip() for t in str(args_obj.redact_concepts).split(",") if t.strip())
    _rs = set(t.strip().lower() for t in str(getattr(args_obj, "redact_splits", "")).split(",") if t.strip())

    if "has_antennae" in _rc and _rs:
        def _redact_antennae_clause_list(lst):
            pat = re.compile(r"(?i)\b(?:with|has)\s+antenna(?:e|s)?\b|\bno\s+antenna(?:e|s)?\b|\bantenna(?:e|s)?\b")

            def clean(z):
                z = re.sub(pat, " ", str(z))
                z = re.sub(r"\s{2,}", " ", z)
                z = re.sub(r"\s+([,.;:!?])", r"\1", z)
                return z.strip()

            return [clean(s) for s in lst]


        if "test" in _rs:
            test_ds = ConceptDatasetSample(X=_redact_antennae_clause_list(test_ds.X), C=test_ds.C, y=test_ds.y,
                                           meta=test_ds.meta)
            pd.DataFrame({"text": [str(x) for x in test_ds.X]}).to_csv(
                run_dir / f"text_samples_postredact_test_{miss_tag}_{seed_tag}.csv", index=False
            )
        if "val" in _rs:
            val_ds = ConceptDatasetSample(X=_redact_antennae_clause_list(val_ds.X), C=val_ds.C, y=val_ds.y,
                                          meta=val_ds.meta)
            pd.DataFrame({"text": [str(x) for x in val_ds.X]}).to_csv(
                run_dir / f"text_samples_postredact_val_{miss_tag}_{seed_tag}.csv", index=False
            )
        if "train" in _rs:
            train_ds = ConceptDatasetSample(X=_redact_antennae_clause_list(train_ds.X), C=train_ds.C, y=train_ds.y,
                                            meta=train_ds.meta)
            pd.DataFrame({"text": [str(x) for x in train_ds.X]}).to_csv(
                run_dir / f"text_samples_postredact_train_{miss_tag}_{seed_tag}.csv", index=False
            )

        _leak_post = {"train": _count_mentions(train_ds), "val": _count_mentions(val_ds),
                      "test": _count_mentions(test_ds)}
        print("Leak check — 'antenna' mentions (post-redact):", _leak_post)
        ds.training = train_ds
        ds.validation = val_ds
        ds.test = test_ds


def _apply_label_flip(sample, p, seed):
    if p <= 0: return sample
    rng = np.random.default_rng(int(seed) + 4242)
    y = sample.y.astype(int).copy()
    flip = rng.random(y.shape[0]) < float(p)
    y[flip] = 1 - y[flip]
    return ConceptDatasetSample(X=sample.X, C=sample.C, y=y,
                                meta={"concepts": sample.concepts, "classes": sample.meta.get("classes", []),
                                      "data_type": "text"})


if float(getattr(args_obj, "test_label_flip", 0.0)) > 0:
    test_ds = _apply_label_flip(test_ds, float(args_obj.test_label_flip), SEED)

if args_obj.corr_pair:
    train_ds = _enforce_corr(train_ds, args_obj.corr_pair, float(args_obj.train_corr), SEED)
    test_ds = _enforce_corr(test_ds, args_obj.corr_pair, max(0.0, 1.0 - float(args_obj.test_break)), SEED + 1)

if args_obj.skew_concept:
    train_ds = _apply_skew(train_ds, args_obj.skew_concept, SEED)
    test_ds = _apply_skew(test_ds, args_obj.skew_concept, SEED + 2)

if VARIANT == "imperfect":
    if IMPERFECT_STRATEGY == "missing_concepts":
        if args_obj.mask_mode == "rowdrop":
            train_ds = _apply_missing_concepts(train_ds, ds.concepts, HELDOUT_CONCEPTS, MASK_P, SEED)
        else:
            pass
    elif IMPERFECT_STRATEGY == "label_prior_shift":
        prior = _parse_label_prior(TEST_LABEL_PRIOR, ds.classes)
        val_ds = _apply_label_prior_shift(val_ds, prior, SEED)

names_all = list(train_ds.concepts)
inc_idx = []
exc_idx = []
if args_obj.concept_include:
    for tok in _csv_list(args_obj.concept_include):
        inc_idx.extend(_indices_for(names_all, tok))
if args_obj.concept_exclude:
    for tok in _csv_list(args_obj.concept_exclude):
        exc_idx.extend(_indices_for(names_all, tok))
if inc_idx or exc_idx:
    if not inc_idx:
        inc_idx = list(range(len(names_all)))
    keep = sorted(set(inc_idx) - set(exc_idx))
    if keep:
        train_ds = _select_concept_columns(train_ds, np.array(keep, dtype=int))
        val_ds = _select_concept_columns(val_ds, np.array(keep, dtype=int))
        test_ds = _select_concept_columns(test_ds, np.array(keep, dtype=int))

detector = TextConceptDetector(
    embed_dim=128,
    hidden_dim=192,
    epochs=6,
    batch_size=64,
    use_bigrams=True,
    lr=2e-3,
    dropout=0.1,
    pos_weight="auto",
    output_mode=CONCEPT_MODE,
    threshold_mode="auto",
    pooling="attn",
    group_unknown_threshold=0.50,
    validate=True,
)

label_mask = None
if VARIANT == "imperfect" and IMPERFECT_STRATEGY == "missing_concepts" and args_obj.mask_mode == "mask" and HELDOUT_CONCEPTS:
    names = list(train_ds.concepts)
    J = len(names)
    label_mask = np.ones((train_ds.C.shape[0], J), dtype=np.int32)
    cols = []
    for spec in HELDOUT_CONCEPTS:
        cols.extend(_indices_for(names, spec))
    cols = sorted(set(cols))
    if cols:
        rngm = np.random.default_rng(SEED + 123)
        for j in cols:
            m = rngm.random(train_ds.C.shape[0]) < float(args_obj.mask_rate)
            label_mask[m, j] = 0

train_ds.meta = dict(getattr(train_ds, "meta", {}) or {})
if label_mask is not None:
    train_ds.meta["observed_mask"] = label_mask

SKIP = int(getattr(args_obj, "skip_fit", 0)) == 1
loaded_cbm = None
if int(args_obj.reuse_detector) and args_obj.detector_model:
    print(f"Loading detector/cbm from: {args_obj.detector_model}")
    obj_det = load_obj(str(args_obj.detector_model))
    if isinstance(obj_det, dict) and "detector" in obj_det:
        detector = obj_det["detector"]
    if hasattr(detector, "output_mode"):
        detector.output_mode = CONCEPT_MODE
    if SKIP and isinstance(obj_det, dict) and "cbm" in obj_det:
        loaded_cbm = obj_det["cbm"]
else:
    is_lfcbm_ma = (args_obj.concept_source == "machine") and (str(args_obj.machine_method) == "lfcbm")
    if SKIP and not is_lfcbm_ma:
        raise ValueError("skip-fit=1 requires --reuse-detector=1 and --detector-model pointing to a saved model.")
    if not is_lfcbm_ma:
        print("Fitting detector")
        detector.fit(train_ds, val_ds)

cbm = loaded_cbm if (SKIP and loaded_cbm is not None) else ConceptBasedModel(concept_detector=detector,
                                                                             front_end_model=FrontEndModel(),
                                                                             propagate=(
                                                                                         args_obj.concept_mode == "soft"))

C_train = train_ds.C
y_train = train_ds.y

if (args_obj.concept_source == "machine") and (str(args_obj.machine_method) == "lfcbm"):
    lf_settings = {
        "concepts_csv": str(args_obj.concepts_csv),
        "lf_alpha": float(args_obj.lf_alpha),
        "lf_threshold": float(args_obj.lf_threshold),
        "lf_mode": "soft",
        "lf_ridge": bool(args_obj.lf_ridge),
        "lf_ridge_alpha": float(args_obj.lf_ridge_alpha),
        "lf_encoder": str(args_obj.lf_encoder),
        "lf_device": str(args_obj.lf_device),
        "lf_batch_size": int(args_obj.lf_batch_size),
    }
    _det_lf = LabelFreeDetector(lf_settings)
    _det_lf.fit([str(x) for x in train_ds.X])
    det_lf = _det_lf

    det_lf = _det_lf
    if args_obj.concept_mode == "soft":
        C_train_used = det_lf.predict([str(x) for x in train_ds.X]).astype(np.float32)
        C_val_used = det_lf.predict([str(x) for x in val_ds.X]).astype(np.float32)
    else:
        old_mode = det_lf.settings["lf_mode"]
        det_lf.settings["lf_mode"] = "hard"
        C_train_used = det_lf.predict([str(x) for x in train_ds.X]).astype(int)
        H_val_lf = det_lf.predict([str(x) for x in val_ds.X]).astype(int)
        H_test_lf = det_lf.predict([str(x) for x in test_ds.X]).astype(int)
        C_val_used = H_val_lf.astype(int)
        det_lf.settings["lf_mode"] = old_mode


else:
    if train_on_detected:
        old_mode = getattr(detector, "output_mode", None)
        try:
            if hasattr(detector, "output_mode"):
                detector.output_mode = "soft"
            C_train_used = detector.predict(train_ds)
        finally:
            if hasattr(detector, "output_mode") and old_mode is not None:
                detector.output_mode = old_mode
    else:
        C_train_used = C_train

noise_mode = merged.get("concept_label_noise_mode", "none")
if noise_mode == "machine":
    confusion_json = merged.get("concept_label_noise_confusion", "")
    confusion = None
    if confusion_json:
        try:
            confusion = json.loads(confusion_json)
        except Exception:
            confusion = None
    C_train_used = apply_machine_noise(C_train_used.astype(int).copy(),
                                       confusion=confusion,
                                       seed=int(merged.get("seed", 0)) + 201)

if not SKIP:
    cbm.front_end_model.fit(C_train_used, y_train)

with np.errstate(invalid="ignore"):
    C_val_true = val_ds.C.astype(np.float32)

if args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm":
    _old_mode = det_lf.settings["lf_mode"]
    det_lf.settings["lf_mode"] = "soft"
    C_val_scores = det_lf.predict([str(x) for x in val_ds.X]).astype(np.float32)
    det_lf.settings["lf_mode"] = _old_mode

else:
    _old = detector.output_mode
    detector.output_mode = "soft"
    C_val_scores = detector.predict(val_ds)
    detector.output_mode = _old

concept_names = list(ds.concepts)
per = {}
if C_val_scores.shape[1] == C_val_true.shape[1]:
    for j, name in enumerate(concept_names):
        yt = C_val_true[:, j]
        ys = C_val_scores[:, j]
        try:
            auprc = average_precision_score(yt, ys)
        except Exception:
            auprc = float("nan")
        try:
            rocauc = roc_auc_score(yt, ys) if len(np.unique(yt)) == 2 else float("nan")
        except Exception:
            rocauc = float("nan")
        per[name] = {"auprc": float(auprc), "roc_auc": float(rocauc)}
else:
    per = {}

auprc_macro = float(np.nanmean([d["auprc"] for d in per.values()])) if per else float("nan")
roc_macro = float(np.nanmean([d["roc_auc"] for d in per.values()])) if per else float("nan")

print("Macro concept metrics:", {"auprc_macro": auprc_macro, "roc_auc_macro": roc_macro})
print("Sample per-concept metrics (first 5):", {k: per[k] for k in list(per.keys())[:5]})

try:
    ear_idx = next(i for i, n in enumerate(concept_names) if str(n).lower().startswith("ears_"))
    oldm = detector.output_mode
    detector.output_mode = "soft"
    C_tr = detector.predict(train_ds);
    C_va = detector.predict(val_ds);
    C_te = detector.predict(test_ds)
    detector.output_mode = oldm


    def _acc(Cp, Ct, gm):
        yp = (Cp[:, ear_idx] >= 0.5).astype(int);
        yt = Ct[:, ear_idx].astype(int)
        return float((yp[~gm] == yt[~gm]).mean()) if (~gm).any() else float("nan"), float(
            (yp[gm] == yt[gm]).mean()) if gm.any() else float("nan")


    gm_tr = getattr(train_ds, "ears_generic_mask", None);
    gm_va = getattr(val_ds, "ears_generic_mask", None);
    gm_te = getattr(test_ds, "ears_generic_mask", None)
    if (gm_tr is not None) and (gm_va is not None) and (gm_te is not None):
        a_tr_spec, a_tr_gen = _acc(C_tr, train_ds.C.astype(int), gm_tr)
        a_va_spec, a_va_gen = _acc(C_va, val_ds.C.astype(int), gm_va)
        a_te_spec, a_te_gen = _acc(C_te, test_ds.C.astype(int), gm_te)
        rep = {"train": {"specific": a_tr_spec, "generic": a_tr_gen},
               "val": {"specific": a_va_spec, "generic": a_va_gen},
               "test": {"specific": a_te_spec, "generic": a_te_gen}}
        print("FE ears acc by split:", json.dumps(rep, indent=2))
except Exception:
    pass

texts_demo = [str(x) for x in ds.X[:3]]
dummy_C = np.zeros((len(texts_demo), len(ds.concepts)), dtype=np.float32)
dummy_y = np.zeros((len(texts_demo),), dtype=int)
demo_ds = ConceptDatasetSample(X=texts_demo, C=dummy_C, y=dummy_y,
                               meta={"concepts": ds.concepts, "classes": ds.classes, "data_type": "text"})
if args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm":
    _old_mode = det_lf.settings["lf_mode"];
    det_lf.settings["lf_mode"] = "soft"
    proba_demo = det_lf.predict([str(x) for x in texts_demo]).astype(np.float32)
    det_lf.settings["lf_mode"] = _old_mode
else:
    proba_demo = detector.predict(demo_ds)
print("Concept order:", concept_names)
print(f"Concept outputs (mode={CONCEPT_MODE}) shape:", proba_demo.shape)
print("First row outputs:", proba_demo[0])

y_train_true = train_ds.y.astype(int)
if args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm" and args_obj.concept_mode == "hard":
    y_train_proba = cbm.front_end_model.predict_proba(C_train_used)
else:
    y_train_proba = cbm.predict_proba(train_ds)

y_train_pred = np.argmax(y_train_proba, axis=1)
acc_train = accuracy_score(y_train_true, y_train_pred)
try:
    cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
    roc_train = roc_auc_score(y_train_true, y_train_proba[:, cls_index_1]) if len(
        np.unique(y_train_true)) == 2 else float("nan")
except Exception:
    roc_train = float("nan")
ba_train = balanced_accuracy_score(y_train_true, y_train_pred)
f1_train = f1_score(y_train_true, y_train_pred, zero_division=0)
print("Label model metrics (train):", {"accuracy": float(acc_train), "roc_auc": float(roc_train),
                                       "f1": float(f1_train), "balanced_acc": float(ba_train),
                                       "ber": float(1.0 - ba_train)})
y_val = val_ds.y.astype(int)
if args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm" and args_obj.concept_mode == "hard":
    y_val_proba = cbm.front_end_model.predict_proba(H_val_lf)
else:
    y_val_proba = cbm.predict_proba(val_ds)

y_val_pred = np.argmax(y_val_proba, axis=1)
acc = accuracy_score(y_val, y_val_pred)
try:
    cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
    roc = roc_auc_score(y_val, y_val_proba[:, cls_index_1]) if len(np.unique(y_val)) == 2 else float("nan")
except Exception:
    roc = float("nan")
ba_val = balanced_accuracy_score(y_val, y_val_pred)
ber_val = 1.0 - ba_val
f1_val = f1_score(y_val, y_val_pred, zero_division=0)
print("Label model metrics (validation):", {"accuracy": float(acc), "roc_auc": float(roc),
                                            "f1": float(f1_val), "balanced_acc": float(ba_val), "ber": float(ber_val)})

y_test = test_ds.y.astype(int)
if args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm" and args_obj.concept_mode == "hard":
    y_test_proba = cbm.front_end_model.predict_proba(H_test_lf)
else:
    y_test_proba = cbm.predict_proba(test_ds)

y_test_pred = np.argmax(y_test_proba, axis=1)
acc_test = accuracy_score(y_test, y_test_pred)
try:
    cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
    roc_test = roc_auc_score(y_test, y_test_proba[:, cls_index_1]) if len(np.unique(y_test)) == 2 else float("nan")
except Exception:
    roc_test = float("nan")
ba_test = balanced_accuracy_score(y_test, y_test_pred)
ber_test = 1.0 - ba_test
f1_test = f1_score(y_test, y_test_pred, zero_division=0)
print("Label model metrics (test):", {"accuracy": float(acc_test), "roc_auc": float(roc_test),
                                      "f1": float(f1_test), "balanced_acc": float(ba_test), "ber": float(ber_test)})

if not (args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm"):
    _old_tr = detector.output_mode
    detector.output_mode = "soft"
    C_train_scores = detector.predict(train_ds)
    detector.output_mode = _old_tr

    _old2 = detector.output_mode
    detector.output_mode = "soft"
    C_test_scores = detector.predict(test_ds)
    detector.output_mode = "hard"
    H_test = detector.predict(test_ds)
    detector.output_mode = _old2
else:
    C_train_scores = None
    C_test_scores = None
    H_test = None

C_train_true = train_ds.C.astype(np.float32)
sel_covs_tr, sel_accs_tr, aucs_tr, auprcs_tr = [], [], [], []
if C_train_scores is not None:
    for j, cname in enumerate(concept_names):
        m = calc_metric(C_train_scores[:, j], C_train_true[:, j], tau=0.5)
        sel_covs_tr.append(m["coverage"])
        sel_accs_tr.append(m["selective_accuracy"])
        try:
            if len(np.unique(C_train_true[:, j])) == 2:
                aucs_tr.append(roc_auc_score(C_train_true[:, j], C_train_scores[:, j]))
                auprcs_tr.append(average_precision_score(C_train_true[:, j], C_train_scores[:, j]))
        except Exception:
            pass

concept_train_metrics = {
    "selective_cov_mean": float(np.nanmean(sel_covs_tr)) if sel_covs_tr else float("nan"),
    "selective_acc_mean": float(np.nanmean(sel_accs_tr)) if sel_accs_tr else float("nan"),
    "auroc_macro": float(np.nanmean(aucs_tr)) if aucs_tr else float("nan"),
    "auprc_macro": float(np.nanmean(auprcs_tr)) if auprcs_tr else float("nan"),
    "tau": 0.5,
}
print("Concept metrics (train):", concept_train_metrics)
if not (args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm"):
    _old2 = detector.output_mode
    detector.output_mode = "soft"
    C_test_scores = detector.predict(test_ds)
    detector.output_mode = "hard"
    H_test = detector.predict(test_ds)
    detector.output_mode = _old2
else:
    if (args_obj.concept_source == "machine") and (str(args_obj.machine_method) == "lfcbm") and (
            not hasattr(detector, "model")):
        if str(args_obj.concept_mode) == "soft":
            C_test_scores = det_lf.predict([str(x) for x in test_ds.X]).astype(np.float32)
            _lf_old = det_lf.settings.get("lf_mode", "soft")
            det_lf.settings["lf_mode"] = "hard"
            H_test = det_lf.predict([str(x) for x in test_ds.X]).astype(int)
            det_lf.settings["lf_mode"] = _lf_old
        else:
            _lf_old = det_lf.settings.get("lf_mode", "soft")
            det_lf.settings["lf_mode"] = "soft"
            C_test_scores = det_lf.predict([str(x) for x in test_ds.X]).astype(np.float32)
            det_lf.settings["lf_mode"] = "hard"
            H_test = det_lf.predict([str(x) for x in test_ds.X]).astype(int)
            det_lf.settings["lf_mode"] = _lf_old
    else:
        _lf_old = det_lf.settings.get("lf_mode", "soft")
        det_lf.settings["lf_mode"] = "soft"
        C_test_scores = det_lf.predict([str(x) for x in test_ds.X]).astype(np.float32)
        det_lf.settings["lf_mode"] = "hard"
        H_test = det_lf.predict([str(x) for x in test_ds.X]).astype(int)
        det_lf.settings["lf_mode"] = _lf_old

C_test_true = test_ds.C.astype(np.float32)
if (C_test_scores is not None) and (C_test_scores.shape[1] == C_test_true.shape[1]):
    sel_covs_t, sel_accs_t, aucs_t, auprcs_t = [], [], [], []
    for j in range(C_test_true.shape[1]):
        m = calc_metric(C_test_scores[:, j], C_test_true[:, j], tau=0.5)
        sel_covs_t.append(m["coverage"])
        sel_accs_t.append(m["selective_accuracy"])
        try:
            if len(np.unique(C_test_true[:, j])) == 2:
                aucs_t.append(roc_auc_score(C_test_true[:, j], C_test_scores[:, j]))
                auprcs_t.append(average_precision_score(C_test_true[:, j], C_test_scores[:, j]))
        except Exception:
            pass
    concept_test_metrics = {
        "selective_cov_mean": float(np.nanmean(sel_covs_t)) if sel_covs_t else float("nan"),
        "selective_acc_mean": float(np.nanmean(sel_accs_t)) if sel_accs_t else float("nan"),
        "auroc_macro": float(np.nanmean(aucs_t)) if aucs_t else float("nan"),
        "auprc_macro": float(np.nanmean(auprcs_t)) if auprcs_t else float("nan"),
        "tau": 0.5,
    }
else:
    concept_test_metrics = {
        "selective_cov_mean": float("nan"),
        "selective_acc_mean": float("nan"),
        "auroc_macro": float("nan"),
        "auprc_macro": float("nan"),
        "tau": 0.5,
    }
print("Concept metrics (test):", concept_test_metrics)


def _groups(names):
    g, singles = {}, []
    for j, n in enumerate(names):
        if "=" in n:
            k = n.split("=", 1)[0]
            g.setdefault(k, []).append(j)
        else:
            singles.append(j)
    return {k: v for k, v in g.items() if len(v) > 1}, singles


def _bin_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, (prec + rec))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": float(acc), "prec": float(prec), "rec": float(rec),
            "f1": float(f1)}


def _concept_error_report(split_name, sample):
    names = list(sample.concepts)
    G, singles = _groups(names)
    oldm = detector.output_mode
    detector.output_mode = "hard"
    H = detector.predict(sample)
    detector.output_mode = oldm
    T = sample.C.astype(int)
    per_concept = {n: _bin_metrics(T[:, j], H[:, j]) for j, n in enumerate(names)}
    per_group = {}
    for k, idxs in G.items():
        t = T[:, idxs]
        h = H[:, idxs]
        unk = (h.sum(1) == 0)
        pred = np.where(unk, -1, h.argmax(1))
        true = t.argmax(1)
        known = ~unk
        acc_known = float((pred[known] == true[known]).mean()) if known.any() else float("nan")
        per_group[k] = {"acc_known": acc_known, "unknown_rate": float(unk.mean())}
    worst = dict(sorted(per_concept.items(), key=lambda kv: kv[1]["acc"])[:5])
    print(f"=== Concept error report [{split_name}] ===")
    print("Per-group:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_group.items()})
    print("Worst 5 concepts by acc:", {k: round(v["acc"], 4) for k, v in worst.items()})
    return {"per_concept": per_concept, "per_group": per_group}


if not (args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm"):
    _ = _concept_error_report("train", train_ds)
    _ = _concept_error_report("val", val_ds)
    _ = _concept_error_report("test", test_ds)

if not (args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm"):
    all_probs = cbm.predict_proba(ds)
    all_preds = np.argmax(all_probs, axis=1)
    label_names = list(ds.classes)
    pred_labels = [label_names[i] for i in all_preds]

metrics_out = {}
try:
    if len(np.unique(y_val)) == 2:
        cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
        lbl_sel = calc_metric(y_val_proba[:, cls_index_1], y_val, tau=0.5)
    else:
        lbl_sel = {"coverage": float("nan"), "selective_accuracy": float("nan"), "tau": 0.5}
except Exception:
    lbl_sel = {"coverage": float("nan"), "selective_accuracy": float("nan"), "tau": 0.5}

metrics_out["label"] = {"accuracy": float(acc), "roc_auc": float(roc),
                        "f1": float(f1_val), "balanced_acc": float(ba_val), "ber": float(ber_val),
                        "selective": lbl_sel}
metrics_out["label_test"] = {"accuracy": float(acc_test), "roc_auc": float(roc_test),
                             "f1": float(f1_test), "balanced_acc": float(ba_test), "ber": float(ber_test)}
metrics_out["label_train"] = {"accuracy": float(acc_train), "roc_auc": float(roc_train),
                              "f1": float(f1_train), "balanced_acc": float(ba_train), "ber": float(1.0 - ba_train)}
metrics_out["label_val"] = metrics_out["label"]

n_concepts = C_val_true.shape[1]
sel_covs, sel_accs, aucs, auprcs = [], [], [], []
for j in range(n_concepts):
    m = calc_metric(C_val_scores[:, j], C_val_true[:, j], tau=0.5)
    sel_covs.append(m["coverage"])
    sel_accs.append(m["selective_accuracy"])
    try:
        if len(np.unique(C_val_true[:, j])) == 2:
            aucs.append(roc_auc_score(C_val_true[:, j], C_val_scores[:, j]))
            auprcs.append(average_precision_score(C_val_true[:, j], C_val_scores[:, j]))
    except Exception:
        pass
concept_metrics = {
    "selective_cov_mean": float(np.nanmean(sel_covs)) if sel_covs else float("nan"),
    "selective_acc_mean": float(np.nanmean(sel_accs)) if sel_accs else float("nan"),
    "auroc_macro": float(np.nanmean(aucs)) if aucs else float("nan"),
    "auprc_macro": float(np.nanmean(auprcs)) if auprcs else float("nan"),
    "tau": 0.5,
}
metrics_out["concepts"] = concept_metrics
metrics_out["concepts_test"] = concept_test_metrics
metrics_out["concepts_train"] = concept_train_metrics
metrics_out["concepts_val"] = metrics_out["concepts"]

run_info = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "concept_mode": CONCEPT_MODE,
    "pos_weight": "auto",
    "n_samples": int(len(ds.X)),
    "n_concepts": int(n_concepts),
    "classes": list(ds.classes),
    "concept_names": list(ds.concepts),
    "variant": VARIANT,
    "strategy": IMPERFECT_STRATEGY,
    "heldout_concepts": HELDOUT_CONCEPTS,
    "mask_p": MASK_P,
    "test_label_prior": TEST_LABEL_PRIOR,
    "seed": SEED,
}

run_meta = {}
if getattr(detector, "thresholds_", None) is not None:
    run_meta["thresholds"] = detector.thresholds_.astype(float).tolist()
    np.savetxt(run_dir / f"thresholds_{miss_tag}_{seed_tag}.csv", detector.thresholds_, delimiter=",")
    np.save(run_dir / f"thresholds_{miss_tag}_{seed_tag}.npy", detector.thresholds_)
if getattr(detector, "concept_acc_", None) is not None:
    run_meta["concept_acc_per_concept"] = detector.concept_acc_.astype(float).tolist()
if getattr(detector, "alignment_", None) is not None:
    run_meta["alignment"] = {k: float(v) for k, v in detector.alignment_.items()}
if getattr(detector, "cross_auroc_", None) is not None:
    A = detector.cross_auroc_
    np.savetxt(run_dir / f"cross_auroc_{miss_tag}_{seed_tag}.csv", A, delimiter=",")
    run_meta["cross_auroc_diag_mean"] = float(np.nanmean(np.diag(A))) if A.size else float("nan")
metrics_out["detector_run_meta"] = run_meta

fe_src_tag = "fe_detected" if train_on_detected else "fe_gt"
model_path = run_dir / f"cbm_{fe_src_tag}_robots_text_{miss_tag}_{seed_tag}.pkl"
metrics_path = run_dir / f"metrics_cbm_{fe_src_tag}_robots_text_{miss_tag}_{seed_tag}.json"
meta_path = run_dir / f"meta_cbm_{fe_src_tag}_robots_text_{miss_tag}_{seed_tag}.json"

payload = {"run": run_info, "metrics": metrics_out}
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

with open(run_dir / f"metrics_label_train_{miss_tag}_{seed_tag}.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out["label_train"], f, indent=2)
with open(run_dir / f"metrics_label_val_{miss_tag}_{seed_tag}.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out["label_val"], f, indent=2)
with open(run_dir / f"metrics_label_test_{miss_tag}_{seed_tag}.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out["label_test"], f, indent=2)

save_obj({"cbm": cbm, "detector": detector, "train_variant": VARIANT, "strategy": IMPERFECT_STRATEGY}, model_path,
         overwrite=True)
_args_save = dict(vars(args_obj))
if "force_rerun" in _args_save:
    del _args_save["force_rerun"]
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({"args": _args_save, "settings_defaults": settings, "artifacts": {"model": str(model_path)}}, f, indent=2)

print("Saved dataset:", ds_path)
print("Saved model:", model_path)
print("Saved metrics:", metrics_path)

MODEL = model_path
DATA = ds_path
obj = load_obj(MODEL)
detector = obj["detector"]
ds = load_obj(DATA)


def ensure_split(ds):
    if hasattr(ds, "training") and hasattr(ds, "validation") and hasattr(ds, "test"):
        if getattr(ds.training, "n", 0) > 0 and getattr(ds.validation, "n", 0) > 0 and getattr(ds.test, "n", 0) > 0:
            return

    row_index = getattr(getattr(ds, "_full", ds), "meta", {}).get("row_index", np.arange(len(ds)))

    def _manual_by_robot_split(ds_obj, row_index_arr, seed=0):
        rng = np.random.default_rng(seed)
        base_ids = np.unique(row_index_arr)
        rng.shuffle(base_ids)

        n_ids = len(base_ids)
        n_val = int(np.floor(0.15 * n_ids))
        n_te = int(np.floor(0.15 * n_ids))
        val_ids = set(base_ids[:n_val])
        te_ids = set(base_ids[n_val:n_val + n_te])

        fold_arr = np.empty(len(row_index_arr), dtype=int)
        for i, rid in enumerate(row_index_arr):
            r = int(rid)
            fold_arr[i] = 0 if r in val_ids else (1 if r in te_ids else 2)  # 0=val, 1=test, 2=train

        def _subset_mask(mask):
            idx = np.where(mask)[0]
            X = [ds_obj.X[i] for i in idx]
            C = ds_obj.C[mask]
            y = ds_obj.y[mask]
            sub = ConceptDatasetSample(
                X=X, C=C, y=y,
                meta={"concepts": ds_obj.concepts, "classes": ds_obj.classes, "data_type": "text"}
            )
            gm = getattr(ds_obj, "ears_generic_mask", None)
            if gm is not None:
                setattr(sub, "ears_generic_mask", np.asarray(gm)[idx])
            return sub

        val_mask = (fold_arr == 0)
        test_mask = (fold_arr == 1)
        train_mask = ~(val_mask | test_mask)

        ds_obj.training = _subset_mask(train_mask)
        ds_obj.validation = _subset_mask(val_mask)
        ds_obj.test = _subset_mask(test_mask)

        try:
            ds_obj.cvindices = {"by_robot": fold_arr}
        except Exception:
            pass

    # try native .split() with explicit 70/15/15, else manual
    try:
        if hasattr(ds, "split"):
            rng = np.random.default_rng(0)
            base_ids = np.unique(row_index)
            rng.shuffle(base_ids)
            n_ids = len(base_ids)
            n_val = int(np.floor(0.15 * n_ids))
            n_te = int(np.floor(0.15 * n_ids))
            val_ids = set(base_ids[:n_val])
            te_ids = set(base_ids[n_val:n_val + n_te])

            fold_arr = np.empty(len(row_index), dtype=int)
            for i, rid in enumerate(row_index):
                r = int(rid)
                fold_arr[i] = 0 if r in val_ids else (1 if r in te_ids else 2)

            if getattr(ds, "cvindices", None) is None:
                ds.cvindices = {}
            ds.cvindices["by_robot"] = fold_arr
            ds.split(fold_id="by_robot", fold_num_validation=0, fold_num_test=1)
            return
    except Exception:
        pass

    _manual_by_robot_split(ds, row_index, seed=0)


def pick_split(ds, name):
    # safeguard if ensure_split hasn’t been called
    if not (hasattr(ds, "training") and hasattr(ds, "validation") and hasattr(ds, "test")):
        ensure_split(ds)
    d = {"train": ds.training, "val": ds.validation, "test": ds.test}[name]
    return d if getattr(d, "n", 0) > 0 else ds


def groups(names):
    g = {}
    for j, n in enumerate(names):
        if "=" in n:
            g.setdefault(n.split("=", 1)[0], []).append(j)
    return {k: v for k, v in g.items() if len(v) > 1}


def metrics(names, hard, C):
    out = {}
    for k, idxs in groups(names).items():
        t = C[:, idxs];
        p = hard[:, idxs]
        if t.shape[0] == 0: out[k] = {"acc_known": float("nan"), "unknown_rate": float("nan")}; continue
        unk = (p.sum(1) == 0)
        known = ~unk
        acc_known = (p.argmax(1)[known] == t.argmax(1)[known]).mean() if known.any() else float("nan")
        out[k] = {"acc_known": float(acc_known), "unknown_rate": float(unk.mean())}
    return out


def worst_examples(split_ds, names, hard, C, concept_key, k=10):
    j = names.index(concept_key)
    fn_idx = np.where((C[:, j] == 1) & (hard[:, j] == 0))[0]
    fp_idx = np.where((C[:, j] == 0) & (hard[:, j] == 1))[0]
    return {"n_FN": int(fn_idx.size), "n_FP": int(fp_idx.size), "FN": [str(split_ds.X[i]) for i in fn_idx[:k]],
            "FP": [str(split_ds.X[i]) for i in fp_idx[:k]]}


def run(name):
    ensure_split(ds)
    split = pick_split(ds, name)
    names = list(split.concepts)
    C = split.C.astype(int)
    H = detector.predict(split)
    print(f"=== {name.upper()} (n={split.n}) ===")
    print("Before:", metrics(names, H, C))
    ex = worst_examples(split, names, H, C, "head_is_square", k=5)
    print("head=square FNs:", ex["n_FN"], "FPs:", ex["n_FP"])
    print("FN examples:", ex["FN"][:3])
    print("FP examples:", ex["FP"][:3])


if not (args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm"):
    run("val"); run("test") #temporary

_old = detector.output_mode
detector.output_mode = "soft"
C_test_scores = detector.predict(test_ds).astype(float)
detector.output_mode = "hard"
H_test = detector.predict(test_ds).astype(int)
detector.output_mode = _old

T_test = test_ds.C.astype(int)
y_test_true = test_ds.y.astype(int)
budgets = [int(x) for x in _csv_list(args_obj.budgets)]
acc_grid = _parse_target_acc_grid(args_obj.target_acc_grid)
target_acc_concepts = _csv_kv_float(args_obj.target_acc_concepts)

vec = None
km = None
P_tr_m = None
P_te_m = None
H_te_m = None
names_m = None
truth_map = None
if args_obj.concept_source == "machine":
    if str(args_obj.machine_method) == "lfcbm":
        lf_settings = {
            "concepts_csv": args_obj.concepts_csv,
            "lf_alpha": float(args_obj.lf_alpha),
            "lf_threshold": float(args_obj.lf_threshold),
            "lf_mode": "soft",
            "lf_ridge": bool(args_obj.lf_ridge),
            "lf_ridge_alpha": float(args_obj.lf_ridge_alpha),
            "lf_encoder": args_obj.lf_encoder,
            "lf_device": args_obj.lf_device,
            "lf_batch_size": int(args_obj.lf_batch_size),
        }
        det_lf = LabelFreeDetector(lf_settings)
        det_lf.fit([str(x) for x in train_ds.X])
        old_mode_lf = det_lf.settings["lf_mode"]

        det_lf.settings["lf_mode"] = "soft"
        P_tr_m = det_lf.predict([str(x) for x in train_ds.X]).astype(np.float32)
        P_val_m = det_lf.predict([str(x) for x in val_ds.X]).astype(np.float32)
        P_te_m = det_lf.predict([str(x) for x in test_ds.X]).astype(np.float32)

        det_lf.settings["lf_mode"] = "hard"
        H_tr_m = det_lf.predict([str(x) for x in train_ds.X]).astype(int)
        H_val_m = det_lf.predict([str(x) for x in val_ds.X]).astype(int)
        H_te_m = det_lf.predict([str(x) for x in test_ds.X]).astype(int)
        det_lf.settings["lf_mode"] = old_mode_lf

        names_m = list(det_lf.concept_names)
        print(f"[lfcbm] concepts={len(names_m)} first5={names_m[:5]}")

        train_ds_lf_hard = ConceptDatasetSample(
            X=[str(x) for x in train_ds.X],
            C=H_tr_m.astype(np.float32),
            y=train_ds.y.astype(int),
            meta={"concepts": tuple(names_m), "classes": train_ds.classes, "data_type": "text"}
        )
        H_val_fit = H_val_m.copy()
        var = (H_val_fit.min(axis=0) != H_val_fit.max(axis=0))
        if not var.all():
            H_val_soft = (P_val_m >= 0.5).astype(int)
            H_val_fit[:, ~var] = H_val_soft[:, ~var]
            var = (H_val_fit.min(axis=0) != H_val_fit.max(axis=0))
            if not var.all():
                med = np.nanmedian(P_val_m, axis=0)
                H_val_soft_med = (P_val_m >= med).astype(int)
                H_val_fit[:, ~var] = H_val_soft_med[:, ~var]

        val_ds_lf_hard = ConceptDatasetSample(
            X=[str(x) for x in val_ds.X],
            C=H_val_fit.astype(np.float32),
            y=val_ds.y.astype(int),
            meta={"concepts": tuple(names_m), "classes": val_ds.classes, "data_type": "text"}
        )
        print("Fitting detector (student) on lfcbm HARD targets")
        detector.fit(train_ds_lf_hard, val_ds_lf_hard)

        truth_map = None
        fe_machine = FrontEndModel()
        fe_machine.fit(H_tr_m, train_ds.y.astype(int))

    else:
        vec, Xtr = _tfidf_fit(train_ds.X, int(args_obj.machine_seed) if int(args_obj.machine_seed) > 0 else SEED)
        Xte = vec.transform([str(t) for t in test_ds.X])
        km = _kmeans_fit(Xtr, int(args_obj.machine_k),
                         int(args_obj.machine_seed) if int(args_obj.machine_seed) > 0 else SEED + 11)
        P_tr_m = _kmeans_soft(Xtr, km)
        P_te_m = _kmeans_soft(Xte, km)
        H_tr_m = np.eye(int(args_obj.machine_k), dtype=int)[np.argmin(km.transform(Xtr), axis=1)]
        H_te_m = np.eye(int(args_obj.machine_k), dtype=int)[np.argmin(km.transform(Xte), axis=1)]
        names_m = [f"machine_{j}" for j in range(int(args_obj.machine_k))]
        if int(args_obj.machine_upper_bound):
            truth_map = _machine_truth_map(H_tr_m, train_ds.C.astype(int))
        fe_machine = FrontEndModel()
        if int(args_obj.machine_soft):
            fe_machine.fit(P_tr_m, train_ds.y.astype(int))
        else:
            fe_machine.fit(H_tr_m, train_ds.y.astype(int))

acc_map = _csv_kv_float(args_obj.human_acc_concepts)
rows = []
rows_all = []
rng = np.random.default_rng(SEED)

def _choose_source():
    if args_obj.concept_source == "detected":
        names_vec = list(test_ds.concepts)
        U_full = C_test_scores * (1 - C_test_scores)
        H_base = H_test

        noise_mode = merged.get("concept_label_noise_mode", "none")
        if noise_mode == "subjective":
            rate = float(merged.get("concept_label_noise_rate", 0.20))
            H_base = apply_subjective_noise(H_base.astype(int).copy(),
                                            rate=rate,
                                            seed=int(merged.get("seed", 0)) + 555)
        elif noise_mode == "machine":
            confusion_json = merged.get("concept_label_noise_confusion", "")
            confusion = None
            if confusion_json:
                try:
                    confusion = json.loads(confusion_json)
                except Exception:
                    confusion = None
            H_base = apply_machine_noise(H_base.astype(int).copy(),
                                         confusion=confusion,
                                         seed=int(merged.get("seed", 0)) + 556)

        T_truth = T_test
        fe = cbm.front_end_model
        return names_vec, U_full, H_base, T_truth, fe
    if args_obj.concept_source == "gt":
        names_vec = list(test_ds.concepts)
        U_full = np.zeros_like(T_test, dtype=float)
        H_base = T_test.copy()
        T_truth = T_test

        fe_gt = FrontEndModel()
        fe_gt.fit(train_ds.C.astype(int), train_ds.y.astype(int))

        noise_mode = merged.get("concept_label_noise_mode", "none")
        if noise_mode == "subjective":
            rate = float(merged.get("concept_label_noise_rate", 0.20))
            H_base = apply_subjective_noise(H_base.astype(int).copy(),
                                            rate=rate,
                                            seed=int(merged.get("seed", 0)) + 100)
        elif noise_mode == "machine":
            confusion_json = merged.get("concept_label_noise_confusion", "")
            confusion = None
            if confusion_json:
                try:
                    confusion = json.loads(confusion_json)
                except Exception:
                    confusion = None
            H_base = apply_machine_noise(H_base.astype(int).copy(),
                                         confusion=confusion,
                                         seed=int(merged.get("seed", 0)) + 200)

        return names_vec, U_full, H_base, T_truth, fe_gt

    names_vec = names_m
    demo_C = np.zeros((len(test_ds.X), len(names_vec)), dtype=np.float32)
    test_ds_cd = ConceptDatasetSample(
        X=[str(x) for x in test_ds.X],
        C=demo_C,
        y=test_ds.y.astype(int),
        meta={"concepts": tuple(names_vec), "classes": test_ds.classes, "data_type": "text"}
    )
    _old_out = detector.output_mode
    detector.output_mode = "soft"
    P_te_cd = detector.predict(test_ds_cd).astype(np.float32)
    detector.output_mode = "hard"
    H_te_cd = detector.predict(test_ds_cd).astype(int)
    detector.output_mode = _old_out

    U_full = P_te_cd * (1.0 - P_te_cd)
    H_base = H_te_cd
    if int(args_obj.machine_upper_bound) and truth_map is not None:
        T_truth = T_test[:, truth_map]
    else:
        T_truth = H_base.copy()
    fe = fe_machine
    return names_vec, U_full, H_base, T_truth, fe


names_vec, U_full_src, H_test_src, T_truth_src, fe_src = _choose_source()
allow_idxs = _allowed_indices(names_vec, args_obj.intervene_allow)

bb_acc = None
if args_obj.blackbox_metrics and Path(args_obj.blackbox_metrics).is_file():
    _m = json.loads(Path(args_obj.blackbox_metrics).read_text())
    bb_acc = float(_m.get("accuracy", _m.get("acc_test", 0.0)))

miss_meta_capture = None
_sel_obj = os.environ.get("SELECTION_OBJ", "reduce_base").lower()

base_proba_cls = None
base_pred_cls = None
base_acc_cls = None

for ta in acc_grid:
    if ta == "raw":
        H0 = H_test_src.copy()
        ta_label = "raw"
    else:
        H0 = _degrade_to_acc(H_test_src, T_truth_src, float(ta), SEED)
        ta_label = ta
    if target_acc_concepts:
        H0 = _apply_per_concept_degrade(H0, T_truth_src, names_vec, target_acc_concepts, SEED + 99)

    # if args_obj.concept_source == "detected":
    #     try:
    #         j_ant = names_vec.index("has_antennae")
    #         H0[:, j_ant] = 0.5
    #     except ValueError:
    #         pass

    tau_val = 0.2 if args_obj.concept_mode == "soft" else 0.5
    if args_obj.concept_mode == "soft":
        if args_obj.concept_source == "machine" and str(args_obj.machine_method) == "lfcbm":
            base_proba = cbm._propagate_predict_proba(P_te_m)
        else:
            _old_mode = getattr(detector, "output_mode", None)
            if hasattr(detector, "output_mode"):
                detector.output_mode = "soft"
            base_proba = cbm.predict_proba(test_ds, propagate=True)
            if hasattr(detector, "output_mode"):
                detector.output_mode = _old_mode
    else:
        base_proba = fe_src.predict_proba(H0)

    base_pred = np.argmax(base_proba, axis=1)
    base_acc = float(accuracy_score(y_test_true, base_pred))
    try:
        cls_index_1 = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
    except Exception:
        cls_index_1 = 1 if base_proba.shape[1] > 1 else 0
    sel_pre = calc_metric(base_proba[:, cls_index_1], y_test_true, tau=tau_val)

    if args_obj.concept_source != "gt":
        concept_err_rate = float((H_test_src != T_truth_src).mean())
        print(f"DBG concept_err_rate (detected vs truth): {concept_err_rate:.4f}")
        run_info["concept_err_rate"] = concept_err_rate
    else:
        run_info["concept_err_rate"] = 0.0
    acc_upper = float(accuracy_score(y_test_true, np.argmax(fe_src.predict_proba(T_truth_src), axis=1)))
    print(f"DBG upper_bound_acc (all concepts set to truth): {acc_upper:.4f}")
    run_info["upper_bound_acc"] = acc_upper

    acc_k0 = None


    def _simulate_mode(H0_local, mode_name):
        recs = []
        conc_recs = []
        pred0 = None
        proba10 = None
        predM = None
        proba1M = None
        for k in budgets:
            Hm = H0_local.copy()
            edit_counts = np.zeros(Hm.shape[0], dtype=int)
            per_concept_edits = np.zeros(Hm.shape[1], dtype=int)
            per_concept_correct = np.zeros(Hm.shape[1], dtype=int)
            per_concept_attempts = np.zeros(Hm.shape[1], dtype=int)
            P_work_loc = (P_te_m.copy() if (args_obj.concept_source == "machine" and str(
                args_obj.machine_method) == "lfcbm") else C_test_scores.copy()) if args_obj.concept_mode == "soft" else None

            if k > 0:
                cols_all = allow_idxs if allow_idxs.size > 0 else np.arange(Hm.shape[1], dtype=int)
                y0 = fe_src.predict(Hm)
                if args_obj.concept_mode == "soft":
                    idxs = \
                        np.where((base_proba[:, cls_index_1] >= tau_val) & (base_proba[:, cls_index_1] <= 1 - tau_val))[
                            0]
                else:
                    idxs = np.arange(Hm.shape[0])
                for i in idxs:
                    x_sel = Hm[i].copy()
                    base = int(y0[i])
                    rem = list(int(c) for c in cols_all)
                    picks = []
                    for _ in range(int(k)):
                        if not rem:
                            break
                        proba = fe_src.predict_proba(x_sel.reshape(1, -1))[0]
                        if _sel_obj == "increase_true":
                            tgt = int(y_test_true[i])
                            p0 = float(proba[tgt])
                            best_j, best_gain = -1, -np.inf
                            for j in rem:
                                tmp = x_sel.copy();
                                tmp[j] = T_truth_src[i, j]
                                p1 = float(fe_src.predict_proba(tmp.reshape(1, -1))[0, tgt])
                                gain = p1 - p0
                                if gain > best_gain:
                                    best_gain, best_j = gain, j
                        else:
                            p0 = float(proba[base])
                            best_j, best_drop = -1, -np.inf
                            for j in rem:
                                tmp = x_sel.copy();
                                tmp[j] = T_truth_src[i, j]
                                p1 = float(fe_src.predict_proba(tmp.reshape(1, -1))[0, base])
                                d = p0 - p1
                                if d > best_drop:
                                    best_drop, best_j = d, j
                        if best_j < 0:
                            break
                        picks.append(best_j)
                        x_sel[best_j] = T_truth_src[i, best_j]
                        rem.remove(best_j)

                    if picks:
                        for j in picks:
                            per_concept_attempts[j] += 1
                        before = Hm[i, picks].copy()
                        Hm[i] = _apply_human_edit(Hm[i], T_truth_src[i], picks, names_vec, float(args_obj.human_acc),
                                                  acc_map, rng, mode=mode_name)
                        after = Hm[i, picks]
                        changed_mask = (after != before)
                        for j, chg in zip(picks, changed_mask):
                            if chg:
                                per_concept_edits[j] += 1
                                if int(Hm[i, j]) == int(T_truth_src[i, j]):
                                    per_concept_correct[j] += 1
                        edit_counts[i] = int(changed_mask.sum())
                        if P_work_loc is not None:
                            for j in picks:
                                P_work_loc[i, j] = float(Hm[i, j])

            H_infer = Hm
            names_infer = list(names_vec)
            H_infer, names_infer, miss_meta = _apply_test_missing(H_infer, names_infer, args_obj.test_miss,
                                                                  float(args_obj.test_miss_rate),
                                                                  args_obj.test_miss_mode, SEED)
            if miss_meta is not None and miss_meta_capture is None:
                miss_meta_capture = dict(miss_meta)

            if miss_meta is not None and args_obj.test_miss_mode == "drop_cols":
                keep_idx = np.array([names_vec.index(n) for n in names_infer], dtype=int)
                if args_obj.concept_source == "detected":
                    Xtr = C_train_used if 'C_train_used' in locals() else detector.predict(train_ds)
                else:
                    Xtr = train_ds.C
                fe_src_drop = FrontEndModel()
                fe_src_drop.fit(Xtr[:, keep_idx], train_ds.y.astype(int))
                y_proba = fe_src_drop.predict_proba(H_infer)
            else:
                if args_obj.concept_mode == "soft":
                    y_proba = cbm._propagate_predict_proba(P_work_loc if P_work_loc is not None else C_test_scores)
                else:
                    y_proba = fe_src.predict_proba(H_infer)

            y_pred = np.argmax(y_proba, axis=1)
            acc_k = float(accuracy_score(y_test_true, y_pred))
            try:
                cls_i1_tmp = int(np.where(cbm.front_end_model.model.classes_ == 1)[0][0])
            except Exception:
                cls_i1_tmp = 1 if y_proba.shape[1] > 1 else 0
            sel_post = calc_metric(y_proba[:, cls_i1_tmp], y_test_true, tau=tau_val)

            interventions = int(np.sum(edit_counts > 0))
            total_applied_edits = int(edit_counts.sum())
            avg_edits_per_case = (float(total_applied_edits) / float(Hm.shape[0])) if Hm.shape[0] > 0 else 0.0
            concepts_per_intervention = float(edit_counts[edit_counts > 0].mean()) if interventions > 0 else 0.0
            incorrect_after = (y_pred != y_test_true)
            failed_interventions = int(np.sum((edit_counts > 0) & incorrect_after))
            failed_interventions_rate = (
                    float(failed_interventions) / float(interventions)) if interventions > 0 else 0.0

            gain_vs_k0 = (acc_k - base_acc) if base_acc is not None else float("nan")
            concept_checks_total = int(k * Hm.shape[0])
            edit_effectiveness = (gain_vs_k0 / max(1, k)) if not np.isnan(gain_vs_k0) else float("nan")
            edit_effectiveness_per_intervention = (gain_vs_k0 / max(1, interventions)) if not np.isnan(
                gain_vs_k0) else float("nan")

            rec = {
                "target_acc": ta,
                "budget": k,
                "acc_cbm_pre": base_acc,
                "acc_cbm_intv": acc_k,
                "raw_gain_vs_k0": gain_vs_k0,
                "gain_acc_human": (acc_k - float(getattr(args_obj, "human_alone", 0.75))) if hasattr(args_obj,
                                                                                                     "human_alone") else float(
                    "nan"),
                "gain_acc_dnn": (acc_k - bb_acc) if bb_acc is not None else float("nan"),
                "delta_vs_blackbox": (acc_k - bb_acc) if bb_acc is not None else float("nan"),
                "concept_checks": concept_checks_total,
                "confirmation_cost": concept_checks_total,
                "edit_effectiveness": edit_effectiveness,
                "edit_effectiveness_per_intervention": edit_effectiveness_per_intervention,
                "interventions_pct": float(interventions) / float(Hm.shape[0]),
                "concepts_per_intervention": concepts_per_intervention,
                "failed_interventions_pct": failed_interventions_rate,
                "avg_edits_per_case": avg_edits_per_case,
                "interventions_total": interventions,
                "applied_edits_total": total_applied_edits,
                "sel_acc_pre": float(sel_pre.get("selective_accuracy", float("nan"))),
                "sel_cov_pre": float(sel_pre.get("coverage", float("nan"))),
                "sel_acc_post": float(sel_post.get("selective_accuracy", float("nan"))),
                "sel_cov_post": float(sel_post.get("coverage", float("nan"))),
                "coverage_automated": float(sel_pre.get("coverage", float("nan"))),
                "coverage_after_confirmation": acc_k,
                "concept_source": args_obj.concept_source,
                "test_miss": args_obj.test_miss,
                "test_miss_rate": float(args_obj.test_miss_rate),
                "test_miss_mode": args_obj.test_miss_mode,
                "concept_err_rate": float(concept_err_rate) if "concept_err_rate" in locals() else float("nan"),
                "upper_bound_acc": float(acc_upper) if "acc_upper" in locals() else float("nan"),
                "corrected_edits_total": int(per_concept_correct.sum()),
                "attempted_edits_total": int(per_concept_attempts.sum()),
            }
            recs.append(rec)

            con_rows = []
            for j, name in enumerate(names_vec):
                n_att = int(per_concept_attempts[j])
                n_app = int(per_concept_edits[j])
                n_ok = int(per_concept_correct[j])
                con_rows.append({
                    "target_acc": ta,
                    "budget": k,
                    "concept": name,
                    "interventions": n_att,
                    "applied": n_app,
                    "correct": n_ok,
                    "correct_rate": (float(n_ok) / float(n_att)) if n_att > 0 else float("nan"),
                })
            if con_rows:
                dfc = pd.DataFrame(con_rows)
                if mode_name == "miss":
                    pcon = run_dir / f"interventions_per_concept_{miss_tag}_{seed_tag}_{args_obj.concept_source}_ta{str(ta_label).replace('.', 'p')}_k{k}.csv"
                else:
                    pcon = run_dir / f"interventions_per_conceptV2_{miss_tag}_{seed_tag}_{args_obj.concept_source}_ta{str(ta_label).replace('.', 'p')}_k{k}.csv"
                dfc.to_csv(pcon, index=False)

            if k == budgets[0]:
                pred0 = y_pred.copy()
                proba10 = y_proba[:, cls_i1_tmp].copy()
            if k == budgets[-1]:
                predM = y_pred.copy()
                proba1M = y_proba[:, cls_i1_tmp].copy()

        return recs, (pred0, proba10, predM, proba1M)


    if str(args_obj.intervention_error_mode) == "both":
        rows_v1, preds_v1 = _simulate_mode(H0, "miss")
        rows_v2, _ = _simulate_mode(H0, "flip")
        rows_all.extend(rows_v1)
        rows = rows_v1

        try:
            pred_k0, proba1_k0, pred_kmax, proba1_kmax = preds_v1
            df_pred = pd.DataFrame({
                "text": [str(x) for x in test_ds.X],
                "y_true": y_test_true.astype(int),
                "pred_k0": pred_k0.astype(int),
                "proba1_k0": proba1_k0.astype(float),
                "pred_kmax": pred_kmax.astype(int),
                "proba1_kmax": proba1_kmax.astype(float),
            })
            df_pred.to_csv(
                run_dir / f"preds_test_k0_k{budgets[-1]}_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv",
                index=False)
        except Exception:
            pass

        viab_v1 = pd.DataFrame(rows_v1)
        viab_v2 = pd.DataFrame(rows_v2)
        viab_path = run_dir / f"viability_robots_text_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        viab_v1.to_csv(viab_path, index=False)
        viab2_path = run_dir / f"viability_v2_robots_text_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        viab_v2.to_csv(viab2_path, index=False)

        v1 = viab_v1.copy()
        v1["check_accuracy"] = v1["corrected_edits_total"] / v1["concept_checks"].replace(0, np.nan)
        v1_path = run_dir / f"intervention_accuracy_v1_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        v1[["target_acc", "budget", "check_accuracy"]].to_csv(v1_path, index=False)

        v2 = viab_v2.copy()
        v2["check_accuracy"] = v2["corrected_edits_total"] / v2["applied_edits_total"].replace(0, np.nan)
        v2_path = run_dir / f"intervention_accuracy_v2_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        v2[["target_acc", "budget", "check_accuracy"]].to_csv(v2_path, index=False)

        viab = viab_v1

    else:
        rows, preds = _simulate_mode(H0, ("flip" if str(args_obj.intervention_error_mode) == "flip" else "miss"))
        rows_all.extend(rows)
        try:
            pred_k0, proba1_k0, pred_kmax, proba1_kmax = preds
            df_pred = pd.DataFrame({
                "text": [str(x) for x in test_ds.X],
                "y_true": y_test_true.astype(int),
                "pred_k0": pred_k0.astype(int),
                "proba1_k0": proba1_k0.astype(float),
                "pred_kmax": pred_kmax.astype(int),
                "proba1_kmax": proba1_kmax.astype(float),
            })
            df_pred.to_csv(
                run_dir / f"preds_test_k0_k{budgets[-1]}_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv",
                index=False)
        except Exception:
            pass
        viab = pd.DataFrame(rows)
        viab_path = run_dir / f"viability_robots_text_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        viab.to_csv(viab_path, index=False)

if miss_meta_capture is not None:
    run_info["test_missing"] = miss_meta_capture
    with open(run_dir / f"test_missing_meta_{miss_tag}_{seed_tag}.json", "w", encoding="utf-8") as f:
        json.dump(miss_meta_capture, f, indent=2)
    if miss_meta_capture.get("kept_cols"):
        with open(run_dir / f"kept_columns_{miss_tag}_{seed_tag}.csv", "w", encoding="utf-8") as f:
            f.write("concept\n")
            for n in miss_meta_capture["kept_cols"]:
                f.write(f"{n}\n")
    if miss_meta_capture.get("realized"):
        mask_rows = [{"concept": k, "realized_rate": v} for k, v in miss_meta_capture["realized"].items()]
        pd.DataFrame(mask_rows).to_csv(run_dir / f"mask_realized_{miss_tag}_{seed_tag}.csv", index=False)

    if str(args_obj.intervention_error_mode) == "miss":
        v1 = viab.copy()
        v1["check_accuracy"] = v1["corrected_edits_total"] / v1["concept_checks"].replace(0, np.nan)
        v1_path = run_dir / f"intervention_accuracy_v1_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        v1[["target_acc", "budget", "check_accuracy"]].to_csv(v1_path, index=False)
        print("Saved check-accuracy (v1):", v1_path)
    elif str(args_obj.intervention_error_mode) == "flip":
        v2 = viab.copy()
        v2["check_accuracy"] = v2["corrected_edits_total"] / v2["applied_edits_total"].replace(0, np.nan)
        v2_path = run_dir / f"intervention_accuracy_v2_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        v2[["target_acc", "budget", "check_accuracy"]].to_csv(v2_path, index=False)
        print("Saved check-accuracy (v2):", v2_path)
    else:
        v1 = viab_v1.copy()
        v1["check_accuracy"] = v1["corrected_edits_total"] / v1["concept_checks"].replace(0, np.nan)
        v1_path = run_dir / f"intervention_accuracy_v1_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        v1[["target_acc", "budget", "check_accuracy"]].to_csv(v1_path, index=False)
        v2 = viab_v2.copy()
        v2["check_accuracy"] = v2["corrected_edits_total"] / v2["applied_edits_total"].replace(0, np.nan)
        v2_path = run_dir / f"intervention_accuracy_v2_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
        v2[["target_acc", "budget", "check_accuracy"]].to_csv(v2_path, index=False)

        print("Saved check-accuracy (v1):", v1_path)
        print("Saved check-accuracy (v2):", v2_path)

    # save test sentences + y + k0/kmax predictions
    import pandas as pd

    df_pred = pd.DataFrame({
        "text": [str(x) for x in test_ds.X],
        "y_true": y_test_true.astype(int),
        "pred_k0": pred_k0.astype(int),
        "proba1_k0": proba1_k0.astype(float),
        "pred_kmax": pred_kmax.astype(int),
        "proba1_kmax": proba1_kmax.astype(float),
    })
    df_pred.to_csv(run_dir / f"preds_test_k0_k{budgets[-1]}_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv", index=False)
    # except Exception as _:
    #     pass

if miss_meta_capture is not None:
    run_info["test_missing"] = miss_meta_capture
    with open(run_dir / f"test_missing_meta_{miss_tag}_{seed_tag}.json", "w", encoding="utf-8") as f:
        json.dump(miss_meta_capture, f, indent=2)

    if miss_meta_capture.get("kept_cols"):
        with open(run_dir / f"kept_columns_{miss_tag}_{seed_tag}.csv", "w", encoding="utf-8") as f:
            f.write("concept\n")
            for n in miss_meta_capture["kept_cols"]:
                f.write(f"{n}\n")

    if miss_meta_capture.get("realized"):
        mask_rows = [{"concept": k, "realized_rate": v} for k, v in miss_meta_capture["realized"].items()]
        pd.DataFrame(mask_rows).to_csv(run_dir / f"mask_realized_{miss_tag}_{seed_tag}.csv", index=False)

_rows_for_write = rows_all if rows_all else rows
if _rows_for_write:
    viab = pd.DataFrame(_rows_for_write)
    viab_path = run_dir / f"viability_robots_text_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
    viab.to_csv(viab_path, index=False)
    print("Saved intervention metrics:", viab_path)

def _first_k_at_least():
    ks = viab.sort_values(["target_acc", "budget"])
    out = []
    for ta_val, grp in ks.groupby("target_acc"):
        k_dnn = grp.loc[grp["acc_cbm_intv"] >= (
            bb_acc if bb_acc is not None else -1), "budget"].min() if bb_acc is not None else np.nan
        human_ref = float(getattr(args_obj, "human_alone", np.nan))
        k_h = grp.loc[grp["acc_cbm_intv"] >= human_ref, "budget"].min() if not np.isnan(human_ref) else np.nan
        out.append({
            "target_acc": ta_val,
            "edits_to_match_dnn": (int(k_dnn) if pd.notna(k_dnn) else -1),
            "edits_to_match_human": (int(k_h) if pd.notna(k_h) else -1),
        })
    return pd.DataFrame(out)

summary_edit = _first_k_at_least()
summary_path = run_dir / f"viability_summary_edits_to_match_{miss_tag}_{seed_tag}_{args_obj.concept_source}.csv"
summary_edit.to_csv(summary_path, index=False)
print("Saved edits-to-match summary:", summary_path)
