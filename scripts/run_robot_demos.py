# scripts/run_robot_demos.py
from __future__ import annotations

import glob
from pathlib import Path
import sys, subprocess, shlex, os
import argparse
from typing import List, Optional
import pandas as pd
import math
import json
from concept_benchmark.paths import repo_dir, results_dir, pkg_dir

ROOT = repo_dir
PY = sys.executable
GEN = ROOT / "scripts" / "gen_text_samples.py"
DNN = ROOT / "scripts" / "robot_baseline.py"
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT) + (os.pathsep + ENV.get("PYTHONPATH", ""))

settings = {
    "seed": 1337,
    "difficulty": "hard",
    "budgets": [0, 1, 2],
    "force": 0,
    "reuse_detector": 1,
    "run_tag": "new_try",

    "modality": "text",
    "text_model": "distilbert-base-uncased",

    "best_human_acc": 1.00,
    "expert_human_accs": [0.80],
    "subjective_human_accs": [0.8],
    "subjective_noise_rates": [0.20],

    "skip_fit": 1,
    "make_plots": 0,

    "redact_concepts": "",
    "redact_splits": "",

    "concepts_csv": str(ROOT / "data" / "robot_text" / "concepts" / "concepts.csv"),
    "run_name_sub": "swapGeneric_footOpenPointy_balanced_v2",

    "generic_enable": 1,
    "generic_rate": 0.7,
    "generic_tol": 0.1,

    "policy": "kflip",
    "k": 2,
    "flip_threshold": 0.30,
    "flip_batch_size": 8192,
    "flip_limit_subsets": None,
    "abstain_only": 0,
    "abstain": "none",
    "calibrate": "platt",

    "seed_test_offset": 1234,
    "image_meta_catalog": "auto",
    "templates_file": str(ROOT / "concept_benchmark" / "synthetic" / "helper" / "static" / "text_templates" / "templates_simple.jsonl"),
}


def parse_cli():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--difficulty", type=str)
    ap.add_argument("--budgets", type=str)
    ap.add_argument("--force", type=int)
    ap.add_argument("--reuse_detector", type=int)
    ap.add_argument("--run_tag", type=str)
    ap.add_argument("--modality", type=str)
    ap.add_argument("--text_model", type=str)
    ap.add_argument("--best_human_acc", type=float)
    ap.add_argument("--expert_human_acc", type=float)
    ap.add_argument("--subjective_human_acc", type=float)
    ap.add_argument("--subjective_noise_rate", type=float)
    ap.add_argument("--label_model_type", type=str)
    ap.add_argument("--label-model-type", dest="label_model_type", type=str)
    ap.add_argument("--label_model_alpha", type=float)
    ap.add_argument("--label-model-alpha", dest="label_model_alpha", type=float)
    ap.add_argument("--label_model_bias", type=float)
    ap.add_argument("--label-model-bias", dest="label_model_bias", type=float)
    ap.add_argument("--label_model_expr", type=str)
    ap.add_argument("--label-model-expr", dest="label_model_expr", type=str)
    ap.add_argument("--templates_file", type=str)
    ap.add_argument("--redact_concepts", type=str)
    ap.add_argument("--redact_splits", type=str)
    ap.add_argument("--redact-masked-only", dest="redact_masked_only", type=int)
    ap.add_argument("--debug-dump", dest="debug_dump", type=int)
    ap.add_argument("--concept_mode", type=str)
    ap.add_argument("--generic-rate", "--generic_rate", dest="generic_rate", type=float)
    ap.add_argument("--generic-tol", "--generic_tol", dest="generic_tol", type=float)
    ap.add_argument("--generic_enable", type=int)
    ap.add_argument("--shared-test", type=int)
    ap.add_argument("--generic-what", dest="generic_what", type=str)
    ap.add_argument("--variants-per-row-minority", dest="variants_per_row_minority", type=int)
    ap.add_argument("--variants-per-row-majority", dest="variants_per_row_majority", type=int)
    ap.add_argument("--train-balance-enable", "--train_balance_enable", dest="train_balance_enable", type=int)
    ap.add_argument("--train-target-pos-frac", "--train_target_pos_frac", dest="train_target_pos_frac", type=float)
    ap.add_argument("--train-target-generic-frac", "--train_target_generic_frac", dest="train_target_generic_frac", type=float)
    ap.add_argument("--train-balance-within-label", "--train_balance_within_label", dest="train_balance_within_label", type=int)
    ap.add_argument("--val-target-generic-frac", type=float)
    ap.add_argument("--test-target-generic-frac", type=float)
    ap.add_argument("--deployment-target-generic-frac", type=float)
    ap.add_argument("--intervention_error_mode", type=str)
    ap.add_argument("--random_intervene", type=int)
    ap.add_argument("--build_hardness", type=int)
    ap.add_argument("--build_consensus", type=int)
    ap.add_argument("--build_disagreement", type=int)
    ap.add_argument("--build_ambiguity", type=int)
    ap.add_argument("--toy_concept_abstain", type=int)
    ap.add_argument("--toy_concept_popular", type=int)
    ap.add_argument("--toy_concept_random", type=int)
    ap.add_argument("--abstain_coverage", type=float)
    ap.add_argument("--abstain_metric", type=str)
    ap.add_argument("--concept_source", type=str)
    ap.add_argument("--machine_method", type=str)
    ap.add_argument("--use_interventions", type=int)
    ap.add_argument("--intervention_budget", type=int)
    ap.add_argument("--intervention_where", type=str)
    ap.add_argument("--intervention_policy", type=str)
    ap.add_argument("--intervention_k", type=int)
    ap.add_argument("--subtype-mode", type=str)
    ap.add_argument("--concept-variant", type=str)
    ap.add_argument("--salient-allowlist", type=str)
    ap.add_argument("--seed-cv", type=int)
    ap.add_argument("--cv-k", type=int)
    ap.add_argument("--cv-fold", type=int)
    ap.add_argument("--dev-per-fold", type=int)
    ap.add_argument("--dev-size", type=int)
    ap.add_argument("--deployment-size", type=int)
    ap.add_argument("--calibrate", choices=["none", "platt", "auto"], default="auto")
    ap.add_argument("--abstain", choices=["none", "conf"], default="none")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--tau-target", type=float)
    ap.add_argument("--cal-select-metric", type=str)  # accuracy | balanced_acc | f1
    ap.add_argument("--save-logits", type=int)  # 1 to save z/y for posthoc
    ap.add_argument("--posthoc-dir", type=str)  # reuse a prior run dir
    ap.add_argument("--lf-alpha", type=float)
    ap.add_argument("--lf-threshold", type=float)
    ap.add_argument("--lf-mode", type=str)
    ap.add_argument("--lf-ridge", action="store_true")
    ap.add_argument("--lf-ridge-alpha", type=float)
    ap.add_argument("--lf-encoder", type=str)
    ap.add_argument("--lf-device", type=str)
    ap.add_argument("--lf-batch-size", type=int)
    ap.add_argument("--val-balance-enable", "--val_balance_enable", dest="val_balance_enable", type=int)
    ap.add_argument("--test-balance-enable", "--test_balance_enable", dest="test_balance_enable", type=int)
    ap.add_argument("--policy", type=str)
    ap.add_argument("--run-name-sub")
    ap.add_argument("--k", type=int)
    ap.add_argument("--flip-threshold", type=float)
    ap.add_argument("--flip-batch-size", type=int)
    ap.add_argument("--flip-limit-subsets")
    ap.add_argument("--abstain-only", type=int)
    ap.add_argument("--image-meta", dest="image_meta", type=str)
    ap.add_argument("--image-meta-catalog", dest="image_meta_catalog", type=str)
    ap.add_argument("--train-variant-mode", choices=["all","one"])
    ap.add_argument("--val-variant-mode", choices=["all","one"])
    ap.add_argument("--test-variant-mode", choices=["all","one"])
    ap.add_argument("--blackbox_metrics", type=str, default="")

    known, _ = ap.parse_known_args()

    for k, v in vars(known).items():
        if v is not None:
            settings[k] = v

    for k in list(settings.keys()):
        k2 = k.replace("_", "-")
        if k2 in vars(known) and vars(known)[k2] is not None:
            settings[k] = vars(known)[k2]
        elif k in vars(known) and vars(known)[k] is not None:
            settings[k] = vars(known)[k]

    if isinstance(settings.get("budgets"), str):
        try:
            settings["budgets"] = [int(x) for x in settings["budgets"].split(",")]
        except Exception:
            settings["budgets"] = [0, 1, 2, 5, 10]


def run_cmd(argv: List[str], cwd: Optional[Path] = None):
    cmd = " ".join(shlex.quote(str(x)) for x in argv)
    print("RUN:", cmd)
    ENV["PYTHONHASHSEED"] = str(int(settings.get("seed", 0)))
    ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    subprocess.check_call(argv, cwd=str(cwd) if cwd else None, env=ENV)

def ensure_baseline(model_id: str, modality: str):
    outdir = results_dir / "robot_baseline" / modality
    outdir.mkdir(parents=True, exist_ok=True)
    seed = int(settings.get("seed", 0))
    model_tag = str(model_id).split("/")[-1]

    # look for existing metrics under any run subfolder
    want = f"baseline_dnn_robots_{modality}_{model_tag}_seed{seed}_metrics_test.json"
    existing = sorted(outdir.rglob(want))
    if existing:
        return

    argv = [str(PY), str(DNN)]
    if modality == "text":
        argv += ["--text_model", str(settings.get("text_model", "distilbert-base-uncased"))]
        if settings.get("templates_file"):
            argv += ["--templates-file", str(settings["templates_file"])]
        if settings.get("generic_what"):
            argv += ["--generic-what", str(settings["generic_what"])]
        if settings.get("image_meta"):
            argv += ["--image-meta", str(settings["image_meta"])]
            if settings.get("image_meta_catalog"):
                argv += ["--image-meta-catalog", str(settings["image_meta_catalog"])]

    # keep corpus and generic target consistent with GEN
    if settings.get("templates_file"):
        argv += ["--templates-file", str(settings["templates_file"])]
    if settings.get("generic_what"):
        argv += ["--generic-what", str(settings["generic_what"])]

    # forward label model so catalog labels match CBM/text GEN
    if settings.get("label_model_expr"):
        argv += ["--label-model-expr", str(settings["label_model_expr"])]
    if settings.get("label_model_type"):
        argv += ["--label-model-type", str(settings["label_model_type"])]
    if settings.get("label_model_alpha") is not None:
        argv += ["--label-model-alpha", str(settings["label_model_alpha"])]
    if settings.get("label_model_bias") is not None:
        argv += ["--label-model-bias", str(settings["label_model_bias"])]

    if settings.get("calibrate"):
        argv += ["--calibrate", str(settings["calibrate"])]
    if settings.get("abstain"):
        argv += ["--abstain", str(settings["abstain"])]
    if settings.get("tau") is not None:
        argv += ["--tau", str(settings["tau"])]

    for flag, key in [
        ("--train-variant-mode", "train_variant_mode"),
        ("--val-variant-mode",   "val_variant_mode"),
        ("--test-variant-mode",  "test_variant_mode"),
        ("--variants-per-row-minority", "variants_per_row_minority"),
        ("--variants-per-row-majority", "variants_per_row_majority"),
        ("--samples_per_instance",      "samples_per_instance"),
        ("--minority_mult",             "minority_mult"),
    ]:
        if settings.get(key) is not None:
            argv += [flag, str(settings[key])]
    if settings.get("tau_target") is not None:
        argv += ["--tau-target", str(settings["tau_target"])]

    if settings.get("deployment_size") is not None:
        argv += ["--deployment-size", str(settings["deployment_size"])]
    if settings.get("dev_size") is not None:
        argv += ["--dev-size", str(settings["dev_size"])]

    if settings.get("seed") is not None:
        argv += ["--seed", str(settings["seed"])]
    if settings.get("seed_cv") is not None:
        argv += ["--seed-cv", str(settings["seed_cv"])]
    if settings.get("cv_k") is not None:
        argv += ["--cv-k", str(settings["cv_k"])]
    if settings.get("cv_fold") is not None:
        argv += ["--cv-fold", str(settings["cv_fold"])]
    if settings.get("dev_per_fold") is not None:
        argv += ["--dev-per-fold", str(settings["dev_per_fold"])]
    if settings.get("redact_masked_only") is not None:
        argv += ["--redact-masked-only", str(settings["redact_masked_only"])]
    if settings.get("debug_dump"):
        argv += ["--debug-dump", str(settings["debug_dump"])]

    for flag, val in [
        ("--generic-rate", settings.get("generic_rate")),
        ("--generic-tol", settings.get("generic_tol")),
        ("--val-balance-enable", settings.get("val_balance_enable")),
        ("--test-balance-enable", settings.get("test_balance_enable")),
    ]:
        if val is not None:
            argv += [flag, str(val)]

    # decision thresholds (optional)
    if settings.get("decision_threshold") is not None:
        argv += ["--decision-threshold", str(settings["decision_threshold"])]
    if settings.get("threshold_masked") is not None:
        argv += ["--threshold-masked", str(settings["threshold_masked"])]
    if settings.get("threshold_unmasked") is not None:
        argv += ["--threshold-unmasked", str(settings["threshold_unmasked"])]

    # calibration passthrough
    if settings.get("cal_select_metric"):
        argv += ["--cal-select-metric", str(settings["cal_select_metric"])]
    if settings.get("save_logits") is not None:
        argv += ["--save-logits", str(settings["save_logits"])]
    if settings.get("posthoc_dir"):
        argv += ["--posthoc-dir", str(settings["posthoc_dir"])]

    # deterministic run folder so we can reuse across runs
    cal = str(settings.get("calibrate", "none"))
    absn = str(settings.get("abstain", "none"))
    tau = settings.get("tau", "na")
    thr = settings.get("decision_threshold", "na")
    run_name = f"baseline_{modality}_{model_tag}_seed{seed}_cal-{cal}_abs-{absn}_tau-{tau}_thr-{thr}"
    argv += ["--run-name", run_name]

    run_cmd(argv)


def find_metrics_json(modality: str, model_id: str, split: str):
    outdir = results_dir / "robot_baseline" / modality
    seed = int(settings.get("seed", 0))
    model_tag = str(model_id).split("/")[-1]
    want = f"baseline_dnn_robots_{modality}_{model_tag}_seed{seed}_metrics_{split}.json"
    hits = sorted(outdir.rglob(want))
    return hits[0] if hits else None

def common_gen_argv():
    argv = [
        str(PY), str(GEN),
        "--modality", str(settings.get("modality", "text")),
        "--machine-method", str(settings.get("machine_method", "lfcbm")),
        "--seed", str(settings.get("seed", 0)),
        "--seed-cv", str(settings.get("seed_cv", int(settings.get("seed", 0)) + 1)),
        "--cv-k", str(settings.get("cv_k", 5)),
        "--cv-fold", str(settings.get("cv_fold", 0)),
        "--dev-per-fold", str(settings.get("dev_per_fold", 1000)),
        "--deployment-size", str(settings.get("deployment_size", 10000)),
        "--shared-test", str(settings.get("shared_test", 1)),
        "--subtype-mode", str(settings.get("subtype_mode", "track")),
        "--policy", str(settings.get("policy", settings.get("intervention_policy", "kflip"))),
        "--k", str(settings.get("k", settings.get("intervention_k", 2))),
        "--flip-threshold", str(settings.get("flip_threshold", 0.30)),
        "--flip-batch-size", str(settings.get("flip_batch_size", 8192)),
        "--budgets", ",".join(str(x) for x in settings.get("budgets", [0, 1, 2, 5, 10])),
    ]
    if settings.get("image_meta"):
        argv += ["--image-meta", str(settings["image_meta"])]
        if settings.get("image_meta_catalog"):
            argv += ["--image-meta-catalog", str(settings["image_meta_catalog"])]
    if settings.get("flip_limit_subsets") is not None:
        argv += ["--flip-limit-subsets", str(settings["flip_limit_subsets"])]
    if settings.get("abstain_only"):
        argv += ["--abstain-only"]
    if settings.get("tau") is not None:
        argv += ["--tau", str(settings["tau"])]

    for flag, key in [
        ("--train-variant-mode", "train_variant_mode"),
        ("--val-variant-mode",   "val_variant_mode"),
        ("--test-variant-mode",  "test_variant_mode"),
    ]:
        if settings.get(key):
            argv += [flag, str(settings[key])]

    # forward balance knobs (train/val/test)
    for flag, val in [
        ("--train-balance-enable", settings.get("train_balance_enable")),
        ("--train-target-pos-frac", settings.get("train_target_pos_frac")),
        ("--train-target-generic-frac", settings.get("train_target_generic_frac")),
        ("--train-balance-within-label", settings.get("train_balance_within_label")),
        ("--val-balance-enable", settings.get("val_balance_enable")),
        ("--test-balance-enable", settings.get("test_balance_enable")),
        ("--val-target-generic-frac", settings.get("val_target_generic_frac")),
        ("--test-target-generic-frac", settings.get("test_target_generic_frac")),
    ]:
        if val is not None:
            argv += [flag, str(val)]

    return argv


def run_spec(prefix: str, regime: str, human_acc: float, blackbox_metrics: str, tag_suffix: str, concept_source: str,
             extra_flags: Optional[List[str]] = None, detector_model: Optional[str] = None):
    budgets = settings.get("budgets", [0, 1, 2, 5, 10])
    force = int(settings.get("force", 0))
    make_plots = int(settings.get("make_plots", 0))
    seed = int(settings.get("seed", 0))

    _sub = str(settings.get("run_name_sub", "")).strip()
    rn_base = f"{prefix}_{regime}_intervene{int(human_acc * 100)}_kset-0_1_2_seed{seed}"
    rn = f"{_sub}_{rn_base}" if _sub else rn_base

    # start with base cmd (includes [PY, GEN] + core flags and kflip knobs)
    argv = common_gen_argv()
    if not blackbox_metrics:
        bb_guess = find_metrics_json(str(settings.get("modality", "text")),
                                     str(settings.get("text_model", "distilbert-base-uncased")), "test")
        blackbox_metrics = str(bb_guess) if bb_guess else ""

    # scenario-specific flags
    argv += [
        "--variant", "perfect",
        "--variants-per-row", "1",
        "--variants-per-row-minority", str(settings.get("variants_per_row_minority", 3)),
        "--variants-per-row-majority", str(settings.get("variants_per_row_majority", 1)),
        "--imperfect-strategy", "missing_concepts",
        "--heldout-concepts", "[]",
        "--mask-p", "0.0",
        "--mask-mode", "mask",
        "--mask-rate", "0.0",
        "--concept-mode", "hard",
        *([] if not str(settings.get("templates_file", "")).strip()
          else ["--templates-file", str(settings["templates_file"]).strip()]),
        "--redact-concepts", str(settings.get("redact_concepts", "")),        "--redact-splits", str(settings.get("redact_splits", "")),
        "--label-model-type", "stochastic",
        "--label-model-alpha", "1.0",
        "--label-model-bias", "0.0",
        "--label-model-expr", "5*int(row['mouth_type']=='closed') + 10*int(str(row['foot_shape']).startswith('pointy_')) - 3",
        "--corr-pair", "",
        "--train-corr", "1.0",
        "--test-break", "1.0",
        "--test-corr", "-1.0",
        "--target-acc-grid", "raw",
        "--target-acc-concepts", "",
        "--intervene-allow", "",
        "--human-acc", str(human_acc),
        "--human-acc-concepts", "",
        "--make-plots", str(make_plots),
        "--concept-label-noise-mode", "none",
        "--concept-label-noise-rate", "0.2",
        "--blackbox-metrics", blackbox_metrics or "",
        *([] if str(settings.get("templates_file", "")).strip()
          else ["--concepts-csv", settings.get("concepts_csv", "")]),
        "--generic-enable", str(int(settings.get("generic_enable", 1))),
        "--generic-rate", str(float(settings.get("generic_rate", 0.7))),
        "--generic-tol", str(float(settings.get("generic_tol", 0.02))),
        "--generic-what", str(settings.get("generic_what", "foot")),
        "--concept-source", concept_source,
        "--skip-fit", str(int(settings.get("skip_fit", 1))),
        "--force-rerun", str(force),
        "--intervention-error-mode", "both",
        "--run-name", rn if not tag_suffix else f"{tag_suffix}_{rn}",
        "--seed-test-offset", str(int(settings.get("seed_test_offset", 1234))),
    ]

    if int(settings.get("reuse_detector", 1)) and detector_model:
        argv += ["--reuse-detector", "1", "--detector-model", detector_model]
    if extra_flags:
        argv += extra_flags

    run_cmd(argv)

def make_run_name():
    tag = settings.get("run_tag", "")
    diff = settings.get("difficulty", "hard")
    seed = int(settings.get("seed", 0))
    return f"{tag}_{diff}_seed{seed}" if tag else f"{diff}_seed{seed}"

def run():
    modality = str(settings.get("modality", "text"))
    model_id = str(settings.get("text_model", "distilbert-base-uncased")) if modality == "text" else "vit"
    ensure_baseline(model_id=model_id, modality=modality)

    bb = str(find_metrics_json(modality, model_id, "test") or "")

    seed = int(settings.get("seed", 0))
    force = int(settings.get("force", 0))

    # ensure detector once (anchor), then reuse
    det_candidates = sorted(
        (results_dir / "robot_text").rglob(f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    det_path = det_candidates[0] if det_candidates else None
    if det_path is None:
        anchor_flags = ["--concept-mode", "hard", "--skip-fit", "0", "--force-rerun", str(force)]
        run_spec(
            prefix="anchor",
            regime="anchor",
            human_acc=float(settings.get("best_human_acc", 1.0)),
            blackbox_metrics=bb,
            tag_suffix="cbm_anchor",
            concept_source="detected",
            extra_flags=anchor_flags,
            detector_model=None,
        )
        det_candidates = sorted(
            (results_dir / "robot_text").rglob(f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        det_path = det_candidates[0] if det_candidates else None
        if det_path is None:
            raise FileNotFoundError("Detector not produced for detected-CBM runs")

    def _listify(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, str):
            return [float(x) for x in v.split(",") if x.strip() != ""]
        return [float(v)]

    # # detected-CBM: best + expert sweeps
    cbm_flags = ["--skip-fit", "1"]
    run_spec(
        prefix="cbm",
        regime="best",
        human_acc=float(settings.get("best_human_acc", 1.0)),
        blackbox_metrics=bb,
        tag_suffix="cbm",
        concept_source="detected",
        extra_flags=cbm_flags,
        detector_model=str(det_path),
    )
    for h in _listify(settings.get("expert_human_accs")):
        run_spec(
            prefix="cbm",
            regime="expert",
            human_acc=float(h),
            blackbox_metrics=bb,
            tag_suffix="cbm",
            concept_source="detected",
            extra_flags=cbm_flags,
            detector_model=str(det_path),
        )

    # machine (LFCBM): best only
    lf_flags = []
    lf_flags += ["--lf-alpha", str(settings.get("lf_alpha", 0.5))]
    lf_flags += ["--lf-threshold", str(settings.get("lf_threshold", 0.5))]
    lf_flags += ["--lf-mode", str(settings.get("lf_mode", "soft"))]
    if bool(settings.get("lf_ridge", False)):
        lf_flags += ["--lf-ridge"]
    lf_flags += ["--lf-ridge-alpha", str(settings.get("lf_ridge_alpha", 1.0))]
    lf_flags += ["--lf-encoder", str(settings.get("lf_encoder", "sentence-transformers/all-MiniLM-L6-v2"))]
    _lf_dev = settings.get("lf_device")
    if _lf_dev is None:
        try:
            import torch
            _lf_dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _lf_dev = "cpu"
    lf_flags += ["--lf-device", str(_lf_dev)]
    lf_flags += ["--lf-batch-size", str(settings.get("lf_batch_size", 64))]
    lf_flags += ["--lf-group-threshold", str(settings.get("lf_group_threshold", 0.9))]
    lf_flags += ["--lf-keep-k", str(settings.get("lf_keep_k", settings.get("lf_topk_concepts", 9)))]

    run_spec(
        prefix="cbm",
        regime="best",
        human_acc=float(settings.get("best_human_acc", 1.0)),
        blackbox_metrics=bb,
        tag_suffix="lfcbm",
        concept_source="machine",
        extra_flags=lf_flags,
        detector_model=None,
    )


def recompute_metrics():
    base = results_dir / "robot_baseline" / "text"
    rows = []
    for run_dir in sorted(base.glob("*")):
        if not run_dir.is_dir():
            continue
        parts = sorted(run_dir.glob("baseline_dnn_robots_text_*_metrics_*.json"))
        for p in parts:
            try:
                data = json.loads(p.read_text())
            except Exception:
                data = {}
            row = {"run_dir": run_dir.name, "file": p.name}
            for k, v in data.items():
                row[k] = v
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        out = results_dir / "robot_text" / "_summary"
        out.mkdir(parents=True, exist_ok=True)
        eff = out / "baseline_metrics_flat.csv"
        df.to_csv(eff, index=False)

def build_final_accuracy_table():
    tbl = []
    base = results_dir / "robot_baseline" / "text"
    for run_dir in sorted(base.glob("*")):
        if not run_dir.is_dir():
            continue
        row = {"run_dir": run_dir.name}
        for split in ("train", "val", "test", "deploy"):
            p = sorted(run_dir.glob(f"baseline_dnn_robots_text_*_metrics_{split}.json"))
            if not p:
                continue
            try:
                data = json.loads(p[0].read_text())
            except Exception:
                data = {}
            row[f"{split}_accuracy"] = data.get("accuracy")
            row[f"{split}_balanced_acc"] = data.get("balanced_acc")
            row[f"{split}_coverage"] = data.get("coverage")
            row[f"{split}_selective_accuracy"] = data.get("selective_accuracy")
            row[f"{split}_tau"] = data.get("tau")
            row[f"{split}_tau_target"] = data.get("tau_target")
        tbl.append(row)
    if tbl:
        df = pd.DataFrame(tbl)
        out = results_dir / "robot_text" / "_summary"
        out.mkdir(parents=True, exist_ok=True)
        eff_path = out / "final_accuracy_table.csv"
        df.to_csv(eff_path, index=False)
        print("Wrote", eff_path)

    concept_recs = []
    robot_text_dir = results_dir / "robot_text"
    sub_all = sorted(robot_text_dir.glob("**/subtype_stats_*_*.csv"))
    for p in sub_all:
        try:
            dfp = pd.read_csv(p)
            dfp["source_file"] = str(p)
            concept_recs.append(dfp)
        except Exception:
            pass
    if concept_recs:
        cdf = pd.concat(concept_recs, ignore_index=True)
        outdir = results_dir / "robot_text" / "_summary"
        c_path = outdir / f"intervention_effectiveness_per_concept_seed{settings['seed']}.csv"
        cdf.to_csv(c_path, index=False)
        print("Wrote", c_path)

parse_cli()
if int(settings.get("recompute_only", 0)) == 1:
    recompute_metrics()
    build_final_accuracy_table()
    try:
        from scripts.report_text_tables import build_text_summary_tables
        build_text_summary_tables()
    except Exception:
        pass
else:
    run()
#     build_final_accuracy_table()
#     try:
#         from scripts.report_text_tables import build_text_summary_tables
#         build_text_summary_tables()
#     except Exception:
#         pass
