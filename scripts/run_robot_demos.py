# scripts/run_robot_demos.py
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
    "budgets": [0, 1, 2, 5, 10],
    "force": 0,
    "reuse_detector": 1,
    "run_tag": "new_try",

    "modality": "text",
    "text_model": "distilbert-base-uncased",

    "best_human_acc": 1.00,
    "expert_human_accs": [0.80, 1.00],
    "subjective_human_accs": [1.00],
    "subjective_noise_rates": [0.20],

    "skip_fit": 1,
    "make_plots": 0,

    "redact_concepts": "",
    "redact_splits": "",

    "concepts_csv": str(ROOT / "data" / "robot_text" / "concepts" / "concepts.csv"),
    "run_name_sub": "",

    "generic_enable": 1,
    "generic_rate": 0.7,
    "generic_tol": 0.1,

    "policy": "kflip",
    "k": 2,
    "flip_threshold": 0.30,
    "flip_batch_size": 8192,
    "flip_limit_subsets": None,
    "abstain_only": 0,

    "seed_test_offset": 1234,
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
    ap.add_argument("--label_model_alpha", type=float)
    ap.add_argument("--label_model_bias", type=float)
    ap.add_argument("--label_model_expr", type=str)
    ap.add_argument("--templates_file", type=str)
    ap.add_argument("--redact_concepts", type=str)
    ap.add_argument("--redact_splits", type=str)
    ap.add_argument("--concept_mode", type=str)
    ap.add_argument("--generic-rate", "--generic_rate", dest="generic_rate", type=float)
    ap.add_argument("--generic-tol", "--generic_tol", dest="generic_tol", type=float)
    ap.add_argument("--generic_enable", type=int)
    ap.add_argument("--train_target_generic_frac", type=float)
    ap.add_argument("--val_target_generic_frac", type=float)
    ap.add_argument("--test_target_generic_frac", type=float)
    ap.add_argument("--deployment_target_generic_frac", type=float)
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
    ap.add_argument("--calibrate", type=str)
    ap.add_argument("--abstain", type=str)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--tau-target", type=float)
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
    subprocess.check_call(argv, cwd=str(cwd) if cwd else None, env=ENV)

def ensure_baseline(model_id: str, modality: str):
    outdir = results_dir / "robot_baseline" / modality
    outdir.mkdir(parents=True, exist_ok=True)
    run_name = f"robots_{modality}_{model_id}_seed{int(settings.get('seed', 0))}"
    mfile = outdir / f"baseline_dnn_robots_{modality}_{model_id}_seed{int(settings.get('seed', 0))}_metrics_test.json"
    if mfile.exists():
        return
    argv = [str(PY), str(DNN)]
    if modality == "text":
        argv += ["--text_model", str(settings.get("text_model", "distilbert-base-uncased"))]
    if settings.get("calibrate"):
        argv += ["--calibrate", str(settings["calibrate"])]
    if settings.get("abstain"):
        argv += ["--abstain", str(settings["abstain"])]
    if settings.get("tau") is not None:
        argv += ["--tau", str(settings["tau"])]
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
    for flag, val in [
        ("--generic-rate", settings.get("generic_rate")),
        ("--generic-tol", settings.get("generic_tol")),
        ("--val-balance-enable", settings.get("val_balance_enable")),
        ("--test-balance-enable", settings.get("test_balance_enable")),
    ]:
        if val is not None:
            argv += [flag, str(val)]
    run_cmd(argv)

def find_metrics_json(modality: str, model_id: str, split: str):
    outdir = results_dir / "robot_baseline" / modality
    f = outdir / f"baseline_dnn_robots_{modality}_{model_id}_seed{int(settings.get('seed', 0))}_metrics_{split}.json"
    return f if f.exists() else None

def common_gen_argv():
    argv = [
        str(PY), str(GEN),
        "--modality", str(settings.get("modality", "text")),
        # "--concept-source", str(settings.get("concept_source", "gt")),
        "--machine-method", str(settings.get("machine_method", "lfcbm")),
        "--seed", str(settings.get("seed", 0)),
        "--seed-cv", str(settings.get("seed_cv", int(settings.get("seed", 0)) + 1)),
        "--cv-k", str(settings.get("cv_k", 5)),
        "--cv-fold", str(settings.get("cv_fold", 0)),
        "--dev-per-fold", str(settings.get("dev_per_fold", 1000)),
        "--deployment-size", str(settings.get("deployment_size", 10000)),
        "--subtype-mode", str(settings.get("subtype_mode", "track")),
        "--policy", str(settings.get("policy", settings.get("intervention_policy", "kflip"))),
        "--k", str(settings.get("k", settings.get("intervention_k", 2))),
        "--flip-threshold", str(settings.get("flip_threshold", 0.30)),
        "--flip-batch-size", str(settings.get("flip_batch_size", 8192)),
        "--budgets", ",".join(str(x) for x in settings.get("budgets", [0, 1, 2, 5, 10])),
    ]
    if settings.get("flip_limit_subsets") is not None:
        argv += ["--flip-limit-subsets", str(settings["flip_limit_subsets"])]
    if settings.get("abstain_only"):
        argv += ["--abstain-only"]
    if settings.get("tau") is not None:
        argv += ["--tau", str(settings["tau"])]
    return argv


def run_spec(prefix: str, regime: str, human_acc: float, blackbox_metrics: str, tag_suffix: str, concept_source: str,
             extra_flags: Optional[List[str]] = None, detector_model: Optional[str] = None):
    budgets = settings.get("budgets", [0, 1, 2, 5, 10])
    force = int(settings.get("force", 0))
    make_plots = int(settings.get("make_plots", 0))
    seed = int(settings.get("seed", 0))

    _sub = str(settings.get("run_name_sub", "")).strip()
    rn_base = f"{prefix}_{regime}_minrule_eval_vpr3_pos1152neg3456_unbalanced_pixelated_fixed_seed{seed}_v1_intervene{int(human_acc * 100)}_kset-0_1_2_5_10_{settings.get('difficulty', 'hard')}_seed{seed}"
    rn = f"{_sub}_{rn_base}" if _sub else rn_base

    # start with base cmd (includes [PY, GEN] + core flags and kflip knobs)
    argv = common_gen_argv()

    # scenario-specific flags
    argv += [
        "--variant", "perfect",
        "--variants-per-row", "1",
        "--imperfect-strategy", "missing_concepts",
        "--heldout-concepts", "[]",
        "--mask-p", "0.0",
        "--mask-mode", "mask",
        "--mask-rate", "0.0",
        "--concept-mode", "hard",
        "--templates-file", "",
        "--redact-concepts", str(settings.get("redact_concepts", "")),
        "--redact-splits", str(settings.get("redact_splits", "")),
        "--label-model-expr", "",
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
        "--concepts-csv", settings.get("concepts_csv", ""),
        "--generic-enable", str(int(settings.get("generic_enable", 1))),
        "--generic-rate", str(float(settings.get("generic_rate", 0.7))),
        "--generic-tol", str(float(settings.get("generic_tol", 0.02))),
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
        anchor_flags = ["--variants-per-row", "1", "--concept-mode", "hard", "--skip-fit", "0", "--force-rerun", str(force)]
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

    # detected-CBM: best + expert sweeps
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
    build_final_accuracy_table()
    try:
        from scripts.report_text_tables import build_text_summary_tables
        build_text_summary_tables()
    except Exception:
        pass
