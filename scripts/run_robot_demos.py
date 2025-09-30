# scripts/run_robot_demos.py
import glob
from pathlib import Path
import sys, subprocess, shlex, os
import argparse
from typing import List, Optional
import pandas as pd
import math
import json
from concept_benchmark.paths import repo_dir, results_dir

ROOT = repo_dir
PY = sys.executable
GEN = ROOT / "scripts" / "gen_text_samples.py"
DNN = ROOT / "scripts" / "robot_baseline.py"

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT) + (os.pathsep + ENV.get("PYTHONPATH", ""))
#
# settings = {
#     "seed": 1337,
#     "difficulty": "hard",
#     "budgets": [0, 1, 2, 5, 10],
#     "force": 0,
#     "reuse_detector": 1,
#     "run_tag": "newwer_run",
#
#     "modality": "text",
#     "text_model": "distilbert-base-uncased",
#
#     "best_human_acc": 1.00,
#     "expert_human_acc": 0.80,
#     "subjective_human_acc": 0.80,
#
#     "subjective_noise_mode": "subjective",
#     "subjective_noise_rate": 0.20,
#
#     "label_model_type": "stochastic",
#     "label_model_alpha": 100.0,
#     "label_model_bias": 0.5,
#     "label_model_expr": "",
#
#     "templates_file": "",
#     "redact_concepts": "has_antennae",
#     "redact_splits": "test",
#
#     "generic_rate": 0.5,
#     "generic_tol": 0.04,
#     "generic_enable": 0,
#
#     "train_target_generic_frac": 0.0,
#     "val_target_generic_frac": 0.0,
#     "test_target_generic_frac": 0.0,
#
#     "recompute_only": 0,
# }

settings = {
    "seed": 1337,
    "difficulty": "hard",
    "budgets": [0, 1, 2, 5, 10],
    "force": 0,
    "reuse_detector": 1,
    "run_tag": "unbalanced_pixelated",

    "modality": "text",
    "text_model": "distilbert-base-uncased",

    "best_human_acc": 1.00,
    "expert_human_acc": 0.80,
    "subjective_human_acc": 0.80,

    "subjective_noise_mode": "subjective",
    "subjective_noise_rate": 0.20,

    "label_model_type": "stochastic",
    "label_model_alpha": 100.0,
    "label_model_bias": 0.2,
    "label_model_expr": "'glorp' if (min(int(str(row['ears_shape'])=='triangle'), int(row['body_shape']=='square')) >= 1) else 'drent'",

    "templates_file": "",
    "redact_concepts": "",
    "redact_splits": "",

    "generic_rate": 0.70,
    "generic_tol": 0.04,
    "generic_enable": 1,

    "train_target_generic_frac": 0.7,
    "val_target_generic_frac": 0.7,
    "test_target_generic_frac": 0.7,

    "recompute_only": 0,
}

def parse_cli():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--difficulty", type=str)
    ap.add_argument("--budgets", type=str)
    ap.add_argument("--force", type=int)
    ap.add_argument("--reuse_detector", type=int)
    ap.add_argument("--run_tag", type=str)
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
    ap.add_argument("--generic_rate", type=float)
    ap.add_argument("--generic_tol", type=float)
    ap.add_argument("--generic_enable", type=int)
    ap.add_argument("--train_target_generic_frac", type=float)
    ap.add_argument("--val_target_generic_frac", type=float)
    ap.add_argument("--test_target_generic_frac", type=float)
    ap.add_argument("--recompute-only", type=int)
    args, _ = ap.parse_known_args()
    d = {k: v for k, v in vars(args).items() if v is not None}
    if "budgets" in d and isinstance(d["budgets"], str):
        d["budgets"] = [int(x) for x in d["budgets"].split(",") if x.strip() != ""]
    settings.update(d)

def run_cmd(label: str, argv: List[str]):
    cmd = [str(PY), *map(str, argv)]
    print(f"[RUN] {label}\n$ {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True, env=ENV)

def ensure_baseline(seed: int, diff: str, tag: str = "", extra_flags: Optional[List[str]] = None) -> Path:
    extra_flags = extra_flags or []
    run_name = f"baseline_text_{diff}_seed{seed}"

    if settings.get("generic_enable", 0) == 1:
        run_name = f"{run_name}_gen{int(100*float(settings.get('generic_rate', 0.0))):02d}"
    if tag:
        run_name = f"{run_name}_{tag}"
        
    out_dir = results_dir / "robot_baseline" / "text" / run_name
    metrics = find_metrics_json(out_dir)
    if metrics is not None:
        return metrics
    argv = [
        str(DNN),
        "--modality", settings["modality"],
        "--seed", str(seed),
        "--template-difficulty", diff,
        "--run-name", run_name,
        "--text_model", settings.get("text_model", "distilbert-base-uncased"),
    ]
    if settings.get("templates_file"):
        argv += ["--templates-file", settings["templates_file"]]
    if settings.get("label_model_type"):
        argv += ["--label-model-type", str(settings["label_model_type"])]
    if settings.get("label_model_alpha") is not None:
        argv += ["--label-model-alpha", str(settings["label_model_alpha"])]
    if settings.get("label_model_bias") is not None:
        argv += ["--label-model-bias", str(settings["label_model_bias"])]
    if settings.get("label_model_expr"):
        argv += ["--label-model-expr", str(settings["label_model_expr"])]
    if settings.get("redact_concepts"):
        argv += ["--redact-concepts", str(settings["redact_concepts"])]
    if settings.get("redact_splits"):
        argv += ["--redact-splits", str(settings["redact_splits"])]
    if settings.get("generic_enable", 0) == 1:
        argv += ["--generic-enable", "1"]
        if settings.get("generic_rate") is not None:
            argv += ["--generic-rate", str(settings["generic_rate"])]
        if settings.get("generic_tol") is not None:
            argv += ["--generic-tol", str(settings["generic_tol"])]
    if extra_flags:
        argv += list(map(str, extra_flags))
    run_cmd(f"baseline ({run_name})", argv)
    metrics = find_metrics_json(out_dir)
    if metrics is None:
        raise FileNotFoundError(f"Could not find baseline metrics json in {out_dir}")
    return metrics

def find_metrics_json(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    named = sorted(run_dir.rglob("baseline_dnn_robots_*_metrics.json"))
    if named:
        return named[0]
    generic = sorted(run_dir.rglob("*metrics*.json"))
    return generic[0] if generic else None

def common_gen_argv(seed: int, diff: str, budgets: List[int]) -> List[str]:
    argv = [
        str(GEN),
        "--variant", "perfect",
        "--seed", str(seed),
        "--template-difficulty", diff,
        "--policy", "uncertainty",
        "--budgets", ",".join(str(k) for k in budgets),
    ]
    if settings.get("concept_mode"):
        argv += ["--concept-mode", settings["concept_mode"]]
    if settings.get("templates_file"):
        argv += ["--templates-file", settings["templates_file"]]
    if settings.get("label_model_type"):
        argv += ["--label-model-type", str(settings["label_model_type"])]
    if settings.get("label_model_alpha") is not None:
        argv += ["--label-model-alpha", str(settings["label_model_alpha"])]
    if settings.get("label_model_bias") is not None:
        argv += ["--label-model-bias", str(settings["label_model_bias"])]
    if settings.get("label_model_expr"):
        argv += ["--label-model-expr", str(settings["label_model_expr"])]
    if settings.get("redact_concepts"):
        argv += ["--redact-concepts", settings["redact_concepts"]]
    if settings.get("redact_splits"):
        argv += ["--redact-splits", settings["redact_splits"]]
    if settings.get("reuse_detector") is not None:
        argv += ["--reuse-detector", str(int(settings["reuse_detector"]))]
    if settings.get("force"):
        argv += ["--force-rerun", str(int(settings["force"]))]
    if settings.get("generic_rate") is not None:
        argv += ["--generic-rate", str(settings["generic_rate"])]
    if settings.get("generic_tol") is not None:
        argv += ["--generic-tol", str(settings["generic_tol"])]
    return argv

def run_spec(view: str, regime: str, human_acc: float, bb_metrics: Path, tag: str, concept_source: str, extra_flags: Optional[List[str]] = None, detector_model: Optional[Path] = None):
    extra_flags = extra_flags or []
    modes = [settings.get("intervention_error_mode", "both")]
    if modes[0] == "both":
        modes = ["miss", "flip"]

    for mode in modes:
        argv = common_gen_argv(settings["seed"], settings["difficulty"], settings["budgets"])
        argv += ["--concept-source", concept_source]
        if detector_model is not None:
            argv += ["--reuse-detector", "1", "--detector-model", str(detector_model)]
        argv += ["--human-acc", str(human_acc)]
        argv += ["--blackbox_metrics", str(bb_metrics)]
        tag2 = f"{tag}_{'v1' if mode=='miss' else 'v2'}"
        argv += ["--run-name", make_run_name(view, regime, human_acc, tag2)]
        argv += ["--intervention-error-mode", mode]
        argv += list(map(str, extra_flags))
        run_cmd(f"{view}:{regime}:{mode}", argv)



def make_run_name(view: str, regime: str, human_acc: float, tag: str) -> str:
    kset = "_".join(str(k) for k in settings["budgets"])
    return f"{regime}_{view}_{tag}_intervene{int(human_acc*100)}_kset-{kset}_{settings['difficulty']}_seed{settings['seed']}"

def run():
    parse_cli()
    seed = settings["seed"]
    diff = settings["difficulty"]

    import hashlib, json as _json
    label_expr = settings.get(
        "label_model_expr") or "'glorp' if (min(int(str(row['has_antennae']).lower()=='true'), int(row['body_shape']=='square')) >= 1) else 'drent'"
    sig = _json.dumps({
        "seed": seed,
        "diff": diff,
        "type": settings.get("label_model_type", "stochastic"),
        "alpha": settings.get("label_model_alpha", 100.0),
        "bias": settings.get("label_model_bias", 0.5),
        "expr": label_expr,
        "templates": settings.get("templates_file") or "",
        "budgets": settings.get("budgets", []),
        "gen_enable": int(settings.get("generic_enable", 0)),
        "gen_rate": float(settings.get("generic_rate", 0.0)),
        "gen_tol": float(settings.get("generic_tol", 0.0)),
    }, sort_keys=True)
    tag_id = (settings.get("run_tag") or hashlib.sha256(sig.encode("utf-8")).hexdigest()[:8])


    label_flags = [
        "--label-model-type", str(settings.get("label_model_type", "stochastic")),
        "--label-model-alpha", str(settings.get("label_model_alpha", 100)),
        "--label-model-bias", str(settings.get("label_model_bias", 0.5)),
    ]
    gen_label_flags = label_flags + ["--label-model-expr", label_expr]

    extra = [
        "--samples_per_instance", "1",
        "--train-balance-enable", "1",
        "--train-target-pos-frac", "-1",
        "--train-target-generic-frac", str(settings.get("generic_rate", 0.5)),
        "--train-balance-within-label", "1",
        "--val-balance-enable", "1",
        "--val-target-generic-frac", str(settings.get("generic_rate", 0.5)),
        "--val-balance-within-label", "1",
        "--test-balance-enable", "1",
        "--test-target-generic-frac", str(settings.get("generic_rate", 0.5)),
        "--test-balance-within-label", "1",
    ]
    bb = ensure_baseline(seed, diff, tag=f"minrule_{tag_id}", extra_flags=extra)

    det_path = None
    csv_file = None

    cs_anchor_tag = f"cs_anchor_{tag_id}"
    cs_anchor_run_v1 = make_run_name("anchor", "anchor", settings["best_human_acc"], f"{cs_anchor_tag}_v1")
    cs_anchor_run_v0 = make_run_name("anchor", "anchor", settings["best_human_acc"], cs_anchor_tag)
    pkl_v1 = results_dir / "robot_text" / cs_anchor_run_v1 / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
    pkl_v0 = results_dir / "robot_text" / cs_anchor_run_v0 / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
    if pkl_v1.exists() or (results_dir / "robot_text" / cs_anchor_run_v1).exists():
        cs_anchor_run = cs_anchor_run_v1
    elif pkl_v0.exists() or (results_dir / "robot_text" / cs_anchor_run_v0).exists():
        cs_anchor_run = cs_anchor_run_v0
    else:
        cs_anchor_run = cs_anchor_run_v1

    def _latest(path_globs):
        c = []
        for gpat in path_globs:
            c += glob.glob(gpat, recursive=True)
        c = [Path(p) for p in c]
        c = [p for p in c if p.exists()]
        c.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return c[0] if c else None

    det_path = results_dir / "robot_text" / cs_anchor_run / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
    if not det_path.exists():
        anchor_flags = ["--variants-per-row", "1", "--force-rerun", "1", "--concept-mode", "soft"]
        run_spec("anchor", "anchor", settings["best_human_acc"], bb, cs_anchor_tag, "detected",
                 extra_flags=anchor_flags)
        det_path = results_dir / "robot_text" / cs_anchor_run / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
        if not det_path.exists():
            raise FileNotFoundError(f"Missing detector at {det_path}")

    patterns = [
        str(results_dir / "robot_text" / cs_anchor_run / f"text_samples_*_seed{settings['seed']}.csv"),
    ]

    cands = []
    for pat in patterns:
        cands += glob.glob(pat, recursive=True)
    cands = sorted(set(cands))
    if not cands:
        raise FileNotFoundError("No text_samples CSV found to compute minority upsample factor.")

    labels_series = None
    picked = None
    for fp in reversed(cands):
        try:
            _df = pd.read_csv(fp)
        except Exception:
            continue
        if "label" in _df.columns:
            ser = _df["label"]
            if ser.dtype != object and ser.dropna().isin([0, 1]).all():
                ser = ser.map({1: "glorp", 0: "drent"})
            labels_series = ser.astype(str)
            picked = fp
            break
        if "y" in _df.columns:
            labels_series = _df["y"].map({1: "glorp", 0: "drent", "1": "glorp", "0": "drent"}).astype(str)
            picked = fp
            break
        if {"ears_shape", "body_shape"}.issubset(_df.columns):
            ser = _df.apply(
                lambda r: "glorp" if (str(r["ears_shape"]) == "square" and str(r["body_shape"]) == "square") else "drent",
                axis=1
            )
            labels_series = ser.astype(str)
            picked = fp
            break

    if labels_series is None:
        raise FileNotFoundError("No suitable text_samples CSV with label, y, or concept columns")

    csv_file = picked
    counts = labels_series.value_counts()
    n_pos = int(counts.get("glorp", 0))
    n_neg = int(counts.get("drent", 0))
    n_min = min(n_pos, n_neg)
    n_maj = max(n_pos, n_neg)
    vpr_min = max(1, math.ceil(n_maj / max(1, n_min)))
    pos_frac_overall = (n_pos / max(1, (n_pos + n_neg)))
    print("vpr_min", vpr_min)

    c_flags = [
        "--variants-per-row", "1",
        "--variants-per-row-minority", str(int(vpr_min)),
        "--skip-fit", "0", "--force-rerun", "1",
        "--concept-mode", "soft",
    ]
    if settings.get("generic_enable", 0) == 1:
        c_flags += [
            "--train-balance-enable", "1",
            "--train-target-generic-frac", str(settings.get("generic_rate", 0.5)),
            "--train-balance-within-label", "1",
            "--train-target-pos-frac", f"{pos_frac_overall:.6f}",
            "--val-balance-enable", "1",
            "--val-target-generic-frac", str(settings.get("generic_rate", 0.5)),
            "--val-balance-within-label", "1",
            "--test-balance-enable", "1",
            "--test-target-generic-frac", str(settings.get("generic_rate", 0.5)),
            "--test-balance-within-label", "1",
        ]

    tag = f"minrule_eval_vpr{int(vpr_min)}_pos{n_pos}neg{n_neg}_{tag_id}_seed{settings['seed']}"

    run_spec("cs", "best", settings["best_human_acc"], bb, tag, "detected",
             extra_flags=c_flags, detector_model=det_path)
    run_spec("cs", "expert", settings["expert_human_acc"], bb, tag, "detected",
             extra_flags=c_flags, detector_model=det_path)
    c_flags_noise = c_flags + ["--concept-label-noise-mode", settings.get("subjective_noise_mode", "subjective"),
                               "--concept-label-noise-rate", str(settings.get("subjective_noise_rate", 0.20)),
                               "--skip-fit", "1", "--force-rerun", "1"]
    run_spec("cs", "subjective", settings["subjective_human_acc"], bb, tag + "_noise", "detected",
             extra_flags=c_flags_noise, detector_model=det_path)

    cbm_anchor_tag = f"cbm_anchor_{tag_id}"
    cbm_anchor_run_v1 = make_run_name("anchor", "anchor", settings["best_human_acc"], f"{cbm_anchor_tag}_v1")
    cbm_anchor_run_v0 = make_run_name("anchor", "anchor", settings["best_human_acc"], cbm_anchor_tag)
    pkl_v1 = results_dir / "robot_text" / cbm_anchor_run_v1 / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
    pkl_v0 = results_dir / "robot_text" / cbm_anchor_run_v0 / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
    if pkl_v1.exists() or (results_dir / "robot_text" / cbm_anchor_run_v1).exists():
        cbm_anchor_run = cbm_anchor_run_v1
        cbm_det_path = pkl_v1
    elif pkl_v0.exists() or (results_dir / "robot_text" / cbm_anchor_run_v0).exists():
        cbm_anchor_run = cbm_anchor_run_v0
        cbm_det_path = pkl_v0
    else:
        cbm_anchor_run = cbm_anchor_run_v1
        cbm_det_path = results_dir / "robot_text" / cbm_anchor_run / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
    if not cbm_det_path.exists():
        cbm_anchor_flags = ["--variants-per-row", "1", "--force-rerun", "1", "--concept-mode", "hard"]
        run_spec("anchor", "anchor", settings["best_human_acc"], bb, cbm_anchor_tag, "detected",
                 extra_flags=cbm_anchor_flags)
        cbm_det_path = results_dir / "robot_text" / cbm_anchor_run_v1 / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
        if not cbm_det_path.exists():
            cbm_det_path = results_dir / "robot_text" / cbm_anchor_run_v0 / f"cbm_fe_gt_robots_text_complete_seed{seed}.pkl"
        if not cbm_det_path.exists():
            raise FileNotFoundError(f"Missing detector at {cbm_det_path}")

    cbm_flags = [
        "--variants-per-row", "1",
        "--variants-per-row-minority", str(int(vpr_min)),
        "--skip-fit", "1", "--force-rerun", "1",
        "--concept-mode", "hard",
    ]

    if settings.get("generic_enable", 0) == 1:
        cbm_flags += [
            "--train-balance-enable", "1",
            "--train-target-generic-frac", str(settings.get("generic_rate", 0.5)),
            "--train-balance-within-label", "1",
            "--train-target-pos-frac", f"{pos_frac_overall:.6f}",
            "--val-balance-enable", "1",
            "--val-target-generic-frac", str(settings.get("generic_rate", 0.5)),
            "--val-balance-within-label", "1",
            "--test-balance-enable", "1",
            "--test-target-generic-frac", str(settings.get("generic_rate", 0.5)),
            "--test-balance-within-label", "1",
        ]

    run_spec("cbm", "best", settings["best_human_acc"], bb, tag, "detected",
             extra_flags=cbm_flags, detector_model=cbm_det_path)
    run_spec("cbm", "expert", settings["expert_human_acc"], bb, tag, "detected",
             extra_flags=cbm_flags, detector_model=cbm_det_path)
    cbm_flags_noise = cbm_flags + ["--concept-label-noise-mode", settings.get("subjective_noise_mode", "subjective"),
                                   "--concept-label-noise-rate", str(settings.get("subjective_noise_rate", 0.20)),
                                   "--skip-fit", "1", "--force-rerun", "1"]
    run_spec("cbm", "subjective", settings["subjective_human_acc"], bb, tag + "_noise", "detected",
             extra_flags=cbm_flags_noise, detector_model=cbm_det_path)


def recompute_metrics():
    seed = int(settings.get("seed", 0))
    base = results_dir / "robot_text"
    meta_glob = f"**/meta_cbm_fe_*_robots_text_complete_seed{seed}.json"
    all_meta = sorted(base.rglob(meta_glob))
    rtag = str(settings.get("run_tag", "")).strip()
    meta_paths = [p for p in all_meta if (not rtag or rtag in str(p.parent))]
    if not meta_paths:
        return
    for mp in meta_paths:
        try:
            args = json.loads(Path(mp).read_text()).get("args", {})
        except Exception:
            continue
        base_argv = [str(GEN)]

        def add_arg(k, v):
            if v is None:
                return
            base_argv.extend([f"--{k.replace('_', '-')}", str(v)])

        add_arg("variant", args.get("variant"))
        add_arg("variants_per_row", args.get("variants_per_row"))
        add_arg("imperfect_strategy", args.get("imperfect_strategy"))
        add_arg("heldout_concepts", args.get("heldout_concepts"))
        add_arg("mask_p", args.get("mask_p"))
        add_arg("mask_mode", args.get("mask_mode"))
        add_arg("mask_rate", args.get("mask_rate"))
        add_arg("seed", args.get("seed"))
        add_arg("concept_mode", args.get("concept_mode"))
        add_arg("templates_file", args.get("templates_file"))
        rc = settings.get("redact_concepts") if "redact_concepts" in settings else args.get("redact_concepts")
        rs = settings.get("redact_splits") if "redact_splits" in settings else args.get("redact_splits")
        add_arg("redact_concepts", rc)
        add_arg("redact_splits", rs)
        expr = settings.get("label_model_expr") if "label_model_expr" in settings else args.get("label_model_expr")
        add_arg("label_model_expr", expr)
        add_arg("corr_pair", args.get("corr_pair"))
        add_arg("train_corr", args.get("train_corr"))
        add_arg("test_break", args.get("test_break"))
        add_arg("test_corr", args.get("test_corr"))
        add_arg("budgets", args.get("budgets"))
        add_arg("target_acc_grid", args.get("target_acc_grid"))
        add_arg("target_acc_concepts", args.get("target_acc_concepts"))
        add_arg("intervene_allow", args.get("intervene_allow"))
        add_arg("human_acc", args.get("human_acc"))
        add_arg("human_acc_concepts", args.get("human_acc_concepts"))
        add_arg("make_plots", args.get("make_plots"))
        add_arg("policy", args.get("policy"))
        ci = args.get("concept_include")
        ce = args.get("concept_exclude")
        if ci not in (None, "", "[]"): add_arg("concept_include", ci)
        if ce not in (None, "", "[]"): add_arg("concept_exclude", ce)

        bm = args.get("blackbox_metrics")
        if not bm or not Path(bm).exists():
            cand = sorted((results_dir / "robot_baseline").rglob("baseline_*metrics.json"))
            if cand:
                bm = str(cand[-1])
        if bm:
            add_arg("blackbox_metrics", bm)

        mobj = json.loads(Path(mp).read_text())
        det = mobj.get("artifacts", {}).get("model") or args.get("detector_model")

        mode = settings.get("intervention_error_mode", "both")
        argv = list(base_argv)
        argv += ["--skip-fit", "1"]
        if det and Path(det).exists():
            argv += ["--reuse-detector", "1", "--detector-model", det]
        else:
            print("Warning: missing detector model", det)
        argv += ["--force-rerun", "1", "--intervention-error-mode", mode]
        rn = args.get("run_name") or ""
        if rn:
            argv += ["--run-name", rn]
        run_cmd("recompute", argv)


def build_final_accuracy_table():
    import re
    files = sorted(glob.glob(str(results_dir / "robot_text" / "**" / "viability_robots_text_*_detected.csv"), recursive=True))
    if not files:
        return

    recs = []
    for f in files:
        run_name = Path(f).parent.name
        m = re.search(r'^(best|expert|subjective)_(cs|cbm)_', run_name)
        if m:
            regime, view = m.group(1), m.group(2)
        else:
            regime = "best" if "best" in run_name else ("expert" if "expert" in run_name else ("subjective" if "subjective" in run_name else "unknown"))
            view = "cs" if "_cs_" in run_name else ("cbm" if "_cbm_" in run_name else "unknown")
        df = pd.read_csv(f)
        if "acc_cbm_intv" not in df.columns:
            continue
        if "budget" not in df.columns and "k" in df.columns:
            df = df.rename(columns={"k":"budget"})
        sub = df[df["budget"].isin(settings["budgets"])]
        if sub.empty:
            continue
        g = sub.groupby("budget", as_index=False).mean(numeric_only=True)
        for _, row in g.iterrows():
            recs.append({
                "regime": regime,
                "budget": int(row["budget"]),
                "method": view.upper(),
                "accuracy": float(row["acc_cbm_intv"])
            })
        if "delta_vs_blackbox" in df.columns:
            dnn_series = df["acc_cbm_intv"] - df["delta_vs_blackbox"]
        elif "gain_acc_dnn" in df.columns:
            dnn_series = df["acc_cbm_intv"] - df["gain_acc_dnn"]
        else:
            dnn_series = None
        if dnn_series is not None and len(dnn_series) > 0:
            if (df["budget"] == 0).any():
                dnn_val = float(dnn_series[df["budget"] == 0].mean())
            else:
                dnn_val = float(dnn_series.mean())
            for b in sorted(sub["budget"].unique()):
                recs.append({
                    "regime": "best",
                    "budget": int(b),
                    "method": "DNN",
                    "accuracy": dnn_val,
                })

    if not recs:
        return

    tidy = pd.DataFrame(recs)
    tidy["method"] = tidy["method"].replace({"ANCHOR": "DNN"})
    tidy = tidy.groupby(["regime","budget","method"], as_index=False)["accuracy"].mean()
    tidy["method"] = tidy["method"].astype(pd.CategoricalDtype(categories=["DNN","CBM","CS"], ordered=True))
    tidy = tidy[tidy["method"].notna()]
    pivot = tidy.pivot_table(index=["budget","method"], columns="regime", values="accuracy", aggfunc="mean").sort_index(level=[0,1])
    if "unknown" in pivot.columns:
        pivot = pivot.drop(columns=["unknown"])
    pivot = pivot.round(4)

    outdir = results_dir / "robot_text" / "_summary"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"final_accuracy_table_seed{settings['seed']}.csv"
    tex_path = outdir / f"final_accuracy_table_seed{settings['seed']}.tex"
    pivot.to_csv(csv_path)
    with open(tex_path, "w") as f:
        f.write(pivot.to_latex(na_rep="--", float_format="%.4f"))
    print("Wrote", csv_path)
    print("Wrote", tex_path)
    print(pivot)

    eff_recs = []
    concept_recs = []
    for f in files:
        run_dir = Path(f).parent
        run_name = run_dir.name
        import re as _re
        m = _re.search(r'^(best|expert|subjective)_(cs|cbm)_', run_name)
        if m:
            regime, view = m.group(1), m.group(2)
        else:
            regime = "best" if "best" in run_name else ("expert" if "expert" in run_name else ("subjective" if "subjective" in run_name else "unknown"))
            view = "cs" if "_cs_" in run_name else ("cbm" if "_cbm_" in run_name else "unknown")

        dfv = pd.read_csv(f)
        df_b = dfv if "budget" in dfv.columns else (dfv.rename(columns={"k": "budget"}) if "k" in dfv.columns else None)
        if df_b is not None:
            sub = df_b[df_b["budget"].isin(settings["budgets"])].copy()
            if not sub.empty:
                sub.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

                if "concept_checks" in sub.columns:
                    cc = pd.to_numeric(sub["concept_checks"], errors="coerce")
                elif "confirmation_cost" in sub.columns:
                    cc = pd.to_numeric(sub["confirmation_cost"], errors="coerce")
                else:
                    cc = pd.Series(pd.NA, index=sub.index, dtype="float64")

                bvec = pd.to_numeric(sub["budget"], errors="coerce")
                sub["_n_cases"] = cc / bvec.replace(0, pd.NA)

                if "interventions_pct" not in sub.columns:
                    sub["interventions_pct"] = pd.NA
                if "concepts_per_intervention" not in sub.columns:
                    sub["concepts_per_intervention"] = pd.NA
                if "avg_edits_per_case" not in sub.columns:
                    sub["avg_edits_per_case"] = pd.NA

                if "interventions_total" in sub.columns:
                    itot = pd.to_numeric(sub["interventions_total"], errors="coerce")
                    sub["interventions_pct"] = sub["interventions_pct"].fillna(itot / sub["_n_cases"])

                if {"applied_edits_total", "interventions_total"}.issubset(sub.columns):
                    etot = pd.to_numeric(sub["applied_edits_total"], errors="coerce")
                    itot = pd.to_numeric(sub["interventions_total"], errors="coerce")
                    sub["concepts_per_intervention"] = sub["concepts_per_intervention"].fillna(
                        etot / itot.replace(0, pd.NA))

                if {"interventions_pct", "concepts_per_intervention"}.issubset(sub.columns):
                    ie = pd.to_numeric(sub["interventions_pct"], errors="coerce")
                    ci = pd.to_numeric(sub["concepts_per_intervention"], errors="coerce")
                    sub["avg_edits_per_case"] = sub["avg_edits_per_case"].fillna(ie * ci)

                if "applied_edits_total" in sub.columns:
                    etot = pd.to_numeric(sub["applied_edits_total"], errors="coerce")
                    sub["avg_edits_per_case"] = sub["avg_edits_per_case"].fillna(etot / sub["_n_cases"])

                sub.loc[bvec == 0, "avg_edits_per_case"] = sub.loc[bvec == 0, "avg_edits_per_case"].fillna(0.0)

                g = sub.groupby("budget", as_index=False).mean(numeric_only=True)

                for _, r in g.iterrows():
                    avgv = r.get("avg_edits_per_case", float("nan"))
                    if not pd.notnull(avgv):
                        continue
                    eff_recs.append({
                        "regime": regime,
                        "method": view.upper(),
                        "budget": int(r["budget"]),
                        "avg_edits_per_case": float(avgv),
                        "interventions_pct": float(r.get("interventions_pct", float("nan"))),
                        "concepts_per_intervention": float(r.get("concepts_per_intervention", float("nan"))),
                        "failed_interventions_pct": float(r.get("failed_interventions_pct", float("nan"))),
                        "edit_effectiveness": float(r.get("edit_effectiveness", float("nan"))),
                        "edit_effectiveness_per_intervention": float(
                            r.get("edit_effectiveness_per_intervention", float("nan"))),
                    })

        conc_files = sorted(run_dir.glob("interventions_per_concept_*_detected*.csv"))
        if conc_files:
            parts = []
            for p in conc_files:
                dfc = pd.read_csv(p)
                if "budget" not in dfc.columns and "k" in dfc.columns:
                    dfc = dfc.rename(columns={"k":"budget"})
                if "budget" in dfc.columns:
                    parts.append(dfc[dfc["budget"].isin(settings["budgets"])])
            if parts:
                dfc = pd.concat(parts, ignore_index=True)
                if not dfc.empty:
                    g2 = dfc.groupby(["budget","concept"], as_index=False).agg({"interventions":"sum","correct":"sum"})
                    g2["correct_rate"] = g2.apply(lambda x: (x["correct"]/x["interventions"]) if x["interventions"]>0 else float("nan"), axis=1)
                    g2["regime"] = regime
                    g2["method"] = view.upper()
                    concept_recs.append(g2)

    if eff_recs:
        eff = pd.DataFrame(eff_recs)
        outdir = results_dir / "robot_text" / "_summary"
        eff_path = outdir / f"intervention_effectiveness_summary_seed{settings['seed']}.csv"
        eff.to_csv(eff_path, index=False)
        print("Wrote", eff_path)

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
else:
    run()
    build_final_accuracy_table()
