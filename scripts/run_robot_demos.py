# scripts/run_robot_demos.py
import glob
from pathlib import Path
import sys, subprocess, shlex, os
import argparse
from typing import List, Optional
import pandas as pd
import math
from concept_benchmark.paths import repo_dir, results_dir

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
    "reuse_detector": 0,

    "modality": "text",
    "text_model": "distilbert-base-uncased",

    "best_human_acc": 1.00,
    "expert_human_acc": 0.80,
    "subjective_human_acc": 0.80,

    "subjective_noise_mode": "subjective",
    "subjective_noise_rate": 0.20,

    "label_model_type": "stochastic",
    "label_model_alpha": 2.772588722239781,
    "label_model_bias": -0.5,
    "label_model_expr": "",

    "templates_file": "",
}

def parse_cli():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--difficulty", type=str)
    ap.add_argument("--budgets", type=str)
    ap.add_argument("--force", type=int)
    ap.add_argument("--reuse_detector", type=int)
    ap.add_argument("--best_human_acc", type=float)
    ap.add_argument("--expert_human_acc", type=float)
    ap.add_argument("--subjective_human_acc", type=float)
    ap.add_argument("--subjective_noise_rate", type=float)
    ap.add_argument("--label_model_type", type=str)
    ap.add_argument("--label_model_alpha", type=float)
    ap.add_argument("--label_model_bias", type=float)
    ap.add_argument("--label_model_expr", type=str)
    ap.add_argument("--templates_file", type=str)
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
    if tag:
        run_name = f"{run_name}_{tag}"
    out_dir = results_dir / "robot_baseline" / "text" / run_name
    metrics = find_metrics_json(out_dir)
    if metrics is not None and settings["force"] == 0:
        return metrics
    argv = [
        str(DNN),
        "--modality", settings["modality"],
        "--seed", str(seed),
        "--template-difficulty", diff,
        "--run-name", run_name,
    ]
    if settings.get("templates_file"):
        argv += ["--templates-file", settings["templates_file"]]
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
    if settings.get("reuse_detector") is not None:
        argv += ["--reuse-detector", str(int(settings["reuse_detector"]))]
    if settings.get("force"):
        argv += ["--force-rerun", str(int(settings["force"]))]
    return argv

def run_spec(view: str, regime: str, human_acc: float, bb_metrics: Path, tag: str, concept_source: str, extra_flags: Optional[List[str]] = None, detector_model: Optional[Path] = None):
    extra_flags = extra_flags or []
    argv = common_gen_argv(settings["seed"], settings["difficulty"], settings["budgets"])
    argv += ["--concept-source", concept_source]
    if detector_model is not None:
        argv += ["--reuse-detector", "1", "--detector-model", str(detector_model)]
    argv += ["--human-acc", str(human_acc)]
    argv += ["--blackbox_metrics", str(bb_metrics)]
    argv += ["--run-name", make_run_name(view, regime, human_acc, tag)]
    argv += list(map(str, extra_flags))
    run_cmd(f"{view}:{regime}", argv)

def make_run_name(view: str, regime: str, human_acc: float, tag: str) -> str:
    kset = "_".join(str(k) for k in settings["budgets"])
    return f"{regime}_{view}_{tag}_intervene{int(human_acc*100)}_kset-{kset}_{settings['difficulty']}_seed{settings['seed']}"

def run():
    parse_cli()
    seed = settings["seed"]
    diff = settings["difficulty"]

    bb = ensure_baseline(seed, diff, tag="balanced_class", extra_flags=["--variants-per-row", "1"])

    # fresh anchor name to avoid collisions
    anchor = make_run_name("anchor", "anchor", settings["best_human_acc"], "trainGT_inferDET_balanced_class_fresh")
    run_spec("anchor", "anchor", settings["best_human_acc"], bb, "trainGT_inferDET_balanced_class_fresh", "detected",
             extra_flags=["--variants-per-row", "1"])
    det_path = results_dir / "robot_text" / anchor / f"cbm_fe_gt_robots_text_complete_seed{settings['seed']}.pkl"

    csv_glob = str(results_dir / "robot_text" / anchor / f"text_samples_*_seed{settings['seed']}.csv")
    csv_matches = sorted(glob.glob(csv_glob))
    if not csv_matches:
        csv_glob_alt = str(results_dir / "robot_text" / "**" / f"text_samples_*_seed{settings['seed']}.csv")
        csv_matches = sorted(glob.glob(csv_glob_alt, recursive=True))
    if not csv_matches:
        raise FileNotFoundError("No text_samples CSV found to compute minority upsample factor.")
    df_c = pd.read_csv(csv_matches[-1])
    counts = df_c["label"].astype(str).value_counts()
    n_pos = int(counts.get("glorp", 0))
    n_neg = int(counts.get("drent", 0))
    n_min = min(n_pos, n_neg)
    n_maj = max(n_pos, n_neg)
    vpr_min = max(1, math.ceil(n_maj / max(1, n_min)))
    print("vpr_min", vpr_min)

    c_flags = ["--variants-per-row", "1", "--variants-per-row-minority", str(int(vpr_min)), "--skip-fit", "1"]

    # fresh tag: includes upsample and class counts to prevent overlap
    tag = f"trainGT_inferDET_balanced_class_up{int(vpr_min)}_pos{n_pos}neg{n_neg}_seed{settings['seed']}"

    run_spec("cs", "best", settings["best_human_acc"], bb, tag, "detected", extra_flags=c_flags,
             detector_model=det_path)
    run_spec("cs", "expert", settings["expert_human_acc"], bb, tag, "detected", extra_flags=c_flags,
             detector_model=det_path)
    run_spec("cs", "subjective", settings["subjective_human_acc"], bb, tag + "_noise", "detected",
             extra_flags=c_flags[:-2] + ["--concept-label-noise-mode", "subjective", "--concept-label-noise-rate",
                                         str(settings["subjective_noise_rate"]), "--skip-fit", "1"],
             detector_model=det_path)

    run_spec("cbm", "best", settings["best_human_acc"], bb, tag, "detected", extra_flags=c_flags,
             detector_model=det_path)
    run_spec("cbm", "expert", settings["expert_human_acc"], bb, tag, "detected", extra_flags=c_flags,
             detector_model=det_path)
    run_spec("cbm", "subjective", settings["subjective_human_acc"], bb, tag + "_noise", "detected",
             extra_flags=c_flags[:-2] + ["--concept-label-noise-mode", "subjective", "--concept-label-noise-rate",
                                         str(settings["subjective_noise_rate"]), "--skip-fit", "1"],
             detector_model=det_path)


# run()

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
            regime = 'best' if 'best' in run_name else ('expert' if 'expert' in run_name else ('subjective' if 'subjective' in run_name else 'unknown'))
            view = 'cs' if '_cs_' in run_name else ('cbm' if '_cbm_' in run_name else 'unknown')
        df = pd.read_csv(f)
        if 'acc_cbm_intv' not in df.columns:
            continue
        if 'delta_vs_blackbox' in df.columns:
            dnn_series = df['acc_cbm_intv'] - df['delta_vs_blackbox']
        elif 'gain_acc_dnn' in df.columns:
            dnn_series = df['acc_cbm_intv'] - df['gain_acc_dnn']
        else:
            dnn_series = pd.Series([float('nan')]*len(df))
        df['_dnn_acc_'] = dnn_series
        sub = df[df['budget'].isin(settings['budgets'])]
        if sub.empty:
            continue
        g = sub.groupby('budget', as_index=False).mean(numeric_only=True)
        for _, row in g.iterrows():
            b = int(row['budget'])
            recs.append({'regime': regime, 'budget': b, 'method': view.upper(), 'accuracy': float(row['acc_cbm_intv'])})
            recs.append({'regime': regime, 'budget': b, 'method': 'DNN', 'accuracy': float(row['_dnn_acc_'])})
    if not recs:
        return
    tidy = pd.DataFrame(recs).groupby(['regime','budget','method'], as_index=False)['accuracy'].mean()
    tidy['method'] = tidy['method'].replace({'ANCHOR':'DNN'})
    tidy['method'] = tidy['method'].astype(pd.CategoricalDtype(categories=['DNN','CBM','CS'], ordered=True))
    tidy = tidy[tidy['method'].notna()]
    pivot = tidy.pivot_table(index=['budget','method'], columns='regime', values='accuracy', aggfunc='mean').sort_index(level=[0,1])
    pivot = pivot.applymap(lambda x: round(x,4) if pd.notnull(x) else x)
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

build_final_accuracy_table()
