from pathlib import Path
import glob
import re
import json
import pandas as pd
from concept_benchmark.paths import results_dir

def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

def _seed_from_text(s):
    m = re.search(r"_seed([0-9]+)", s)
    return int(m.group(1)) if m else None

def _model_from_text(s):
    m = re.search(r"robots_text_([^_]+)_seed", s)
    return m.group(1) if m else None

def _split_from_name(name):
    m = re.search(r"_metrics_(train|val|test|deploy)\.json$", name)
    return m.group(1) if m else None

def _load_baseline_tables():
    base = results_dir / "robot_baseline" / "text"
    if not base.exists():
        return pd.DataFrame()
    rows = []
    for run_dir in sorted(base.glob("*")):
        if not run_dir.is_dir():
            continue
        js = sorted(run_dir.glob("baseline_dnn_robots_text_*_metrics_*.json"))
        if not js:
            j = sorted(run_dir.glob("baseline_dnn_robots_text_*_metrics.json"))
            if j:
                js = j
        for p in js:
            data = _read_json(p)
            name = p.name
            split = _split_from_name(name) or "all"
            seed = _seed_from_text(name) or _seed_from_text(str(run_dir))
            model = _model_from_text(name)
            row = {
                "run_name": run_dir.name,
                "file": str(p),
                "seed": seed,
                "model": model,
                "split": split,
                "accuracy": data.get("accuracy"),
                "balanced_acc": data.get("balanced_acc"),
                "ber": data.get("ber"),
                "f1": data.get("f1"),
                "roc_auc": data.get("roc_auc"),
                "selective_accuracy": data.get("selective_accuracy"),
                "coverage": data.get("coverage"),
                "tau": data.get("tau"),
                "tau_target": data.get("tau_target"),
            }
            rows.append(row)
    return pd.DataFrame(rows)

def _concat_subtype_stats():
    base = results_dir / "robot_text"
    if not base.exists():
        return pd.DataFrame()
    files = sorted(set(glob.glob(str(base / "**" / "subtype_stats_*_*.csv"), recursive=True)))
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f
            parts.append(df)
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)

def _parse_inv_key(k):
    pol = None
    bud = None
    m1 = re.search(r"budget[_-]?([0-9]+)", k)
    if m1:
        bud = int(m1.group(1))
    m2 = re.search(r"(top-1|top-k|kflip|greedy)", k)
    if m2:
        pol = m2.group(1)
    return pol, bud

def _load_cbm_interventions():
    base = results_dir / "robot_text"
    if not base.exists():
        return pd.DataFrame()
    files = sorted(set(glob.glob(str(base / "**" / "*.json"), recursive=True)))
    rows = []
    for f in files:
        name = Path(f).name
        if "baseline_dnn" in name:
            continue
        data = _read_json(f)
        if not isinstance(data, dict):
            continue
        inv = data.get("interventions")
        if not inv:
            continue
        acc_before = data.get("cbm_acc_detected") or data.get("cbm_acc") or None
        run_name = Path(f).parent.name
        seed = _seed_from_text(name) or _seed_from_text(run_name) or None
        for key, rec in inv.items():
            pol = rec.get("policy")
            bud = rec.get("budget")
            if pol is None or bud is None:
                pol2, bud2 = _parse_inv_key(str(key))
                pol = pol or pol2
                bud = bud or bud2
            row = {
                "run_name": run_name,
                "file": f,
                "seed": seed,
                "policy": pol,
                "budget": bud,
                "threshold": rec.get("threshold") or rec.get("flip_threshold"),
                "human_accuracy": rec.get("human_accuracy"),
                "accuracy_before": acc_before,
                "accuracy_after": rec.get("accuracy") or rec.get("accuracy_after_intervention"),
                "accuracy_gain": rec.get("accuracy_gain"),
                "intervention_rate": rec.get("interventions_rate") or rec.get("intervention_rate"),
                "avg_edits_per_intervention": rec.get("avg_edits_per_intervention"),
                "total_concept_checks": rec.get("total_concept_checks"),
                "total_concept_edits_made": rec.get("total_concept_edits_made"),
            }
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

def build_text_summary_tables():
    outdir = results_dir / "robot_text" / "_summary"
    outdir.mkdir(parents=True, exist_ok=True)
    dfb = _load_baseline_tables()
    if not dfb.empty:
        dfb.to_csv(outdir / "baseline_selective_metrics.csv", index=False)
        for seed, sub in dfb.groupby("seed"):
            if pd.notnull(seed):
                sub.to_csv(outdir / f"baseline_selective_metrics_seed{int(seed)}.csv", index=False)
    dfs = _concat_subtype_stats()
    if not dfs.empty:
        dfs.to_csv(outdir / "subtype_stats_all.csv", index=False)
    dfi = _load_cbm_interventions()
    if not dfi.empty:
        dfi.to_csv(outdir / "cbm_interventions.csv", index=False)
        g = dfi.groupby(["run_name", "policy", "budget"], dropna=False).agg(
            n=("file", "count"),
            avg_accuracy_before=("accuracy_before", "mean"),
            avg_accuracy_after=("accuracy_after", "mean"),
            avg_accuracy_gain=("accuracy_gain", "mean"),
            avg_intervention_rate=("intervention_rate", "mean"),
            avg_edits_per_intervention=("avg_edits_per_intervention", "mean"),
            total_concept_checks=("total_concept_checks", "sum"),
            total_concept_edits_made=("total_concept_edits_made", "sum"),
        ).reset_index()
        g.to_csv(outdir / "cbm_interventions_summary.csv", index=False)
