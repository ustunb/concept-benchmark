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
#
# }

settings = {
    "seed": 1337,
    "difficulty": "hard",
    "budgets": [0, 1, 2, 5, 10],
    "force": 0,
    "reuse_detector": 1,
    "run_tag": "",

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
}

def parse_cli():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seed", type=int)
    p.add_argument("--difficulty")
    p.add_argument("--budgets")
    p.add_argument("--force", type=int)
    p.add_argument("--reuse-detector", type=int)
    p.add_argument("--run-tag")
    p.add_argument("--modality")
    p.add_argument("--text-model")
    p.add_argument("--best-human-acc", type=float)
    p.add_argument("--expert-human-accs")
    p.add_argument("--subjective-human-accs")
    p.add_argument("--subjective-noise-rates")
    p.add_argument("--skip-fit", type=int)
    p.add_argument("--make-plots", type=int)
    p.add_argument("--recompute-only", type=int)
    p.add_argument("--redact-concepts")
    p.add_argument("--redact-splits")
    p.add_argument("--concepts-csv")
    p.add_argument("--force-rerun", type=int)
    args, _ = p.parse_known_args()
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in {"budgets", "expert_human_accs", "subjective_human_accs", "subjective_noise_rates"}:
            if isinstance(v, str):
                vals = [x for x in v.split(",") if x.strip() != ""]
                if k in {"budgets"}:
                    settings[k] = [int(float(x)) for x in vals]
                else:
                    settings[k] = [float(x) for x in vals]
            else:
                settings[k] = v
        else:
            kk = k.replace("-", "_")
            settings[kk] = v

def run_cmd(tag: str, argv: List[str], cwd: Optional[Path] = None):
    cmd = [str(PY), str(GEN)] + argv
    subprocess.run(cmd, check=True, env=ENV)

def _csv_list(v: str) -> List[str]:
    if not v:
        return []
    return [x for x in str(v).split(",") if x.strip() != ""]

def run():
    seed = int(settings.get("seed", 0))
    budgets = [int(x) for x in settings.get("budgets", [0, 1, 2, 5, 10])]
    force = int(settings.get("force", 0))
    make_plots = int(settings.get("make_plots", 0))

    tag = str(settings.get("run_tag", "")).strip()
    if not tag:
        tag = "cbm"

    bb = str(sorted((results_dir / "robot_baseline").rglob("baseline_*metrics.json"))[-1]) if list((results_dir / "robot_baseline").rglob("baseline_*metrics.json")) else ""

    def run_spec(prefix: str, regime: str, human_acc: float, blackbox_metrics: str, tag_suffix: str, concept_source: str, extra_flags: Optional[List[str]] = None, detector_model: Optional[str] = None):
        rn = f"{prefix}_{regime}_minrule_eval_vpr3_pos1152neg3456_unbalanced_pixelated_fixed_seed{seed}_v1_intervene{int(human_acc*100)}_kset-0_1_2_5_10_{settings.get('difficulty','hard')}_seed{seed}"
        argv = [
            "--variant", "perfect",
            "--variants-per-row", "1",
            "--imperfect-strategy", "missing_concepts",
            "--heldout-concepts", "[]",
            "--mask-p", "0.0",
            "--mask-mode", "mask",
            "--mask-rate", "0.0",
            "--seed", str(seed),
            "--concept-mode", "hard",
            "--templates-file", "",
            "--redact-concepts", str(settings.get("redact_concepts", "")),
            "--redact-splits", str(settings.get("redact_splits", "")),
            "--label-model-expr", "",
            "--corr-pair", "",
            "--train-corr", "1.0",
            "--test-break", "1.0",
            "--test-corr", "-1.0",
            "--budgets", ",".join(str(x) for x in budgets),
            "--target-acc-grid", "raw",
            "--target-acc-concepts", "",
            "--intervene-allow", "",
            "--human-acc", str(human_acc),
            "--human-acc-concepts", "",
            "--make-plots", str(make_plots),
            "--policy", "uncertainty",
            "--concept-label-noise-mode", "none",
            "--concept-label-noise-rate", "0.2",
            "--blackbox-metrics", blackbox_metrics,
            "--concepts-csv", settings.get("concepts_csv", ""),
            "--concept-source", concept_source,
            "--skip-fit", str(int(settings.get("skip_fit", 1))),
            "--force-rerun", str(force),
            "--intervention-error-mode", "both",
            "--run-name", rn if not tag_suffix else f"{tag_suffix}_{rn}",
        ]
        if int(settings.get("reuse_detector", 1)) and detector_model:
            argv += ["--reuse-detector", "1", "--detector-model", detector_model]
        if extra_flags:
            argv += extra_flags
        run_cmd("run", argv)

    run_spec("best", "best", settings.get("best_human_acc", 1.0), bb, "best", "detected", detector_model=None)

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
        add_arg("concept_label_noise_mode", args.get("concept_label_noise_mode"))
        add_arg("concept_label_noise_rate", args.get("concept_label_noise_rate"))
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
        ccsv = settings.get("concepts_csv") or args.get("concepts_csv")
        add_arg("concepts_csv", ccsv)

        mobj = json.loads(Path(mp).read_text())
        det = mobj.get("artifacts", {}).get("model") or args.get("detector_model")

        cs = args.get("concept_source")
        if cs in (None, "", "none", "human"):
            machine_keys = ["machine_method","machine_k","machine_soft","machine_seed","machine_upper_bound","lf_alpha","lf_threshold","lf_mode","lf_ridge","lf_ridge_alpha","lf_encoder","lf_device","lf_batch_size"]
            if any(args.get(k) for k in machine_keys):
                cs = "machine"
            elif det:
                cs = "detected"
            elif args.get("use_gt_concepts") in [1, "1", True, "true"]:
                cs = "gt"
        if cs:
            add_arg("concept_source", cs)
        if cs == "machine":
            for k in ["machine_method","machine_k","machine_soft","machine_seed","machine_upper_bound","lf_alpha","lf_threshold","lf_mode","lf_ridge","lf_ridge_alpha","lf_encoder","lf_device","lf_batch_size"]:
                add_arg(k, args.get(k))

        mode = "both"
        rn0 = args.get("run_name") or ""

        import re as _re
        if rn0.startswith("best_") or rn0.startswith("anchor_"):
            regime = "best"
        elif rn0.startswith("expert_"):
            regime = "expert"
        elif rn0.startswith("subjective_"):
            regime = "subjective"
        else:
            regime = "unknown"

        def _listify(v):
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
            if isinstance(v, str):
                return [float(x) for x in v.split(",") if x.strip() != ""]
            if v is None:
                return []
            return [float(v)]

        if regime == "expert":
            haccs = _listify(settings.get("expert_human_accs")) or _listify(args.get("human_acc"))
        elif regime == "subjective":
            haccs = _listify(settings.get("subjective_human_accs")) or _listify(args.get("human_acc"))
        else:
            haccs = _listify(args.get("human_acc")) or _listify(settings.get("best_human_acc"))

        if regime == "subjective":
            rates = _listify(settings.get("subjective_noise_rates"))
            if not rates:
                rates = _listify(args.get("concept_label_noise_rate")) or _listify(settings.get("subjective_noise_rate", 0.20))
        else:
            rates = [None]

        for ha in haccs:
            for rate in rates:
                argv = list(base_argv)
                _skip = int(settings.get("skip_fit", 1))
                if _skip == 1 and not (det and Path(det).exists()):
                    raise RuntimeError(f"skip_fit=1 but detector missing: {det}")
                argv += ["--skip-fit", str(_skip)]
                if det and Path(det).exists():
                    argv += ["--reuse-detector", "1", "--detector-model", det]
                else:
                    print("Warning: missing detector model", det)
                argv += ["--force-rerun", str(int(settings.get("force", 0))), "--intervention-error-mode", mode]

                argv += ["--human-acc", str(float(ha))]
                rn = rn0
                rn = _re.sub(r"_intervene\d+", f"_intervene{int(float(ha)*100)}", rn)
                if regime == "subjective" and rate is not None:
                    argv += ["--concept-label-noise-rate", str(float(rate))]
                    dd = int(round(float(rate)*100))
                    if dd == 20:
                        if "_noise20" in rn:
                            cand = rn.replace("_noise20", "_noise")
                            if (results_dir / "robot_text" / cand).exists():
                                rn = cand
                        elif "_noise" not in rn:
                            rn = rn + "_noise"
                    else:
                        if "_noise" in rn and not _re.search(r"_noise\d{2}", rn):
                            rn = rn.replace("_noise", f"_noise{dd:02d}")
                        elif _re.search(r"_noise\d{2}", rn):
                            rn = _re.sub(r"_noise\d{2}", f"_noise{dd:02d}", rn)
                        else:
                            rn = rn + f"_noise{dd:02d}"
                argv += ["--run-name", rn]
                run_cmd("recompute", argv)

def build_final_accuracy_table():
    import re
    files_v1 = glob.glob(str(results_dir / "robot_text" / "**" / "viability_robots_text_*_detected.csv"),
                         recursive=True) + \
               glob.glob(str(results_dir / "robot_text" / "**" / "viability_v2_robots_text_*_detected.csv"),
                         recursive=True)
    rows = []
    for fp in files_v1:
        p = Path(fp)
        try:
            parts = p.stem.split("_")
        except Exception:
            continue
        hacc = None
        mm = re.search(r"intervene(\d+)", p.as_posix())
        if mm:
            try:
                hacc = int(mm.group(1)) / 100.0
            except Exception:
                hacc = None
        mode = "v2" if "viability_v2_" in p.name else "v1"
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        df["human_acc"] = hacc
        df["mode"] = mode
        df["run_dir"] = str(p.parent)
        rows.append(df)
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    out["budget"] = out["budget"].astype(int)
    out = out.sort_values(["run_dir", "mode", "human_acc", "budget"]).reset_index(drop=True)
    out_csv = results_dir / "robot_text" / "final_intervention_accuracy_table.csv"
    out_tex = results_dir / "robot_text" / "final_intervention_accuracy_table.tex"

    def _fmt(x):
        try:
            xi = int(x)
            return f"{xi}"
        except Exception:
            try:
                xf = float(x)
                return f"{xf:.4f}"
            except Exception:
                return str(x)

    cols = [c for c in out.columns if c not in {"run_dir"}]
    out.to_csv(out_csv, index=False)
    with open(out_tex, "w") as f:
        hdr = " & ".join(cols)
        f.write("\\begin{tabular}{%s}\n" % ("l" * len(cols)))
        f.write(hdr + " \\\\\n\\hline\n")
        for _, row in out.iterrows():
            f.write(" & ".join(_fmt(row[c]) for c in cols) + " \\\\\n")
        f.write("\\end{tabular}\n")

if __name__ == "__main__":
    parse_cli()
    if int(settings.get("recompute_only", 0)) == 1:
        recompute_metrics()
        build_final_accuracy_table()
    else:
        run()
        build_final_accuracy_table()
