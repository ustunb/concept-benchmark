from pathlib import Path
import sys, subprocess, shlex, os
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
    "variants_per_row": {"hard": 3},
    "label_model_type": "stochastic",
    "label_model_alpha": 10.0,
    "label_model_bias": 0.0,
    "label_model_expr": "'glorp' if (int(row[\"body_shape\"]==\"square\") + int(row[\"has_antennae\"]==\"true\") - 2 >= 0) else \"drent\"",
    "budgets": [0, 1, 2, 5, 10],
    "human_acc": {"best": 1.00, "expert": 0.80, "subjective": 0.80},
    "subjective_noise_rate": 0.20,
    "reuse_mode": "cd",
    "force": 1,
}

def run_cmd(name, argv, cwd=ROOT):
    cmd = [PY] + [str(x) for x in argv]
    print(f"{name}\nCMD: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
    r = subprocess.run(cmd, env=ENV, cwd=str(cwd))
    if r.returncode != 0:
        print(f"FAILED: {name} (code {r.returncode})"); sys.exit(r.returncode)
    print(f"DONE: {name}\n", flush=True)

def exists_viability(run_dir: Path) -> bool:
    return any(run_dir.glob("viability_robots_text_*_*.csv"))

def ktag(budgets):
    if isinstance(budgets, (list, tuple)):
        return "kset-" + "_".join(str(int(k)) for k in budgets)
    return f"k{int(budgets)}"

def make_name(regime, model_type, train_tag, infer_tag, human_acc, budgets, diff, seed):
    aa = int(round(float(human_acc) * 100))
    return f"{regime}_{model_type}_train{train_tag}_infer{infer_tag}_intervene{aa}_{ktag(budgets)}_{diff}_seed{seed}"

def add_common(argv, diff):
    argv += [
        "--label-model-type", settings["label_model_type"],
        "--label-model-alpha", str(settings["label_model_alpha"]),
        "--label-model-bias", str(settings["label_model_bias"]),
        "--label-model-expr", settings["label_model_expr"],
        "--template-difficulty", diff,
        "--variants-per-row", str(settings["variants_per_row"][diff]),
        "--target-acc-grid", "raw",
        "--budgets", ",".join(str(k) for k in settings["budgets"]),
        "--policy", "uncertainty",
    ]
    if settings["force"]:
        argv += ["--force-rerun","1"]
    return argv

def add_reuse_detector(argv):
    argv += ["--reuse-detector","0"]
    return argv

def run_dnn(diff, seed):
    run_name = f"dnn_text_{diff}_seed{seed}"
    run_dir = results_dir / "robot_text" / run_name
    if settings["force"] == 0 and run_dir.exists():
        print(f"[SKIP existing] {run_name}")
        return
    argv = [
        DNN,
        "--modality","text",
        "--seed", str(seed),
        "--template-difficulty", diff,
        "--run-name", run_name,
    ]
    run_cmd(f"[DNN] {run_name}", argv)

def run_setup(regime, model_type, diff, train_tag, infer_tag, human_acc, budgets, subjective=False):
    seed = settings["seed"]
    run_name = make_name(regime, model_type, train_tag, infer_tag, human_acc, budgets, diff, seed)
    run_dir = results_dir / "robot_text" / run_name
    if settings["force"] == 0 and exists_viability(run_dir):
        print(f"[SKIP existing] {run_name}")
        return
    argv = [GEN, "--variant","perfect", "--seed", str(seed)]
    argv += ["--concept-source","detected"]
    argv = add_common(argv, diff)
    if isinstance(budgets, (list, tuple)):
        argv = [a for a in argv if a != "--budgets" and not (isinstance(a, str) and a.startswith("0,1,2"))]
        argv += ["--budgets", ",".join(str(k) for k in budgets)]
    else:
        argv = [a for a in argv if a != "--budgets" and not (isinstance(a, str) and a.startswith("0,1,2"))]
        argv += ["--budgets", str(int(budgets))]
    argv += ["--human-acc", str(human_acc)]
    if subjective:
        argv += ["--concept-label-noise-mode","subjective","--concept-label-noise-rate", str(settings["subjective_noise_rate"])]
    argv += ["--run-name", run_name]
    argv = add_reuse_detector(argv)
    run_cmd(f"[{regime} {model_type} {diff}] {run_name}", argv)

def main():
    diff = settings["difficulty"]
    seed = settings["seed"]

    # run_dnn(diff, seed)

    kgrid = settings["budgets"]
    aa = settings["human_acc"]

    run_setup("best",       "cs",  diff, "GT", "Detected", aa["best"],       kgrid)
    run_setup("expert",     "cs",  diff, "GT", "Detected", aa["expert"],     kgrid)
    run_setup("subjective", "cs",  diff, "GT", "Detected", aa["subjective"], kgrid, subjective=True)

    run_setup("best",       "cbm", diff, "GT", "Detected", aa["best"],       kgrid)
    run_setup("expert",     "cbm", diff, "GT", "Detected", aa["expert"],     kgrid)
    run_setup("subjective", "cbm", diff, "GT", "Detected", aa["subjective"], kgrid, subjective=True)

main()
