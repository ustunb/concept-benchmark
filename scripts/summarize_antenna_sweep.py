"""Summarize a balanced-rule sweep (experiments/antenna_weight_sweep.py, experiments/skew_sweep.py).

Reads run directories `run_{tag}_s{S}_{preset}/` (as downloaded from Bridges) and prints, per tag (antenna weight or skew level),
DNN and CBM accuracy under perfect interventions (mean ± SE over seeds), plus the paired differences
true − human and CBM − DNN at k=max with how many seeds agree in sign.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

PRESETS = {"ground_truth": ("ideal", "true"), "foot_subtypes": ("subconcept", "human")}


def mean_se(values: list[float]) -> str:
    m = 100 * st.mean(values)
    se = 100 * st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else float("nan")
    return f"{m:5.1f} ± {se:3.1f}"


def read_run(run_dir: Path, variant: str) -> tuple[float, dict[str, float], dict]:
    results = run_dir / "results"
    collect = next(results.glob(f"robot_{variant}_seed*_results.csv"))
    dnn = next(float(r["accuracy"]) for r in csv.DictReader(collect.open()) if r["model"] == "dnn")
    cell = next(results.glob(f"robot_image_stochastic_{variant}_cbm_seed*_results.csv"))
    rows = [r for r in csv.DictReader(cell.open()) if r["intervention_source"] == "perfect"]
    rows.sort(key=lambda r: int(r["budget"]))
    cbm = dict(zip(["k=0", "k=1", "k=3", "k=max"], (float(r["accuracy"]) for r in rows)))
    check = json.loads((results / "rule_check.json").read_text())
    return dnn, cbm, check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Directory holding the run_w*_s*_* folders.")
    args = ap.parse_args()

    runs = defaultdict(dict)  # (tag, seed) -> preset -> (dnn, cbm, check)
    n_runs = 0
    for run_dir in sorted(args.root.glob("run_*_s*_*")):
        m = re.fullmatch(r"run_(.+?)_s(\d+)_(ground_truth|foot_subtypes)", run_dir.name)
        if not m or not (run_dir / "DONE").exists():
            continue
        tag, seed, preset = m[1], int(m[2]), m[3]
        runs[(tag, seed)][preset] = read_run(run_dir, PRESETS[preset][0])
        n_runs += 1
    print(f"finished runs: {n_runs}")

    for tag in sorted({t for t, _ in runs}):
        seeds = sorted(s for t, s in runs if t == tag and len(runs[(t, s)]) == 2)
        if not seeds:
            continue
        by = {p: [runs[(tag, s)][p] for s in seeds] for p in PRESETS}
        checks = [c for p in PRESETS for _, _, c in by[p]]
        rule = f"intercept {checks[0]['intercept']:.1f}, " if "intercept" in checks[0] else ""
        print(f"\n== {tag}  (n={len(seeds)} seeds, {rule}"
              f"class balance {min(c['class_balance'] for c in checks):.3f}–{max(c['class_balance'] for c in checks):.3f}, "
              f"labels against score {100 * st.mean(c['against_score_rate'] for c in checks):.1f}%)")
        print(f"   DNN                  {mean_se([d for d, _, _ in by['ground_truth']])}")
        for p, (_, name) in PRESETS.items():
            print(f"   CBM {name:5}  " + "  ".join(f"{k} {mean_se([c[k] for _, c, _ in by[p]])}" for k in ["k=0", "k=1", "k=3", "k=max"]))
        gap = [t[1]["k=max"] - h[1]["k=max"] for t, h in zip(by["ground_truth"], by["foot_subtypes"])]
        for p, (_, name) in PRESETS.items():
            diff = [c["k=max"] - d for d, c, _ in by[p]]
            print(f"   CBM {name:5} k=max − DNN {mean_se(diff)}  (CBM ahead in {sum(x > 0 for x in diff)}/{len(diff)})")
        print(f"   true − human k=max     {mean_se(gap)}  (true ahead in {sum(x > 0 for x in gap)}/{len(gap)})")
        # targets at k=max, required on every seed: true − human ≥ 5, true − DNN ≥ 5, human ≤ DNN (points)
        met = [
            (t[1]["k=max"] - h[1]["k=max"] >= 0.05, t[1]["k=max"] - t[0] >= 0.05, h[1]["k=max"] <= h[0])
            for t, h in zip(by["ground_truth"], by["foot_subtypes"])
        ]
        counts = [sum(m[i] for m in met) for i in range(3)]
        verdict = "ALL TARGETS MET" if all(all(m) for m in met) else "not met"
        print(f"   targets on seeds: true−human≥5 {counts[0]}/{len(met)}, true−DNN≥5 {counts[1]}/{len(met)}, "
              f"human≤DNN {counts[2]}/{len(met)}  → {verdict}")


if __name__ == "__main__":
    main()
