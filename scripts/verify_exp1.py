#!/usr/bin/env python
"""Experiment 1: Concept Discovery (Section 5.1)

Tests effect of concept granularity on CBM performance and interventions.
- Ideal (7 concepts) vs subconcept (12 concepts)
- Baseline regime, kflip strategy
- seed=1014, threshold=0.2, budgets=[1, 3, -1]

Usage:
    ./venv/bin/python scripts/verify_exp1.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow sibling imports (robot_pipeline lives in scripts/)
_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_exp1")

# Paper reference (Table in Section 5.1)
PAPER = {
    "ideal":      {"dnn": 0.8746, "k0": 0.8673, "k1": 0.9734, "k3": 0.9767, "kmax": 0.9767},
    "subconcept": {"dnn": 0.8746, "k0": 0.7812, "k1": 0.9212, "k3": 0.9439, "kmax": 0.9439},
}


def main():
    import robot_pipeline as robot
    from concept_benchmark.config import RobotBenchmarkConfig
    from experiments.models import RobotClassifierCNN
    from concept_benchmark.utils import (
        compute_accuracy,
        determine_device,
        get_loader_config,
        patch_macos_dataloader,
    )

    patch_macos_dataloader()
    device = determine_device()
    loader_config = get_loader_config(device)
    t0 = time.time()

    results = {}
    for subconcept in [False, True]:
        label = "subconcept" if subconcept else "ideal"
        logger.info("--- %s ---", label)

        cfg = RobotBenchmarkConfig(
            seed=1014,
            subconcept=subconcept,
            intervention_budgets=[1, 3, -1],
            intervention_thresholds=[0.2],
            intervention_regimes=["baseline"],
            intervention_strategy="kflip",
        )
        data = robot.setup_dataset(cfg)
        cbm = robot.train_cbm(cfg, data)
        dnn_weights = robot.train_dnn(cfg, data)

        # DNN test accuracy
        dnn = RobotClassifierCNN(input_size=cfg.input_size).to(device)
        dnn.load_state_dict(dnn_weights)
        dnn_acc = compute_accuracy(dnn, data.test.loader(shuffle=False, **loader_config), device)

        # Baseline kflip interventions
        interv_df = robot.run_interventions(cfg, cbm, data)
        bl = interv_df[(interv_df["regime"] == "baseline") & (interv_df["threshold"] == 0.2)]

        k0 = float(bl[bl["budget"] == 0]["accuracy"].iloc[0])
        k1 = float(bl[bl["budget"] == 1]["accuracy"].iloc[0])
        k3 = float(bl[bl["budget"] == 3]["accuracy"].iloc[0])
        kmax_budget = bl["budget"].max()
        kmax = float(bl[bl["budget"] == kmax_budget]["accuracy"].iloc[0])

        results[label] = {"dnn": dnn_acc, "k0": k0, "k1": k1, "k3": k3, "kmax": kmax}

    # Print results
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("Experiment 1: Concept Discovery (Section 5.1)")
    print("=" * 70)
    print(f"| {'Setup':<12} | {'Source':<6} | {'DNN':>7} | {'k=0':>7} | {'k=1':>7} | {'k=3':>7} | {'k=max':>7} |")
    print(f"|{'-'*14}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|")
    for label in ["ideal", "subconcept"]:
        r = results[label]
        p = PAPER[label]
        print(f"| {label:<12} | {'ours':<6} | {r['dnn']:.4f} | {r['k0']:.4f} | {r['k1']:.4f} | {r['k3']:.4f} | {r['kmax']:.4f} |")
        print(f"| {label:<12} | {'paper':<6} | {p['dnn']:.4f} | {p['k0']:.4f} | {p['k1']:.4f} | {p['k3']:.4f} | {p['kmax']:.4f} |")
    print(f"\nDone in {elapsed/60:.1f} min")

    # Write markdown
    out = Path("results/verify_exp1.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment 1: Concept Discovery (Section 5.1)\n",
             "seed=1014, threshold=0.2, strategy=kflip\n",
             "| Setup | Source | DNN | CBM k=0 | k=1 | k=3 | k=max |",
             "|-------|--------|-----|---------|-----|-----|-------|"]
    for label in ["ideal", "subconcept"]:
        r = results[label]
        p = PAPER[label]
        lines.append(f"| {label} | ours  | {r['dnn']:.4f} | {r['k0']:.4f} | {r['k1']:.4f} | {r['k3']:.4f} | {r['kmax']:.4f} |")
        lines.append(f"| {label} | paper | {p['dnn']:.4f} | {p['k0']:.4f} | {p['k1']:.4f} | {p['k3']:.4f} | {p['kmax']:.4f} |")
    lines.append(f"\n*Time: {elapsed/60:.1f} min*\n")
    out.write_text("\n".join(lines))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
