#!/usr/bin/env python
"""Experiment 2: Alignment (Section 5.2)

Tests whether alignment constraints (forcing has_knees weight to +1)
preserve intervention benefit.
- Ideal (7 concepts) vs subconcept (12 concepts)
- alignment_constraints={"has_knees": 1}
- seed=1014, threshold=0.2
- Gains are relative to DNN accuracy (matching collect_results)

Usage:
    ./venv/bin/python scripts/verify_exp2.py
"""
from __future__ import annotations

import copy
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
logger = logging.getLogger("verify_exp2")

# Paper reference (Table in Section 5.2)
# Gains are relative to DNN accuracy (= 0.8746)
PAPER = {
    "ideal":      {"cbm_k0": 0.8673, "aligned_k0": 0.8657, "cbm_k3_gain": +0.102, "aligned_k3_gain": -0.003},
    "subconcept": {"cbm_k0": 0.7812, "aligned_k0": 0.7656, "cbm_k3_gain": +0.069, "aligned_k3_gain": -0.080},
}


def main():
    import robot_pipeline as robot
    from experiments.alignment import align_frontend_weights
    from concept_benchmark.config import RobotBenchmarkConfig
    from experiments.models import RobotClassifierCNN
    from concept_benchmark.utils import (
        compute_accuracy, determine_device, get_loader_config, patch_macos_dataloader,
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
            intervention_budgets=[3],
            intervention_thresholds=[0.2],
            intervention_regimes=["baseline"],
            intervention_strategy="kflip",
            alignment_constraints={"has_knees": 1},
        )
        data = robot.setup_dataset(cfg)
        cbm = robot.train_cbm(cfg, data)
        dnn_weights = robot.train_dnn(cfg, data)

        # DNN test accuracy (gain reference, matching collect_results line 1323)
        dnn = RobotClassifierCNN(input_size=cfg.input_size).to(device)
        dnn.load_state_dict(dnn_weights)
        dnn_acc = compute_accuracy(dnn, data.test.loader(shuffle=False, **loader_config), device)

        # CBM k=0 accuracy
        cbm_k0 = float((cbm.predict(data.test) == data.test.y).mean())

        # CBM k=3 via baseline interventions
        interv_df = robot.run_interventions(cfg, cbm, data)
        bl = interv_df[(interv_df["regime"] == "baseline") & (interv_df["threshold"] == 0.2)]
        cbm_k3 = float(bl[bl["budget"] == 3]["accuracy"].iloc[0])

        # Alignment
        align_data = robot.align(cfg, cbm, data)
        aligned_k0 = float(align_data["aligned_accuracy"])

        # Aligned k=3: replicate collect_results logic (robot.py:1382-1422)
        aligned_fe = copy.deepcopy(cbm.front_end_model)
        aligned_fe = align_frontend_weights(
            aligned_fe, list(data.test.concepts), align_data["aligned_weights"],
        )
        c_preds = cbm.concept_detector.predict(data.test)
        isettings = robot.InterventionSettings(
            seed=cfg.seed,
            budgets=[3],
            intervention_accuracy=cfg.intervention_accuracy,
            intervention_threshold=0.2,
        )
        _, _, int_results = robot._test_interventions(
            prob_test=c_preds,
            settings=isettings,
            acc_det=aligned_k0,
            fe=aligned_fe,
            test=data.test,
        )
        aligned_k3 = float(next(iter(int_results.values()))["accuracy"])

        # Gains relative to DNN accuracy (matching collect_results)
        results[label] = {
            "dnn_acc": dnn_acc,
            "cbm_k0": cbm_k0,
            "cbm_k3": cbm_k3,
            "aligned_k0": aligned_k0,
            "aligned_k3": aligned_k3,
            "cbm_k3_gain": cbm_k3 - dnn_acc,
            "aligned_k3_gain": aligned_k3 - dnn_acc,
        }

    # Print results
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("Experiment 2: Alignment (Section 5.2)")
    print("Gains are relative to DNN accuracy (matching collect_results)")
    print("=" * 70)
    print(f"| {'Setup':<12} | {'Source':<6} | {'CBM k=0':>8} | {'Align k=0':>9} | {'CBM k=3 Δ':>10} | {'Align k=3 Δ':>11} |")
    print(f"|{'-'*14}|{'-'*8}|{'-'*10}|{'-'*11}|{'-'*12}|{'-'*13}|")
    for label in ["ideal", "subconcept"]:
        r = results[label]
        p = PAPER[label]
        print(f"| {label:<12} | {'ours':<6} | {r['cbm_k0']:.4f}  | {r['aligned_k0']:.4f}   | {r['cbm_k3_gain']:+.1%}     | {r['aligned_k3_gain']:+.1%}      |")
        print(f"| {label:<12} | {'paper':<6} | {p['cbm_k0']:.4f}  | {p['aligned_k0']:.4f}   | {p['cbm_k3_gain']:+.1%}     | {p['aligned_k3_gain']:+.1%}      |")

    # Also print absolute accuracies for debugging
    print("\nAbsolute accuracies:")
    for label in ["ideal", "subconcept"]:
        r = results[label]
        print(f"  {label}: DNN={r['dnn_acc']:.4f}, CBM k=0={r['cbm_k0']:.4f}, "
              f"CBM k=3={r['cbm_k3']:.4f}, Aligned k=0={r['aligned_k0']:.4f}, "
              f"Aligned k=3={r['aligned_k3']:.4f}")
    print(f"\nDone in {elapsed/60:.1f} min")

    # Write markdown
    out = Path("results/verify_exp2.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment 2: Alignment (Section 5.2)\n",
             "seed=1014, threshold=0.2, alignment_constraints={has_knees: 1}",
             "Gains relative to DNN accuracy\n",
             "| Setup | Source | CBM k=0 | Aligned k=0 | CBM k=3 ΔAcc | Aligned k=3 ΔAcc |",
             "|-------|--------|---------|-------------|--------------|------------------|"]
    for label in ["ideal", "subconcept"]:
        r = results[label]
        p = PAPER[label]
        lines.append(f"| {label} | ours  | {r['cbm_k0']:.4f} | {r['aligned_k0']:.4f} | {r['cbm_k3_gain']:+.1%} | {r['aligned_k3_gain']:+.1%} |")
        lines.append(f"| {label} | paper | {p['cbm_k0']:.4f} | {p['aligned_k0']:.4f} | {p['cbm_k3_gain']:+.1%} | {p['aligned_k3_gain']:+.1%} |")
    lines.append("")
    lines.append("### Absolute Accuracies")
    for label in ["ideal", "subconcept"]:
        r = results[label]
        lines.append(f"- {label}: DNN={r['dnn_acc']:.4f}, CBM k=0={r['cbm_k0']:.4f}, "
                     f"CBM k=3={r['cbm_k3']:.4f}, Aligned k=0={r['aligned_k0']:.4f}, "
                     f"Aligned k=3={r['aligned_k3']:.4f}")
    lines.append(f"\n*Time: {elapsed/60:.1f} min*\n")
    out.write_text("\n".join(lines))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
