#!/usr/bin/env python
"""Verify robot benchmark experiments against paper results.

Experiments:
  1. Concept Discovery (Section 5.1) — ideal vs subconcept
  2. Alignment (Section 5.2) — alignment destroys intervention benefit
  3. Intervention Regimes (Section 5.3, Figure 7) — noise degrades interventions

Toggle experiments via the flags below. Results appended to
results/verification_2026-02-26.md.

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/verify_robot.py
"""
from __future__ import annotations

import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Toggle experiments ────────────────────────────────────────────────
RUN_EXP1 = True   # Concept Discovery
RUN_EXP2 = True   # Alignment
RUN_EXP3 = True   # Intervention Regimes (all 6, both strategies)
SKIP_LLM_REGIMES = True  # Skip clip/llm (require Gemini API calls)

GEMINI_API_KEY = "AIzaSyCttbGa-rYjMB3jOFrYLPN4EP7OWIskuHo"

# ── Paper reference values ────────────────────────────────────────────
PAPER_EXP1 = {
    "ideal":      {"dnn": 0.8746, "k0": 0.8673, "k1": 0.9736, "k3": 0.9769, "kmax": 0.9769},
    "subconcept": {"dnn": 0.8746, "k0": 0.7812, "k1": 0.9212, "k3": 0.9439, "kmax": 0.9439},
}
PAPER_EXP2 = {
    "ideal":      {"cbm_k0": 0.8673, "aligned_k0": 0.8657, "cbm_k3_gain": 0.102, "aligned_k3_gain": -0.004},
    "subconcept": {"cbm_k0": 0.7812, "aligned_k0": 0.7656, "cbm_k3_gain": 0.069, "aligned_k3_gain": -0.080},
}
PAPER_EXP3 = {  # ΔAccuracy (gain over k=0 baseline)
    "baseline":   {"k1": 0.140, "k2": 0.163, "k5": 0.163, "mean": 0.155},
    "expert":     {"k1": 0.097, "k2": 0.115, "k5": 0.109, "mean": 0.107},
    "subjective": {"k1": 0.003, "k2": 0.006, "k5": 0.000, "mean": 0.003},
    "llm":        {"k1": -0.262, "k2": -0.314, "k5": -0.232, "mean": -0.269},
    "clip":       {"k1": -0.320, "k2": -0.320, "k5": -0.290, "mean": -0.310},
    "machine":    {"k1": -0.336, "k2": -0.367, "k5": -0.346, "mean": -0.350},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_robot")


def main():
    from concept_benchmark.utils import (
        compute_accuracy,
        determine_device,
        get_loader_config,
        patch_macos_dataloader,
        set_deterministic_seed,
    )
    import robot_pipeline as robot
    from concept_benchmark.config import RobotBenchmarkConfig
    from concept_benchmark.ext.fileutils import load
    from concept_benchmark.models import RobotClassifierCNN

    patch_macos_dataloader()
    device = determine_device()
    loader_config = get_loader_config(device)

    md_lines = []
    md_lines.append(f"# Robot Verification — {datetime.now():%Y-%m-%d %H:%M}\n")
    t0_total = time.time()

    # ══════════════════════════════════════════════════════════════════
    # Phase A: Shared setup + training
    # ══════════════════════════════════════════════════════════════════
    logger.info("Phase A: Setup and training")

    # -- Ideal config --
    cfg_ideal = RobotBenchmarkConfig(
        seed=1014,
        subconcept=False,
        intervention_budgets=[1, 3, -1],
        intervention_thresholds=[0.2],
        alignment_constraints={"has_knees": 1},
    )
    logger.info("Setting up ideal dataset...")
    data_ideal = robot.setup_dataset(cfg_ideal)
    logger.info("Training ideal CBM...")
    cbm_ideal = robot.train_cbm(cfg_ideal, data_ideal)
    logger.info("Training ideal DNN...")
    dnn_weights_ideal = robot.train_dnn(cfg_ideal, data_ideal)

    # -- Subconcept config --
    cfg_sub = RobotBenchmarkConfig(
        seed=1014,
        subconcept=True,
        intervention_budgets=[1, 3, -1],
        intervention_thresholds=[0.2],
        alignment_constraints={"has_knees": 1},
    )
    logger.info("Setting up subconcept dataset...")
    data_sub = robot.setup_dataset(cfg_sub)
    logger.info("Training subconcept CBM...")
    cbm_sub = robot.train_cbm(cfg_sub, data_sub)
    logger.info("Training subconcept DNN...")
    dnn_weights_sub = robot.train_dnn(cfg_sub, data_sub)

    # Helper: compute DNN test accuracy
    def dnn_accuracy(weights, cfg, data):
        dnn = RobotClassifierCNN(input_size=cfg.input_size).to(device)
        dnn.load_state_dict(weights)
        test_loader = data.test.loader(shuffle=False, **loader_config)
        return compute_accuracy(dnn, test_loader, device)

    dnn_acc_ideal = dnn_accuracy(dnn_weights_ideal, cfg_ideal, data_ideal)
    dnn_acc_sub = dnn_accuracy(dnn_weights_sub, cfg_sub, data_sub)
    logger.info("DNN accuracy: ideal=%.4f, subconcept=%.4f", dnn_acc_ideal, dnn_acc_sub)

    # ══════════════════════════════════════════════════════════════════
    # Experiment 1: Concept Discovery
    # ══════════════════════════════════════════════════════════════════
    if RUN_EXP1:
        logger.info("=" * 60)
        logger.info("Experiment 1: Concept Discovery")
        logger.info("=" * 60)
        t0 = time.time()

        # Run baseline interventions (kflip)
        cfg_ideal_exp1 = copy.deepcopy(cfg_ideal)
        cfg_ideal_exp1.intervention_regimes = ["baseline"]
        cfg_ideal_exp1.intervention_strategy = "kflip"
        interv_ideal = robot.run_interventions(cfg_ideal_exp1, cbm_ideal, data_ideal)

        cfg_sub_exp1 = copy.deepcopy(cfg_sub)
        cfg_sub_exp1.intervention_regimes = ["baseline"]
        cfg_sub_exp1.intervention_strategy = "kflip"
        interv_sub = robot.run_interventions(cfg_sub_exp1, cbm_sub, data_sub)

        # Extract results
        def extract_exp1(interv_df, dnn_acc, label):
            bl = interv_df[(interv_df["regime"] == "baseline") & (interv_df["threshold"] == 0.2)]
            k0_acc = float(bl[bl["budget"] == 0]["accuracy"].iloc[0])
            k1_acc = float(bl[bl["budget"] == 1]["accuracy"].iloc[0])
            k3_acc = float(bl[bl["budget"] == 3]["accuracy"].iloc[0])
            kmax = bl["budget"].max()
            kmax_acc = float(bl[bl["budget"] == kmax]["accuracy"].iloc[0])
            return {
                "setup": label,
                "dnn": round(dnn_acc, 4),
                "k0": round(k0_acc, 4),
                "k1": round(k1_acc, 4),
                "k3": round(k3_acc, 4),
                "kmax": round(kmax_acc, 4),
            }

        r_ideal = extract_exp1(interv_ideal, dnn_acc_ideal, "ideal")
        r_sub = extract_exp1(interv_sub, dnn_acc_sub, "subconcept")

        # Build markdown table
        md_lines.append("## Experiment 1: Concept Discovery (Section 5.1)\n")
        md_lines.append("| Setup | Source | DNN | CBM k=0 | k=1 | k=3 | k=max |")
        md_lines.append("|-------|--------|-----|---------|-----|-----|-------|")
        for label, r in [("ideal", r_ideal), ("subconcept", r_sub)]:
            p = PAPER_EXP1[label]
            md_lines.append(
                f"| {label} | ours  | {r['dnn']:.4f} | {r['k0']:.4f} | "
                f"{r['k1']:.4f} | {r['k3']:.4f} | {r['kmax']:.4f} |"
            )
            md_lines.append(
                f"| {label} | paper | {p['dnn']:.4f} | {p['k0']:.4f} | "
                f"{p['k1']:.4f} | {p['k3']:.4f} | {p['kmax']:.4f} |"
            )
        md_lines.append("")
        logger.info("Experiment 1 done in %.1fs", time.time() - t0)

    # ══════════════════════════════════════════════════════════════════
    # Experiment 2: Alignment
    # ══════════════════════════════════════════════════════════════════
    if RUN_EXP2:
        logger.info("=" * 60)
        logger.info("Experiment 2: Alignment")
        logger.info("=" * 60)
        t0 = time.time()

        from concept_benchmark.alignment import align_frontend_weights

        align_ideal = robot.align(cfg_ideal, cbm_ideal, data_ideal)
        align_sub = robot.align(cfg_sub, cbm_sub, data_sub)

        # For aligned k=3 interventions: replicate collect_results logic
        def aligned_k3_acc(cfg, cbm, data, align_data):
            aligned_weights = align_data.get("aligned_weights")
            if aligned_weights is None:
                return None
            aligned_fe = copy.deepcopy(cbm.front_end_model)
            aligned_fe = align_frontend_weights(
                aligned_fe, list(data.test.concepts), aligned_weights,
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
                acc_det=float(align_data["aligned_accuracy"]),
                fe=aligned_fe,
                test=data.test,
            )
            # Extract accuracy from first (only) result
            for key, res in int_results.items():
                return float(res["accuracy"])
            return None

        aligned_k3_ideal = aligned_k3_acc(cfg_ideal, cbm_ideal, data_ideal, align_ideal)
        aligned_k3_sub = aligned_k3_acc(cfg_sub, cbm_sub, data_sub, align_sub)

        # CBM k=0 and k=3 accuracy (from Exp 1 if available, else compute)
        cbm_k0_ideal = float((cbm_ideal.predict(data_ideal.test) == data_ideal.test.y).mean())
        cbm_k0_sub = float((cbm_sub.predict(data_sub.test) == data_sub.test.y).mean())

        # k=3 accuracy from intervention results
        def get_k3_acc(interv_df):
            bl = interv_df[(interv_df["regime"] == "baseline") & (interv_df["threshold"] == 0.2)]
            row = bl[bl["budget"] == 3]
            if len(row) == 0:
                return None
            return float(row["accuracy"].iloc[0])

        if RUN_EXP1:
            cbm_k3_ideal = get_k3_acc(interv_ideal)
            cbm_k3_sub = get_k3_acc(interv_sub)
        else:
            # Need to run baseline interventions to get k=3
            cfg_i = copy.deepcopy(cfg_ideal)
            cfg_i.intervention_regimes = ["baseline"]
            cfg_i.intervention_budgets = [3]
            interv_i = robot.run_interventions(cfg_i, cbm_ideal, data_ideal)
            cbm_k3_ideal = get_k3_acc(interv_i)

            cfg_s = copy.deepcopy(cfg_sub)
            cfg_s.intervention_regimes = ["baseline"]
            cfg_s.intervention_budgets = [3]
            interv_s = robot.run_interventions(cfg_s, cbm_sub, data_sub)
            cbm_k3_sub = get_k3_acc(interv_s)

        md_lines.append("## Experiment 2: Alignment (Section 5.2)\n")
        md_lines.append("| Setup | Source | CBM k=0 | Aligned k=0 | CBM k=3 ΔAcc | Aligned k=3 ΔAcc |")
        md_lines.append("|-------|--------|---------|-------------|--------------|------------------|")
        for label, cbm_k0, cbm_k3, al_data, al_k3 in [
            ("ideal", cbm_k0_ideal, cbm_k3_ideal, align_ideal, aligned_k3_ideal),
            ("subconcept", cbm_k0_sub, cbm_k3_sub, align_sub, aligned_k3_sub),
        ]:
            al_k0 = float(al_data["aligned_accuracy"])
            cbm_gain = (cbm_k3 - cbm_k0) if cbm_k3 is not None else None
            al_gain = (al_k3 - al_k0) if al_k3 is not None else None
            p = PAPER_EXP2[label]

            cbm_gain_str = f"{cbm_gain:+.1%}" if cbm_gain is not None else "N/A"
            al_gain_str = f"{al_gain:+.1%}" if al_gain is not None else "N/A"
            md_lines.append(
                f"| {label} | ours  | {cbm_k0:.4f} | {al_k0:.4f} | "
                f"{cbm_gain_str} | {al_gain_str} |"
            )
            md_lines.append(
                f"| {label} | paper | {p['cbm_k0']:.4f} | {p['aligned_k0']:.4f} | "
                f"{p['cbm_k3_gain']:+.1%} | {p['aligned_k3_gain']:+.1%} |"
            )
        md_lines.append("")
        logger.info("Experiment 2 done in %.1fs", time.time() - t0)

    # ══════════════════════════════════════════════════════════════════
    # Experiment 3: Intervention Regimes
    # ══════════════════════════════════════════════════════════════════
    if RUN_EXP3:
        logger.info("=" * 60)
        logger.info("Experiment 3: Intervention Regimes")
        logger.info("=" * 60)
        t0 = time.time()

        # Train regime-specific models once (cached for both strategies)
        # train_lfcbm needs data=None so it loads from disk with correct base_dir paths
        logger.info("Pre-training regime models...")
        cfg_regime_base = RobotBenchmarkConfig(
            seed=1014,
            subconcept=True,
            intervention_budgets=[1, 2, 5],
            intervention_thresholds=[0.2],
        )
        try:
            robot.train_cbm_subjective(cfg_regime_base, data_sub)
        except Exception as e:
            logger.warning("Failed to train subjective CBM: %s", e)
        try:
            robot.train_lfcbm(cfg_regime_base)  # data=None: loads from disk
        except Exception as e:
            logger.warning("Failed to train LFCBM: %s", e)

        for strategy in ["exact_k", "kflip"]:
            logger.info("--- Strategy: %s ---", strategy)
            regime_results = {}  # regime -> {k1, k2, k5, mean, k0}

            # D1: baseline + expert
            try:
                logger.info("Running baseline + expert (%s)...", strategy)
                cfg_be = RobotBenchmarkConfig(
                    seed=1014,
                    subconcept=True,
                    intervention_budgets=[1, 2, 5],
                    intervention_thresholds=[0.2],
                    intervention_strategy=strategy,
                    intervention_regimes=["baseline", "expert"],
                )
                df_be = robot.run_interventions(cfg_be, cbm_sub, data_sub)
                for regime in ["baseline", "expert"]:
                    regime_results[regime] = _extract_regime_gains(df_be, regime)
            except Exception as e:
                logger.error("baseline+expert failed: %s", e, exc_info=True)

            # D2: subjective
            try:
                logger.info("Running subjective (%s)...", strategy)
                cfg_subj = RobotBenchmarkConfig(
                    seed=1014,
                    subconcept=True,
                    intervention_budgets=[1, 2, 5],
                    intervention_thresholds=[0.2],
                    intervention_strategy=strategy,
                    intervention_regimes=["subjective"],
                )
                df_subj = robot.run_interventions(cfg_subj, cbm_sub, data_sub)
                regime_results["subjective"] = _extract_regime_gains(df_subj, "subjective")
            except Exception as e:
                logger.error("subjective failed: %s", e, exc_info=True)

            # D3: machine
            try:
                logger.info("Running machine (%s)...", strategy)
                cfg_mach = RobotBenchmarkConfig(
                    seed=1014,
                    subconcept=True,
                    intervention_budgets=[1, 2, 5],
                    intervention_thresholds=[0.2],
                    intervention_strategy=strategy,
                    intervention_regimes=["machine"],
                )
                df_mach = robot.run_interventions(cfg_mach, cbm_sub, data_sub)
                regime_results["machine"] = _extract_regime_gains(df_mach, "machine")
            except Exception as e:
                logger.error("machine failed: %s", e, exc_info=True)

            # D4: clip (Gemini 2.5 Flash Lite)
            if not SKIP_LLM_REGIMES:
                try:
                    logger.info("Running clip (%s)...", strategy)
                    cfg_clip = RobotBenchmarkConfig(
                        seed=1014,
                        subconcept=True,
                        intervention_budgets=[1, 2, 5],
                        intervention_thresholds=[0.2],
                        intervention_strategy=strategy,
                        intervention_regimes=["clip"],
                        llm_model="gemini-2.5-flash-lite",
                        llm_api_key=GEMINI_API_KEY,
                    )
                    df_clip = robot.run_interventions(cfg_clip, cbm_sub, data_sub)
                    regime_results["clip"] = _extract_regime_gains(df_clip, "clip")
                except Exception as e:
                    logger.error("clip failed: %s", e, exc_info=True)
            else:
                logger.info("Skipping clip (%s) — SKIP_LLM_REGIMES=True", strategy)

            # D5: llm (Gemini 2.5 Flash)
            if not SKIP_LLM_REGIMES:
                try:
                    logger.info("Running llm (%s)...", strategy)
                    cfg_llm = RobotBenchmarkConfig(
                        seed=1014,
                        subconcept=True,
                        intervention_budgets=[1, 2, 5],
                        intervention_thresholds=[0.2],
                        intervention_strategy=strategy,
                        intervention_regimes=["llm"],
                        llm_model="gemini-2.5-flash",
                        llm_api_key=GEMINI_API_KEY,
                    )
                    df_llm = robot.run_interventions(cfg_llm, cbm_sub, data_sub)
                    regime_results["llm"] = _extract_regime_gains(df_llm, "llm")
                except Exception as e:
                    logger.error("llm failed: %s", e, exc_info=True)
            else:
                logger.info("Skipping llm (%s) — SKIP_LLM_REGIMES=True", strategy)

            # Build markdown table
            strat_label = "exact_k" if strategy == "exact_k" else "kflip (up-to-k)"
            md_lines.append(f"## Experiment 3: Regimes — {strat_label} (Section 5.3)\n")
            md_lines.append("| Regime | Source | Δk=1 | Δk=2 | Δk=5 | Mean ΔAcc |")
            md_lines.append("|--------|--------|------|------|------|-----------|")
            for regime in ["baseline", "expert", "subjective", "llm", "clip", "machine"]:
                r = regime_results.get(regime)
                if r is None:
                    md_lines.append(f"| {regime} | ours  | FAILED | FAILED | FAILED | FAILED |")
                    p = PAPER_EXP3[regime]
                    md_lines.append(
                        f"| {regime} | paper | {p['k1']:+.1%} | {p['k2']:+.1%} | "
                        f"{p['k5']:+.1%} | {p['mean']:+.1%} |"
                    )
                    continue
                md_lines.append(
                    f"| {regime} | ours  | {r['k1']:+.1%} | {r['k2']:+.1%} | "
                    f"{r['k5']:+.1%} | {r['mean']:+.1%} |"
                )
                p = PAPER_EXP3[regime]
                md_lines.append(
                    f"| {regime} | paper | {p['k1']:+.1%} | {p['k2']:+.1%} | "
                    f"{p['k5']:+.1%} | {p['mean']:+.1%} |"
                )
            md_lines.append("")
            logger.info("Strategy %s done in %.1fs", strategy, time.time() - t0)

        logger.info("Experiment 3 done in %.1fs", time.time() - t0)

    # ══════════════════════════════════════════════════════════════════
    # Write results
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0_total
    md_lines.append(f"\n---\n*Total robot verification time: {elapsed/60:.1f} min*\n")

    out_path = Path("results/verification_2026-02-26.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing content (sudoku may have written already)
    existing = out_path.read_text() if out_path.exists() else ""
    # Remove any previous robot section
    if "# Robot Verification" in existing:
        # Replace robot section (everything from "# Robot" to next "# " or end)
        import re
        existing = re.sub(
            r"# Robot Verification.*?(?=# [A-Z]|\Z)", "", existing, flags=re.DOTALL
        ).strip()

    combined = "\n".join(md_lines) + "\n"
    if existing:
        combined = combined + "\n" + existing + "\n"

    out_path.write_text(combined)
    logger.info("Results written to %s", out_path)
    print(f"\nResults saved to {out_path}")


def _extract_regime_gains(df, regime):
    """Extract ΔAccuracy for k=1,2,5 from an intervention DataFrame."""
    bl = df[(df["regime"] == regime) & (df["threshold"] == 0.2)]
    k0_acc = float(bl[bl["budget"] == 0]["accuracy"].iloc[0])
    gains = {}
    for k in [1, 2, 5]:
        rows = bl[bl["budget"] == k]
        if len(rows) == 0:
            gains[f"k{k}"] = float("nan")
        else:
            gains[f"k{k}"] = float(rows["accuracy"].iloc[0]) - k0_acc
    gains["mean"] = np.mean([gains["k1"], gains["k2"], gains["k5"]])
    gains["k0"] = k0_acc
    return gains


if __name__ == "__main__":
    main()
