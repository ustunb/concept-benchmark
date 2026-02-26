#!/usr/bin/env python
"""Reproduce the intervention study (baseline + expert regimes).

Runs the full robot subconcept pipeline from scratch:
  1. Generate dataset (30k robot images, ~90s)
  2. Train CBM + DNN (~3 min)
  3. Run interventions for selected regimes (~seconds with vectorized code)
  4. Run alignment
  5. Collect results CSV

Edit REGIMES and STRATEGY below to control what gets run.

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/reproduce_interventions.py

Expected results (seed=1014, subconcept, exact_k, threshold=0.2):
    baseline: k=0 ~0.78, k=1 ~0.92, k=3 ~0.94, k=12 ~0.94
    expert:   k=1 gain ~+10%, k=3 gain ~+7%
"""
import time
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

# ===================== EDIT THESE =====================
SEED = 1014
REGIMES = ["baseline", "expert"]       # add "subjective", "machine", etc.
STRATEGY = "kflip"                   # "exact_k" (paper) or "kflip" (up-to-k)
BUDGETS = [1, 2, 3, 5]                # intervention budgets (k values)
THRESHOLDS = [0.2]                     # score thresholds
SUBCONCEPT = True                      # True = 12 subconcepts, False = 7 ideal
# ======================================================

cfg = RobotBenchmarkConfig(
    seed=SEED,
    data_type="image",
    size="medium",
    samples_per_instance=4,
    model_type="stochastic",
    subconcept=SUBCONCEPT,
    spurious_features=["has_elbows", "hand_shape"],
    additional_features=["foot_shape_subtype"],
    epochs=50,
    lr=1e-3,
    patience=10,
    batch_size=32,
    intervention_budgets=BUDGETS,
    intervention_thresholds=THRESHOLDS,
    intervention_accuracy=1.0,
    intervention_strategy=STRATEGY,
    intervention_regimes=REGIMES,
)

variant = "subconcept" if SUBCONCEPT else "ideal"
print(f"=== Intervention Study: {variant}, strategy={STRATEGY}, regimes={REGIMES} ===\n")

# --- Stage 1: Setup ---
t0 = time.time()
print("[1/5] Generating dataset...")
data = robot.setup_dataset(cfg)
t_setup = time.time() - t0
print(f"  Done in {t_setup:.1f}s — {data.training.n} train, {data.test.n} test, "
      f"{data.n_concepts} concepts: {data.concepts}\n")

# --- Stage 2: Train CBM ---
t0 = time.time()
print("[2/5] Training CBM (concept detector + frontend)...")
cbm = robot.train_cbm(cfg, data)
t_cbm = time.time() - t0
print(f"  Done in {t_cbm:.1f}s\n")

# --- Stage 3: Train DNN ---
t0 = time.time()
print("[3/5] Training DNN baseline...")
dnn = robot.train_dnn(cfg, data)
t_dnn = time.time() - t0
print(f"  Done in {t_dnn:.1f}s\n")

# --- Stage 4: Run interventions ---
t0 = time.time()
print(f"[4/5] Running interventions ({REGIMES}, budgets={BUDGETS})...")
intervention_df = robot.run_interventions(cfg, cbm, data)
t_int = time.time() - t0
print(f"  Done in {t_int:.1f}s\n")

# Print results
cols = ["regime", "budget", "threshold", "accuracy", "predictions_intervened_on",
        "predictions_changed"]
available = [c for c in cols if c in intervention_df.columns]
print("Intervention results:")
print(intervention_df[available].to_string(index=False))

# Compute gains
if "accuracy" in intervention_df.columns and "regime" in intervention_df.columns:
    print("\n--- ΔAccuracy (change from k=0, no intervention) ---")
    for regime in intervention_df["regime"].unique():
        rdf = intervention_df[intervention_df["regime"] == regime]
        base_acc = rdf[rdf["budget"] == 0]["accuracy"].iloc[0]
        print(f"\n  {regime} (k=0 acc = {base_acc:.4f}):")
        for _, row in rdf.iterrows():
            delta = row["accuracy"] - base_acc
            print(f"    k={int(row['budget']):>2}: acc={row['accuracy']:.4f}  "
                  f"Δ={delta:+.4f} ({delta*100:+.1f}%)")

# --- Summary ---
total = t_setup + t_cbm + t_dnn + t_int
print(f"\n=== Timing Summary ===")
print(f"  Setup:         {t_setup:6.1f}s")
print(f"  Train CBM:     {t_cbm:6.1f}s")
print(f"  Train DNN:     {t_dnn:6.1f}s")
print(f"  Interventions: {t_int:6.1f}s")
print(f"  TOTAL:         {total:6.1f}s")
