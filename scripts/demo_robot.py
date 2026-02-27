"""Demo: Robot benchmark — concept discovery, alignment, and intervention regimes.

Classifies synthetic robots into 'glorp' vs 'drent' using visual concepts.
Demonstrates the key experiments from the paper: how concept granularity,
alignment constraints, and annotation quality affect CBM performance.

All 9 robot features:
  head_shape (square/round), body_shape (square/round), has_knees (bool),
  has_elbows (bool), has_antennae (bool), ears_shape (square/triangle),
  mouth_type (closed/open), hand_shape (6 types), foot_shape (10 subtypes)

Two concept sets (controlled via `subconcept` flag):
  - Ideal (7 concepts): binary foot_shape + 6 other features. Drops all 10 subtypes.
  - Subconcept (12 concepts): 6 foot subtypes replace binary foot_shape.

Spurious features (has_elbows, hand_shape) are excluded from the concept set.

Label rule: glorp if (mouth_closed + foot_pointy + has_knees) >= 3

Run from the command line:
  cbm-benchmark robot --seed 1014 --subconcept --budgets 1 3 max
"""
import numpy as np
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig

# ── Experiment 1: Concept Discovery ──────────────────────────────────
# Compare ideal (7 concepts) vs subconcept (12 concepts) performance.

cfg = RobotBenchmarkConfig(
    seed=1014,
    subconcept=True,                            # 12 subconcepts (set False for 7 ideal)
    # --- Interventions ---
    intervention_budgets=[1, 3, -1],            # k=1, k=3, k=max (resolves to n_concepts)
    intervention_thresholds=[0.2],              # uncertainty threshold for kflip
    # --- Alignment ---
    alignment_constraints={"has_knees": 1},     # force has_knees weight positive
)

# Stage 1: Generate dataset (32x32 robot images, stochastic labeling)
data = robot.setup_dataset(cfg)

print(f"Training: {data.training.n} samples, Test: {data.test.n} samples")
print(f"Concepts ({data.n_concepts}): {data.concepts}")

# Stage 2: Train concept bottleneck model
# ConceptDetector: image -> concept probabilities
# FrontEndModel: concept probabilities -> P(glorp)
cbm = robot.train_cbm(cfg, data)

# Stage 3: Train DNN baseline (image -> label, no concept layer)
dnn_weights = robot.train_dnn(cfg, data)

# Stage 4: Run k-flip interventions
# For each test sample, find up to k concepts whose correction maximally
# changes the label prediction. k=0 (no intervention) is always included.
intervention_df = robot.run_interventions(cfg, cbm, data)

print("\nIntervention results:")
print(intervention_df[["budget", "threshold", "accuracy",
                        "predictions_intervened_on"]].to_string(index=False))

# Stage 5: Alignment
# Retrain FrontEndModel with sign constraint: has_knees weight must be positive.
# Tests whether alignment preserves or destroys intervention benefit.
align_stats = robot.align(cfg, cbm, data)
print(f"\nAlignment: {align_stats}")

# ── Experiment 2: Intervention Regimes ───────────────────────────────
# Uncomment to test different annotation quality levels.
# Requires --regimes flag or regimes= config parameter.
#
# cfg_regimes = RobotBenchmarkConfig(
#     seed=1014,
#     subconcept=True,
#     intervention_budgets=[1, 3, -1],
#     intervention_thresholds=[0.2],
#     intervention_strategy="exact_k",          # enumerate all size-k subsets (paper mode)
#     intervention_regimes=["baseline", "expert"],
# )
# robot.run(cfg_regimes)

# ── Experiment 3: Concept Missingness ────────────────────────────────
# Uncomment to test MCAR (missing completely at random) annotation noise.
#
# cfg_mcar = RobotBenchmarkConfig(
#     seed=1014,
#     subconcept=True,
#     concept_missing=0.2,                      # mask 20% of concept labels in training
#     concept_missing_mech="mcar",              # missing completely at random
#     intervention_budgets=[1, 3, -1],
#     intervention_thresholds=[0.2],
# )
# robot.run(cfg_mcar)

# ── Or run everything + collect into CSV in one call ─────────────────
# robot.run(cfg)
