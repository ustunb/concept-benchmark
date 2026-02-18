"""Demo: Robot benchmark — reproduce the subconcept + MCAR experiment (Table 2).

Classifies synthetic robots into 'glorp' vs 'drent' using visual concepts.
This experiment uses the subconcept set with 20% MCAR missingness.

All 9 robot features:
  head_shape (square/round), body_shape (square/round), has_knees (bool),
  has_elbows (bool), has_antennae (bool), ears_shape (square/triangle),
  mouth_type (closed/open), hand_shape (6 types), foot_shape (10 subtypes)

foot_shape subtypes (5 flat + 5 pointy):
  flat_trapezoid, flat_rounded, flat_square, flat_5sided, flat_lshaped,
  pointy_trapezoid, pointy_rounded, pointy_square, pointy_3sided, pointy_4sided

Concept sets are controlled via drop_concepts:
  - IDEAL_DROP: drops all 10 foot subtypes -> keeps binary foot_shape (pointy/flat)
  - SUBCONCEPT_DROP: drops parent foot_shape + 5 subtypes -> keeps 5 subtypes

Label rule: glorp if (mouth_closed + foot_pointy + has_knees) >= 3

Spurious features (not in label rule, but correlated): has_elbows, hand_shape
"""
from concept_benchmark.benchmarks import robot
from concept_benchmark.config import RobotBenchmarkConfig, SUBCONCEPT_DROP, IDEAL_DROP

# --- Choose which concepts to include by specifying which to DROP ---
# IDEAL_DROP: drop all 10 foot subtypes -> 9 concepts (with binary foot_shape)
# SUBCONCEPT_DROP: drop parent foot_shape + 5 subtypes -> 11 concepts (with 5 subtypes)
# Custom: drop any subset you want, e.g. drop only 3 subtypes
chosen_drop = list(SUBCONCEPT_DROP)  # subconcept variant

cfg = RobotBenchmarkConfig(
    seed=1014,
    # --- Data generation ---
    data_type="image",
    size="medium",                          # 32x32 pixel images
    samples_per_instance=4,                 # 4 color variations per robot design
    model_type="stochastic",                # stochastic label model (vs "deterministic")
    # --- Concept set ---
    drop_concepts=chosen_drop,              # which concepts to exclude from the model
    subconcept=True,                        # naming flag only -- adds "_subconcept" to file paths
    # After SUBCONCEPT_DROP, remaining concepts are:
    #   head_shape, body_shape, has_knees, has_antennae, ears_shape, mouth_type,
    #   foot_shape_pointy_square, foot_shape_pointy_4sided,
    #   foot_shape_pointy_rounded, foot_shape_flat_trapezoid, foot_shape_flat_square
    spurious_features=["has_elbows", "hand_shape"],  # in data but not in label rule
    additional_features=["foot_shape_subtype"],       # extra column for analysis
    # --- Training ---
    epochs=50,
    lr=1e-3,
    patience=10,
    batch_size=32,
    # --- Concept missingness ---
    concept_missing=0.2,                    # mask 20% of concept labels during training
    concept_missing_mech="mcar",            # missing completely at random (vs "mnar")
    # --- Interventions ---
    intervention_budgets=[1, 3],            # intervene on k=1 and k=3 concepts per sample
    intervention_thresholds=[0.2],          # uncertainty threshold for k-flip strategy
    intervention_accuracy=1.0,              # oracle accuracy (perfect human interventions)
    # --- Alignment ---
    alignment_constraints={"has_knees": 1}, # monotonicity: more knees -> more glorp
)

# Stage 1: Generate dataset
# Creates 30,720 robot images with skewed train/val/test splits.
# Returns ConceptDataset with:
#   data.X  -- (30720,) image file paths (dtype=object)
#   data.C  -- (30720, n_concepts) binary concept matrix (int8, served as float32)
#   data.y  -- (30720,) labels: 0=drent, 1=glorp
#   data.meta -- {'classes': ['drent','glorp'], 'concepts': [...], 'data_type': 'image'}
#   data.training / data.validation / data.test -- split views
data = robot.setup_dataset(cfg)

print(f"Training: {data.training.n} samples, Test: {data.test.n} samples")
print(f"Concepts ({data.n_concepts}): {data.concepts}")
print(f"Classes: {data.classes}")   # ['drent', 'glorp']

# Stage 2: Train concept bottleneck model
# ConceptDetector: image -> concept probabilities (one per concept)
# FrontEndModel: concept probabilities -> P(glorp)
# With MCAR, 20% of concept labels are masked during training.
cbm = robot.train_cbm(cfg, data)

# Stage 3: Train DNN baseline (image -> label, bypasses concepts entirely)
dnn_weights = robot.train_dnn(cfg, data)

# Stage 4: Run k-flip interventions
# For each test sample, KFlipInterventionStrategy finds the k concepts whose
# correction maximally changes the label prediction. Intervenes if the
# flip probability exceeds 0.2 threshold.
# Returns DataFrame with:
#   budget, threshold, accuracy, predictions_intervened_on,
#   predictions_changed, total_concept_edits_made, ...
intervention_df = robot.run_interventions(cfg, cbm, data)

print("\nIntervention results (subconcept + MCAR):")
print(intervention_df[["budget", "threshold", "accuracy",
                        "predictions_intervened_on"]].to_string(index=False))

# Stage 5: Alignment test
# Retrains FrontEndModel with monotonicity constraint (has_knees -> +glorp).
# Checks if learned concept->label direction matches domain knowledge.
align_stats = robot.align(cfg, cbm, data)
print(f"\nAlignment: {align_stats}")

# --- Or run everything + collect into CSV in one call ---
# robot.run(cfg)
