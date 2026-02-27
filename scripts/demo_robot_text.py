"""Demo: Robot text benchmark — text-based robot classification.

Same robot classification task (glorp vs drent) but from natural language
descriptions instead of images. Text is rendered from a template corpus
with deterministic synonym selection.

Concepts (9 binary, derived from the same robot attributes):
  head_square, body_square, has_knees, has_elbows, foot_pointy,
  has_antennae, ears_triangle, mouth_open, hand_edgy

Example text: "This robot has a boxy head and a rounded body. Its feet come
              to sharp points and it has small triangular ears..."

Label rule: glorp if (mouth_closed + foot_pointy + has_knees) >= 3

Generic descriptions: 70% of the test set uses concept-ambiguous text to
test detector robustness on out-of-distribution descriptions.

Run from the command line:
  cbm-benchmark robot-text --seed 1337
"""
import numpy as np
from concept_benchmark.benchmarks import robot_text
from concept_benchmark.config import RobotTextBenchmarkConfig

cfg = RobotTextBenchmarkConfig(
    seed=1337,
    # --- Data generation ---
    difficulty="hard",                          # label model variant
    generic_rate=0.7,                           # 70% of test set uses ambiguous text
    # --- Interventions ---
    intervention_budgets=[1, 2, 5, -1],         # k=1, 2, 5, max (resolves to 9)
    flip_threshold=0.30,                        # minimum flip probability for intervention
)

# Stage 1: Generate text dataset
# Enumerates all 512 robot concept combinations (9 binary features),
# renders text from templates, splits by robot identity (no leakage).
data = robot_text.setup_dataset(cfg)

print(f"Training: {data.training.n} samples, Test: {data.test.n} samples")
print(f"Concepts ({len(data.concepts)}): {data.concepts}")
print(f"\nExample text:\n  {data.X[0][:120]}...")
print(f"  Label: {data.y[0]} ({'glorp' if data.y[0]==1 else 'drent'})")

# Stage 2: Train TextConceptDetector + FrontEndModel
# TextConceptDetector: text -> 9 concept probabilities (attention-pooled bigrams)
# FrontEndModel: concept probabilities -> P(glorp)
cbm = robot_text.train_cbm(cfg, data)

# Stage 3: Fine-tune DistilBERT (text -> label, no concept layer)
dnn = robot_text.train_dnn(cfg, data)

# Stage 4: K-flip interventions
# k=0 (no intervention) is always included automatically.
results = robot_text.run_interventions(cfg, cbm, data)

print("\nIntervention results:")
print(results[["budget", "accuracy", "predictions_intervened_on",
               "predictions_changed"]].to_string(index=False))

# Stage 5: Alignment
align_stats = robot_text.align(cfg, cbm, data)
print(f"\nAlignment: {align_stats}")

# ── With intervention regimes ────────────────────────────────────────
# Uncomment to test noisy human interventions.
#
# cfg_expert = RobotTextBenchmarkConfig(
#     seed=1337,
#     intervention_budgets=[1, 2, 5, -1],
#     intervention_regimes=["baseline", "expert"],
# )
# robot_text.run(cfg_expert)

# ── Or run everything + collect into CSV in one call ─────────────────
# robot_text.run(cfg)
