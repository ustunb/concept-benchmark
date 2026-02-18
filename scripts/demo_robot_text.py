"""Demo: Robot text benchmark — text-based robot classification.

Same robot classification task (glorp vs drent) but from natural language
descriptions instead of images. Text is rendered from a JSONL template corpus
with SHA-256 deterministic synonym selection.

Concepts (9 binary, derived from the same robot attributes):
  head_square    -- head shape is square (vs round)
  body_square    -- body shape is square (vs round)
  has_knees      -- robot has visible knee joints
  has_elbows     -- robot has visible elbow joints
  foot_pointy    -- foot shape starts with "pointy_" (vs "flat_")
  has_antennae   -- robot has antennae
  ears_triangle  -- ear shape is triangular (vs square)
  mouth_open     -- mouth is open (vs closed)
  hand_edgy      -- hand shape starts with "edgy_" (vs "round_")

Example text: "This robot has a boxy head and a rounded body. Its feet come
             to sharp points and it has small triangular ears..."

Label rule: glorp if (mouth_closed + foot_pointy + has_knees) >= 3

Generic descriptions: test set mixes in 70% generic (concept-ambiguous)
descriptions to test detector robustness on out-of-distribution text.
"""
from concept_benchmark.benchmarks import robot_text
from concept_benchmark.config import RobotTextBenchmarkConfig

cfg = RobotTextBenchmarkConfig(
    seed=1337,
    # --- Data generation ---
    difficulty="hard",                      # label model variant ("easy" or "hard")
    variants_per_row_minority=3,            # 3 text variations per minority-class robot
    variants_per_row_majority=1,            # 1 text variation per majority-class robot
    generic_enable=True,                    # mix in generic (concept-ambiguous) descriptions
    generic_rate=0.7,                       # 70% of deployment/test set uses generic text
    generic_target="foot",                  # which concept the generic corpus makes ambiguous
    # --- K-fold splitting (by robot identity) ---
    cv_k=5,                                 # 5-fold cross-validation
    cv_fold=0,                              # use fold 0 as dev/test
    dev_per_fold=1000,                      # development set size per fold
    deployment_size=10000,                  # deployment (test) set size
    # --- TextConceptDetector (attention-pooled bigram model) ---
    detector_epochs=6,
    detector_batch_size=64,
    detector_lr=2e-3,
    concept_mode="hard",                    # hard concept predictions (argmax, vs "soft")
    # --- DNN baseline (DistilBERT fine-tuned on text -> label directly) ---
    dnn_model_name="distilbert-base-uncased",
    dnn_epochs=3,
    dnn_batch_size=16,
    dnn_lr=5e-5,
    # --- K-flip interventions ---
    intervention_budgets=[0, 1, 2, 5, 10], # intervene on k concepts per sample
    intervention_accuracy=1.0,              # oracle: human always corrects correctly
    flip_threshold=0.30,                    # minimum flip probability to trigger intervention
)

# Stage 1: Generate text dataset
# Enumerates all robot concept combinations (512 unique robots for 9 binary concepts),
# renders text from JSONL corpus (concept_benchmark/synthetic/helper/static/text_templates/),
# splits by robot identity (no robot appears in both train and test).
# Returns ConceptDatasetSample with:
#   data.X  -- (n_samples,) text strings (dtype=object)
#   data.C  -- (n_samples, 9) binary concept matrix (float32)
#   data.y  -- (n_samples,) labels: 0=drent, 1=glorp
#   data.meta -- {'concepts': ('head_square',...), 'classes': (0,1), 'data_type': 'text'}
data = robot_text.setup_dataset(cfg)

print(f"Training: {data.training.n} samples, Test: {data.test.n} samples")
print(f"Concepts: {data.concepts}")
print(f"\nExample text:\n  {data.X[0][:120]}...")
print(f"  Concepts: {data.C[0]}")
print(f"  Label: {data.y[0]} ({'glorp' if data.y[0]==1 else 'drent'})")

# Stage 2: Train TextConceptDetector + FrontEndModel
# TextConceptDetector: text -> 9 concept probabilities (attention-pooled bigrams)
# FrontEndModel: concept probabilities -> P(glorp)
cbm = robot_text.train_cbm(cfg, data)

# Stage 3: Fine-tune DistilBERT (text -> label, bypasses concept layer)
dnn = robot_text.train_dnn(cfg, data)

# Stage 4: K-flip interventions at budgets [0,1,2,5,10]
# k=0 is no intervention (baseline), k=10 intervenes on all 9 concepts.
results = robot_text.run_interventions(cfg, cbm, data)

print("\nIntervention results:")
print(results[["budget", "accuracy", "predictions_intervened_on",
               "predictions_changed"]].to_string(index=False))

# Stage 5: Alignment test
align_stats = robot_text.align(cfg, cbm, data)
print(f"\nAlignment: {align_stats}")

# --- Or run everything + collect into CSV in one call ---
# robot_text.run(cfg)
