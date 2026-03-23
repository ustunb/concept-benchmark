"""Robot classification — full neural CBM pipeline with interventions.

Tier 2: requires cloning the repo (uses the ``experiments/`` package).

This example walks through the complete concept bottleneck model workflow:
  1. Generate a robot image dataset
  2. Train a ConceptDetector (images → concept probabilities)
  3. Train a FrontEndModel (concepts → label)
  4. Combine into a ConceptBasedModel and evaluate
  5. Run interventions using ConceptInterventionRunner + KFlipInterventionStrategy
  6. Train a DNN baseline for comparison
  7. Run alignment — retrain with sign constraints

Timing: ~3 min on MPS (Apple Silicon), ~5–10 min on CPU.

Usage:
    ./venv/bin/python examples/robot_pipeline_example.py

Note: ``uv sync`` makes ``experiments/`` importable automatically.
"""

import numpy as np

from concept_benchmark.robot import DatasetGenerator
from concept_benchmark.utils import set_deterministic_seed
from experiments.intervention import ConceptInterventionRunner, InterventionConfig
from experiments.kflip import KFlipInterventionStrategy
from experiments.models import (
    ConceptBasedModel,
    ConceptDetector,
    FrontEndModel,
    RobotClassifierCNN,
    RobotConceptClassifier,
)
from experiments.utils import (
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
    run_alignment,
    train_dnn,
)

SEED = 1014

# ---------------------------------------------------------------------------
# 0. Reproducibility and device setup
# ---------------------------------------------------------------------------
set_deterministic_seed(SEED)
patch_macos_dataloader()
device = determine_device()
loader_config = get_loader_config()
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Generate dataset with rendered images
# ---------------------------------------------------------------------------
print("Generating robot image dataset (concept_preset='foot_subtypes', 12 concepts)...")
gen = DatasetGenerator(
    seed=SEED,
    concept_preset="foot_subtypes",  # 12 fine-grained concepts (default: "ground_truth" = 7)
    use_stochastic_labels=True,  # probabilistic labeling
    # render_images=True is the default — renders robot images
)
dataset = gen.generate()
from concept_benchmark.config import PRESET_EXCLUDED_CONCEPTS

dataset.drop_concepts(PRESET_EXCLUDED_CONCEPTS["foot_subtypes"])
dataset.sample(test_size=10000, val_size=0.2, train_size=3800, seed=SEED)

train, val, test = dataset.train, dataset.validation, dataset.test
print(f"  Training:    {train.n} samples, {train.n_concepts} concepts")
print(f"  Validation:  {val.n} samples")
print(f"  Test:        {test.n} samples")
print(f"  Concepts:    {train.concepts}")

# ---------------------------------------------------------------------------
# 2. Train concept detector (images → concept probabilities)
# ---------------------------------------------------------------------------
print("\nTraining ConceptDetector...")
n_concepts = train.n_concepts
cd = ConceptDetector(
    model=RobotConceptClassifier(num_concepts=n_concepts, input_size=32),
)
cd.fit(
    train,
    val,
    fit_params={
        "epochs": 50,
        "lr": 1e-3,
        "patience": 10,
        "device": str(device),
        **loader_config,
    },
)
print("  Done.")

# ---------------------------------------------------------------------------
# 3. Train label predictor (concepts → label)
# ---------------------------------------------------------------------------
print("Training FrontEndModel (logistic regression on ground-truth concepts)...")
fe = FrontEndModel()
fe.fit(train.C, train.y)

print("  Learned concept weights:")
for name, w in zip(train.concepts, fe.model.coef_[0]):
    print(f"    {name:30s}  {w:+.3f}")

# ---------------------------------------------------------------------------
# 4. Combine into a CBM and evaluate
# ---------------------------------------------------------------------------
cbm = ConceptBasedModel(concept_detector=cd, label_predictor=fe)
predictions = cbm.predict(test)
baseline_acc = np.mean(predictions == test.y)
print(f"\nCBM accuracy (k=0, no interventions): {baseline_acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Interventions using ConceptInterventionRunner + KFlipInterventionStrategy
# ---------------------------------------------------------------------------
# The KFlip strategy evaluates all subsets of up to k concepts per sample,
# computing the probability that flipping each subset changes the prediction.
# This matches the paper's intervention protocol exactly.
# Replace KFlipInterventionStrategy() with your own InterventionStrategy
# subclass to benchmark custom intervention policies (see docs/interventions.md).
runner = ConceptInterventionRunner(model=cbm)
concept_probs = cd.predict_proba(test)

print("\nKFlip interventions (correct k most impactful concepts per sample):")
print(f"  {'k':>3s}   {'accuracy':>8s}   {'gain':>8s}")
print(f"  {'---':>3s}   {'--------':>8s}   {'--------':>8s}")

budgets = [0, 1, 3, n_concepts]
for k in budgets:
    if k == 0:
        acc = baseline_acc
    else:
        result = runner.run(
            strategy=KFlipInterventionStrategy(),
            config=InterventionConfig(
                max_concepts_per_instance=k,
                score_threshold=0.2,
            ),
            dataset=test,
            concept_proba=concept_probs,
        )
        acc = np.mean(result.y_pred_after == test.y)

    gain = acc - baseline_acc
    k_str = str(k) if k != n_concepts else f"{k} (max)"
    print(f"  {k_str:>8s}   {acc:>8.4f}   {gain:>+8.4f}")

# Expected results (seed=1014, subconcept, KFlip with threshold=0.2):
#   k=0: 0.7812  |  k=1: 0.9212  |  k=3: 0.9439  |  k=12 (max): 0.9439

# ---------------------------------------------------------------------------
# 6. DNN baseline — end-to-end image classifier (no concepts)
# ---------------------------------------------------------------------------
print("\nTraining DNN baseline (images → label, no concepts)...")
set_deterministic_seed(SEED)
dnn = RobotClassifierCNN(input_size=32)
dnn_acc = train_dnn(dnn, train, val, test, device, loader_config=loader_config)
print(f"  DNN accuracy: {dnn_acc:.4f}")
# Expected: 0.8746

# ---------------------------------------------------------------------------
# 7. Alignment — retrain frontend with sign constraints
# ---------------------------------------------------------------------------
# The paper shows that forcing has_knees to have a positive weight
# (matching the intuitive direction) preserves training accuracy but
# destroys intervention benefit.
print("\nRunning alignment (has_knees constrained to +1)...")
alignment_results = run_alignment(
    concept_based_model=cbm,
    train_dataset=train,
    test_dataset=test,
    monotonicity_constraints={"has_knees": 1},
)
print(f"  Original accuracy: {alignment_results['original_accuracy']:.4f}")
print(f"  Aligned accuracy:  {alignment_results['aligned_accuracy']:.4f}")
print(f"  Change:            {alignment_results['accuracy_change']:+.4f}")
# Expected: original 0.7812, aligned 0.7656 (-0.0156)

print("\nDone!")
