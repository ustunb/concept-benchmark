"""Sudoku validation — full neural CS model pipeline with interventions.

Tier 2: requires cloning the repo (uses the ``experiments/`` package).

This example walks through the complete concept-supervised (CS) workflow:
  1. Generate a Sudoku dataset with handwritten digit images
  2. Train a ConceptDetector (board digits → 27 validity concepts)
  3. Train a FrontEndModel (concepts → valid/invalid label)
  4. Evaluate selective classification (abstain on uncertain predictions)
  5. Run oracle interventions and observe AND-fragility

Timing: Data generation ~5 min, model training ~30 s, evaluation ~2 min.

Usage:
    ./venv/bin/python examples/sudoku_pipeline_example.py

Note: ``uv sync`` makes ``experiments/`` importable automatically.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from concept_benchmark import SudokuDatasetGenerator
from concept_benchmark.utils import set_deterministic_seed
from experiments.intervention import ConceptInterventionRunner, InterventionConfig
from experiments.kflip import KFlipInterventionStrategy
from experiments.models import (
    ConceptBasedModel,
    ConceptDetector,
    FrontEndModel,
    GroupPoolingConceptSudokuCNN,
)
from experiments.utils import (
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
)

SEED = 171

# ---------------------------------------------------------------------------
# 0. Reproducibility and device setup
# ---------------------------------------------------------------------------
set_deterministic_seed(SEED)
patch_macos_dataloader()
device = determine_device()
loader_config = get_loader_config(device)
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Generate dataset
# ---------------------------------------------------------------------------
print("Generating Sudoku dataset (1000 boards, max_corrupt=9)...")
dataset = SudokuDatasetGenerator(
    seed=SEED,
    n_samples=1000,
    max_corrupt=9,     # cells swapped in invalid boards (higher = subtler)
    valid_ratio=0.5,   # 50% valid, 50% invalid
).generate()

train, val, test = dataset.training, dataset.validation, dataset.test
print(f"  Training:    {train.n} samples, {train.n_concepts} concepts")
print(f"  Validation:  {val.n} samples")
print(f"  Test:        {test.n} samples")
print(f"  Concepts:    {train.concepts[:5]} ... ({train.n_concepts} total)")

# ---------------------------------------------------------------------------
# 2. Train concept detector (board → 27 validity concepts)
# ---------------------------------------------------------------------------
print("\nTraining ConceptDetector (GroupPoolingConceptSudokuCNN)...")
cd = ConceptDetector(model=GroupPoolingConceptSudokuCNN())
cd.fit(
    train, val,
    fit_params={
        "epochs": 100, "lr": 1e-3, "patience": 20,
        "device": str(device), **loader_config,
    },
)
print("  Done.")

# ---------------------------------------------------------------------------
# 3. Train label predictor (concepts → valid/invalid)
# ---------------------------------------------------------------------------
print("Training FrontEndModel...")
fe = FrontEndModel()
fe.fit(train.C, train.y)

# Show the AND structure: all weights should be positive
weights = fe.model.coef_[0]
print(f"  All concept weights positive: {(weights > 0).all()}")
print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")

# ---------------------------------------------------------------------------
# 4. Combine into a CS model and evaluate
# ---------------------------------------------------------------------------
cbm = ConceptBasedModel(concept_detector=cd, front_end_model=fe)
predictions = cbm.predict(test)
raw_acc = np.mean(predictions == test.y)
print(f"\nCS model raw accuracy: {raw_acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Selective classification — abstain on uncertain predictions
# ---------------------------------------------------------------------------
# The key metric for Sudoku is selective classification: the model only
# answers when confident, achieving high accuracy on kept predictions.
# KEY: cd.predict() returns concept probabilities in [0, 1], NOT binary.
concept_probs = cd.predict(test)
C_binary = (concept_probs > 0.5).astype(np.float32)
label_proba = fe.predict_proba(C_binary)[:, 1]
y_pred = fe.predict(C_binary)

print("\nSelective classification (abstain when uncertain):")
print(f"  {'target_acc':>12s}   {'sel_acc':>8s}   {'coverage':>8s}")
print(f"  {'----------':>12s}   {'-------':>8s}   {'--------':>8s}")

for target_acc in [0.90, 0.95, 0.99]:
    confidence = np.abs(label_proba - 0.5)
    best_tau = 0.0
    for tau in np.linspace(0, 0.5, 500):
        keep = confidence >= tau
        if keep.sum() == 0:
            continue
        if accuracy_score(test.y[keep], y_pred[keep]) >= target_acc:
            best_tau = tau
            break

    keep = confidence >= best_tau
    coverage = keep.mean()
    sel_acc = accuracy_score(test.y[keep], y_pred[keep]) if keep.sum() > 0 else 0.0
    print(f"  {target_acc:>12.2f}   {sel_acc:>8.4f}   {coverage:>7.1%}")

# ---------------------------------------------------------------------------
# 6. Oracle interventions — demonstrate AND-fragility
# ---------------------------------------------------------------------------
# In Sudoku, a board is valid iff ALL 27 concepts are true. This AND
# structure means fixing a single wrong concept may not help: if another
# concept is also wrong, the board is still predicted invalid.
print("\nOracle interventions (correct k most uncertain concepts):")
print(f"  {'k':>8s}   {'accuracy':>8s}   {'gain':>8s}")
print(f"  {'---':>8s}   {'--------':>8s}   {'--------':>8s}")

baseline_acc = np.mean(y_pred == test.y)
n_concepts = test.n_concepts
runner = ConceptInterventionRunner(model=cbm)

for k in [0, 1, 3, n_concepts]:
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

# Note: Unlike robot classification, interventions show diminishing returns
# in Sudoku due to the AND structure. Fixing one concept rarely flips the
# final prediction unless ALL remaining concepts are already correct.

print("\nDone!")
