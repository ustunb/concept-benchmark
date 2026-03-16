"""Robot classification — full neural CBM pipeline with interventions.

Tier 2: requires cloning the repo (uses the ``experiments/`` package).

This example walks through the complete concept bottleneck model workflow:
  1. Generate a robot image dataset
  2. Train a ConceptDetector (images → concept probabilities)
  3. Train a FrontEndModel (concepts → label)
  4. Combine into a ConceptBasedModel and evaluate
  5. Run oracle interventions — correct the k most uncertain concepts
  6. Train a DNN baseline for comparison
  7. Run alignment — retrain with sign constraints

Usage:
    ./venv/bin/python examples/robot_pipeline_example.py

Note: ``uv sync`` makes ``experiments/`` importable automatically.
"""

import numpy as np
import torch
import torch.nn as nn

from concept_benchmark import RobotDatasetGenerator
from concept_benchmark.utils import set_deterministic_seed
from experiments.models import (
    ConceptBasedModel,
    ConceptDetector,
    FrontEndModel,
    RobotClassifierCNN,
    RobotConceptClassifier,
)
from experiments.utils import (
    compute_accuracy,
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
    run_alignment,
)

SEED = 1014

# ---------------------------------------------------------------------------
# 0. Reproducibility and device setup
# ---------------------------------------------------------------------------
set_deterministic_seed(SEED)
patch_macos_dataloader()
device = determine_device()
loader_config = get_loader_config(device)
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Generate dataset with rendered images
# ---------------------------------------------------------------------------
print("Generating robot image dataset (subconcept=True, 12 concepts)...")
dataset = RobotDatasetGenerator(
    seed=SEED,
    subconcept=True,           # 12 fine-grained concepts (default: 7 coarse)
    model_type="stochastic",   # probabilistic labeling
    # draw=True is the default — renders robot images
).generate()

train, val, test = dataset.training, dataset.validation, dataset.test
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
    train, val,
    fit_params={
        "epochs": 50, "lr": 1e-3, "patience": 10,
        "device": str(device), **loader_config,
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
cbm = ConceptBasedModel(concept_detector=cd, front_end_model=fe)
predictions = cbm.predict(test)
baseline_acc = np.mean(predictions == test.y)
print(f"\nCBM accuracy (k=0, no interventions): {baseline_acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Oracle interventions — correct the k most uncertain concepts
# ---------------------------------------------------------------------------
# KEY: cd.predict() returns concept probabilities in [0, 1], NOT binary
# predictions. This is different from the sklearn convention where predict()
# returns class labels. Use (probs > 0.5) to get binary predictions.
concept_probs = cd.predict(test)

print("\nOracle interventions (correct k most uncertain concepts per sample):")
print(f"  {'k':>3s}   {'accuracy':>8s}   {'gain':>8s}")
print(f"  {'---':>3s}   {'--------':>8s}   {'--------':>8s}")

budgets = [0, 1, 3, n_concepts]
for k in budgets:
    if k == 0:
        acc = baseline_acc
    else:
        # Copy predicted probabilities
        C_intervened = concept_probs.copy()

        # Uncertainty = distance from decision boundary (0.5)
        uncertainty = np.abs(concept_probs - 0.5)

        # For each sample, replace the k most uncertain concepts with
        # ground-truth values
        for i in range(len(test)):
            most_uncertain = np.argsort(uncertainty[i])[:k]
            C_intervened[i, most_uncertain] = test.C[i, most_uncertain]

        # Threshold to binary and predict with the label predictor
        C_binary = (C_intervened > 0.5).astype(np.float32)
        preds = fe.predict(C_binary)
        acc = np.mean(preds == test.y)

    gain = acc - baseline_acc
    k_str = str(k) if k != n_concepts else f"{k} (max)"
    print(f"  {k_str:>8s}   {acc:>8.4f}   {gain:>+8.4f}")

# Expected results (seed=1014, subconcept, simple oracle):
#   k=0: 0.7812  |  k=1: 0.9206  |  k=3: 0.9487  |  k=12 (max): 0.9487

# ---------------------------------------------------------------------------
# 6. DNN baseline — end-to-end image classifier (no concepts)
# ---------------------------------------------------------------------------
print("\nTraining DNN baseline (images → label, no concepts)...")
set_deterministic_seed(SEED)
dnn = RobotClassifierCNN(input_size=32)
dnn.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-3)

train_loader = train.loader(shuffle=True, **loader_config)
val_loader = val.loader(shuffle=False, **loader_config)
test_loader = test.loader(shuffle=False, **loader_config)

best_val_loss = float("inf")
best_state_dict = None
epochs_no_improve = 0

for epoch in range(50):
    dnn.train()
    for X, _, y in train_loader:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        outputs = dnn(X)
        loss = criterion(outputs.squeeze(), y.float())
        loss.backward()
        optimizer.step()

    dnn.eval()
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad():
        for X, _, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = dnn(X)
            batch_loss = criterion(outputs.squeeze(), y.float())
            val_loss_sum += batch_loss.item()
            val_batches += 1

    val_loss = val_loss_sum / val_batches
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.clone() for k, v in dnn.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 10:
            break

dnn.load_state_dict(best_state_dict)
dnn_acc = compute_accuracy(dnn, test_loader, device)
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
    cbm=cbm,
    train_dataset=train,
    test_dataset=test,
    monotonicity_constraints={"has_knees": 1},
)
print(f"  Original accuracy: {alignment_results['original_accuracy']:.4f}")
print(f"  Aligned accuracy:  {alignment_results['aligned_accuracy']:.4f}")
print(f"  Change:            {alignment_results['accuracy_change']:+.4f}")
# Expected: original 0.7812, aligned 0.7656 (-0.0156)

print("\nDone!")
