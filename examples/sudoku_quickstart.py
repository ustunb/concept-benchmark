"""Sudoku validation quickstart — exploring the concept bottleneck.

Tier 1: works with ``pip install concept-benchmark`` (no repo clone needed).

Generates a Sudoku dataset with board images, inspects its structure, trains
a label predictor on perfect concepts, and demonstrates selective
classification and how concept noise degrades predictions.

For the full neural CS model pipeline (digit recognition + selective
classification + interventions), see ``examples/sudoku_pipeline_example.py``
(requires cloning the repo).

Usage:
    python examples/sudoku_quickstart.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from concept_benchmark import SudokuDatasetGenerator

# ---------------------------------------------------------------------------
# 1. Generate dataset (renders board images by default, ~35 s for 100 boards)
# ---------------------------------------------------------------------------
print("Generating Sudoku dataset (100 boards with handwritten digit images)...")
dataset = SudokuDatasetGenerator(seed=171, n_samples=100).generate()

train, test = dataset.training, dataset.test
print(f"  Training:  {train.n} samples, {train.n_concepts} concepts")
print(f"  Test:      {test.n} samples")
print(f"  Concepts:  {train.concepts[:5]} ... ({train.n_concepts} total)")
print(f"  Classes:   {train.classes}")

# ---------------------------------------------------------------------------
# 2. Explore the dataset — opens an interactive viewer with board images
# ---------------------------------------------------------------------------
# Uncomment to launch the Spotlight viewer:
# dataset.training.explore()

df = train.to_dataframe()
print(f"\nDataFrame preview ({len(df)} rows, showing first 5 concepts):")
show_cols = list(train.concepts[:5]) + ["label"]
print(df[show_cols].head(8).to_string(index=False))

# Count valid vs invalid
n_valid = (train.y == 1).sum()
print(f"\nTraining set: {n_valid} valid, {train.n - n_valid} invalid boards")

# ---------------------------------------------------------------------------
# 3. Train a concept → label predictor
# ---------------------------------------------------------------------------
# In Sudoku, a board is valid iff ALL 27 concepts (row/col/block validity) are 1.
# This is an AND function — a single violated concept invalidates the board.
clf = LogisticRegression(max_iter=1000)
clf.fit(train.C, train.y)

acc = accuracy_score(test.y, clf.predict(test.C))
print(f"\nLabel predictor accuracy (perfect concepts): {acc:.4f}")

# Show the AND structure: all concept weights should be positive
weights = clf.coef_[0]
print(f"  All weights positive: {(weights > 0).all()}")
print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")

# ---------------------------------------------------------------------------
# 4. Impact of concept noise — the AND fragility
# ---------------------------------------------------------------------------
print("\nEffect of concept noise on accuracy:")
print("  (A single wrong concept can flip the prediction)")

rng = np.random.default_rng(42)
for noise_rate in [0.0, 0.02, 0.05, 0.10, 0.20]:
    C_noisy = test.C.copy()
    if noise_rate > 0:
        flip = rng.random(C_noisy.shape) < noise_rate
        C_noisy = np.where(flip, 1 - C_noisy, C_noisy)
    acc = accuracy_score(test.y, clf.predict(C_noisy))
    print(f"  noise={noise_rate:.0%}: accuracy={acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Selective classification: abstain on uncertain predictions
# ---------------------------------------------------------------------------
print("\nSelective classification demo:")
print("  (With perfect concept detectors, confidence = label predictor margin)")

proba = clf.predict_proba(test.C)[:, 1]
y_pred = clf.predict(test.C)

for target_acc in [0.90, 0.95, 0.99]:
    confidence = np.abs(proba - 0.5)
    # Find threshold that achieves target accuracy
    best_tau = 0.5
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
    print(f"  target={target_acc:.2f}:  sel_acc={sel_acc:.4f},  coverage={coverage:.1%}")

print("\nDone!")

# ---------------------------------------------------------------------------
# Next steps: for the full neural CS model pipeline with concept detectors,
# selective classification, and interventions, see
# examples/sudoku_pipeline_example.py.
# ---------------------------------------------------------------------------
