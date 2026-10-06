"""Sudoku validation quickstart — exploring the concept bottleneck.

Tier 1: works with ``pip install concept-benchmark`` (no repo clone needed).

Generates a small Sudoku dataset with board images, inspects its structure,
trains a label predictor on perfect concepts, and demonstrates selective
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

from concept_benchmark.evaluation import abstention_threshold, selective_at
from concept_benchmark.sudoku import DatasetGenerator

# ---------------------------------------------------------------------------
# 1. Generate dataset (renders board images by default, about 20 s for 50 boards)
# ---------------------------------------------------------------------------
print("Generating Sudoku dataset (50 boards with handwritten digit images)...")
dataset = DatasetGenerator(seed=171, n_boards=50).generate()
dataset.sample(test_size=0.2, val_size=0.2, stratify=dataset.y, seed=171)

train, val, test = dataset.train, dataset.val, dataset.test
print(f"  Training:  {train.n} samples, {train.n_concepts} concepts")
print(f"  Test:      {test.n} samples")
print(f"  Concepts:  {train.concepts[:5]} ... ({train.n_concepts} total)")
print(f"  Classes:   {train.classes}")

# ---------------------------------------------------------------------------
# 2. Explore the dataset — opens an interactive viewer with board images
# ---------------------------------------------------------------------------
# Uncomment to launch the Spotlight viewer:
# dataset.train.explore()

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

for target_accuracy in [0.90, 0.95, 0.99]:
    # the threshold at which the kept predictions reach the target (fit on the validation boards)
    threshold, _ = abstention_threshold(
        val.y, clf.predict_proba(val.C)[:, 1], target_accuracy
    )
    if threshold is None:
        print(f"  target={target_accuracy:.2f}:  out of reach")
        continue
    sel_acc, cov = selective_at(test.y, proba, threshold, 0.5)
    print(
        f"  target={target_accuracy:.2f}:  sel_acc={sel_acc:.4f},  coverage={cov:.1%}"
    )

print("\nDone!")

# ---------------------------------------------------------------------------
# Next steps: for the full neural CS model pipeline with concept detectors,
# selective classification, and interventions, see
# examples/sudoku_pipeline_example.py.
# ---------------------------------------------------------------------------
