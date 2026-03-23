"""Robot classification quickstart — end-to-end CBM workflow.

Tier 1: works with ``pip install concept-benchmark`` (no repo clone needed).

Generates a robot dataset, inspects it, trains a simple concept-based model
(sklearn on ground-truth concepts), and demonstrates how oracle interventions
improve accuracy.

For the full neural CBM pipeline (image concept detector + interventions),
see ``examples/robot_pipeline_example.py`` (requires cloning the repo).

Usage:
    python examples/robot_quickstart.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from concept_benchmark.robot import DatasetGenerator

# ---------------------------------------------------------------------------
# 1. Generate dataset (render_images=False skips image rendering for speed)
# ---------------------------------------------------------------------------
print("Generating robot dataset...")
gen = DatasetGenerator(seed=1014, render_images=False)
dataset = gen.generate()
from concept_benchmark.config import PRESET_EXCLUDED_CONCEPTS

dataset.drop_concepts(PRESET_EXCLUDED_CONCEPTS["ground_truth"])
dataset.sample(test_size=10000, val_size=0.2, train_size=3800, seed=1014)

train, test = dataset.train, dataset.test
print(f"  Training:  {train.n} samples, {train.n_concepts} concepts")
print(f"  Test:      {test.n} samples")
print(f"  Concepts:  {train.concepts}")
print(f"  Classes:   {train.classes}")

# ---------------------------------------------------------------------------
# 2. Inspect with to_dataframe()
# ---------------------------------------------------------------------------
df = train.to_dataframe()
print(f"\nDataFrame preview ({len(df)} rows):")
print(df.head(8).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Train a concept → label model (the label predictor in a CBM)
# ---------------------------------------------------------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(train.C, train.y)

print("\nLearned concept weights:")
for name, w in zip(train.concepts, clf.coef_[0]):
    print(f"  {name:20s}  {w:+.3f}")

baseline_acc = clf.score(test.C, test.y)
print(f"\nBaseline accuracy (perfect concepts): {baseline_acc:.4f}")

# ---------------------------------------------------------------------------
# 4. Simulate noisy concept predictions
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
noise_rate = 0.15
C_noisy = test.C.copy()
flip_mask = rng.random(C_noisy.shape) < noise_rate
C_noisy = np.where(flip_mask, 1 - C_noisy, C_noisy)

noisy_acc = (clf.predict(C_noisy) == test.y).mean()
print(f"Noisy concept accuracy (noise={noise_rate}):  {noisy_acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Oracle interventions: correct k most uncertain concepts per sample
# ---------------------------------------------------------------------------
print("\nOracle interventions (correct k wrong concepts):")

for k in [1, 3, 7]:
    C_intervened = C_noisy.copy()
    # Identify wrong positions and fix the first k per sample
    diff = C_noisy != test.C
    for i in range(len(test.C)):
        wrong_idx = np.where(diff[i])[0]
        fix = wrong_idx[:k]
        C_intervened[i, fix] = test.C[i, fix]
    acc = (clf.predict(C_intervened) == test.y).mean()
    gain = acc - noisy_acc
    print(f"  k={k}: accuracy={acc:.4f}  (gain={gain:+.4f})")

print("\nDone!")

# ---------------------------------------------------------------------------
# Next steps: for the full neural CBM pipeline with image concept detectors
# and programmatic interventions, see examples/robot_pipeline_example.py.
# ---------------------------------------------------------------------------
