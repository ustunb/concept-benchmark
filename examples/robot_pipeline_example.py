"""Robots: train a CBM and a DNN, correct concepts with the paper's policy, draw the decision-support panel.

Requires cloning the repo (uses ``experiments/``). Each step is one block of the benchmark; the pipeline
``scripts/robot_pipeline.py`` runs the same blocks.

    uv run python examples/robot_pipeline_example.py

About ten minutes on a laptop.
"""

from concept_benchmark.evaluation import accuracy, plot_intervention_curve
from concept_benchmark.robots import DatasetGenerator
from experiments.evaluate import (
    intervention_table,
    predict_labels,
    train_cbm,
    train_dnn,
)
from experiments.models import RobotClassifierCNN
from experiments.utils import run_alignment

SEED = 1014

# 1. Data: the paper's split, with the 12 human concepts
dataset = DatasetGenerator(seed=SEED, concept_preset="foot_subtypes").generate_splits()
train, val, test = dataset.train, dataset.val, dataset.test
print(f"train {train.n}, validation {val.n}, test {test.n}; concepts: {train.concepts}")

# 2. CBM: concept detector (image -> concepts) and label predictor (concepts -> label)
cbm = train_cbm(train, val, seed=SEED)
print(
    "concept weights:",
    dict(zip(train.concepts, cbm.label_predictor.model.coef_[0].round(2))),
)

# 3. Interventions: correct up to k concepts per robot with the paper's policy
results = intervention_table(cbm, test, budgets=(1, 3, "max"), seed=SEED)
print(
    results[["budget", "accuracy", "predictions_intervened_on"]].to_string(index=False)
)
# Over the paper's 10 seeds the CBM goes from 77.6% (k=0) to 85.7% (k=max).

# 4. DNN: image -> label, no concepts
dnn = train_dnn(lambda: RobotClassifierCNN(input_size=32), train, val, seed=SEED)
dnn_accuracy = accuracy(predict_labels(dnn, test), test.y)
print(f"DNN accuracy: {dnn_accuracy:.4f}")  # about 0.88 over the paper's 10 seeds

# 5. The paper's panel for this run
fig, ax = plot_intervention_curve(
    results, label="CBM, human_concepts", baseline_accuracy=dnn_accuracy
)
fig.savefig("robot_pipeline_example.png", dpi=150, bbox_inches="tight")
print("saved robot_pipeline_example.png")

# 6. Alignment: retrain the label predictor with has_knees forced to a positive weight
alignment = run_alignment(
    concept_based_model=cbm,
    train_dataset=train,
    test_dataset=test,
    monotonicity_constraints={"has_knees": 1},
)
print(
    f"alignment: {alignment['original_accuracy']:.4f} -> {alignment['aligned_accuracy']:.4f} "
    f"({alignment['accuracy_change']:+.4f})"
)
# Over the paper's 10 seeds the constraint raises accuracy before interventions (77.6% -> 81.0%).
