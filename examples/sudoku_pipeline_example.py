"""Sudoku: train a CBM and a DNN, let them abstain, check concepts, draw the automation panel.

Requires cloning the repo (uses ``experiments/``). Each step is one block of the benchmark; the pipeline
``scripts/sudoku_pipeline.py`` runs the same blocks on the images of the boards (through the digit
recognizer); this example reads the digits of each board directly.

    ./venv/bin/python examples/sudoku_pipeline_example.py

About a minute.
"""

from concept_benchmark.evaluation import plot_automation
from concept_benchmark.sudoku import DatasetGenerator
from experiments.evaluate import (
    automation_table,
    coverage_at_target,
    train_cbm,
    train_dnn,
)
from experiments.models import GroupPoolingConceptSudokuCNN, SudokuValidatorCNN

SEED, TARGET = 171, 0.95

# 1. Data: 1,000 boards, half of them invalid, 9 cells swapped in each invalid board
dataset = DatasetGenerator(
    seed=SEED, n_boards=1000, max_cell_swaps=9, data_type="tabular"
).generate_splits()
train, val, test = dataset.train, dataset.val, dataset.test
print(
    f"train {train.n}, validation {val.n}, test {test.n}; {train.n_concepts} concepts"
)

# 2. CBM: board -> 27 validity concepts -> valid. A board is valid iff every concept holds (an AND).
cbm = train_cbm(
    train,
    val,
    detector=GroupPoolingConceptSudokuCNN,
    epochs=100,
    patience=20,
    seed=SEED,
    should_propagate=True,
)
weights = cbm.label_predictor.model.coef_[0]
print(f"all concept weights positive: {(weights > 0).all()}")

# 3. DNN: board -> valid, no concepts
dnn = train_dnn(SudokuValidatorCNN(), train, val, seed=SEED)

# 4. Automation: both abstain until they are right on 95% of the boards they keep;
#    a human checks up to k concepts of each board the CBM would defer
results = automation_table(
    cbm, val, test, budgets=(1, 3, "max"), target_accuracy=TARGET, seed=SEED
)
dnn_accuracy, dnn_coverage = coverage_at_target(dnn, val, test, TARGET)
print(
    results[
        ["budget", "selective_accuracy_after", "coverage_after", "total_concept_checks"]
    ].to_string(index=False)
)
print(f"DNN: selective accuracy {dnn_accuracy:.3f}, coverage {dnn_coverage:.3f}")
# Checking concepts rarely raises coverage: one wrong concept keeps the board invalid (the AND).

# 5. The paper's panel for this run
fig, ax = plot_automation(
    results,
    n_instances=test.n,
    n_concepts=train.n_concepts,
    baseline_coverage=dnn_coverage,
    target_accuracy=TARGET,
)
fig.savefig("sudoku_pipeline_example.png", dpi=150, bbox_inches="tight")
print("saved sudoku_pipeline_example.png")
