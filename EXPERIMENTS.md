# Reproducing Paper Experiments

All commands assume `uv sync` has been run from the repo root.

## Robot Benchmark

### Experiment 1: Concept Discovery (Section 5.1)

```bash
# Ideal (7 concepts)
python scripts/robot_pipeline.py --seed 1014 --stages setup cbm dnn intervene collect

# Subconcept (12 concepts)
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --stages setup cbm dnn intervene collect
```

### Experiment 2: Alignment (Section 5.2)

```bash
# Ideal
python scripts/robot_pipeline.py --seed 1014 --stages align collect

# Subconcept
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --stages align collect
```

### Experiment 3: Intervention Regimes (Section 5.3)

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes \
    --regimes baseline expert subjective machine \
    --strategy exactly_k --budgets 1 2 3 4 5 \
    --stages intervene collect
```

LLM and CLIP regimes require a Gemini API key:

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes \
    --regimes llm clip \
    --strategy exactly_k --budgets 1 2 3 4 5 \
    --stages intervene collect \
    --llm-api-key $GEMINI_API_KEY
```

## Sudoku Benchmark

### Experiment 4: Selective Classification (Section 4)

```bash
python scripts/sudoku_pipeline.py --seed 171 --data-type tabular
```

## Generating Datasets Only

```bash
pip install concept-benchmark
```

```python
from concept_benchmark.robots import DatasetGenerator

dataset = DatasetGenerator(seed=1014, concept_preset="foot_subtypes").generate()
dataset.sample(test_size=0.2, val_size=0.2, seed=1014)

X_train, C_train, y_train = dataset.train.X, dataset.train.C, dataset.train.y
```

```python
from concept_benchmark.sudoku import DatasetGenerator

dataset = DatasetGenerator(seed=171, n_boards=1000, data_type="tabular").generate()
dataset.sample(test_size=0.2, val_size=0.2, stratify=dataset.y, seed=171)
```
