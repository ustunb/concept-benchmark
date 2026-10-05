# Reproducing Paper Experiments

All commands assume `uv sync` has been run from the repo root (and `./scripts/install_cem_repo.sh` for CEM and ProbCBM).

The paper reports every result as mean ± SE over 10 seeds: `1014`–`1023` for robots and `171`–`180` for sudoku. The commands below run one seed; loop over the seeds to reproduce a table. Numbers differ slightly across hardware, so expect to match the paper's means within their standard errors, not digit for digit.

Once the runs exist, [`scripts/paper/README.md`](scripts/paper/README.md) maps every table, figure and test of the paper to the script that produces it.

## Robot Benchmark

The default label rule is the paper's `balanced` rule; `--concept-preset ground_truth` gives `true_concepts` (7) and `foot_subtypes` gives `human_concepts` (12).

### Misspecified concepts (CBM against the DNN)

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset ground_truth --budgets 1 3 max
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --budgets 1 3 max
```

### Other architectures

```bash
for family in cem probcbm ecbm; do
    python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes \
        --cbm-family $family --budgets 1 3 max --stages cbm intervene collect
done
```

ProbCBM is trained with `--training-mode joint`.

### Automated concepts and interventions

```bash
# every concept source, with perfect and expert interventions and the model's own answers
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes \
    --concept-sources human_concepts machine_annotation llm_concepts clip_concepts \
    --intervention-sources perfect expert self \
    --intervention-encoding percentile --budgets 1 3 max \
    --dump-interventions results/intervention_records \
    --stages intervene collect
```

`--intervention-encoding percentile` applies to the label-free CBM; add `--cbm-family` for the other architectures.

LLM interventions use Gemini by default and need an API key:

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes \
    --concept-sources human_concepts --intervention-sources llm \
    --budgets 1 3 max --llm-api-key YOUR_KEY --stages intervene collect
```

`scripts/paper/build_llm_caches.py` queries the LLM once per test robot and concept, so that every model reads the same answers with `--llm-cache-only`.

### Alignment

```bash
python scripts/robot_pipeline.py --seed 1014 --concept-preset ground_truth --stages align collect
python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --stages align collect
```

`scripts/paper/evaluate_alignment_balanced.py` evaluates the constrained and unconstrained models before and after interventions on all seeds.

### Rare class (appendix)

```bash
python scripts/robot_pipeline.py --seed 1014 --label-rule sparse --concept-preset foot_subtypes --budgets 1 3 max
```

### Real datasets (appendix)

```bash
python experiments/real_data.py --out results/paper/real_datasets \
    --derm-root <Derm7pt> --cub-root <CUB_200_2011>
```

## Sudoku Benchmark

```bash
for px in 50 18; do
    for family in cbm cem probcbm ecbm; do
        python scripts/sudoku_pipeline.py --seed 171 --cell-px $px --cbm-family $family
    done
done
```

The `selective` stage scores every model at the selective-accuracy targets 0.90 to 0.99; the paper reports 0.95. Add the optional `diagnose` stage to save each model's confidence:

```bash
python scripts/sudoku_pipeline.py --seed 171 --cell-px 18 --stages cs intervene selective diagnose
```

Data generation (`setup`, `ocr`) takes about five minutes per seed; skip it on later runs with `--stages cs dnn intervene selective collect`.

## Generating Datasets Only

```python
from concept_benchmark.robots import DatasetGenerator

dataset = DatasetGenerator(seed=1014, concept_preset="foot_subtypes").generate_splits()
print(dataset.train.C.shape)  # (3800, 12)
```

```python
from concept_benchmark.sudoku import DatasetGenerator

dataset = DatasetGenerator(seed=171, n_boards=1000).generate_splits()
print(dataset.train.C.shape)  # (600, 27)
```

The same datasets are on the Hugging Face Hub: [`robots-true-concepts`](https://huggingface.co/datasets/juliannski/robots-true-concepts), [`robots-human-concepts`](https://huggingface.co/datasets/juliannski/robots-human-concepts), [`sudoku`](https://huggingface.co/datasets/juliannski/sudoku).
