# Reproducing Paper Experiments

All commands assume `uv sync` has been run from the repo root (and `./scripts/install_cem_repo.sh` for CEM and ProbCBM).

The paper reports every result as mean ± SE over 10 seeds: `1014`–`1023` for robots and `171`–`180` for sudoku. The commands below run one seed; loop over the seeds to reproduce a table. Numbers differ slightly across hardware, so expect to match the paper's means within their standard errors, not digit for digit.

[`scripts/paper/README.md`](scripts/paper/README.md) maps every table, figure and test of the paper to the script that builds it, from the paper's results (a download) or from your own runs:

```bash
python scripts/paper/collect_pipeline_runs.py --runs results --results-root my_results
python scripts/paper/make_robot_big_table.py --results-root my_results --out robot_table.tex
```

`collect_pipeline_runs.py` names the files after the settings of the runs; pass `--encoding percentile` for the runs of "Automated concepts and interventions" and `--target-accuracy 0.95` for the sudoku runs.

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
        python scripts/sudoku_pipeline.py --seed 171 --cell-px $px --cbm-family $family --target-accuracy 0.95
    done
done
```

The paper reports a selective-accuracy target of 0.95 (`--target-accuracy`; the default is 0.90). The `selective` stage also scores every model without interventions at the targets 0.90 to 0.99. Add the optional `diagnose` stage to save each model's confidence:

```bash
python scripts/sudoku_pipeline.py --seed 171 --cell-px 18 --target-accuracy 0.95 --stages cs intervene selective diagnose
```

The sudoku pipeline calibrates each model's label probability on the validation split before abstaining (`calibrate: true` in the config); the automation numbers depend on it. Training the same seed on another GPU model gives a slightly different model (raw sudoku accuracy moves by one or two points), so compare means over the ten seeds rather than single seeds. ProbCBM also samples at prediction time, and these samples are not seeded when the evaluation stages run on their own, so its sudoku numbers vary by a few boards between runs.

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
