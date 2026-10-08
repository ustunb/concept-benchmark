# Installation

The package requires the **cairo** graphics library. Install it first:

```bash
# macOS
brew install cairo pkg-config

# Ubuntu / Debian
sudo apt-get install libcairo2-dev pkg-config python3-dev

# Fedora / RHEL
sudo dnf install cairo-devel pkg-config python3-devel
```

Then install the package:

```bash
pip install concept-benchmark
```

**Dataset generation, metrics, and plots** work out of the box with `pip install`. **Model training and interventions** (`experiments/` package and the pipelines in `scripts/`) require cloning the repo:

```bash
git clone https://github.com/ustunb/concept-benchmark.git
cd concept-benchmark
uv sync
```

Verify the installation:

```bash
python3 -c "import concept_benchmark; print(concept_benchmark.__version__)"
```

The datasets of the paper are also on the Hugging Face Hub ([`robots-true-concepts`](https://huggingface.co/datasets/juliannski/robots-true-concepts), [`robots-human-concepts`](https://huggingface.co/datasets/juliannski/robots-human-concepts), [`sudoku`](https://huggingface.co/datasets/juliannski/sudoku)):

```python
# pip install datasets
from datasets import load_dataset
ds = load_dataset("juliannski/robots-human-concepts")
```

## CEM, ProbCBM and ECBM

`./scripts/install_cem_repo.sh` installs the official [`mateoespinosa/cem`](https://github.com/mateoespinosa/cem) package that CEM and ProbCBM need; ECBM is included. Select a family with `--cbm-family cem|probcbm|ecbm` in the robot and sudoku pipelines (the text pipeline trains the CBM only; alignment runs on the CBM only). Their settings are fields of the config:

| Parameter | Default | Description |
|---|---|---|
| `cem_emb_size` | 16 | Concept embedding dimension for CEM |
| `cem_training_intervention_prob` | 0.25 | Intervention probability during CEM training |
| `training_mode` | `independent` | How the concept and label parts are trained: `independent`, `sequential` or `joint` (CLI: `--training-mode`) |
| `probcbm_n_samples_inference` | 50 | Monte Carlo samples during ProbCBM inference |
