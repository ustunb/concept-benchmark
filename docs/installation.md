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

Or install from source (includes training/evaluation code and pipeline scripts):

```bash
git clone https://github.com/ustunb/concept-benchmark.git
cd concept-benchmark
pip install -e ".[experiments]"
```

If you use [uv](https://docs.astral.sh/uv/), `uv sync` works too and also installs dev/docs dependencies.

```{note}
`pip install concept-benchmark` gives you **dataset generation only** (`concept_benchmark/`). To run the full training/evaluation pipelines, clone the repo and install with `.[experiments]` — the `experiments/` directory contains model training, interventions, alignment, and LFCBM code.
```

Verify the installation:

```bash
python3 -c "import concept_benchmark; print('OK')"
```
