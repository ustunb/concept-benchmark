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
git clone https://anonymous.4open.science/r/concept-benchmark-84D2.git
cd concept-benchmark
uv sync
```

```{note}
`pip install concept-benchmark` gives you **dataset generation only** (`concept_benchmark/`). To run the full training/evaluation pipelines, clone the repo and use `uv sync` — this installs all dependencies including dev tools and pipeline scripts.
```

Verify the installation:

```bash
python3 -c "import concept_benchmark; print('OK')"
```
