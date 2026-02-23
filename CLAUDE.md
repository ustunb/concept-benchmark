# Concept Benchmark - Project Instructions

## What This Project Is

A **toolbox paper** ("Measuring What Matters" by Skirzynski et al., UCSD) providing synthetic benchmarks for evaluating Concept Bottleneck Models (CBMs).

**The problem:** CBM research is bottlenecked by scarce concept-annotated datasets. Existing evaluations reuse ~10 datasets (CUB-200, CelebA, etc.) and don't reflect real prediction problems where interpretability matters.

**The solution:** Synthetic benchmarks with fully-specified ground truth, enabling controlled evaluation of CBM architectures across:
- Data modality (images, text, tabular)
- Concept granularity (ground-truth vs. misspecified concepts)
- Annotation quality (noise, missingness, subjectivity)
- Intervention regimes (expert, machine-discovered, LLM-based, CLIP-based)

### Two Benchmarks, Two Use Cases

**Robot Classification (Decision Support):** Classify fictional robots (Glorps vs. Drents) from images. 9 binary body features + subconcepts. Measures gain from human interventions correcting concept predictions — like a dermatologist verifying detected patterns.

**Sudoku Validation (Automation):** Determine if a 9x9 Sudoku board is valid. 27 binary concepts (row/column/block validity). Measures net automated work via selective abstention — the model handles routine cases and defers uncertain ones.

### Key Findings the Benchmarks Reveal
- Finer-grained concepts can underperform coarse ground-truth concepts
- Alignment constraints that preserve training accuracy can destroy intervention benefit
- Interventions degrade rapidly with concept noise (from +15.5% to -35% with machine-discovered concepts)

## Project Context

This is a **toolbox paper**, not a methods paper. When editing or reviewing, focus on usability, API design, and practical value — not novelty of methods. Keep writing smooth and precise; avoid overclaiming or clunky phrasing.

Always understand the paper's exact experimental setup before generating comparison results or writing reproduction scripts:
- Two setups: **concepts** (7 ideal) and **subconcepts** (12 subconcepts). These are distinct and must not be confused.
- All regimes work with both ideal and subconcept variants. LFCBM regimes (machine/llm/clip) operate in their own 12-concept space independent of the GT concept count.
- Never fabricate or assume result structures. If unsure, read the source code or ask.

## Environment

- Python 3.10+, PyTorch-based
- Venv: `./venv/bin/python` — always use this, with `PYTHONPATH=.`
- Tests: `./venv/bin/python -m pytest tests/ -v`
- Package management: `uv`
- Install: `./install.sh && source venv/bin/activate`

## Quick Start — Running Pipelines

The CLI entry point is `cbm-benchmark` (defined in `pyproject.toml`).

### Robot (image)
```bash
# Basic subconcept run (paper default)
cbm-benchmark robot --seed 1014 --subconcept

# With intervention regimes
cbm-benchmark robot --seed 1014 --subconcept --regimes baseline expert subjective machine

# Paper-matching exact-k interventions
cbm-benchmark robot --seed 1014 --subconcept --strategy exact_k --regimes baseline expert

# Run specific stages only (default: setup cbm dnn intervene align collect)
cbm-benchmark robot --stages setup cbm dnn intervene
```

### Sudoku
```bash
cbm-benchmark sudoku --seed 171

# Skip data regeneration (reuse existing boards), only retrain models
cbm-benchmark sudoku --seed 171 --stages cs dnn selective intervene align collect
```

### Robot Text
```bash
cbm-benchmark robot-text --seed 1337
cbm-benchmark robot-text --regimes baseline expert subjective
```

### Key CLI Flags

| Flag | Benchmarks | Description |
|------|-----------|-------------|
| `--seed` | all | Random seed (robot: 1014, sudoku: 171, robot-text: 1337) |
| `--stages` | all | Which stages to run |
| `--subconcept` | robot | Use 12 subconcepts instead of 7 ideal concepts |
| `--regimes` | robot, robot-text | Intervention regimes: `baseline expert subjective machine llm clip` |
| `--strategy` | robot, robot-text | `kflip` (up-to-k) or `exact_k` (exactly k concepts) |
| `--no-missing` | robot | Skip MCAR/MNAR missingness variants |
| `--force-retrain` | robot | Retrain LFCBM/subjective models even if cached |
| `--llm-api-key` | robot | API key for LLM provider |

### Pipeline Stages

**Robot:** setup → cbm → dnn → intervene → align → collect
**Sudoku:** setup → ocr → cs → dnn → intervene → selective → align → collect

### Demo Scripts
```bash
PYTHONPATH=. python scripts/demo_robot.py    # Subconcept + MCAR (Table 2)
PYTHONPATH=. python scripts/demo_sudoku.py   # Sudoku mc21 (Table 3)
```

## Paper Experiments & Results

All paper results: seed=1014 (robot) / seed=171 (sudoku), threshold=0.2.

### Reproduction Status

| Experiment | Status | Notes |
|-----------|--------|-------|
| Concept Discovery (robot) | **Exact match** | ideal + subconcept baseline results match paper |
| Alignment (robot) | **Exact match** | aligned CBM numbers match paper |
| Regime: baseline | **Exact match** | Deterministic ground-truth interventions, identical |
| Regime: expert | **Exact match (exact_k)** | Matches paper with `--strategy exact_k`; kflip differs slightly |
| Regime: subjective | **Trend matches** | Same direction (+small gain), different magnitude (reimplemented noisy CBM) |
| Regime: machine | **Trend matches** | Same direction (backfire), different magnitude (our: ~0.56, paper: ~0.45 at k=1) |
| Regime: clip | **Trend matches** | Backfires as expected (0.87→0.42-0.59); LLM not yet run (no cache) |
| Sudoku | **Not reproduced** | Numbers don't match paper |

The plan is to regenerate results with the new codebase and show the same trends rather than reproduce exact paper numbers for the noisy/machine regimes.

### Experiment 1: Concept Discovery (Robot, Section 5.1)

Tests effect of concept granularity on CBM performance and interventions.

**Parameters:** 32px images, stochastic labeling (`P(Glorp|c) = σ(5·MouthType + 8·FootShape - 5·HasKnees - 3)`), 3800 train / 10000 test.

| Setup | DNN | CBM (k=0) | CBM (k=1) | CBM (k=3) | CBM (k=max) |
|-------|-----|-----------|-----------|-----------|-------------|
| ideal (7 concepts) | 0.8746 | 0.8673 | 0.9736 | 0.9769 | 0.9769 |
| subconcept (12 concepts) | 0.8746 | 0.7812 | 0.9212 | 0.9439 | 0.9439 |

**Finding:** Finer-grained subconcepts degrade baseline (-9.3%) but interventions recover most of the gap.

### Experiment 2: Alignment (Robot, Section 5.2)

Tests whether alignment constraints (forcing `has_knees` weight to +1) preserve intervention benefit.

| Setup | CBM (k=0) | Aligned (k=0) | CBM (k=3) gain | Aligned (k=3) gain |
|-------|-----------|----------------|-----------------|---------------------|
| ideal | 0.8673 | 0.8657 | +10.2% | -0.4% |
| subconcept | 0.7812 | 0.7656 | +6.9% | -8.0% |

**Finding:** Alignment preserves training accuracy but **destroys** intervention benefit.

### Experiment 3: Concept Noise / Intervention Regimes (Robot, Section 5.3, Figure 7)

Tests how annotation quality affects interventions. Subconcept setup only. Figure 7 plots mean ΔAccuracy averaged over k∈{1,2,5}.

**Per-budget ΔAccuracy (gain over no-intervention baseline):**

| Regime | k=1 | k=2 | k=5 | Mean(k=1,2,5) |
|--------|-----|-----|-----|----------------|
| **baseline** | +14.0% | +16.3% | +16.3% | **+15.5%** |
| **expert** | +9.7% | +11.5% | +10.9% | **+10.7%** |
| **subjective** | +0.3% | +0.6% | +0.0% | **+0.3%** |
| **llm** | -26.2% | -31.4% | -23.2% | **-26.9%** |
| **clip** | -32.0% | -32.0% | -29.0% | **-31.0%** |
| **machine** | -33.6% | -36.7% | -34.6% | **-35.0%** |

Source CSV: `concept-benchmark-paper/gain_vs_perfect_all.csv`. Figure 7 plots `np.mean([gain_k1, gain_k2, gain_k5]) * 100` with min/max error bars.

**Finding:** Interventions help only under specific conditions. Degradation: perfect (+15.5%) → expert (+10.7%) → subjective (+0.3%) → LLM (-26.9%) → CLIP (-31.0%) → machine (-35.0%).

### Experiment 4: Sudoku Automation (Section 4)

1000 boards, 50:50 valid/invalid, seed=171, handwritten digits, 27 concepts (row/column/block validity). Paper uses mc9 (max_corrupt=9), target_accuracy=0.95.

| Metric | DNN | CBM (k=0) | CBM (k=1) | CBM (k=3) | CBM (k=max) |
|--------|-----|-----------|-----------|-----------|-------------|
| Selective Accuracy | 81.8% | 98.1% | 98.2% | 97.8% | 96.2% |
| Coverage | 5.5% | 87.6% | 87.7% | 88.0% | 90.5% |

**Finding:** CBM achieves 98.1% selective accuracy vs DNN's 81.8%. Interventions have diminishing returns due to AND structure (single concept error fails entire board).

**`target_accuracy` parameter:** The minimum selective accuracy demanded on the validation set. A confidence threshold τ is found such that kept predictions achieve ≥ target_accuracy. Higher target_accuracy → more abstention → lower coverage but higher accuracy on kept predictions.

### Intervention Regimes — Config Reference

| Regime | Concept Source | Intervention Source | `intervention_accuracy` | Extra Config |
|--------|---------------|--------------------|-----------------------|--------------|
| baseline | ground truth | ground truth | 1.0 | — |
| expert | ground truth | noisy human | 0.8 | — |
| subjective | noisy CBM (20% label noise) | noisy human | 0.8 | `subjective_noise_rate=0.2` |
| machine | LFCBM on GT descriptions | noisy human | 0.8 | `lfcbm_concepts_file` |
| llm | LFCBM on LLM descriptions | LLM (Gemini) | — | `llm_concepts_file`, `GEMINI_API_KEY` |
| clip | LFCBM on CLIP keywords | LLM (Gemini) | — | `clip_concepts_file`, `GEMINI_API_KEY` |

Concept description files in `data/robot_images/`: `gt_concepts.jsonl`, `gt_concepts_subconcept.jsonl`, `llm.jsonl`, `clip.jsonl` (12 concepts each for subconcept).

## Sudoku Pipeline Notes

### What the seed controls
The `--seed` flag controls **everything**: board generation (which sudoku solutions, which cells corrupted, which handwritten digit images), CV splits (train/val/test), and model training randomness. To sweep training seeds without regenerating data (~5 min/seed), load data once from a fixed seed directory and vary only the training seed (controls splits + model init).

### Data generation is the bottleneck
`setup` + `ocr` stages take ~5 min per seed (generates 1000 boards with handwritten digit images). Training (`cs` + `dnn` + `selective`) takes ~30s. To skip data regeneration, use `--stages cs dnn selective intervene align collect`.

### DNN non-determinism (fixed)
`nn.Embedding` backward pass uses scatter-add which is non-deterministic on MPS (Apple Silicon). Fixed by replacing with `nn.Linear` on one-hot encoded input in `models.py` (both `SudokuValidatorCNN` and `GroupPoolingConceptSudokuCNN`). Robot DNN was never affected (uses `Conv2d` on float tensors, no embedding).

### Sudoku model behavior (mc9, target_accuracy=0.95, seed sweep 130-199)
- **CS model is very stable**: sel_acc ~0.977±0.010, coverage ~0.995±0.007 across all seeds
- **DNN is highly variable**: ~39% of seeds produce random-chance DNN (50% acc, 0 coverage). When it learns, sel_acc is 0.8-1.0 but at very low coverage (1-8%)
- Our CS model is stronger than the paper's (coverage ~100% vs paper's 87.6%), likely due to model/training differences
- Sweep results saved in `results/sudoku_seed_sweep_mc9_ta095.csv`

## Debugging Principles

- Check configuration/settings dictionaries and simple parameter errors **FIRST** before exploring complex hypotheses.
- Do not go down rabbit holes (e.g., MPS GPU hangs, deep framework internals) without first ruling out straightforward causes like wrong values, missing params, or config mismatches.
- If 3 hypotheses fail, **STOP** and present findings so far. Ask which direction to pursue rather than continuing to guess.

## Figures & Visualization

When working on figures (matplotlib/LaTeX), make **ONE change at a time** and show the result before proceeding. Do not stack multiple visual adjustments in a single edit — annotations, spacing, and styling changes frequently conflict. After each change, regenerate the figure and confirm before the next edit.

## API & External Services

Be aware of API rate limits (especially Gemini free tier: RPM and RPD). Before proposing batch sizes or retry strategies, check what tier/limits apply and calculate whether the approach will exceed them.

## Documentation Style

When writing READMEs or documentation, keep it concise and scannable. Prefer short descriptions, command examples, and table-based result summaries over detailed prose.
