# Sudoku Validation

This benchmark targets automation settings where the system handles routine cases and defers uncertain ones to a human. The task is to determine whether a 9×9 Sudoku board is valid, i.e., contains the digits 1–9 exactly once in each row, column, and block. The 27 concepts correspond to the validity of each row, column, and 3×3 block. A board is valid if and only if all 27 concepts are true (AND structure), so a single violated concept is enough to invalidate the board. When the model abstains, a human can verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty.

```{image} assets/sudoku_handwritten.png
:width: 400px
:align: center
:alt: Sudoku board with handwritten digits and concept annotations
```

## Parameters

All parameters can be passed to `DatasetGenerator(...)` (imported from `concept_benchmark.sudoku`); `sudoku_pipeline.py` takes `seed`, `cell_px`, `target_accuracy` and `cbm_family` as flags and every other parameter (`n_boards`, `max_cell_swaps`, `valid_board_ratio`, ...) through a YAML `--config`:

```python
from concept_benchmark.sudoku import DatasetGenerator

dataset = DatasetGenerator(
    seed=171,                  # random seed
    data_type="image",         # "image" (renders board PNGs) or "tabular" (digit vectors)
    render_images=True,        # set False to skip rendering PNGs (faster, image only)
    block_size=3,              # block size (3 = standard 9×9 board)
    n_boards=1000,             # number of boards to generate
    max_cell_swaps=9,          # corruptions applied to each invalid board, 1 to 9 (each swaps or duplicates digits)
    valid_board_ratio=0.5,     # fraction of valid boards
    blank_cell_ratio=0.0,      # share of cells left blank on each invalid board (breaks its row, column, block)
    symbol_cell_ratio=0.0,     # share of cells holding a letter on each invalid board (breaks them too)
    # ── Rendering (image only) ──
    font_style="handwritten",  # "handwritten" or "printed"
    font_size=25,              # printed-digit font size at 50 px per cell; every size is this board scaled
    cell_px=50,                # cell size in pixels
    cell_margin_px=2,          # cell margin in pixels
    gridline_px=2,             # grid line width in pixels
    block_border_px=5,         # block border width in pixels
    pencil_mark_ratio=0.22,    # share of cells with pencil marks (candidate digits)
    pencil_mark_sizes=(2,),    # candidate digits per pencil mark
    printed_digit_ratio=0.35,  # share of digits printed as givens; the rest are handwritten
    render_options=None,       # other image_transform arguments, e.g. {"fill_color": (0, 0, 0)}
).generate()
```

Blank cells and letters appear only on invalid boards, so the share of valid boards stays as set. In tabular data and for the OCR recognizer, both are the value 0 ("other").

`generate()` returns the unsplit dataset; `generate_splits()` returns the train/val/test split of the paper (60/20/20, stratified on the label).

## Pipeline

To train models and run the full evaluation (selective classification, interventions, alignment) without writing Python, use the pipeline script:

```bash
python scripts/sudoku_pipeline.py --seed 171
python scripts/sudoku_pipeline.py --seed 171 --cbm-family cem

# Harder concept detection: 10 pixels per cell instead of 50
python scripts/sudoku_pipeline.py --seed 171 --cell-px 10

# Save each model's confidence for plot_confidence (optional stage)
python scripts/sudoku_pipeline.py --seed 171 --stages cs intervene selective diagnose plot
```

At 10 px and the paper's target of 0.99, the abstention threshold is fitted on the validation boards; on the test boards of seed 171 the answered predictions are 99% correct without checks and 98% with all 27, since a misread digit makes the model confidently wrong on a few boards that it never defers.

| Option | Description |
|--------|-------------|
| `--cbm-family` | `cbm`, `cem`, `probcbm` or `ecbm` |
| `--cell-px` | Pixels per cell (50 by default; at 10 the recognizer misreads a digit on half the boards) |
| `--pencil-mark-ratio`, `--printed-digit-ratio` | Share of cells with pencil marks (0.22) and of printed givens (0.35) |
| `--blank-cell-ratio`, `--symbol-cell-ratio` | Share of cells left blank or holding a letter on invalid boards (0); the recognizer reads both as "other" |
| `--target-accuracy` | Selective accuracy that kept predictions must reach (0.90 by default; the paper uses 0.99) |
| `calibrate` (config) | Platt-scale each model's label probability on the validation split before it abstains (on by default; the safeguard rule needs calibrated probabilities) |
| `--stages` | `setup ocr cs dnn intervene selective align collect plot`, plus the optional `diagnose` |
| `--budgets` | Intervention budgets to evaluate (`1 3 max` by default) |
| `--data-type` | `image` (default) or `tabular` (the digits of each board, no recognizer) |
| `--no-handwriting` | Printed digits instead of handwritten ones (`--handwriting` restores the default) |
| `--direct-image` | A ViT on the board images instead of the digit recognizer |
| `--force-setup` | Regenerate the dataset even if one with the same parameters exists |

Run `python scripts/sudoku_pipeline.py --help` for the full list of flags; training parameters (`cs_epochs`, `cs_patience`, `ocr_epochs`, `ocr_patience`, `epochs`, `batch_size`) are set in a YAML config passed with `--config`.

Image datasets are written to `data/sudoku/sudoku_image_…_seed<seed>_r<RENDER_VERSION>`; the suffix changes when the renderer changes (3 since 0.4.1, which gives every board its own layout of printed digits), so boards from different renderers never share a folder. The paper's boards are those of `v0.4.0`.
