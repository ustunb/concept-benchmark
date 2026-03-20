# Sudoku Validation

This benchmark targets automation settings where the system handles routine cases and defers uncertain ones to a human. The task is to determine whether a 9×9 Sudoku board is valid, i.e., contains the digits 1–9 exactly once in each row, column, and block. The 27 concepts correspond to the validity of each row, column, and 3×3 block. A board is valid if and only if all 27 concepts are true (AND structure), so a single violated concept is enough to invalidate the board. When the model abstains, a human can verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty.

```{image} assets/sudoku_handwritten.png
:width: 400px
:align: center
:alt: Sudoku board with handwritten digits and concept annotations
```

## Parameters

All parameters can be passed to `DatasetGenerator("sudoku", ...)` or as CLI flags to `sudoku_pipeline.py`:

```python
from concept_benchmark import DatasetGenerator

dataset = DatasetGenerator(
    "sudoku",
    seed=171,                  # random seed
    data_type="image",         # "image" (renders board PNGs) or "tabular" (digit vectors)
    render_images=True,        # set False to skip rendering PNGs (faster, image only)
    block_size=3,              # block size (3 = standard 9×9 board)
    n_boards=1000,             # number of boards to generate
    max_cell_swaps=9,          # cells swapped in invalid boards (higher = subtler errors)
    valid_board_ratio=0.5,     # fraction of valid boards
    # ── Rendering (image only) ──
    font_style="handwritten",  # "handwritten" or "printed"
    font_size=25,              # digit font size in pixels
    cell_px=50,                # cell size in pixels
    cell_margin_px=2,          # cell margin in pixels
    gridline_px=2,             # grid line width in pixels
    block_border_px=5,         # block border width in pixels
).generate()
```

## Pipeline

To train models and run the full evaluation (selective classification, interventions, alignment) without writing Python, use the pipeline script:

```bash
python scripts/sudoku_pipeline.py --seed 171
```

Run `python scripts/sudoku_pipeline.py --help` for the full list of options.
