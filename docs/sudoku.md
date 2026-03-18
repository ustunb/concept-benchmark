# Sudoku Validation

This benchmark targets automation settings where the system handles routine cases and defers uncertain ones to a human. The task is to determine whether a 9×9 Sudoku board is valid, i.e., contains the digits 1–9 exactly once in each row, column, and block. The 27 concepts correspond to the validity of each row, column, and 3×3 block. A board is valid if and only if all 27 concepts are true (AND structure), so a single violated concept is enough to invalidate the board. When the model abstains, a human can verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty.

```{image} assets/sudoku_handwritten.png
:width: 400px
:align: center
:alt: Sudoku board with handwritten digits and concept annotations
```

## Parameters

All parameters can be passed to `SudokuDatasetGenerator()` or as CLI flags to `sudoku_pipeline.py`:

```python
from concept_benchmark import SudokuDatasetGenerator

dataset = SudokuDatasetGenerator(
    seed=171,             # random seed
    n_samples=1000,       # number of boards to generate
    max_corrupt=9,        # cells swapped in invalid boards (higher = subtler errors)
    valid_ratio=0.5,      # fraction of valid boards
    data_type="image",    # "image" (OCR-inferred digits) or "tabular" (ground-truth values)
    handwriting=True,     # render digits in handwritten style (image only)
    target_accuracy=0.9,  # minimum accuracy on kept predictions (selective classification)
    # intervention_thresholds=[0.2, 0.4, 0.6, 0.8]  # concept confidence thresholds
).generate()
```

Run the full pipeline from the CLI:

```bash
python scripts/sudoku_pipeline.py --seed 171
python scripts/sudoku_pipeline.py --seed 171 --stages cs dnn selective intervene align collect  # skip data regen
# run --help for all flags
```
