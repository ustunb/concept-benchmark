# Sudoku Validation

This benchmark targets automation settings where the system handles routine cases and defers uncertain ones to a human. The task is to determine whether a 9×9 Sudoku board is valid, i.e., contains the digits 1–9 exactly once in each row, column, and block. The 27 concepts correspond to the validity of each row, column, and 3×3 block. A board is valid if and only if all 27 concepts are true (AND structure), so a single violated concept is enough to invalidate the board. When the model abstains, a human can verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty.

```{image} assets/sudoku_handwritten.png
:width: 400px
:align: center
:alt: Sudoku board with handwritten digits and concept annotations
```

## Expected Results

Generate the paper dataset (see {doc}`quickstart` for the full training code):

```python
from concept_benchmark import SudokuDatasetGenerator

dataset = SudokuDatasetGenerator(
    seed=171,
    n_samples=1000,       # number of boards
    max_corrupt=9,        # cells swapped in invalid boards
    valid_ratio=0.5,      # fraction of valid boards
).generate()
```

Selective classification results — the model abstains on uncertain predictions to meet a target accuracy (seed=171, target_accuracy=0.95):

| model | selective accuracy | coverage |
|-------|-------------------|----------|
| DNN | 0.875 | 0.04 |
| CS | 0.915 | 1.00 |

Or run the entire pipeline from the command line:

```bash
python scripts/sudoku_pipeline.py --seed 171
```

To skip data regeneration and only retrain models, use `--stages cs dnn selective intervene align collect`. Run `--help` for the full list of flags.

## Parameters

All parameters below can be passed directly to `SudokuDatasetGenerator()` or as CLI flags to `sudoku_pipeline.py`. For the full list, see `concept_benchmark/config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_corrupt` | `9` | Number of cells corrupted in invalid boards (higher values produce subtler errors) |
| `data_type` | `"image"` | `"image"` evaluates on OCR-inferred digits (adds OCR stage); `"tabular"` evaluates on ground-truth digit values (no OCR). Training always uses ground-truth values. |
| `handwriting` | `True` | Render digits in handwritten style (only applies when `data_type="image"`) |
| `target_accuracy` | `0.9` | Minimum accuracy required on kept predictions |

<details>
<summary>Remaining parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `171` | Random seed |
| `n_samples` | `1000` | Number of boards to generate |
| `valid_ratio` | `0.5` | Fraction of valid boards |
| `intervention_thresholds` | `[0.2, 0.4, 0.6, 0.8]` | Concept confidence thresholds that determine which concepts are candidates for verification |

</details>
