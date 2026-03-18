# Sudoku Validation

This benchmark targets automation settings where the system handles routine cases and defers uncertain ones to a human. The task is to determine whether a 9×9 Sudoku board is valid, i.e., contains the digits 1–9 exactly once in each row, column, and block. The 27 concepts correspond to the validity of each row, column, and 3×3 block. A board is valid if and only if all 27 concepts are true (AND structure), so a single violated concept is enough to invalidate the board. When the model abstains, a human can verify specific concepts (e.g., "is row 5 valid?") to resolve the uncertainty.

```{image} assets/sudoku_handwritten.png
:width: 400px
:align: center
:alt: Sudoku board with handwritten digits and concept annotations
```

## Setup and Evaluation

Generate a dataset, train a concept-supervised (CS) model — the Sudoku equivalent of a CBM — and evaluate selective classification. The CS model predicts 27 concepts, then a label predictor determines board validity. The selective classification stage finds a confidence threshold so that kept predictions achieve at least the target accuracy:

```python
import numpy as np
from concept_benchmark import SudokuDatasetGenerator
from concept_benchmark.utils import set_deterministic_seed
from experiments.models import (
    ConceptDetector, FrontEndModel, ConceptBasedModel, GroupPoolingConceptSudokuCNN,
)
from concept_benchmark.utils import determine_device, get_loader_config, patch_macos_dataloader

set_deterministic_seed(171)
patch_macos_dataloader()
device = determine_device()
loader_config = get_loader_config(device)

# Generate dataset — max_corrupt controls how subtle invalid boards are
dataset = SudokuDatasetGenerator(
    seed=171,
    n_samples=1000,       # number of boards
    max_corrupt=9,        # cells swapped in invalid boards
    valid_ratio=0.5,      # fraction of valid boards
).generate()

# Train concept detector (board digits → 27 validity concepts)
cd = ConceptDetector(model=GroupPoolingConceptSudokuCNN())
cd.fit(dataset.training, dataset.validation,
       fit_params={"epochs": 100, "lr": 1e-3, "patience": 20, "device": str(device), **loader_config})

# Train label predictor and combine into a CBM
fe = FrontEndModel()
fe.fit(dataset.training.C, dataset.training.y)
cbm = ConceptBasedModel(concept_detector=cd, front_end_model=fe)

# Evaluate
predictions = cbm.predict(dataset.test)
print(f"CS accuracy: {np.mean(predictions == dataset.test.y):.4f}")
```

The pipeline also evaluates selective classification — the model abstains on uncertain predictions to achieve a minimum accuracy on kept samples (`target_accuracy`). Expected results (seed=171, target_accuracy=0.95):

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
