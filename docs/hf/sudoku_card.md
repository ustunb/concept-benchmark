---
license: mit
task_categories:
- image-classification
language:
- en
tags:
- concept-bottleneck-models
- interpretability
- benchmark
- synthetic
size_categories:
- 1K<n<10K
source_datasets:
- original
pretty_name: Sudoku Validation
---

# Sudoku Validation

Synthetic benchmark for evaluating Concept Bottleneck Models on sudoku validation. Each example is an image of a 9x9 board; the label says whether the board is valid. The 27 binary concepts encode the validity of each row, column and block.

## Generated from

This dataset is the **exact output** of the [`concept-benchmark`](https://pypi.org/project/concept-benchmark/) Python package, with the train/validation/test split of the paper: 1,000 boards at 50 px per cell (454 x 454 px), each written in its own handwriting, with printed starters and candidate marks. The export is `scripts/export_hf_datasets.py` in the repository.

```python
# pip install concept-benchmark==0.4.0
from concept_benchmark.sudoku import DatasetGenerator
dataset = DatasetGenerator(seed=171).generate_splits()
```

📓 **[Quickstart notebook](./quickstart.ipynb)** — load from the Hub and train a model, end-to-end in one notebook.

🔗 [PyPI](https://pypi.org/project/concept-benchmark/) · [GitHub](https://github.com/ustunb/concept-benchmark)

## Quick start (datasets library)

```python
from datasets import load_dataset
ds = load_dataset("juliannski/sudoku")
row = ds["train"][0]
row["input"]  # PIL.Image
row["concepts"]    # 27 binary values, ordered as concept_names below
row["label"]       # 0 or 1
row["label_name"]  # "invalid" or "valid"
```

## Schema

| Field        | Type | Notes |
|---           |---   |---    |
| `input` | `datasets.Image()` | PNG, embedded |
| `concepts`   | `Sequence(int8, length=27)` | binary; index order = `concept_names` below |
| `label`      | `ClassLabel("invalid", "valid")` | |
| `label_name` | `string` | display alias for `label` |

**`concept_names`** (column order in `concepts`):

```python
['row_valid_1', 'row_valid_2', 'row_valid_3', 'row_valid_4', 'row_valid_5', 'row_valid_6', 'row_valid_7', 'row_valid_8', 'row_valid_9', 'col_valid_1', 'col_valid_2', 'col_valid_3', 'col_valid_4', 'col_valid_5', 'col_valid_6', 'col_valid_7', 'col_valid_8', 'col_valid_9', 'block_valid_1', 'block_valid_2', 'block_valid_3', 'block_valid_4', 'block_valid_5', 'block_valid_6', 'block_valid_7', 'block_valid_8', 'block_valid_9']
```

## Splits

| Split        | n      |
|---           |---     |
| `train`      | 600 |
| `validation` | 200 |
| `test`       | 200 |

Seed 171. 50% of the training examples are `valid`.

## License

MIT.
