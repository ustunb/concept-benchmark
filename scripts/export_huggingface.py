"""Export a benchmark dataset in the layout of the Hugging Face Hub, and optionally upload it.

Builds the paper's train/validation/test split with ``DatasetGenerator.generate_splits()`` and writes
``<out>/<dataset>/data/{train,validation,test}-00000-of-00001.parquet`` and a dataset card (``README.md``).

    python scripts/export_huggingface.py --dataset robots-human-concepts
    python scripts/export_huggingface.py --dataset robots-true-concepts --push --owner <hub user>

``--push`` uploads the folder to ``<owner>/<dataset>`` and needs a Hub token (``huggingface-cli login``).
Requires ``datasets`` and ``huggingface_hub`` (``uv sync --dev``).
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np

from concept_benchmark.config import ROBOT_LABEL_RULES
from concept_benchmark.generators import DatasetGenerator

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")

DATASETS = {
    "robots-true-concepts": dict(
        benchmark="robot", seed=1014, options=dict(concept_preset="ground_truth"), image_field="image",
        class_names=("drent", "glorp"), title="Robots — True Concepts",
        summary="Synthetic benchmark for evaluating Concept Bottleneck Models (CBMs) with the true concepts. "
        "Robot images are generated deterministically; `foot_shape` is the binary pointy/flat concept that the "
        "labeling rule uses.",
    ),
    "robots-human-concepts": dict(
        benchmark="robot", seed=1014, options=dict(concept_preset="foot_subtypes"), image_field="image",
        class_names=("drent", "glorp"), title="Robots — Human Concepts",
        summary="Synthetic benchmark for evaluating Concept Bottleneck Models (CBMs) with misspecified concepts. "
        "Same robots, labels and splits as `robots-true-concepts`, but `foot_shape` is replaced by the six foot "
        "subtypes that appear in the training split. The test split also holds four subtypes never seen in "
        "training, for which every foot concept is 0.",
    ),
    "sudoku": dict(
        benchmark="sudoku", seed=171, options={}, image_field="input", class_names=("invalid", "valid"),
        title="Sudoku Validation",
        summary="Synthetic benchmark for evaluating Concept Bottleneck Models on sudoku validation. Each example "
        "is an image of a 9x9 board; the label says whether the board is valid. The 27 binary concepts encode "
        "the validity of each row, column and block.",
    ),
}


def build_dataset_dict(name: str):
    """The splits of `name` as a `datasets.DatasetDict`, plus the concept names."""
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value

    spec = DATASETS[name]
    data = DatasetGenerator(spec["benchmark"], seed=spec["seed"], **spec["options"]).generate_splits()
    concept_names = list(data.train.concepts)
    features = Features({
        spec["image_field"]: Image(),
        "concepts": Sequence(Value("int8"), length=len(concept_names)),
        "label": ClassLabel(names=list(spec["class_names"])),
        "label_name": Value("string"),
    })
    splits = {}
    for split in SPLITS:
        part = getattr(data, split)
        labels = np.asarray(part.y).astype(int)
        splits[split] = Dataset.from_dict(
            {
                # image bytes, so that the parquet file holds the images and not local paths
                spec["image_field"]: [
                    {"bytes": (Path(part.base_dir) / path).read_bytes(), "path": None} for path in part.inputs
                ],
                "concepts": np.asarray(part.C).astype(np.int8).tolist(),
                "label": labels.tolist(),
                "label_name": [spec["class_names"][label] for label in labels],
            },
            features=features,
        )
    return DatasetDict(splits), concept_names


def write_card(name: str, dataset_dict, concept_names: list[str], owner: str) -> str:
    """Dataset card describing how the data was generated and what it contains."""
    spec = DATASETS[name]
    sizes = {split: len(dataset_dict[split]) for split in SPLITS}
    n_total = sum(sizes.values())
    positive_share = float(np.mean(dataset_dict["train"]["label"]))
    size_category = "10K<n<100K" if n_total >= 10_000 else "1K<n<10K" if n_total >= 1_000 else "n<1K"
    options = "".join(f", {key}={value!r}" for key, value in spec["options"].items())
    module = "robots" if spec["benchmark"] == "robot" else "sudoku"
    if spec["benchmark"] == "robot":
        rule = (
            "\n## Labeling rule\n\n"
            "Labels follow the package's default `balanced` rule: no single concept decides the label, and both "
            "classes are equally likely. `has_elbows` also carries weight in the rule but is not among the "
            "concepts, so it is visible in the image and cannot be intervened on. The training split is skewed "
            "toward six foot subtypes ("
            + ", ".join(
                f"{next(iter(c['concepts'])).removeprefix('foot_shape_')} {c['min_fraction']:.0%}"
                for c in ROBOT_LABEL_RULES["balanced"].sampling_constraints
            )
            + "); the test split is drawn from all ten. Pass `label_rule=\"sparse\"` to the generator for the "
            "rule with a rare class.\n"
        )
    else:
        rule = ""
    return f"""---
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
- {size_category}
source_datasets:
- original
pretty_name: {spec["title"]}
---

# {spec["title"]}

{spec["summary"]}

## Generated from

This dataset is the **exact output** of the [`concept-benchmark`](https://pypi.org/project/concept-benchmark/) Python package, with the train/validation/test split of the paper.

```python
# pip install concept-benchmark=={version("concept-benchmark")}
from concept_benchmark.{module} import DatasetGenerator
dataset = DatasetGenerator(seed={spec["seed"]}{options}).generate_splits()
```

📓 **[Quickstart notebook](./quickstart.ipynb)** — load from the Hub and train a model, end-to-end in one notebook.

🔗 [PyPI](https://pypi.org/project/concept-benchmark/) · [GitHub](https://github.com/ustunb/concept-benchmark)

## Quick start (datasets library)

```python
from datasets import load_dataset
ds = load_dataset("{owner}/{name}")
row = ds["train"][0]
row["{spec["image_field"]}"]  # PIL.Image
row["concepts"]    # {len(concept_names)} binary values, ordered as concept_names below
row["label"]       # 0 or 1
row["label_name"]  # "{spec["class_names"][0]}" or "{spec["class_names"][1]}"
```

## Schema

| Field        | Type | Notes |
|---           |---   |---    |
| `{spec["image_field"]}` | `datasets.Image()` | PNG, embedded |
| `concepts`   | `Sequence(int8, length={len(concept_names)})` | binary; index order = `concept_names` below |
| `label`      | `ClassLabel("{spec["class_names"][0]}", "{spec["class_names"][1]}")` | |
| `label_name` | `string` | display alias for `label` |

**`concept_names`** (column order in `concepts`):

```python
{concept_names}
```
{rule}
## Splits

| Split        | n      |
|---           |---     |
| `train`      | {sizes["train"]:,} |
| `validation` | {sizes["validation"]:,} |
| `test`       | {sizes["test"]:,} |

Seed {spec["seed"]}. {positive_share:.0%} of the training examples are `{spec["class_names"][1]}`.

## License

MIT.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--out", type=Path, default=REPO / "huggingface_export")
    ap.add_argument("--owner", default="juliannski", help="Hub user or organization the dataset belongs to.")
    ap.add_argument("--push", action="store_true", help="Upload the exported folder to the Hub.")
    args = ap.parse_args()

    dataset_dict, concept_names = build_dataset_dict(args.dataset)
    folder = args.out / args.dataset
    (folder / "data").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        dataset_dict[split].to_parquet(folder / "data" / f"{split}-00000-of-00001.parquet")
    (folder / "README.md").write_text(write_card(args.dataset, dataset_dict, concept_names, args.owner))
    print(f"wrote {folder} ({', '.join(f'{split} {len(dataset_dict[split]):,}' for split in SPLITS)})")

    if args.push:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=folder, repo_id=f"{args.owner}/{args.dataset}", repo_type="dataset",
            commit_message=f"Regenerate with concept-benchmark {version('concept-benchmark')}",
        )
        print(f"uploaded to https://huggingface.co/datasets/{args.owner}/{args.dataset}")


if __name__ == "__main__":
    main()
