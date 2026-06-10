"""Dataset construction and splitting for robot text benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.synthetic.robot.text.catalog import CORE_CONCEPT_NAMES
from concept_benchmark.synthetic.robot.text.corpus import (
    concept_vector_from_row,
    load_jsonl,
    render_from_corpus,
)


def build_text_dataset(
    catalog_df: pd.DataFrame,
    corpus_path: Path,
    variants_per_row: int,
    seed: int,
    concept_names: list[str] | None = None,
    row_variants: list[int] | None = None,
) -> ConceptDatasetSample:
    """Build a text dataset from a concept catalog and JSONL corpus.

    Args:
        catalog_df: DataFrame with concept columns and a 'label' column.
        corpus_path: Path to the main JSONL corpus.
        variants_per_row: Default number of text variants per catalog row.
        seed: Random seed for reproducibility.
        concept_names: List of concept column names. Defaults to CORE_CONCEPT_NAMES.
        row_variants: Optional per-row variant counts (overrides variants_per_row).

    Returns:
        ConceptDatasetSample with text X, binary concepts C, and integer labels y.
    """
    corpus_spec = load_jsonl(corpus_path)
    concepts = catalog_df.columns.drop("label", errors="ignore").tolist()
    names = (
        list(concept_names) if concept_names is not None else list(CORE_CONCEPT_NAMES)
    )
    X: list[str] = []
    C: list[np.ndarray] = []
    y: list[int] = []
    row_index: list[int] = []

    for i, sr in catalog_df.iterrows():
        row = {k: sr[k] for k in concepts}
        repeats = (
            int(row_variants[i]) if row_variants is not None else int(variants_per_row)
        )
        for v in range(repeats):
            text = render_from_corpus(row, corpus_spec, seed + v)
            X.append(text)
            C.append(concept_vector_from_row(row, names))
            y.append(1 if str(sr["label"]) == "glorp" else 0)
            row_index.append(int(i))

    C_arr = np.stack(C, axis=0).astype(np.float32)
    y_arr = np.asarray(y, dtype=int)
    ds = ConceptDatasetSample(
        inputs=[str(t) for t in X],
        C=C_arr,
        y=y_arr,
        meta={"concepts": tuple(names), "classes": (0, 1)},
        input_type="text",
        classes=(0, 1),
    )
    ds._row_index = np.asarray(row_index, dtype=int)
    return ds


def kfold_by_robot_identity(
    ds: ConceptDatasetSample,
    cv_k: int = 5,
    cv_fold: int = 0,
    dev_per_fold: int = 1000,
    deployment_size: int = 10000,
    seed: int = 0,
    corpus_path: Path | None = None,
    catalog_df: pd.DataFrame | None = None,
    concept_names: list[str] | None = None,
) -> ConceptDatasetSample:
    """Split dataset into train/validation/test by robot identity.

    Uses k-fold cross-validation on robot identities (row indices) to ensure
    no robot appears in both train and test. The test set is independently
    sampled (optionally from a shared corpus) for consistent evaluation.

    Args:
        ds: Full dataset from build_text_dataset.
        cv_k: Number of CV folds.
        cv_fold: Which fold to use as validation (0-indexed maps to fold 1+).
        dev_per_fold: Max samples per development fold.
        deployment_size: Size of the deployment/test set.
        seed: Random seed.
        corpus_path: Path to main corpus (for generating shared test set).
        catalog_df: Catalog DataFrame (for generating shared test set).
        concept_names: Concept column names (passed through to test set builder).
    """
    seed_cv = seed + 1
    row_index = getattr(ds, "_row_index", np.arange(len(ds.y)))
    K = cv_k
    val_fold = (cv_fold or 0) or ((seed_cv % K) + 1)
    if val_fold < 1:
        val_fold = 1

    # Assign folds by robot identity
    rng = np.random.default_rng(seed_cv)
    ids = np.unique(row_index)
    y_arr = np.asarray(ds.y, dtype=int)

    # Map robot id → label
    y_by_id = {}
    for i2, rid in enumerate(row_index):
        r = int(rid)
        if r not in y_by_id:
            y_by_id[r] = int(y_arr[i2])

    ids0 = [r for r in ids if y_by_id.get(r, 0) == 0]
    ids1 = [r for r in ids if y_by_id.get(r, 0) == 1]
    rng.shuffle(ids0)
    rng.shuffle(ids1)

    # Reserve ~15% of each class for test
    n_t0 = int(np.floor(0.15 * len(ids0)))
    n_t1 = int(np.floor(0.15 * len(ids1)))
    test_ids = set(list(ids0[:n_t0]) + list(ids1[:n_t1]))

    # Assign remaining to folds
    rem0 = ids0[n_t0:]
    rem1 = ids1[n_t1:]
    buckets0 = [rem0[i::K] for i in range(K)]
    buckets1 = [rem1[i::K] for i in range(K)]

    fold_by_id = {}
    for k in range(K):
        fold_ids = list(buckets0[k]) + list(buckets1[k])
        if dev_per_fold and len(fold_ids) > dev_per_fold:
            rng.shuffle(fold_ids)
            fold_ids = fold_ids[:dev_per_fold]
        for r in fold_ids:
            fold_by_id[r] = k + 1

    fold_arr = np.zeros(len(row_index), dtype=int)
    for i, rid in enumerate(row_index):
        r = int(rid)
        if r in test_ids:
            fold_arr[i] = 0
        else:
            fold_arr[i] = fold_by_id.get(r, ((i % K) + 1))

    def _subset(mask):
        idx = np.where(mask)[0]
        X_sub = [ds.inputs[i] for i in idx]
        C_sub = ds.C[idx]
        y_sub = ds.y[idx]
        return ConceptDatasetSample(
            inputs=X_sub,
            C=C_sub,
            y=y_sub,
            meta={"concepts": ds.concepts, "classes": ds.classes},
            input_type="text",
            classes=ds.classes,
        )

    val_mask = fold_arr == val_fold
    test_mask = fold_arr == 0
    train_mask = ~(val_mask | test_mask)

    ds.train = _subset(train_mask)
    ds.validation = _subset(val_mask)

    # Generate deployment/test set
    if deployment_size > 0 and corpus_path is not None and catalog_df is not None:
        # Shared test: generate fresh from full catalog
        standard_size = len(catalog_df)
        vpr_t = max(1, int((deployment_size + standard_size - 1) // standard_size))
        test_ds = build_text_dataset(
            catalog_df=catalog_df,
            corpus_path=corpus_path,
            variants_per_row=vpr_t,
            seed=seed,
            concept_names=concept_names,
        )
        rng_dep = np.random.default_rng(seed + 1234)
        pool = np.arange(len(test_ds.y), dtype=int)
        replace = deployment_size > pool.size
        idx_dep = rng_dep.choice(pool, size=deployment_size, replace=replace)
        X_dep = [test_ds.inputs[i] for i in idx_dep]
        C_dep = test_ds.C[idx_dep]
        y_dep = test_ds.y[idx_dep]
        dep = ConceptDatasetSample(
            inputs=X_dep,
            C=C_dep,
            y=y_dep,
            meta={
                "concepts": test_ds.concepts,
                "classes": test_ds.classes,
            },
            input_type="text",
            classes=test_ds.classes,
        )
        ds.test = dep
    else:
        ds.test = _subset(test_mask)

    return ds
