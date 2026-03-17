"""Tests for to_dataframe(include_X=) and explore() guard."""

import numpy as np
import pandas as pd
import pytest

from concept_benchmark.data import ConceptDatasetSample


# ---------- Backward compatibility ----------


def test_to_dataframe_default_matches_old(tab_small):
    """include_X=False (default) returns same columns as before."""
    sample = tab_small.training
    df_default = sample.to_dataframe()
    df_explicit = sample.to_dataframe(include_X=False)
    pd.testing.assert_frame_equal(df_default, df_explicit)
    expected_cols = list(sample.concepts) + ["label", "class"]
    assert list(df_default.columns) == expected_cols


# ---------- Tabular include_X ----------


def test_to_dataframe_include_X_tabular_columns(tab_small):
    sample = tab_small.training
    df = sample.to_dataframe(include_X=True)
    d = sample.X.shape[1]
    expected_x_cols = [f"x_{j}" for j in range(d)]
    expected_cols = expected_x_cols + list(sample.concepts) + ["label", "class"]
    assert list(df.columns) == expected_cols


def test_to_dataframe_include_X_tabular_values(tab_small):
    sample = tab_small.training
    df = sample.to_dataframe(include_X=True)
    d = sample.X.shape[1]
    x_cols = [f"x_{j}" for j in range(d)]
    np.testing.assert_array_almost_equal(df[x_cols].values, sample.X)


def test_to_dataframe_include_X_tabular_row_count(tab_small):
    sample = tab_small.training
    df = sample.to_dataframe(include_X=True)
    assert len(df) == sample.n


# ---------- Text include_X ----------


def test_to_dataframe_include_X_text():
    """Text data_type should produce a 'text' column."""
    texts = np.array(["hello world", "foo bar", "baz qux"], dtype=object)
    C = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    y = np.array([0, 1, 0], dtype=np.int32)
    meta = {"classes": ["a", "b"], "concepts": ["c0", "c1"], "data_type": "text"}
    sample = ConceptDatasetSample(X=texts, C=C, y=y, meta=meta)
    df = sample.to_dataframe(include_X=True)
    assert "text" in df.columns
    assert list(df["text"]) == ["hello world", "foo bar", "baz qux"]
    assert list(df.columns) == ["text", "c0", "c1", "label", "class"]


# ---------- Image include_X ----------


def test_to_dataframe_include_X_image(img_small):
    sample = img_small.training
    df = sample.to_dataframe(include_X=True)
    assert "image" in df.columns
    # Paths should be resolved via base_dir
    for path_str in df["image"]:
        assert isinstance(path_str, str)
    expected_cols = ["image"] + list(sample.concepts) + ["label", "class"]
    assert list(df.columns) == expected_cols
    assert len(df) == sample.n


def test_to_dataframe_image_without_X(img_small):
    """Image to_dataframe(include_X=False) should NOT have image column."""
    sample = img_small.training
    df = sample.to_dataframe(include_X=False)
    assert "image" not in df.columns
    expected_cols = list(sample.concepts) + ["label", "class"]
    assert list(df.columns) == expected_cols


# ---------- explore() guard ----------


def test_explore_raises_import_error(tab_small, monkeypatch):
    """explore() should raise ImportError with helpful message when spotlight is missing."""
    import concept_benchmark.data as data_module  # noqa: F401 — ensures module is loaded before monkeypatching __import__

    # Patch the import to fail
    original_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def mock_import(name, *args, **kwargs):
        if name == "renumics" or name.startswith("renumics."):
            raise ImportError("No module named 'renumics'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    sample = tab_small.training
    with pytest.raises(ImportError, match="concept-benchmark\\[explore\\]"):
        sample.explore()
