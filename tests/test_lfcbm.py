"""Tests for concept_benchmark.lfcbm module (file parsing, pickle)."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from concept_benchmark.lfcbm import LFConceptSet, LFTrainingConfig, LabelFreeCBM


# ── LFConceptSet.from_file ───────────────────────────────────────────

class TestLFConceptSetParsing:
    def test_jsonl_parsing(self, tmp_path):
        p = tmp_path / "concepts.jsonl"
        lines = [
            json.dumps({"key": "c0", "text": "red body"}),
            json.dumps({"key": "c1", "text": "round head"}),
            json.dumps({"key": "c2", "text": "has wings"}),
        ]
        p.write_text("\n".join(lines))
        cs = LFConceptSet.from_file(p)
        assert cs.keys == ["c0", "c1", "c2"]
        assert cs.texts == ["red body", "round head", "has wings"]

    def test_json_list_parsing(self, tmp_path):
        p = tmp_path / "concepts.json"
        data = [
            {"key": "a", "text": "big ears"},
            {"key": "b", "text": "small nose"},
        ]
        p.write_text(json.dumps(data))
        cs = LFConceptSet.from_file(p)
        assert cs.keys == ["a", "b"]
        assert cs.texts == ["big ears", "small nose"]

    def test_json_concepts_wrapper(self, tmp_path):
        p = tmp_path / "concepts.json"
        data = {"concepts": [{"key": "x", "text": "tail"}]}
        p.write_text(json.dumps(data))
        cs = LFConceptSet.from_file(p)
        assert cs.keys == ["x"]
        assert cs.texts == ["tail"]

    def test_csv_two_columns(self, tmp_path):
        p = tmp_path / "concepts.csv"
        p.write_text("key,text\nc0,red body\nc1,round head\n")
        cs = LFConceptSet.from_file(p)
        assert cs.keys == ["c0", "c1"]
        assert cs.texts == ["red body", "round head"]

    def test_csv_single_column(self, tmp_path):
        """Single column with non-standard header → auto-numbered keys."""
        p = tmp_path / "concepts.csv"
        p.write_text("feature\nred body\nround head\n")
        cs = LFConceptSet.from_file(p)
        assert cs.keys == ["0", "1"]
        assert cs.texts == ["red body", "round head"]

    def test_csv_text_header_uses_text_column(self, tmp_path):
        """CSV with 'text' header is treated as two-column path."""
        p = tmp_path / "concepts.csv"
        p.write_text("text\nred body\nround head\n")
        cs = LFConceptSet.from_file(p)
        assert cs.texts == ["red body", "round head"]

    def test_txt_one_per_line(self, tmp_path):
        p = tmp_path / "concepts.txt"
        p.write_text("red body\nround head\nhas wings\n")
        cs = LFConceptSet.from_file(p)
        assert len(cs.keys) == 3
        assert cs.texts == ["red body", "round head", "has wings"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LFConceptSet.from_file(tmp_path / "no_such_file.jsonl")


class TestLFConceptSetAlignment:
    def test_dataset_keys_reorder(self, tmp_path):
        p = tmp_path / "concepts.jsonl"
        lines = [
            json.dumps({"key": "b", "text": "beta"}),
            json.dumps({"key": "a", "text": "alpha"}),
            json.dumps({"key": "c", "text": "gamma"}),
        ]
        p.write_text("\n".join(lines))
        cs = LFConceptSet.from_file(p, dataset_keys=["a", "b", "c"])
        assert cs.keys == ["a", "b", "c"]
        assert cs.texts == ["alpha", "beta", "gamma"]

    def test_dataset_keys_mismatch_raises(self, tmp_path):
        p = tmp_path / "concepts.jsonl"
        lines = [json.dumps({"key": "x", "text": "unknown"})]
        p.write_text("\n".join(lines))
        with pytest.raises(ValueError, match="Cannot align"):
            LFConceptSet.from_file(p, dataset_keys=["z"])


class TestLFConceptSetInit:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            LFConceptSet(keys=["a", "b"], texts=["one"])


# ── LabelFreeCBM pickle ─────────────────────────────────────────────

class TestLabelFreeCBMPickle:
    def test_pickle_round_trip(self):
        cfg = LFTrainingConfig(device="cpu")
        cbm = LabelFreeCBM(cfg)
        # Manually set artefacts to test pickle
        cbm.Wc = np.random.default_rng(0).random((5, 10)).astype(np.float32)
        cbm.Wc = __import__("torch").from_numpy(cbm.Wc)
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        cbm.scaler = StandardScaler()
        cbm.scaler.fit(np.random.default_rng(1).random((20, 5)))
        cbm.classifier = LogisticRegression()
        cbm.classifier.fit(np.random.default_rng(2).random((20, 5)), np.random.default_rng(2).integers(0, 2, 20))

        data = pickle.dumps(cbm)
        restored = pickle.loads(data)
        assert restored.Wc is not None
        assert restored.scaler is not None
        assert restored.classifier is not None

    def test_getstate_drops_cache(self):
        cfg = LFTrainingConfig(device="cpu")
        cbm = LabelFreeCBM(cfg)
        cbm._img_train = np.zeros((10, 5))
        cbm._img_valid = np.zeros((5, 5))
        cbm._img_test = np.zeros((3, 5))
        cbm._txt_concepts = np.zeros((4, 5))
        state = cbm.__getstate__()
        for k in ("_img_train", "_img_valid", "_img_test", "_txt_concepts"):
            assert k not in state
