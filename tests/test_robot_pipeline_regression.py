"""Regression tests that lock down the robot demo pipeline outputs.

These tests load saved artifacts (datasets, models, results CSVs) and
verify that key metrics match the expected values from the paper.
They serve as a safety net during refactoring.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

from concept_benchmark.utils import compute_accuracy, determine_device
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load
import experiments.models as _exp_models
import experiments.train as _exp_train
from experiments.models import RobotClassifierCNN

# Backward compatibility: allow unpickling models saved under old module paths.
for _old, _new in [
    ("concept_benchmark.models", _exp_models),
    ("concept_benchmark.train", _exp_train),
]:
    if _old not in sys.modules:
        sys.modules[_old] = _new


# Model-loading tests need cached artifacts; reference CSV tests don't.
_ideal_cfg = RobotBenchmarkConfig.default_ideal()
_has_artifacts = _ideal_cfg.get_dataset_path().exists()
_needs_artifacts = pytest.mark.skipif(
    not _has_artifacts,
    reason="Robot demo artifacts not found; run the pipeline first.",
)


@pytest.fixture(scope="module")
def ideal_config():
    return RobotBenchmarkConfig.default_ideal()


@pytest.fixture(scope="module")
def dataset(ideal_config):
    return load(ideal_config.get_dataset_path())


@pytest.fixture(scope="module")
def device():
    return determine_device()


# ── Dataset shape checks ──────────────────────────────────────────────


@_needs_artifacts
class TestDatasetShape:
    def test_ideal_dataset_has_training_split(self, dataset):
        assert hasattr(dataset, "train")
        assert dataset.train.n > 0

    def test_ideal_dataset_has_test_split(self, dataset):
        assert hasattr(dataset, "test")
        assert dataset.test.n == 10000

    def test_ideal_dataset_n_concepts(self, dataset):
        # Ideal drops 10 foot_shape subtypes -> 7 concepts remain
        assert dataset.test.n_concepts == 7


# ── CBM accuracy ──────────────────────────────────────────────────────


@_needs_artifacts
class TestCBMAccuracy:
    def test_ideal_cbm_accuracy(self, ideal_config):
        cbm = load(ideal_config.get_model_path("cbm"))
        data = load(ideal_config.get_dataset_path())
        acc = (cbm.predict(data.test) == data.test.y).mean().item()
        assert abs(acc - 0.8673) < 0.001, f"Expected ~0.8673, got {acc}"


# ── DNN accuracy ──────────────────────────────────────────────────────


@_needs_artifacts
class TestDNNAccuracy:
    def test_ideal_dnn_accuracy(self, ideal_config, dataset, device):
        dnn_weights = load(ideal_config.get_model_path("dnn"))
        dnn = RobotClassifierCNN(input_size=ideal_config.input_size).to(device)
        dnn.load_state_dict(dnn_weights)
        loader_config = {"batch_size": 32, "num_workers": 0, "pin_memory": False}
        test_loader = dataset.test.loader(shuffle=False, **loader_config)
        acc = compute_accuracy(dnn, test_loader, device)
        assert abs(acc - 0.8746) < 0.001, f"Expected ~0.8746, got {acc}"


# ── Results CSV spot-checks ───────────────────────────────────────────


class TestResultsCSV:
    @pytest.fixture(scope="class")
    def results_df(self):
        csv_path = Path(__file__).parent / "data" / "ideal_results_reference.csv"
        if not csv_path.exists():
            pytest.skip("Reference ideal results CSV not found at tests/data/")
        return pd.read_csv(csv_path)

    def test_dnn_accuracy_in_csv(self, results_df):
        row = results_df[
            (results_df["model"] == "dnn") & (results_df["dataset"] == "ideal")
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["accuracy"] - 0.8746) < 0.001

    def test_cbm_no_int_accuracy_ideal(self, results_df):
        row = results_df[
            (results_df["model"] == "cbm")
            & (results_df["dataset"] == "ideal")
            & (results_df["budget"] == 0)
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["accuracy"] - 0.8673) < 0.001

    def test_cbm_no_int_accuracy_subconcept(self):
        csv_path = Path(__file__).parent / "data" / "subconcept_results_reference.csv"
        if not csv_path.exists():
            pytest.skip("Reference subconcept results CSV not found at tests/data/")
        sub_df = pd.read_csv(csv_path)
        row = sub_df[
            (sub_df["model"] == "cbm")
            & (sub_df["dataset"] == "subconcept")
            & (sub_df["budget"] == 0)
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["accuracy"] - 0.7812) < 0.002

    def test_cbm_with_int_1_ideal_t02(self, results_df):
        row = results_df[
            (results_df["model"] == "cbm")
            & (results_df["dataset"] == "ideal")
            & (results_df["budget"] == 1)
            & (results_df["threshold"] == 0.2)
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["accuracy"] - 0.9736) < 0.001

    def test_cbm_with_int_1_ideal_t02_changed(self, results_df):
        row = results_df[
            (results_df["model"] == "cbm")
            & (results_df["dataset"] == "ideal")
            & (results_df["budget"] == 1)
            & (results_df["threshold"] == 0.2)
        ]
        assert len(row) == 1
        assert row.iloc[0]["predictions_changed"] == 1417

    def test_aligned_cbm_ideal(self, results_df):
        row = results_df[
            (results_df["model"] == "aligned_cbm")
            & (results_df["dataset"] == "ideal")
            & (results_df["budget"] == 0)
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["accuracy"] - 0.8657) < 0.001

    def test_gain_computed(self, results_df):
        """Gain column = accuracy - dnn_accuracy for all rows."""
        dnn_row = results_df[
            (results_df["model"] == "dnn") & (results_df["dataset"] == "ideal")
        ]
        dnn_acc = dnn_row.iloc[0]["accuracy"]
        cbm_row = results_df[
            (results_df["model"] == "cbm")
            & (results_df["dataset"] == "ideal")
            & (results_df["budget"] == 1)
            & (results_df["threshold"] == 0.2)
        ]
        expected_gain = cbm_row.iloc[0]["accuracy"] - dnn_acc
        assert abs(cbm_row.iloc[0]["gain"] - expected_gain) < 0.001


# ── Intervention results CSV (per-variant) ────────────────────────────


class TestInterventionCSV:
    @pytest.fixture(scope="class")
    def ideal_interv_df(self):
        csv_path = Path(__file__).parent / "data" / "ideal_intervention_reference.csv"
        if not csv_path.exists():
            pytest.skip("Reference ideal intervention CSV not found at tests/data/")
        df = pd.read_csv(csv_path)
        # Filter to baseline regime if column present
        if "regime" in df.columns:
            df = df[df["regime"] == "baseline"]
        return df

    def test_ideal_interv_shape(self, ideal_interv_df):
        # At least budget 0 (no-intervention) + budget 1
        assert len(ideal_interv_df) >= 2

    def test_ideal_interv_kmax_t02_accuracy(self, ideal_interv_df):
        max_budget = ideal_interv_df["budget"].max()
        row = ideal_interv_df[
            (ideal_interv_df["budget"] == max_budget)
            & (ideal_interv_df["threshold"] == 0.2)
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["accuracy"] - 0.9769) < 0.001


# ── Alignment regression tests ─────────────────────────────────────


class TestAlignment:
    @pytest.fixture(scope="class")
    def ideal_alignment(self):
        import json

        path = Path(__file__).parent / "data" / "ideal_alignment_reference.json"
        if not path.exists():
            pytest.skip("Reference ideal alignment JSON not found at tests/data/")
        with open(path) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def subconcept_alignment(self):
        import json

        path = Path(__file__).parent / "data" / "subconcept_alignment_reference.json"
        if not path.exists():
            pytest.skip("Reference subconcept alignment JSON not found at tests/data/")
        with open(path) as f:
            return json.load(f)

    def test_ideal_original_accuracy(self, ideal_alignment):
        assert abs(ideal_alignment["original_accuracy"] - 0.8673) < 0.001

    def test_ideal_aligned_accuracy(self, ideal_alignment):
        assert abs(ideal_alignment["aligned_accuracy"] - 0.8657) < 0.001

    def test_subconcept_original_accuracy(self, subconcept_alignment):
        assert abs(subconcept_alignment["original_accuracy"] - 0.7812) < 0.001

    def test_subconcept_aligned_accuracy(self, subconcept_alignment):
        assert abs(subconcept_alignment["aligned_accuracy"] - 0.7656) < 0.001


# ── Experiment 3: Regime intervention results ─────────────────────────


class TestRegimeCSV:
    """Spot-check intervention regime results against paper CSV values.

    These tests verify the cached results CSV — they do NOT re-run the
    pipeline.  If the CSV is missing, the tests are skipped.
    """

    @pytest.fixture(scope="class")
    def regime_df(self):
        csv_path = Path(__file__).parent / "data" / "regime_results_reference.csv"
        if not csv_path.exists():
            pytest.skip("Reference regime CSV not found at tests/data/")
        df = pd.read_csv(csv_path)
        return df[df["threshold"] == 0.2]

    def _get_acc(self, df, regime, budget):
        row = df[(df["regime"] == regime) & (df["budget"] == budget)]
        assert len(row) == 1, f"Expected 1 row for {regime} k={budget}, got {len(row)}"
        return float(row.iloc[0]["accuracy"])

    # Baseline (perfect interventions)
    def test_baseline_k0(self, regime_df):
        assert abs(self._get_acc(regime_df, "baseline", 0) - 0.7812) < 0.001

    def test_baseline_k1(self, regime_df):
        assert abs(self._get_acc(regime_df, "baseline", 1) - 0.9212) < 0.001

    def test_baseline_k2(self, regime_df):
        assert abs(self._get_acc(regime_df, "baseline", 2) - 0.9439) < 0.001

    # Expert (80% accurate interventions)
    def test_expert_k1(self, regime_df):
        assert abs(self._get_acc(regime_df, "expert", 1) - 0.8782) < 0.001

    def test_expert_k2(self, regime_df):
        assert abs(self._get_acc(regime_df, "expert", 2) - 0.8960) < 0.001

    # Subjective (noisy CBM + 80% accurate interventions)
    def test_subjective_k0(self, regime_df):
        assert abs(self._get_acc(regime_df, "subjective", 0) - 0.8762) < 0.001

    def test_subjective_k1(self, regime_df):
        assert abs(self._get_acc(regime_df, "subjective", 1) - 0.8834) < 0.001

    # Machine (LFCBM + 80% accurate interventions)
    def test_machine_k0(self, regime_df):
        assert abs(self._get_acc(regime_df, "machine", 0) - 0.8739) < 0.001

    def test_machine_k1(self, regime_df):
        assert abs(self._get_acc(regime_df, "machine", 1) - 0.5188) < 0.001
