"""Regression tests that lock down the robot demo pipeline outputs.

These tests load saved artifacts (datasets, models, results CSVs) and
verify that key metrics match the expected values from the paper.
They serve as a safety net during refactoring.
"""
import numpy as np
import pandas as pd
import pytest

from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import results_dir
from scripts.robot_demo.utils import (
    DEFAULT_ROBOT_SETTINGS,
    INPUT_MAP,
    RobotClassifierCNN,
    compute_accuracy,
    determine_device,
    get_dataset_file,
    get_model_file,
    get_results_file,
)


# Skip entire module if artifacts are not present
_settings = DEFAULT_ROBOT_SETTINGS.copy()
_dataset_path = get_dataset_file(**_settings)
_has_artifacts = _dataset_path.exists()

pytestmark = pytest.mark.skipif(
    not _has_artifacts,
    reason="Robot demo artifacts not found; run the pipeline first.",
)


@pytest.fixture(scope="module")
def settings():
    return DEFAULT_ROBOT_SETTINGS.copy()


@pytest.fixture(scope="module")
def dataset(settings):
    return load(get_dataset_file(**settings))


@pytest.fixture(scope="module")
def device():
    return determine_device()


# ── Dataset shape checks ──────────────────────────────────────────────

class TestDatasetShape:
    def test_ideal_dataset_has_training_split(self, dataset):
        assert hasattr(dataset, "training")
        assert dataset.training.n > 0

    def test_ideal_dataset_has_test_split(self, dataset):
        assert hasattr(dataset, "test")
        assert dataset.test.n == 10000

    def test_ideal_dataset_n_concepts(self, dataset):
        # Ideal drops 10 foot_shape subtypes -> 7 concepts remain
        assert dataset.test.n_concepts == 7


# ── CBM accuracy ──────────────────────────────────────────────────────

class TestCBMAccuracy:
    def test_ideal_cbm_accuracy(self, settings):
        cbm = load(get_model_file(model_class="cbm", **settings))
        data = load(get_dataset_file(**settings))
        acc = (cbm.predict(data.test) == data.test.y).mean().item()
        assert abs(acc - 0.8673) < 0.001, f"Expected ~0.8673, got {acc}"


# ── DNN accuracy ──────────────────────────────────────────────────────

class TestDNNAccuracy:
    def test_ideal_dnn_accuracy(self, settings, dataset, device):
        dnn_weights = load(get_model_file(model_class="dnn", **settings))
        dnn = RobotClassifierCNN(input_size=INPUT_MAP[settings["size"]]).to(device)
        dnn.load_state_dict(dnn_weights)
        loader_config = {"batch_size": 32, "num_workers": 0, "pin_memory": False}
        test_loader = dataset.test.loader(shuffle=False, **loader_config)
        acc = compute_accuracy(dnn, test_loader, device)
        assert abs(acc - 0.8746) < 0.001, f"Expected ~0.8746, got {acc}"


# ── Results CSV spot-checks ───────────────────────────────────────────

class TestResultsCSV:
    @pytest.fixture(scope="class")
    def results_df(self):
        csv_path = results_dir / "robot_demo_results.csv"
        if not csv_path.exists():
            pytest.skip("robot_demo_results.csv not found")
        return pd.read_csv(csv_path)

    def test_dnn_accuracy_in_csv(self, results_df):
        row = results_df[
            (results_df["model"] == "dnn")
            & (results_df["data_name"] == "ideal")
            & (results_df["metric"] == "accuracy")
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["value"] - 0.8746) < 0.001

    def test_cbm_no_int_accuracy_ideal(self, results_df):
        row = results_df[
            (results_df["model"] == "cbm_no_int")
            & (results_df["data_name"] == "ideal")
            & (results_df["concept_missing"] == 0.0)
            & (results_df["metric"] == "accuracy")
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["value"] - 0.8673) < 0.001

    def test_cbm_no_int_accuracy_subconcept(self, results_df):
        row = results_df[
            (results_df["model"] == "cbm_no_int")
            & (results_df["data_name"] == "subconcept")
            & (results_df["concept_missing"] == 0.0)
            & (results_df["metric"] == "accuracy")
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["value"] - 0.7812) < 0.001

    def test_cbm_with_int_1_ideal_t02(self, results_df):
        row = results_df[
            (results_df["model"] == "cbm_with_int_1")
            & (results_df["data_name"] == "ideal")
            & (results_df["concept_missing"] == 0.0)
            & (results_df["threshold"] == 0.2)
            & (results_df["metric"] == "accuracy")
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["value"] - 0.9736) < 0.001

    def test_cbm_with_int_1_ideal_t02_changed(self, results_df):
        row = results_df[
            (results_df["model"] == "cbm_with_int_1")
            & (results_df["data_name"] == "ideal")
            & (results_df["concept_missing"] == 0.0)
            & (results_df["threshold"] == 0.2)
            & (results_df["metric"] == "predictions_changed")
        ]
        assert len(row) == 1
        assert row.iloc[0]["value"] == 1417.0


# ── Intervention results CSV (per-variant) ────────────────────────────

class TestInterventionCSV:
    @pytest.fixture(scope="class")
    def ideal_interv_df(self, settings):
        csv_path = get_results_file(model_class="cbm", **settings)
        if not csv_path.exists():
            pytest.skip("Ideal CBM results CSV not found")
        return pd.read_csv(csv_path)

    def test_ideal_interv_shape(self, ideal_interv_df):
        # 3 budgets x 2 thresholds = 6 rows
        assert len(ideal_interv_df) == 6

    def test_ideal_interv_k1_t02_accuracy(self, ideal_interv_df):
        row = ideal_interv_df[
            (ideal_interv_df["budget"] == 1)
            & (ideal_interv_df["threshold"] == 0.2)
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["accuracy"] - 0.9736) < 0.001

    def test_ideal_interv_k1_t02_changed(self, ideal_interv_df):
        row = ideal_interv_df[
            (ideal_interv_df["budget"] == 1)
            & (ideal_interv_df["threshold"] == 0.2)
        ]
        assert row.iloc[0]["predictions_changed"] == 1417

    def test_ideal_interv_k3_t02_accuracy(self, ideal_interv_df):
        row = ideal_interv_df[
            (ideal_interv_df["budget"] == 3)
            & (ideal_interv_df["threshold"] == 0.2)
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]["accuracy"] - 0.9769) < 0.001
