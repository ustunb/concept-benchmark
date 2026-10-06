import copy
import torch.nn as nn

from experiments.train import (
    train_concept_heads,
)
from tests.conftest import _any_state_diff


def test_train_heads_with_encoder_finetunes_encoder_changes(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    enc = nn.Linear(d, 6)
    before = copy.deepcopy(enc.state_dict())
    _ = train_concept_heads(
        train_dataset=train,
        valid_dataset=valid,
        embedding_model=enc,
        input_dim=None,
        hidden_layer_size=8,
        freeze_backbone=False,
        fit_params={
            "epochs": 2,
            "device": "cpu",
            "batch_size": 8,
            "lr_encoder": 1e-2,
            "lr_heads": 1e-2,
        },
    )
    after = enc.state_dict()
    assert _any_state_diff(before, after), "Encoder should update when not frozen"


## Calibration is handled in ConceptDetector; no wrapper-based calibrators here.


def test_nan_validation_metric_is_never_the_best_epoch(
    tabular_train_valid, monkeypatch
):
    import math

    import numpy as np
    from experiments import train as train_module
    from experiments.train import DefaultConceptTrainer

    train, valid, d, k = tabular_train_valid
    real_f1 = train_module.f1_score
    calls = {"n": 0}

    def _f1_nan_first_epoch(*args, **kwargs):
        calls["n"] += 1
        return float("nan") if calls["n"] <= k else real_f1(*args, **kwargs)

    monkeypatch.setattr(train_module, "f1_score", _f1_nan_first_epoch)
    result = DefaultConceptTrainer()(
        nn.Linear(d, k),
        train,
        valid,
        num_concepts=k,
        params={"epochs": 3, "device": "cpu", "batch_size": 8, "patience": 5},
    )
    history = result.history["val_f1"]
    assert math.isnan(history[0])
    finite = [v for v in history if not math.isnan(v)]
    assert finite and result.best_metric == max(finite)
    assert not np.isnan(result.best_metric)
