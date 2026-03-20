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
