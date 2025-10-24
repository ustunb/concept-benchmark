import copy
import numpy as np
import torch
import torch.nn as nn

from concept_benchmark.train import (
    train_concept_heads,
)


def _any_state_diff(state_a, state_b):
    for k in state_a:
        ta, tb = state_a[k], state_b[k]
        if ta.dtype != tb.dtype or ta.shape != tb.shape:
            return True
        if not torch.allclose(ta, tb):
            return True
    return False


# TODO: Uncomment and fix after fixing models.py
# def test_train_heads_no_encoder_builds_heads_architecture(tabular_train_valid):
#     train, valid, d, k = tabular_train_valid
#     heads = train_concept_heads(
#         train_dataset=train,
#         valid_dataset=valid,
#         embedding_model=None,
#         input_dim=None,
#         l1_size=16,
#         freeze=True,
#         fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
#     )
#     assert isinstance(heads, nn.ModuleList)
#     assert len(heads) == k
#     first = heads[0]
#     assert isinstance(first[0], nn.Linear)
#     assert first[0].in_features == d
#     assert isinstance(first[-1], nn.Linear)
#     assert first[-1].out_features == 1


# def test_train_heads_with_encoder_freeze_true_keeps_encoder_constant(tabular_train_valid):
#     train, valid, d, k = tabular_train_valid
#     enc = nn.Linear(d, 6)
#     before = copy.deepcopy(enc.state_dict())
#     heads = train_concept_heads(
#         train_dataset=train,
#         valid_dataset=valid,
#         embedding_model=enc,
#         input_dim=None,
#         l1_size=8,
#         freeze=True,
#         fit_params={"epochs": 1, "device": "cpu", "batch_size": 16},
#     )
#     after = enc.state_dict()
#     assert not _any_state_diff(before, after), "Encoder should not change when frozen"
#     assert heads[0][0].in_features == 6


def test_train_heads_with_encoder_finetunes_encoder_changes(tabular_train_valid):
    train, valid, d, k = tabular_train_valid
    enc = nn.Linear(d, 6)
    before = copy.deepcopy(enc.state_dict())
    _ = train_concept_heads(
        train_dataset=train,
        valid_dataset=valid,
        embedding_model=enc,
        input_dim=None,
        l1_size=8,
        freeze=False,
        fit_params={"epochs": 2, "device": "cpu", "batch_size": 8, "lr_encoder": 1e-2, "lr_heads": 1e-2},
    )
    after = enc.state_dict()
    assert _any_state_diff(before, after), "Encoder should update when not frozen"


## Calibration is handled in ConceptDetector; no wrapper-based calibrators here.
