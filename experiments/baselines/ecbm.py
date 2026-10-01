"""ECBM (Energy-based Concept Bottleneck Model), ported from the authors' code.

Reference: Xu et al., "Energy-Based Concept Bottleneck Models", ICLR 2024; code at github.com/xmed-lab/ECBM
(`networks/EBM.py`, `loss.py`, `LitModel.py`, `GradientInference.py`, `utils.py`). The energy network, losses,
concept augmentation, gradient inference and intervention procedure follow that code; line references below
point to it. Adapted to this benchmark: the image encoder is the shared backbone used by the CBM, CEM and
ProbCBM (instead of an ImageNet ResNet-101), hidden size is `ecbm_hid_size`, and training uses Adam with early
stopping on the validation loss (instead of SGD for 300 epochs).
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.utils import determine_device

from experiments.baselines._common import (
    _EARLY_STOP_EPS,
    _OfficialBenchmarkModelBase,
    _PredictionCache,
    _infer_backbone_spec,
    _make_backbone_factory,
    _prepare_batch_concepts,
    _prepare_batch_features,
    _prepare_batch_labels,
    _resolve_epochs,
    _resolve_learning_rate,
    _resolve_loader_config,
    _resolve_patience,
    _stack_numpy,
    _stack_tensors,
)

# Gradient inference settings from the authors' inference configs (configs/*_inference.json) and
# GradientInference.py: energy weights (xy, xc, cy) without and with interventions, learning rate for the
# label/concept variables, and the early-stopping rule (utils.EarlyStopping(patience=10, delta=1)).
INFERENCE_WEIGHTS = (1.0, 1.0, 0.01)
INTERVENTION_WEIGHTS = (0.0, 0.0, 3.0)  # GradientInference.py:134-136
INFERENCE_LR = 0.1
INFERENCE_PATIENCE = 10
INFERENCE_DELTA = 1.0
INFERENCE_MAX_STEPS = 1000  # safety cap; the authors' loop has none
INTERVENTION_LOGIT = 5.0  # (one_hot(c) - 0.5) * 10, GradientInference.py:130-131
# Concept -> label augmentation during training (main.py: cy_perturb_prob, cy_permute_prob).
CY_CONCEPT_FLIP_SHARE = 0.2
CY_SAMPLE_FLIP_SHARE = 0.2


# ---------------------------------------------------------------------------
# ECBM network (networks/EBM.py: EBM_GL)
# ---------------------------------------------------------------------------

class _ECBMNet(nn.Module):
    def __init__(
        self,
        *,
        n_concepts: int,
        n_tasks: int,
        hid_size: int,
        feature_dim: int,
        lambda_xy: float,
        lambda_xc: float,
        lambda_cy: float,
        c_extractor_arch: Any,
    ) -> None:
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.num_classes = int(n_tasks)
        self.hid_size = int(hid_size)
        self.feature_dim = int(feature_dim)
        self.lambda_xy = float(lambda_xy)
        self.lambda_xc = float(lambda_xc)
        self.lambda_cy = float(lambda_cy)

        self.backbone = c_extractor_arch(self.feature_dim)
        self.y_embedding = nn.Parameter(torch.randn((self.num_classes, self.hid_size)))
        self.c_embedding = nn.Parameter(torch.randn((self.n_concepts * 2, self.hid_size)))
        self.classifier_xc = nn.ModuleList([nn.Linear(self.hid_size, 1) for _ in range(self.n_concepts)])
        self.concept_proj = nn.Linear(self.n_concepts * self.hid_size, self.hid_size)
        self.xy_fc1 = nn.Linear(self.feature_dim, self.hid_size)
        self.xc_fc1 = nn.Linear(self.feature_dim, self.hid_size)
        self.classifier_xy = nn.Linear(self.hid_size, 1)
        self.classifier_cy = nn.Linear(self.hid_size, 1)
        self.dropout = nn.Dropout(p=0.2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.float())

    def _concept_code(self, present: torch.Tensor, c_embed_cy: torch.Tensor) -> torch.Tensor:
        """Concept -> label code: the first embedding block if a concept is present, else the second (EBM.py:173-180, 190-196)."""
        n = self.n_concepts
        chosen = torch.where(present.unsqueeze(-1), c_embed_cy[:, :n, :], c_embed_cy[:, n:, :])
        return self.concept_proj(chosen.reshape(chosen.shape[0], -1))

    def training_energies(
        self, features: torch.Tensor, concepts: torch.Tensor, *, augment: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Class-wise and state-wise energies for training (EBM.py:89-212, is_training=True).

        Returns xy [bs, n_classes], cy [bs, n_classes], xc [bs, n_concepts, 2]; lower energy = more likely.
        """
        bs, n = features.shape[0], self.n_concepts
        y_embed = F.normalize(self.y_embedding.unsqueeze(0).repeat(bs, 1, 1), p=2, dim=-1)
        x_embed = self.dropout(self.xy_fc1(features))[:, None, :].expand_as(y_embed)
        xy_energy = self.classifier_xy(F.relu(x_embed + x_embed * y_embed)).view(bs, -1)

        c_embed_cy = self.c_embedding.unsqueeze(0).repeat(bs, 1, 1)
        c_embed = F.normalize(c_embed_cy, p=2, dim=-1)
        x_embed = self.dropout(self.xc_fc1(features))[:, None, :].expand_as(c_embed)
        xc_energy = []
        for i in range(n):
            pos = F.relu(x_embed[:, i] + x_embed[:, i] * c_embed[:, i])
            neg = F.relu(x_embed[:, i + n] + x_embed[:, i + n] * c_embed[:, i + n])
            xc_energy.append(self.classifier_xc[i](torch.stack([pos, neg], dim=1)).view(bs, 1, 2))
        xc_energy = torch.cat(xc_energy, dim=1)

        present = concepts >= 0.5
        if augment:
            present = _flip_concepts(present)
        c_code = self._concept_code(present, c_embed_cy)[:, None, :].expand_as(y_embed)
        cy_energy = self.classifier_cy(F.relu(c_code + c_code * y_embed)).view(bs, -1)
        return xy_energy, cy_energy, xc_energy

    def inference_energies(
        self, features: torch.Tensor, y_logits: torch.Tensor, c_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Energies of the current label/concept variables (EBM.py:89-212, is_training=False).

        `y_logits` [bs, n_classes] and `c_logits` [bs, n_concepts, 2] are the optimized variables (the authors'
        `y_prob` and `c_prob` parameters before softmax). Returns xy [bs], cy [bs], xc [bs, n_concepts].
        """
        bs, n = features.shape[0], self.n_concepts
        y_prob = torch.softmax(y_logits, dim=-1)
        y_embed = (y_prob.unsqueeze(-1) * self.y_embedding.unsqueeze(0)).sum(dim=1)
        y_embed = F.normalize(y_embed, p=2, dim=-1)
        x_embed = self.dropout(self.xy_fc1(features))
        xy_energy = self.classifier_xy(F.relu(x_embed + x_embed * y_embed)).view(bs)

        c_prob = torch.softmax(c_logits, dim=-1)
        c_embed_cy = self.c_embedding.unsqueeze(0).repeat(bs, 1, 1)
        c_embed = c_embed_cy[:, :n] * c_prob[:, :, 0:1] + c_embed_cy[:, n:] * c_prob[:, :, 1:2]
        c_embed = F.normalize(c_embed, p=2, dim=-1)
        x_embed = self.dropout(self.xc_fc1(features))[:, None, :].expand_as(c_embed)
        xc_energy = torch.cat(
            [self.classifier_xc[i](F.relu(x_embed[:, i] + x_embed[:, i] * c_embed[:, i])) for i in range(n)],
            dim=1,
        )

        c_code = self._concept_code(c_prob[:, :, 1] > 0.5, c_embed_cy)
        cy_energy = self.classifier_cy(F.relu(c_code + c_code * y_embed)).view(bs)
        return xy_energy, cy_energy, xc_energy


def _flip_concepts(present: torch.Tensor) -> torch.Tensor:
    """Flip a random 20% of concepts in a random 20% of examples (EBM.py:63-86, `cy_augment`)."""
    bs, n = present.shape
    n_concepts = math.ceil(n * CY_CONCEPT_FLIP_SHARE)
    n_samples = math.ceil(bs * CY_SAMPLE_FLIP_SHARE)
    concept_idx = torch.randperm(n, device=present.device)[:n_concepts]
    sample_idx = torch.randperm(bs, device=present.device)[:n_samples]
    flipped = present.clone()
    rows = sample_idx[:, None]
    flipped[rows, concept_idx[None, :]] = ~flipped[rows, concept_idx[None, :]]
    return flipped


# ---------------------------------------------------------------------------
# ECBM losses (loss.py) and gradient inference (GradientInference.py)
# ---------------------------------------------------------------------------

def _energy_nll(energy: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """E(target) + logsumexp(-E): the authors' EBMLoss (= cross-entropy on -energy)."""
    return F.cross_entropy(-energy, target)


def _ecbm_losses(
    model: _ECBMNet,
    features: torch.Tensor,
    concepts: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Training loss (LitModel.training_step); the unweighted sum is the validation loss the authors monitor."""
    xy_energy, cy_energy, xc_energy = model.training_energies(features, concepts)
    loss_xy = _energy_nll(xy_energy, labels)
    loss_cy = _energy_nll(cy_energy, labels)
    target_c = (concepts >= 0.5).long()
    loss_xc = sum(_energy_nll(xc_energy[:, i], target_c[:, i]) for i in range(model.n_concepts))
    total = model.lambda_xc * loss_xc + model.lambda_xy * loss_xy + model.lambda_cy * loss_cy
    metrics = {
        "loss_xy": float(loss_xy.detach().cpu()),
        "loss_xc": float(loss_xc.detach().cpu()),
        "loss_cy": float(loss_cy.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
        "loss_unweighted": float((loss_xy + loss_xc + loss_cy).detach().cpu()),
    }
    return total, metrics


def _run_gradient_inference(
    model: _ECBMNet,
    features: torch.Tensor,
    y_logits: torch.Tensor,
    c_logits: torch.Tensor,
    weights: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize the label and concept variables until the stopping rule fires (GradientInference.run_optim).

    The rule is utils.EarlyStopping(patience=10, delta=1) on the mean x->y energy: stop after 10 consecutive
    steps in which it does not drop by more than 1 below its best value.
    """
    lambda_xy, lambda_xc, lambda_cy = weights
    y_logits = nn.Parameter(y_logits.detach().clone())
    c_logits = nn.Parameter(c_logits.detach().clone())
    optimizer = torch.optim.Adam(
        [{"params": [c_logits], "lr": INFERENCE_LR}, {"params": [y_logits], "lr": INFERENCE_LR}]
    )
    requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    best, counter = None, 0
    try:
        with torch.enable_grad():
            for _ in range(INFERENCE_MAX_STEPS):
                optimizer.zero_grad()
                xy_en, cy_en, xc_en = model.inference_energies(features, y_logits, c_logits)
                loss = lambda_xy * xy_en.mean() + lambda_xc * xc_en.mean(dim=0).sum() + lambda_cy * cy_en.mean()
                score = -float(xy_en.mean().detach())
                if best is None:
                    best = score
                elif score <= best + INFERENCE_DELTA:
                    counter += 1
                else:
                    best, counter = score, 0
                loss.backward()
                optimizer.step()
                if counter >= INFERENCE_PATIENCE:
                    break
    finally:
        for p, flag in zip(model.parameters(), requires_grad):
            p.requires_grad_(flag)
    return y_logits.detach(), c_logits.detach()


def _infer_labels_and_concepts(model: _ECBMNet, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prediction without interventions: start from uniform variables (GradientInference.inference:80-97)."""
    bs = features.shape[0]
    y0 = torch.zeros(bs, model.num_classes, device=features.device)
    c0 = torch.zeros(bs, model.n_concepts, 2, device=features.device)
    return _run_gradient_inference(model, features, y0, c0, INFERENCE_WEIGHTS)


def _intervene(
    model: _ECBMNet,
    features: torch.Tensor,
    c_logits: torch.Tensor,
    concepts: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Prediction after an intervention (GradientInference.inference:126-143).

    Intervened concepts are set to +/-5 logits, the label variable is reset to uniform, and inference reruns
    with only the concept -> label energy (weights 0/0/3). Returns label logits.
    """
    target = F.one_hot((concepts >= 0.5).long(), num_classes=2).float()
    forced = (target - 0.5) * (2 * INTERVENTION_LOGIT)
    c_start = torch.where(mask.unsqueeze(-1), forced, c_logits)
    y0 = torch.zeros(features.shape[0], model.num_classes, device=features.device)
    y_logits, _ = _run_gradient_inference(model, features, y0, c_start, INTERVENTION_WEIGHTS)
    return y_logits


def _run_ecbm_epoch(
    model: _ECBMNet,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    n_examples = 0
    for batch_x, batch_c, batch_y in loader:
        batch_x = _prepare_batch_features(batch_x, device=device)
        batch_c = _prepare_batch_concepts(batch_c, device=device)
        batch_y = _prepare_batch_labels(batch_y, device=device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            loss, metrics = _ecbm_losses(model, model.extract_features(batch_x), batch_c, batch_y)
            if is_train:
                loss.backward()
                optimizer.step()
        batch_size = int(batch_y.shape[0])
        n_examples += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    return {key: value / max(1, n_examples) for key, value in totals.items()}


# ---------------------------------------------------------------------------
# ECBM benchmark model
# ---------------------------------------------------------------------------

class ECBMBenchmarkModel(_OfficialBenchmarkModelBase):
    family = "ecbm"

    def _inference_batch_size(self) -> int:
        return int(self._loader_kwargs()["batch_size"])

    def _run_official_model(
        self,
        dataset: ConceptDatasetSample,
    ) -> tuple[np.ndarray, np.ndarray, _PredictionCache]:
        model = self._require_official_model()
        model.eval()
        device = self._inference_device()
        loader = dataset.loader(shuffle=False, **self._loader_kwargs())

        label_chunks: list[np.ndarray] = []
        concept_chunks: list[np.ndarray] = []
        feature_chunks: list[torch.Tensor] = []
        logit_chunks: list[torch.Tensor] = []
        model.to(device)
        for batch_x, _, _ in loader:
            batch_x = _prepare_batch_features(batch_x, device=device)
            with torch.no_grad():
                features = model.extract_features(batch_x)
            y_logits, c_logits = _infer_labels_and_concepts(model, features)
            label_chunks.append(torch.softmax(y_logits, dim=-1).cpu().numpy().astype(np.float32))
            concept_chunks.append(torch.softmax(c_logits, dim=-1)[..., 1].cpu().numpy().astype(np.float32))
            feature_chunks.append(features.detach().cpu())
            logit_chunks.append(c_logits.cpu())
        model.cpu()

        label_probs = _stack_numpy(label_chunks, cols=self.n_classes)
        concept_probs = _stack_numpy(concept_chunks, cols=self.n_concepts)
        cache = _PredictionCache(
            dataset_id=id(dataset),
            concept_probs=concept_probs,
            label_probs=label_probs,
            ecbm_features=_stack_tensors(feature_chunks),
            ecbm_concept_logits=_stack_tensors(logit_chunks),
        )
        return label_probs, concept_probs, cache

    def _predict_from_cached_concepts(
        self,
        concepts: np.ndarray,
        cache: _PredictionCache,
        *,
        baseline_concepts: np.ndarray,
        intervention_mask: np.ndarray | None,
    ) -> np.ndarray:
        model = self._require_official_model()
        if cache.ecbm_features is None or cache.ecbm_concept_logits is None:
            raise RuntimeError("Missing cached ECBM features for intervention replay.")
        if intervention_mask is None:
            intervention_mask = ~np.isclose(concepts, baseline_concepts, atol=1e-6, rtol=1e-6)
        intervention_mask = np.asarray(intervention_mask, dtype=bool)
        out = cache.label_probs.copy()
        rows = np.flatnonzero(intervention_mask.any(axis=1))
        if rows.size == 0:
            return out

        device = self._inference_device()
        if next(model.parameters()).device != device:
            model.to(device)
        model.eval()
        batch_size = self._inference_batch_size()
        for start in range(0, rows.size, batch_size):
            idx = rows[start:start + batch_size]
            t_idx = torch.as_tensor(idx, dtype=torch.long)
            y_logits = _intervene(
                model,
                cache.ecbm_features.index_select(0, t_idx).to(device),
                cache.ecbm_concept_logits.index_select(0, t_idx).to(device),
                torch.as_tensor(concepts[idx], dtype=torch.float32, device=device),
                torch.as_tensor(intervention_mask[idx], dtype=torch.bool, device=device),
            )
            out[idx] = torch.softmax(y_logits, dim=-1).cpu().numpy().astype(np.float32)
        return out

    def _rebuild_model(
        self,
        *,
        model_init_kwargs: dict[str, Any],
        backbone_spec: dict[str, Any],
    ) -> Any:
        kwargs = copy.deepcopy(model_init_kwargs)
        kwargs["c_extractor_arch"] = _make_backbone_factory(backbone_spec)
        return _ECBMNet(**kwargs)

    def compute_interpretation_summary(
        self,
        dataset: ConceptDatasetSample,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return compute_ecbm_interpretation_summary(self, dataset, top_k=top_k)


# ---------------------------------------------------------------------------
# ECBM interpretation
# ---------------------------------------------------------------------------

def compute_ecbm_interpretation_summary(
    model: ECBMBenchmarkModel,
    dataset: ConceptDatasetSample,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    y_prob, c_prob = model.predict_proba(dataset, return_concepts=True)
    y_pred = y_prob.argmax(axis=1)
    y_true = np.asarray(dataset.y, dtype=int).reshape(-1)
    c_true = np.asarray(dataset.C, dtype=np.float32)

    overall_pred = c_prob.mean(axis=0) if len(c_prob) else np.zeros(model.n_concepts)
    overall_true = c_true.mean(axis=0) if len(c_true) else np.zeros(model.n_concepts)

    rows: list[dict[str, Any]] = []
    top_concepts: dict[str, list[dict[str, Any]]] = {}
    for class_idx, class_name in enumerate(model.class_names):
        pred_mask = y_pred == class_idx
        true_mask = y_true == class_idx
        pred_mean = (
            c_prob[pred_mask].mean(axis=0)
            if np.any(pred_mask)
            else np.zeros(model.n_concepts, dtype=np.float32)
        )
        oracle_mean = (
            c_true[true_mask].mean(axis=0)
            if np.any(true_mask)
            else np.zeros(model.n_concepts, dtype=np.float32)
        )
        lift = pred_mean - overall_pred
        oracle_lift = oracle_mean - overall_true
        order = np.argsort(-(lift + oracle_lift))
        top_concepts[class_name] = [
            {
                "concept": model.concept_names[int(idx)],
                "predicted_conditional_prob": float(pred_mean[int(idx)]),
                "oracle_conditional_prob": float(oracle_mean[int(idx)]),
                "predicted_lift": float(lift[int(idx)]),
                "oracle_lift": float(oracle_lift[int(idx)]),
                "absolute_error": float(abs(pred_mean[int(idx)] - oracle_mean[int(idx)])),
            }
            for idx in order[: max(1, int(top_k))]
        ]
        for concept_idx, concept_name in enumerate(model.concept_names):
            rows.append(
                {
                    "class_name": class_name,
                    "concept_name": concept_name,
                    "predicted_conditional_prob": float(pred_mean[concept_idx]),
                    "oracle_conditional_prob": float(oracle_mean[concept_idx]),
                    "predicted_lift": float(lift[concept_idx]),
                    "oracle_lift": float(oracle_lift[concept_idx]),
                    "absolute_error": float(
                        abs(pred_mean[concept_idx] - oracle_mean[concept_idx])
                    ),
                    "predicted_support": int(pred_mask.sum()),
                    "oracle_support": int(true_mask.sum()),
                }
            )

    return {
        "family": "ecbm",
        "n_examples": int(dataset.n),
        "overall_predicted_concept_mean": {
            name: float(overall_pred[idx]) for idx, name in enumerate(model.concept_names)
        },
        "overall_oracle_concept_mean": {
            name: float(overall_true[idx]) for idx, name in enumerate(model.concept_names)
        },
        "top_concepts_by_class": top_concepts,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ecbm_model(
    *,
    train_dataset: ConceptDatasetSample,
    valid_dataset: ConceptDatasetSample,
    benchmark: str,
    config: Any,
    device: torch.device | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> ECBMBenchmarkModel:
    """Train an ECBM (authors' energy network and losses) on a benchmark split."""

    device = determine_device() if device is None else torch.device(device)
    loader_cfg = _resolve_loader_config(
        batch_size=int(getattr(config, "batch_size", 32)),
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loader_kwargs = {k: v for k, v in loader_cfg.items() if k != "device"}
    train_loader = train_dataset.loader(shuffle=True, **loader_kwargs)
    valid_loader = valid_dataset.loader(shuffle=False, **loader_kwargs)

    backbone_spec = _infer_backbone_spec(train_dataset, benchmark=benchmark, config=config)
    feature_dim = int(backbone_spec.get("default_output_dim", 128))
    model_init_kwargs = {
        "n_concepts": train_dataset.n_concepts,
        "n_tasks": train_dataset.n_classes,
        "hid_size": int(getattr(config, "ecbm_hid_size", 64)),
        "feature_dim": feature_dim,
        "lambda_xy": float(getattr(config, "ecbm_lambda_xy", 3.0)),
        "lambda_xc": float(getattr(config, "ecbm_lambda_xc", 1.0)),
        "lambda_cy": float(getattr(config, "ecbm_lambda_cy", 1.0)),
        "c_extractor_arch": _make_backbone_factory(backbone_spec),
    }
    model = _ECBMNet(**model_init_kwargs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_resolve_learning_rate(config),
        weight_decay=float(getattr(config, "ecbm_weight_decay", 1e-4)),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    max_epochs = _resolve_epochs(config, benchmark=benchmark, family="ecbm")
    patience = _resolve_patience(config, benchmark=benchmark)
    epochs_no_improve = 0
    best_epoch = 0
    for epoch in range(max_epochs):
        _run_ecbm_epoch(model, train_loader, optimizer=optimizer, device=device)
        valid_metrics = _run_ecbm_epoch(model, valid_loader, optimizer=None, device=device)
        current_val = float(valid_metrics["loss_unweighted"])  # the authors monitor the unweighted sum
        if current_val < best_val_loss - _EARLY_STOP_EPS:
            best_val_loss = current_val
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    model.cpu()

    wrapped_kwargs = copy.deepcopy(model_init_kwargs)
    wrapped_kwargs.pop("c_extractor_arch", None)
    return ECBMBenchmarkModel(
        official_model=model,
        benchmark=benchmark,
        concept_names=list(train_dataset.concepts),
        class_names=list(train_dataset.classes),
        backbone_spec=backbone_spec,
        model_init_kwargs=wrapped_kwargs,
        eval_config=dict(loader_cfg),
        training_summary={"max_epochs": best_epoch + 1},
    )
