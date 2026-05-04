"""ECBM (Energy-based Concept Bottleneck Model) — standalone implementation."""
from __future__ import annotations

import copy
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
    _safe_binary_logit,
    _safe_class_logit,
    _soft_cross_entropy_from_probs,
    _stack_numpy,
    _stack_tensors,
)


# ---------------------------------------------------------------------------
# ECBM network
# ---------------------------------------------------------------------------

class _ECBMNet(nn.Module):
    def __init__(
        self,
        *,
        n_concepts: int,
        n_tasks: int,
        emb_size: int,
        hid_size: int,
        feature_dim: int,
        lambda_xy: float,
        lambda_xc: float,
        lambda_cy: float,
        c_extractor_arch: Any,
    ) -> None:
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.n_tasks = int(n_tasks)
        self.emb_size = int(emb_size)
        self.hid_size = int(hid_size)
        self.feature_dim = int(feature_dim)
        self.lambda_xy = float(lambda_xy)
        self.lambda_xc = float(lambda_xc)
        self.lambda_cy = float(lambda_cy)

        self.backbone = c_extractor_arch(self.feature_dim)
        self.xy_fc1 = nn.Linear(self.feature_dim, self.hid_size)
        self.xc_fc1 = nn.Linear(self.feature_dim, self.hid_size)
        self.classifier_xc = nn.Linear(self.hid_size, self.n_concepts)
        self.pos_concept_embeddings = nn.Parameter(
            torch.randn(self.n_concepts, self.emb_size) * 0.05
        )
        self.neg_concept_embeddings = nn.Parameter(
            torch.randn(self.n_concepts, self.emb_size) * 0.05
        )
        self.concept_proj = nn.Linear(self.n_concepts * self.emb_size, self.hid_size)
        self.class_embeddings = nn.Parameter(
            torch.randn(self.n_tasks, self.hid_size) * 0.05
        )
        self.xy_bias = nn.Parameter(torch.zeros(self.n_tasks))
        self.cy_bias = nn.Parameter(torch.zeros(self.n_tasks))
        self.xy_scale = nn.Parameter(torch.tensor(5.0))
        self.cy_scale = nn.Parameter(torch.tensor(5.0))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.float())

    def _class_logits(
        self,
        hidden: torch.Tensor,
        *,
        bias: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.normalize(hidden, p=2, dim=-1)
        class_embeddings = F.normalize(self.class_embeddings, p=2, dim=-1)
        scaled = F.softplus(scale) + 1.0
        return scaled * (hidden @ class_embeddings.T) + bias

    def xy_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.xy_fc1(features))
        return self._class_logits(hidden, bias=self.xy_bias, scale=self.xy_scale)

    def xc_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.xc_fc1(features))
        return self.classifier_xc(hidden)

    def concept_embeddings_from_probs(self, concept_probs: torch.Tensor) -> torch.Tensor:
        pos = self.pos_concept_embeddings.unsqueeze(0)
        neg = self.neg_concept_embeddings.unsqueeze(0)
        return concept_probs.unsqueeze(-1) * pos + (1.0 - concept_probs).unsqueeze(
            -1
        ) * neg

    def cy_logits_from_concepts(self, concept_probs: torch.Tensor) -> torch.Tensor:
        concept_embeddings = self.concept_embeddings_from_probs(concept_probs)
        hidden = F.relu(self.concept_proj(concept_embeddings.flatten(start_dim=1)))
        return self._class_logits(hidden, bias=self.cy_bias, scale=self.cy_scale)


# ---------------------------------------------------------------------------
# ECBM loss and inference
# ---------------------------------------------------------------------------

def _apply_ecbm_losses(
    model: _ECBMNet,
    *,
    xy_logits: torch.Tensor,
    xc_logits: torch.Tensor,
    cy_logits: torch.Tensor,
    concepts: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_xy = F.cross_entropy(xy_logits, labels)
    loss_xc = F.binary_cross_entropy_with_logits(xc_logits, concepts)
    loss_cy = F.cross_entropy(cy_logits, labels)
    total = (
        model.lambda_xy * loss_xy
        + model.lambda_xc * loss_xc
        + model.lambda_cy * loss_cy
    )
    metrics = {
        "loss_xy": float(loss_xy.detach().cpu()),
        "loss_xc": float(loss_xc.detach().cpu()),
        "loss_cy": float(loss_cy.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }
    return total, metrics


def _run_ecbm_inference(
    model: _ECBMNet,
    features: torch.Tensor,
    *,
    steps: int,
    lr: float,
    concept_init: torch.Tensor | None = None,
    label_init: torch.Tensor | None = None,
    forced_concept_probs: torch.Tensor | None = None,
    forced_concept_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        xy_logits = model.xy_logits_from_features(features)
        xc_logits = model.xc_logits_from_features(features)
        xy_target = torch.softmax(xy_logits, dim=-1)
        xc_target = torch.sigmoid(xc_logits)

    if label_init is None:
        label_logits = _safe_class_logit(xy_target)
    else:
        label_logits = _safe_class_logit(label_init)
    if concept_init is None:
        concept_logits = _safe_binary_logit(xc_target)
    else:
        concept_logits = _safe_binary_logit(concept_init)

    label_logits = nn.Parameter(label_logits.detach().clone())
    concept_logits = nn.Parameter(concept_logits.detach().clone())
    optimizer = torch.optim.Adam([label_logits, concept_logits], lr=float(lr))

    fixed_logits = None
    if forced_concept_probs is not None:
        fixed_logits = _safe_binary_logit(forced_concept_probs)
    intervened_rows = None
    if forced_concept_mask is not None and fixed_logits is not None:
        intervened_rows = forced_concept_mask.any(dim=-1)
        with torch.no_grad():
            concept_logits.data = torch.where(
                forced_concept_mask,
                fixed_logits,
                concept_logits.data,
            )
            if intervened_rows.any():
                label_logits.data[intervened_rows] = 0.0

    original_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)
    try:
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad()
            y_prob = torch.softmax(label_logits, dim=-1)
            c_prob = torch.sigmoid(concept_logits)
            if forced_concept_mask is not None and forced_concept_probs is not None:
                c_prob = torch.where(forced_concept_mask, forced_concept_probs, c_prob)
            cy_logits = model.cy_logits_from_concepts(c_prob)
            cy_prob = torch.softmax(cy_logits, dim=-1)

            loss_xy = _soft_cross_entropy_from_probs(label_logits, xy_target)
            xc_loss = F.binary_cross_entropy_with_logits(
                concept_logits,
                xc_target,
                reduction="none",
            )
            if forced_concept_mask is not None:
                xc_loss = xc_loss.masked_fill(forced_concept_mask, 0.0)
                free_count = int((~forced_concept_mask).sum().item())
                loss_xc = (
                    xc_loss.sum() / free_count if free_count > 0 else xc_loss.sum() * 0.0
                )
            else:
                loss_xc = xc_loss.mean()
            loss_cy = 0.5 * (
                _soft_cross_entropy_from_probs(label_logits, cy_prob.detach())
                + _soft_cross_entropy_from_probs(cy_logits, y_prob.detach())
            )
            loss = (
                model.lambda_xy * loss_xy
                + model.lambda_xc * loss_xc
                + model.lambda_cy * loss_cy
            )
            loss.backward()
            if forced_concept_mask is not None and concept_logits.grad is not None:
                concept_logits.grad.masked_fill_(forced_concept_mask, 0.0)
            if intervened_rows is not None and label_logits.grad is not None:
                non_intervened = ~intervened_rows
                if non_intervened.any():
                    label_logits.grad.data[non_intervened] = 0.0
            optimizer.step()
            if forced_concept_mask is not None and fixed_logits is not None:
                with torch.no_grad():
                    concept_logits.data = torch.where(
                        forced_concept_mask,
                        fixed_logits,
                        concept_logits.data,
                    )

        with torch.no_grad():
            y_prob = torch.softmax(label_logits, dim=-1)
            c_prob = torch.sigmoid(concept_logits)
            if forced_concept_mask is not None and forced_concept_probs is not None:
                c_prob = torch.where(forced_concept_mask, forced_concept_probs, c_prob)
        return y_prob.detach(), c_prob.detach()
    finally:
        for param, required in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(required)


def _run_ecbm_epoch(
    model: _ECBMNet,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss_xy": 0.0, "loss_xc": 0.0, "loss_cy": 0.0, "loss_total": 0.0}
    n_examples = 0

    for batch_x, batch_c, batch_y in loader:
        batch_x = _prepare_batch_features(batch_x, device=device)
        batch_c = _prepare_batch_concepts(batch_c, device=device)
        batch_y = _prepare_batch_labels(batch_y, device=device)

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            features = model.extract_features(batch_x)
            xy_logits = model.xy_logits_from_features(features)
            xc_logits = model.xc_logits_from_features(features)
            cy_logits = model.cy_logits_from_concepts(batch_c)
            loss, metrics = _apply_ecbm_losses(
                model,
                xy_logits=xy_logits,
                xc_logits=xc_logits,
                cy_logits=cy_logits,
                concepts=batch_c,
                labels=batch_y,
            )
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(batch_y.shape[0])
        n_examples += batch_size
        for key, value in metrics.items():
            totals[key] += value * batch_size

    if n_examples == 0:
        return {key: 0.0 for key in totals}
    return {key: value / n_examples for key, value in totals.items()}


# ---------------------------------------------------------------------------
# ECBM benchmark model
# ---------------------------------------------------------------------------

class ECBMBenchmarkModel(_OfficialBenchmarkModelBase):
    family = "ecbm"

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
        inference_steps = int(self.eval_config.get("ecbm_inference_steps", 10))
        inference_lr = float(self.eval_config.get("ecbm_inference_lr", 0.1))

        model.to(device)
        for batch_x, _, _ in loader:
            batch_x = _prepare_batch_features(batch_x, device=device)
            with torch.no_grad():
                features = model.extract_features(batch_x)
            y_prob, c_prob = _run_ecbm_inference(
                model,
                features,
                steps=inference_steps,
                lr=inference_lr,
            )
            label_chunks.append(y_prob.cpu().numpy().astype(np.float32))
            concept_chunks.append(c_prob.cpu().numpy().astype(np.float32))
            feature_chunks.append(features.detach().cpu())
        model.cpu()

        label_probs = _stack_numpy(label_chunks, cols=self.n_classes)
        concept_probs = _stack_numpy(concept_chunks, cols=self.n_concepts)
        cache = _PredictionCache(
            dataset_id=id(dataset),
            concept_probs=concept_probs,
            label_probs=label_probs,
            ecbm_features=_stack_tensors(feature_chunks),
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
        features = cache.ecbm_features
        if features is None:
            raise RuntimeError("Missing cached ECBM features for intervention replay.")

        effective = (
            np.where(intervention_mask, concepts, baseline_concepts)
            if intervention_mask is not None
            else concepts
        )
        if np.allclose(effective, baseline_concepts, atol=1e-6, rtol=1e-6):
            return cache.label_probs.copy()

        if intervention_mask is None:
            intervention_mask = ~np.isclose(
                concepts,
                baseline_concepts,
                atol=1e-6,
                rtol=1e-6,
            )

        device = self._inference_device()
        features = features.to(device)
        baseline_tensor = torch.as_tensor(
            baseline_concepts, dtype=torch.float32, device=device
        )
        label_init = torch.as_tensor(cache.label_probs, dtype=torch.float32, device=device)
        forced_probs = torch.as_tensor(effective, dtype=torch.float32, device=device)
        forced_mask = torch.as_tensor(intervention_mask, dtype=torch.bool, device=device)

        if next(model.parameters()).device != device:
            model.to(device)
        model.eval()
        y_prob, _ = _run_ecbm_inference(
            model,
            features,
            steps=int(self.eval_config.get("ecbm_inference_steps", 10)),
            lr=float(self.eval_config.get("ecbm_inference_lr", 0.1)),
            concept_init=baseline_tensor,
            label_init=label_init,
            forced_concept_probs=forced_probs,
            forced_concept_mask=forced_mask,
        )
        return y_prob.cpu().numpy().astype(np.float32)

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
    """Train a local ECBM-style wrapped model on a benchmark split."""

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
    feature_dim = max(
        int(backbone_spec.get("default_output_dim", 64)),
        int(getattr(config, "ecbm_hid_size", 64)),
    )
    model_init_kwargs = {
        "n_concepts": train_dataset.n_concepts,
        "n_tasks": train_dataset.n_classes,
        "emb_size": int(getattr(config, "ecbm_emb_size", 8)),
        "hid_size": int(getattr(config, "ecbm_hid_size", 64)),
        "feature_dim": feature_dim,
        "lambda_xy": float(getattr(config, "ecbm_lambda_xy", 1.0)),
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
        current_val = float(valid_metrics["loss_total"])
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
    eval_config = {
        **loader_cfg,
        "ecbm_inference_steps": int(getattr(config, "ecbm_inference_steps", 10)),
        "ecbm_inference_lr": float(getattr(config, "ecbm_inference_lr", 0.1)),
    }
    return ECBMBenchmarkModel(
        official_model=model,
        benchmark=benchmark,
        concept_names=list(train_dataset.concepts),
        class_names=list(train_dataset.classes),
        backbone_spec=backbone_spec,
        model_init_kwargs=wrapped_kwargs,
        eval_config=eval_config,
        training_summary={"max_epochs": best_epoch + 1},
    )
