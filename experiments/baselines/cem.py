"""CEM (Concept Embedding Model) wrapper and training."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.utils import determine_device

from experiments.baselines._common import (
    _OfficialBenchmarkModelBase,
    _PredictionCache,
    _build_trainer,
    _infer_backbone_spec,
    _make_backbone_factory,
    _resolve_epochs,
    _resolve_learning_rate,
    _resolve_loader_config,
    _resolve_patience,
    _stack_numpy,
    _stack_tensors,
    _to_numpy_task_proba,
    make_cem_loader,
    require_cem_dependencies,
)


class CEMBenchmarkModel(_OfficialBenchmarkModelBase):
    family = "cem"

    def _run_official_model(
        self,
        dataset: ConceptDatasetSample,
    ) -> tuple[np.ndarray, np.ndarray, _PredictionCache]:
        model = self._require_official_model()
        model.eval()
        device = self._inference_device()
        loader = make_cem_loader(dataset, shuffle=False, **self._loader_kwargs())

        concept_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        pos_chunks: list[torch.Tensor] = []
        neg_chunks: list[torch.Tensor] = []

        model.to(device)
        with torch.no_grad():
            for batch_x, _, _ in loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x, output_embeddings=True)
                concept_probs = outputs[0]
                task_outputs = outputs[2]
                pos_embeddings = outputs[3]
                neg_embeddings = outputs[4]

                concept_chunks.append(concept_probs.detach().cpu().numpy())
                label_chunks.append(_to_numpy_task_proba(task_outputs, self.n_classes))
                pos_chunks.append(pos_embeddings.detach().cpu())
                neg_chunks.append(neg_embeddings.detach().cpu())
        model.cpu()

        concept_probs = _stack_numpy(concept_chunks, cols=self.n_concepts)
        label_probs = _stack_numpy(label_chunks, cols=self.n_classes)
        cache = _PredictionCache(
            dataset_id=id(dataset),
            concept_probs=concept_probs,
            label_probs=label_probs,
            pos_embeddings=_stack_tensors(pos_chunks),
            neg_embeddings=_stack_tensors(neg_chunks),
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
        pos = cache.pos_embeddings
        neg = cache.neg_embeddings
        if pos is None or neg is None:
            raise RuntimeError("Missing cached CEM embeddings for intervention replay.")

        device = self._inference_device()
        if next(model.parameters()).device != device:
            model.to(device)
        concept_tensor = torch.as_tensor(concepts, dtype=torch.float32, device=device)
        baseline_tensor = torch.as_tensor(
            baseline_concepts, dtype=torch.float32, device=device
        )
        if intervention_mask is not None:
            mask_tensor = torch.as_tensor(intervention_mask, dtype=torch.bool, device=device)
            concept_tensor = torch.where(mask_tensor, concept_tensor, baseline_tensor)
        pos = pos.to(device)
        neg = neg.to(device)

        with torch.no_grad():
            bottleneck = pos * concept_tensor.unsqueeze(-1) + neg * (
                1.0 - concept_tensor.unsqueeze(-1)
            )
            logits = model.c2y_model(torch.flatten(bottleneck, start_dim=1, end_dim=-1))
        return _to_numpy_task_proba(logits, self.n_classes)

    def _rebuild_model(
        self,
        *,
        model_init_kwargs: dict[str, Any],
        backbone_spec: dict[str, Any],
    ) -> Any:
        deps = require_cem_dependencies()
        kwargs = copy.deepcopy(model_init_kwargs)
        kwargs["c_extractor_arch"] = _make_backbone_factory(backbone_spec)
        return deps.ConceptEmbeddingModel(**kwargs)


def train_cem_model(
    *,
    train_dataset: ConceptDatasetSample,
    valid_dataset: ConceptDatasetSample,
    benchmark: str,
    config: Any,
    device: torch.device | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> CEMBenchmarkModel:
    """Train an official Concept Embedding Model on a benchmark split."""

    deps = require_cem_dependencies()
    device = determine_device() if device is None else torch.device(device)
    loader_cfg = _resolve_loader_config(
        batch_size=int(getattr(config, "batch_size", 32)),
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loader_kwargs = {k: v for k, v in loader_cfg.items() if k != "device"}
    train_loader = make_cem_loader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = make_cem_loader(valid_dataset, shuffle=False, **loader_kwargs)

    backbone_spec = _infer_backbone_spec(train_dataset, benchmark=benchmark, config=config)
    model_init_kwargs = {
        "n_concepts": train_dataset.n_concepts,
        "n_tasks": train_dataset.n_classes,
        "emb_size": int(getattr(config, "cem_emb_size", 16)),
        "training_intervention_prob": float(
            getattr(config, "cem_training_intervention_prob", 0.25)
        ),
        "concept_loss_weight": float(
            getattr(config, "cem_concept_loss_weight", 1.0)
        ),
        "task_loss_weight": float(getattr(config, "cem_task_loss_weight", 1.0)),
        "learning_rate": _resolve_learning_rate(config),
        "optimizer": "adam",
        "weight_decay": 4e-5,
        "c_extractor_arch": _make_backbone_factory(backbone_spec),
    }
    model = deps.ConceptEmbeddingModel(**model_init_kwargs)

    trainer = _build_trainer(
        pl_module=deps.pl,
        max_epochs=_resolve_epochs(config, benchmark=benchmark, family="cem"),
        patience=_resolve_patience(config, benchmark=benchmark),
        device=device,
    )
    trainer.fit(model, train_loader, valid_loader)
    model.eval()
    model.cpu()

    wrapped_kwargs = copy.deepcopy(model_init_kwargs)
    wrapped_kwargs.pop("c_extractor_arch", None)
    return CEMBenchmarkModel(
        official_model=model,
        benchmark=benchmark,
        concept_names=list(train_dataset.concepts),
        class_names=list(train_dataset.classes),
        backbone_spec=backbone_spec,
        model_init_kwargs=wrapped_kwargs,
        eval_config=loader_cfg,
        training_summary={"max_epochs": trainer.current_epoch + 1},
    )
