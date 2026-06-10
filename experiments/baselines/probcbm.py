"""ProbCBM (Probabilistic Concept Bottleneck Model) wrapper and training."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.types import CBMTrainingMode
from concept_benchmark.utils import determine_device

from experiments.baselines._common import (
    _OfficialBenchmarkModelBase,
    _PredictionCache,
    _build_trainer,
    _device_to_pl_args,
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

_REPO_ROOT = Path(__file__).resolve().parents[2]


class ProbCBMBenchmarkModel(_OfficialBenchmarkModelBase):
    family = "probcbm"

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
        pred_embedding_chunks: list[torch.Tensor] = []
        pred_mean_chunks: list[torch.Tensor] = []
        pred_logsigma_chunks: list[torch.Tensor] = []

        model.to(device)
        with torch.no_grad():
            for batch_x, _, _ in loader:
                batch_x = batch_x.to(device)
                # Official ProbCBM training uses `_forward`; in current torch
                # stacks the public `forward()` path is not reliable because it
                # calls a `ModuleList` directly after init-time head swaps.
                outputs = model._forward(batch_x, train=False, output_latent=True)
                concept_probs = outputs[0]
                pred_embeddings = outputs[1]
                task_outputs = outputs[2]
                latent_dict = outputs[3]

                concept_chunks.append(concept_probs.detach().cpu().numpy())
                label_chunks.append(_to_numpy_task_proba(task_outputs, self.n_classes))
                pred_embedding_chunks.append(pred_embeddings.detach().cpu())
                pred_mean_chunks.append(latent_dict["pred_mean"].detach().cpu())
                if latent_dict.get("pred_logsigma") is not None:
                    pred_logsigma_chunks.append(latent_dict["pred_logsigma"].detach().cpu())
        model.cpu()

        concept_probs = _stack_numpy(concept_chunks, cols=self.n_concepts)
        label_probs = _stack_numpy(label_chunks, cols=self.n_classes)
        cache = _PredictionCache(
            dataset_id=id(dataset),
            concept_probs=concept_probs,
            label_probs=label_probs,
            probcbm_pred_embeddings=_stack_tensors(pred_embedding_chunks),
            probcbm_pred_mean=_stack_tensors(pred_mean_chunks),
            probcbm_pred_logsigma=_stack_tensors(pred_logsigma_chunks) if pred_logsigma_chunks else None,
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
        device = self._inference_device()
        if next(model.parameters()).device != device:
            model.to(device)
        concept_tensor = torch.as_tensor(concepts, dtype=torch.float32, device=device)
        baseline_probs = torch.as_tensor(
            baseline_concepts, dtype=torch.float32, device=device
        )

        # Sample from the predicted concept distribution (proper ProbCBM inference).
        # For intervened concepts we replace with deterministic prototype (zero variance).
        n_samples = getattr(model, "n_samples_inference", 50)
        pred_mean = cache.probcbm_pred_mean
        pred_logsigma = cache.probcbm_pred_logsigma

        if pred_mean is not None and pred_logsigma is not None:
            mu = pred_mean.to(device)
            logsigma = pred_logsigma.to(device)
            eps = torch.randn(
                mu.size(0), mu.size(1), n_samples, mu.size(2),
                dtype=mu.dtype, device=device,
            )
            pred_embeddings = eps.mul(torch.exp(logsigma.unsqueeze(2) * 0.5)).add_(mu.unsqueeze(2))
        else:
            # Fallback: use cached embeddings (mean only, n_samples=1)
            pred_embeddings = cache.probcbm_pred_embeddings
            if pred_embeddings is None:
                raise RuntimeError(
                    "Missing cached ProbCBM concept embeddings for intervention replay."
                )
            pred_embeddings = pred_embeddings.to(device)

        concept_mean = F.normalize(model.concept_vectors, p=2, dim=-1)
        if getattr(model, "use_neg_concept", True) and concept_mean.shape[0] < 2:
            concept_mean = torch.cat([-concept_mean, concept_mean], dim=0)
        neg_proto = concept_mean[0]
        pos_proto = concept_mean[1] if concept_mean.shape[0] > 1 else concept_mean[0]

        # Deterministic prototype replacement for intervened concepts (zero variance)
        replacement = (
            concept_tensor.unsqueeze(-1).unsqueeze(-1)
            * pos_proto.unsqueeze(0).unsqueeze(2)
            + (1.0 - concept_tensor).unsqueeze(-1).unsqueeze(-1)
            * neg_proto.unsqueeze(0).unsqueeze(2)
        )
        if intervention_mask is not None:
            changed = torch.as_tensor(
                intervention_mask, dtype=torch.bool, device=device
            ).unsqueeze(-1).unsqueeze(-1)
        else:
            changed = (
                ~torch.isclose(concept_tensor, baseline_probs, atol=1e-6, rtol=1e-6)
            ).unsqueeze(-1).unsqueeze(-1)
        concept_embeddings_for_class = torch.where(changed, replacement, pred_embeddings)

        # Expected shape: (batch, n_concepts, n_classes, emb_size)
        assert concept_embeddings_for_class.ndim == 4, (
            f"Expected 4-D concept embeddings, got shape {concept_embeddings_for_class.shape}"
        )
        concept_embeddings_for_class = (
            concept_embeddings_for_class.permute(0, 2, 1, 3).contiguous().view(
                concept_embeddings_for_class.shape[0],
                concept_embeddings_for_class.shape[2],
                -1,
            )
        )

        with torch.no_grad():
            class_embeddings = model.head(concept_embeddings_for_class)
            class_mean = model.class_mean.unsqueeze(1).unsqueeze(0)
            distance = torch.sqrt(
                (
                    (class_embeddings.unsqueeze(1) - class_mean) ** 2
                ).mean(-1)
                + 1e-10
            )
            if getattr(model, "use_scale", False):
                distance = model.class_negative_scale * distance
            class_probs = F.softmax(-distance, dim=1).mean(dim=-1)
        return class_probs.detach().cpu().numpy().astype(np.float32)

    def _rebuild_model(
        self,
        *,
        model_init_kwargs: dict[str, Any],
        backbone_spec: dict[str, Any],
    ) -> Any:
        deps = require_cem_dependencies()
        kwargs = copy.deepcopy(model_init_kwargs)
        kwargs["c_extractor_arch"] = _make_backbone_factory(backbone_spec)
        return deps.ProbCBM(**kwargs)


def train_probcbm_model(
    *,
    train_dataset: ConceptDatasetSample,
    valid_dataset: ConceptDatasetSample,
    benchmark: str,
    config: Any,
    device: torch.device | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> ProbCBMBenchmarkModel:
    """Train an official ProbCBM, preferring the upstream helper when available."""

    deps = require_cem_dependencies(include_probcbm_training_helper=True)
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
        "concept_loss_weight": float(
            getattr(config, "cem_concept_loss_weight", 1.0)
        ),
        "task_loss_weight": float(getattr(config, "cem_task_loss_weight", 1.0)),
        "hidden_dim": int(getattr(config, "probcbm_hidden_dim", 8)),
        "class_hidden_dim": int(getattr(config, "probcbm_class_hidden_dim", 64)),
        "intervention_prob": float(
            getattr(config, "probcbm_intervention_prob", 0.25)
        ),
        "c_extractor_arch": _make_backbone_factory(backbone_spec),
        "pretrained": False,
        "n_samples_inference": int(
            getattr(config, "probcbm_n_samples_inference", 1)
        ),
        "use_neg_concept": True,
        "pred_class": True,
        "use_scale": True,
        "train_class_mode": config.training_mode,
        "latent_dim": int(getattr(config, "probcbm_latent_dim", 8)),
        "learning_rate": _resolve_learning_rate(config),
        "optimizer": "adam",
    }
    used_official_helper = deps.train_prob_cbm is not None
    if used_official_helper:
        helper_config = copy.deepcopy(model_init_kwargs)
        helper_config["architecture"] = "ProbCBM"
        helper_config["max_epochs"] = _resolve_epochs(
            config, benchmark=benchmark, family="probcbm"
        )
        helper_config["check_val_every_n_epoch"] = 1
        helper_config["early_stopping_best_model"] = False
        with tempfile.TemporaryDirectory(dir=_REPO_ROOT) as tmpdir:
            model, helper_metrics = deps.train_prob_cbm(
                input_shape=np.asarray(train_dataset.inputs[0]).shape,
                n_concepts=train_dataset.n_concepts,
                n_tasks=train_dataset.n_classes,
                config=helper_config,
                train_dl=train_loader,
                val_dl=valid_loader,
                run_name="ProbCBM",
                result_dir=tmpdir,
                split=0,
                seed=int(getattr(config, "seed", 0)),
                save_model=False,
                logger=False,
                enable_checkpointing=False,
                **_device_to_pl_args(device),
            )
        trainer_epochs = int(helper_metrics.get("num_epochs", helper_config["max_epochs"]))
    else:
        # Fallback for environments where the upstream helper cannot be imported
        # because optional helper-only deps (for example TensorFlow) are absent.
        model = deps.ProbCBM(**model_init_kwargs)
        patience = _resolve_patience(config, benchmark=benchmark)

        if model.train_class_mode == CBMTrainingMode.Sequential:
            # Phase 1: train concept predictor only
            model.stage = "concept"
            classify_params = set(model.params_to_classify())
            trainable_before = {
                name: param.requires_grad
                for name, param in model.named_parameters()
            }
            for name, param in model.named_parameters():
                if name in classify_params:
                    param.requires_grad = False

            try:
                concept_trainer = _build_trainer(
                    pl_module=deps.pl,
                    max_epochs=_resolve_epochs(config, benchmark=benchmark, family="probcbm"),
                    patience=patience,
                    device=device,
                )
                concept_trainer.fit(model, train_loader, valid_loader)
                trainer_epochs = concept_trainer.current_epoch + 1

                # Phase 2: train class predictor only
                model.stage = "class"
                for name, param in model.named_parameters():
                    param.requires_grad = name in classify_params

                class_epochs = int(getattr(config, "probcbm_epochs_class", 20))
                class_trainer = _build_trainer(
                    pl_module=deps.pl,
                    max_epochs=class_epochs,
                    patience=patience,
                    device=device,
                )
                class_trainer.fit(model, train_loader, valid_loader)
                trainer_epochs += class_trainer.current_epoch + 1
            finally:
                # Restore requires_grad even if training fails
                for name, param in model.named_parameters():
                    if name in trainable_before:
                        param.requires_grad = trainable_before[name]
        else:
            # Joint mode: single training pass
            trainer = _build_trainer(
                pl_module=deps.pl,
                max_epochs=_resolve_epochs(config, benchmark=benchmark, family="probcbm"),
                patience=patience,
                device=device,
            )
            trainer.fit(model, train_loader, valid_loader)
            trainer_epochs = trainer.current_epoch + 1
    model.eval()
    model.cpu()

    wrapped_kwargs = copy.deepcopy(model_init_kwargs)
    wrapped_kwargs.pop("c_extractor_arch", None)
    return ProbCBMBenchmarkModel(
        official_model=model,
        benchmark=benchmark,
        concept_names=list(train_dataset.concepts),
        class_names=list(train_dataset.classes),
        backbone_spec=backbone_spec,
        model_init_kwargs=wrapped_kwargs,
        eval_config=loader_cfg,
        training_summary={
            "max_epochs": trainer_epochs,
            "used_official_train_prob_cbm": used_official_helper,
            "official_train_prob_cbm_available": deps.train_prob_cbm is not None,
        },
    )
