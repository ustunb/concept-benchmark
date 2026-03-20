from __future__ import annotations

__all__ = [
    "RobotClassifierCNN",
    "RobotConceptClassifier",
    "RobotViTConceptClassifier",
    "JointConceptModel",
    "SudokuValidatorCNN",
    "GroupPoolingConceptSudokuCNN",
    "ConceptDetector",
    "FrontEndModel",
    "ConceptBasedModel",
]

import copy
import itertools
import os
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.ext.fileutils import load as load_object, save as save_object
from experiments.train import (
    ConceptTrainer,
    DefaultConceptTrainer,
    TrainerResult,
    _extract_logits,
    _prepare_inputs,
)


class RobotClassifierCNN(nn.Module):
    def __init__(self, num_classes=1, input_size=224):
        super(RobotClassifierCNN, self).__init__()

        # --- Feature Extractor ---
        # Input images are assumed to be 3-channel RGB
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves the dimensions
        # Block 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Halves the dimensions again
        # Block 3
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.backbone = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            nn.ReLU(),
            self.pool3,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            dummy_out = self.backbone(dummy)
            feature_size = dummy_out.view(1, -1).size(1)

        # self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc1 = nn.Linear(feature_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass through the feature extractor
        x = self.backbone(x)

        # Flatten the feature maps for the classifier
        x = torch.flatten(x, 1)

        # Pass through the classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # For binary classification, we apply a sigmoid function to the output
        return torch.sigmoid(x)


class RobotConceptClassifier(nn.Module):
    def __init__(self, num_concepts: int, input_size: int):
        super(RobotConceptClassifier, self).__init__()

        if input_size >= 128:
            # Large images: use pooling to reduce spatial dimensions
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
        else:
            # Small images: No pooling, keep all spatial information
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        # Automatically compute feature size with dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            dummy_out = self.backbone(dummy)
            feature_size = dummy_out.view(1, -1).size(1)

        # Concept heads
        self.heads = nn.ModuleList(
            [nn.Linear(feature_size, 1) for _ in range(num_concepts)]
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        logits = torch.cat([head(features) for head in self.heads], dim=1)
        return logits


class RobotViTConceptClassifier(nn.Module):
    def __init__(self, num_concepts: int):
        super(RobotViTConceptClassifier, self).__init__()
        from transformers import ViTModel

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        feature_size = 768  # ViT base model feature size

        # 2) One head per concept (order matches input labels), wrapped in nn.Sequential
        self.heads = nn.ModuleList(
            [nn.Linear(feature_size, 1) for _ in range(num_concepts)]
        )

    def forward(self, x):
        vit_outputs = self.vit(pixel_values=x)
        features = vit_outputs.last_hidden_state[:, 0, :]  # (N, 768)
        # Concatenate per-concept logits into shape (N, num_concepts)
        logits = torch.cat(
            [head(features) for head in self.heads], dim=1
        )  # (N, num_concepts)
        return logits


class JointConceptModel(nn.Module):
    """Simple wrapper that links an optional backbone with a concept head."""

    def __init__(
        self,
        backbone: nn.Module | None,
        head: nn.Module,
        *,
        should_flatten: bool = True,
    ) -> None:
        """Create a joint concept model.

        Args:
            backbone: Optional feature extractor applied before the concept head.
                If ``None`` the raw input is forwarded directly to ``head``.
            head: Module that maps backbone features to per-concept logits.
            should_flatten: Whether to flatten high-rank tensors before feeding them to
                ``head``. Disable when ``head`` expects structured inputs (e.g. CNN).
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.should_flatten = bool(should_flatten)

    def forward(self, x: Any) -> torch.Tensor:
        """Return per-concept logits for the provided batch.

        Args:
            x: Mini-batch of raw inputs consumed by the backbone/head stack.

        Returns:
            torch.Tensor: Logits of shape ``(batch, n_concepts)``.

        Raises:
            ValueError: If the backbone returns an empty sequence of features.
            TypeError: If the backbone output dict lacks recognised keys.
        """
        features = x
        if self.backbone is not None:
            features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError("Backbone returned an empty sequence of features.")
            features = features[0]
        if isinstance(features, dict):
            if "logits" in features:
                features = features["logits"]
            elif "last_hidden_state" in features:
                features = features["last_hidden_state"][:, 0]
            else:
                raise TypeError(
                    "Unable to extract tensor from backbone output dictionary."
                )
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        if self.should_flatten and features.ndim > 2:
            features = features.view(features.size(0), -1)
        return self.head(features)


# ── Sudoku models ────────────────────────────────────────────────────


class SudokuValidatorCNN(nn.Module):
    """End-to-end DNN baseline for sudoku board validation."""

    _NUM_DIGITS = 10

    def __init__(self, embedding_dim=16, hidden_dim=128):
        super(SudokuValidatorCNN, self).__init__()
        self.embedding = nn.Linear(self._NUM_DIGITS, embedding_dim, bias=False)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = F.one_hot(x.long(), self._NUM_DIGITS).float()
        x = self.embedding(x)
        x = x.permute(0, 2, 1).view(-1, x.size(2), 9, 9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = self.mlp(x)
        return torch.sigmoid(x)


class GroupPoolingConceptSudokuCNN(nn.Module):
    """Concept-based sudoku model using group pooling over rows/cols/blocks."""

    _NUM_DIGITS = 10

    def __init__(self, embedding_dim=16, hidden_dim=64):
        super(GroupPoolingConceptSudokuCNN, self).__init__()
        self.embedding = nn.Linear(self._NUM_DIGITS, embedding_dim, bias=False)
        self.head = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _pool_groups(self, x, dim):
        mean = x.mean(dim=dim)
        maxv = x.amax(dim=dim)
        return torch.cat([mean, maxv], dim=-1)

    def forward(self, x):
        x = F.one_hot(x.long(), self._NUM_DIGITS).float()
        x = self.embedding(x)  # (N, 81, D)
        x = x.view(x.size(0), 9, 9, -1)  # (N, 9, 9, D)

        row_feats = self._pool_groups(x, dim=2)  # (N, 9, 2D)
        col_feats = self._pool_groups(x, dim=1)  # (N, 9, 2D)

        blocks = x.view(x.size(0), 3, 3, 3, 3, x.size(-1))
        blocks = blocks.permute(0, 1, 3, 2, 4, 5).contiguous()
        block_cells = blocks.view(x.size(0), 9, 9, x.size(-1))
        block_feats = self._pool_groups(block_cells, dim=2)  # (N, 9, 2D)

        row_logits = self.head(row_feats).squeeze(-1)
        col_logits = self.head(col_feats).squeeze(-1)
        block_logits = self.head(block_feats).squeeze(-1)

        logits = torch.cat([row_logits, col_logits, block_logits], dim=1)
        return logits


class ConceptDetector:
    """Concept detector with optional calibration and pluggable trainer."""

    def __init__(
        self,
        embedding_model: nn.Module | None = None,
        *,
        model: nn.Module | None = None,
        trainer: ConceptTrainer | None = None,
        model_builder: Callable[[int, int, int], nn.Module] | None = None,
    ) -> None:
        """Initialise a concept detector.

        Args:
            embedding_model: Optional backbone that produces intermediate
                representations. If provided, it is wrapped inside a
                ``JointConceptModel`` when the detector builds its default head.
            model: Pre-built joint concept model. When supplied the detector
                skips automatic model construction.
            trainer: Custom training callable. Defaults to
                :class:`~concept_benchmark.train.DefaultConceptTrainer`.
            model_builder: Optional factory used to create a joint model when
                ``model`` is not provided. Receives ``(train_dataset, device,
                hidden_dim)`` and must return an ``nn.Module`` that emits
                ``n_concepts`` logits.
        """
        self.embedding_model = embedding_model
        self.model = model
        self.model_builder = model_builder
        self.trainer = trainer or DefaultConceptTrainer()
        self.calibration_params: list[dict | None] | None = None
        self.training_result: TrainerResult | None = None
        self._n_concepts: int | None = None
        self._eval_config: dict[str, Any] = {}

    def to(self, device: str | torch.device) -> "ConceptDetector":
        """Move the underlying torch modules to a target device."""
        target = torch.device(device)
        if self.model is not None:
            self.model.to(target)
            if isinstance(self.model, JointConceptModel):
                self.embedding_model = self.model.backbone
        elif self.embedding_model is not None:
            self.embedding_model.to(target)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly snapshot of the detector."""
        model_copy = None
        if self.model is not None:
            model_copy = copy.deepcopy(self.model)
            model_copy.to("cpu")
        embedding_copy = None
        if self.embedding_model is not None:
            embedding_copy = copy.deepcopy(self.embedding_model)
            if isinstance(embedding_copy, nn.Module):
                embedding_copy.to("cpu")

        training_summary = None
        if self.training_result is not None:
            training_summary = {
                "history": copy.deepcopy(self.training_result.history),
                "best_metric": self.training_result.best_metric,
            }

        return {
            "version": 1,
            "model": model_copy,
            "embedding_model": embedding_copy,
            "calibration_params": copy.deepcopy(self.calibration_params),
            "eval_config": copy.deepcopy(self._eval_config),
            "n_concepts": self._n_concepts,
            "training_summary": training_summary,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore detector state saved with :meth:`state_dict`."""
        version = state.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported ConceptDetector state version: {version}")

        self.model = state.get("model")
        if isinstance(self.model, JointConceptModel):
            self.embedding_model = self.model.backbone
        else:
            self.embedding_model = state.get("embedding_model")
            if isinstance(self.embedding_model, nn.Module):
                self.embedding_model.to("cpu")
        if self.model is not None and isinstance(self.model, nn.Module):
            self.model.to("cpu")
            self.model.eval()

        self.calibration_params = state.get("calibration_params")
        self._eval_config = state.get("eval_config", {})
        self._n_concepts = state.get("n_concepts")

        summary = state.get("training_summary")
        if summary is not None:
            self.training_result = TrainerResult(
                model=None,
                history=copy.deepcopy(summary.get("history")),
                best_metric=summary.get("best_metric"),
            )
        else:
            self.training_result = None

    def save(
        self,
        path: str | os.PathLike,
        *,
        overwrite: bool = False,
        msg: bool = True,
    ) -> os.PathLike:
        """Persist the detector using concept_benchmark.ext.fileutils.save."""
        payload = {"version": 1, "state": self.state_dict()}
        return save_object(payload, path, overwrite=overwrite, msg=msg)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        map_location: str | torch.device | None = "cpu",
    ) -> "ConceptDetector":
        """Restore a detector previously saved with ``save``."""
        payload = load_object(path)
        if isinstance(payload, dict) and "state" in payload:
            state = payload["state"]
        elif isinstance(payload, dict) and "detector" in payload:
            detector = payload["detector"]
            if map_location is not None:
                detector.to(map_location)
            return detector
        else:
            state = payload
        detector = cls()
        detector.load_state_dict(state)
        if map_location is not None:
            detector.to(map_location)
        return detector

    def _default_head(
        self, feature_dim: int, hidden_dim: int, num_concepts: int
    ) -> nn.Module:
        """Build a simple MLP concept head when the caller does not provide one.

        Args:
            feature_dim: Size of the flattened backbone feature vector.
            hidden_dim: Width of the intermediate hidden layer.
            num_concepts: Number of concept logits to predict.

        Returns:
            nn.Module: Two-layer MLP that produces ``num_concepts`` logits.
        """
        layers = [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_concepts),
        ]
        return nn.Sequential(*layers)

    def _build_default_model(
        self,
        train_dataset: ConceptDatasetSample,
        *,
        device: torch.device,
        hidden_dim: int,
    ) -> nn.Module:
        """Infer feature dimensions and assemble a ``JointConceptModel``.

        Args:
            train_dataset: Dataset used to probe feature dimensionality.
            device: Device where the probe forward pass should execute.
            hidden_dim: Width of the default concept head hidden layer.

        Returns:
            nn.Module: Joint model combining the configured backbone and default
            head.

        Raises:
            ValueError: If ``train_dataset`` contains no samples or the backbone
                returns no features.
            TypeError: If the backbone output type is unsupported for shape
                inference.
        """
        sample_loader = train_dataset.loader(
            batch_size=min(8, max(1, getattr(train_dataset, "n", 8))),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        try:
            sample_X, _, _ = next(iter(sample_loader))
        except StopIteration:
            raise ValueError("Training dataset is empty; cannot build default model.")

        # Use a small probe batch to discover the size of the backbone features.
        tensor_X = _prepare_inputs(sample_X, device=torch.device("cpu"))

        feature_source = tensor_X
        if self.embedding_model is not None:
            module = self.embedding_model
            was_training = module.training
            try:
                param = next(module.parameters())
                original_device = param.device
            except StopIteration:
                original_device = torch.device("cpu")
            module = module.to(device)
            module.eval()
            with torch.no_grad():
                feature_source = module(_prepare_inputs(sample_X, device))
            if was_training:
                module.train()
            module.to(original_device)
        if isinstance(feature_source, (list, tuple)):
            if not feature_source:
                raise ValueError("Embedding model returned empty output.")
            feature_source = feature_source[0]
        if isinstance(feature_source, dict):
            if "logits" in feature_source:
                feature_source = feature_source["logits"]
            elif "last_hidden_state" in feature_source:
                feature_source = feature_source["last_hidden_state"][:, 0]
            else:
                raise TypeError("Cannot infer feature dimension from embedding output.")
        if isinstance(feature_source, torch.Tensor):
            feature_source = feature_source.detach().cpu()
        else:
            feature_source = torch.as_tensor(feature_source)
        flat = feature_source.view(feature_source.size(0), -1)
        feature_dim = int(flat.size(1))

        head = self._default_head(feature_dim, hidden_dim, train_dataset.n_concepts)
        flatten_inputs = self.embedding_model is None
        if flatten_inputs:
            # absorb flatten into head for raw features
            head = nn.Sequential(nn.Flatten(), head)

        return JointConceptModel(
            self.embedding_model, head, should_flatten=not flatten_inputs
        )

    def _set_trainable(self, freeze_backbone: bool) -> None:
        """Freeze or unfreeze the backbone parameters depending on user request."""
        if not isinstance(self.model, JointConceptModel):
            return
        backbone = self.model.backbone
        if backbone is None:
            return
        for param in backbone.parameters():
            param.requires_grad = not freeze_backbone

    def fit(
        self,
        train_dataset: ConceptDatasetSample,
        valid_dataset: ConceptDatasetSample,
        freeze_backbone: bool = False,
        embedding_params: dict | None = None,
        fit_params: dict | None = None,
        hidden_layer_size: int | None = 100,
        should_calibrate: bool = False,
        should_log_training: bool = False,
        log_interval: int | None = None,
        *,
        model: nn.Module | None = None,
        trainer: ConceptTrainer | None = None,
    ) -> None:
        """Train the detector and optionally calibrate the resulting logits.

        Parameters
        ----------
        train_dataset : ConceptDatasetSample
            Dataset providing supervised concept labels for optimisation.
        valid_dataset : ConceptDatasetSample
            Held-out dataset used for early stopping and calibration.
        freeze_backbone : bool
            When ``True`` the detector disables gradient updates on the
            backbone parameters.
        embedding_params : dict, optional
            Unused in the joint setup; accepted for parity with older call
            sites.
        fit_params : dict, optional
            Keyword overrides passed to the trainer. Recognised keys include
            ``epochs``, ``batch_size``, ``device``, ``lr_encoder``,
            ``lr_heads``, and ``optimizer_factory``.
        hidden_layer_size : int, optional
            Hidden width for the default MLP head when a custom model is not
            supplied.
        should_calibrate : bool
            When ``True`` runs ``calibrate`` on the validation split after
            training completes.
        should_log_training : bool
            Convenience flag that sets ``fit_params['verbose']``.
        log_interval : int, optional
            How often (in steps) to emit training loss when verbose logging
            is active.
        model : nn.Module, optional
            Optional joint model to train instead of constructing the default.
        trainer : ConceptTrainer, optional
            Custom trainer overriding ``self.trainer`` for this call.

        Raises
        ------
        ValueError
            If the training dataset is empty when constructing a default model.
        TypeError
            If the trainer returns an unexpected object.
        """
        if fit_params is None:
            fit_params = {}
        fit_params = dict(fit_params)
        fit_params.setdefault("verbose", bool(should_log_training))
        if log_interval is not None:
            fit_params.setdefault("log_interval", int(log_interval))

        from concept_benchmark.utils import determine_device, get_loader_config

        _loader_defaults = get_loader_config()
        training_device = torch.device(
            fit_params.get("device", None) or determine_device()
        )
        eval_batch = int(
            fit_params.get("eval_batch_size", fit_params.get("batch_size", _loader_defaults["batch_size"]))
        )
        # Snapshot evaluation configuration so inference and calibration share settings.
        self._eval_config = {
            "batch_size": eval_batch,
            "num_workers": fit_params.get("num_workers", _loader_defaults["num_workers"]),
            "pin_memory": fit_params.get("pin_memory", _loader_defaults["pin_memory"]),
            "device": training_device,
        }

        if model is not None:
            self.model = model
        if self.model is None:
            hidden = int(hidden_layer_size or fit_params.get("hidden_dim", 100))
            builder = self.model_builder or self._build_default_model
            self.model = builder(
                train_dataset, device=training_device, hidden_dim=hidden
            )

        self._n_concepts = train_dataset.n_concepts

        lr_encoder = fit_params.pop("lr_encoder", None)
        lr_heads = fit_params.pop("lr_heads", None)
        if (
            lr_encoder is not None or lr_heads is not None
        ) and "optimizer_factory" not in fit_params:

            def _make_optimizer(param_iterable):
                """Build an Adam optimizer honouring distinct backbone/head learning rates."""
                params_list = list(param_iterable)
                base_lr = fit_params.get("lr", lr_heads or lr_encoder or 1e-3)
                weight_decay = fit_params.get("weight_decay", 0.0)

                if isinstance(self.model, JointConceptModel):
                    groups = []
                    if self.model.backbone is not None:
                        enc_params = [
                            p
                            for p in self.model.backbone.parameters()
                            if p.requires_grad
                        ]
                        if enc_params:
                            groups.append(
                                {"params": enc_params, "lr": lr_encoder or base_lr}
                            )
                    head_params = [
                        p for p in self.model.head.parameters() if p.requires_grad
                    ]
                    if head_params:
                        groups.append(
                            {"params": head_params, "lr": lr_heads or base_lr}
                        )
                    if not groups:
                        groups = [{"params": params_list}]
                    return torch.optim.Adam(
                        groups, lr=base_lr, weight_decay=weight_decay
                    )

                trainable = [p for p in params_list if p.requires_grad] or params_list
                return torch.optim.Adam(
                    trainable,
                    lr=lr_heads or lr_encoder or base_lr,
                    weight_decay=weight_decay,
                )

            fit_params["optimizer_factory"] = _make_optimizer

        self._set_trainable(freeze_backbone=freeze_backbone)

        trainer_fn = trainer if trainer is not None else self.trainer
        result = trainer_fn(
            self.model,
            train_dataset,
            valid_dataset,
            num_concepts=train_dataset.n_concepts,
            params=fit_params,
        )

        trained_model, trainer_metadata = self._parse_trainer_output(result)
        self.model = trained_model
        if isinstance(self.model, JointConceptModel):
            self.embedding_model = self.model.backbone
        self.training_result = trainer_metadata
        self.model.eval()
        self.model.cpu()

        if should_calibrate:
            self.calibrate(valid_dataset)
        else:
            self.calibration_params = None

    def _parse_trainer_output(
        self, output: nn.Module | tuple[nn.Module, dict[str, Any]] | TrainerResult
    ) -> tuple[nn.Module, TrainerResult | None]:
        """Normalise trainer outputs into a `(model, TrainerResult|None)` tuple."""
        if isinstance(output, TrainerResult):
            return output.model, output
        if isinstance(output, tuple):
            model, meta = output
            if isinstance(meta, TrainerResult):
                return model, meta
            result = TrainerResult(
                model=model,
                history=getattr(meta, "history", meta),
                best_metric=getattr(meta, "best_metric", None),
            )
            return model, result
        if isinstance(output, nn.Module):
            return output, None
        raise TypeError(
            "Trainer must return an nn.Module, (nn.Module, info), or TrainerResult instance."
        )

    def calibrate(
        self,
        valid_dataset: ConceptDatasetSample,
    ) -> None:
        """Fit Platt-scaling parameters on validation logits."""
        if self.model is None or self._n_concepts is None:
            raise RuntimeError("Must call fit(...) before calibrating.")

        logits = self._predict_logits(valid_dataset)
        concepts = valid_dataset.C.copy()

        if (
            bool(getattr(valid_dataset, "concept_missing", False))
            and getattr(valid_dataset, "concept_missing_mask", None) is not None
        ):
            fill_value = getattr(valid_dataset, "concept_missing_fill_value", np.nan)
            if isinstance(fill_value, float) and np.isnan(fill_value):
                observed_mask = ~np.isnan(concepts)
            else:
                observed_mask = ~np.isclose(concepts, fill_value)
        else:
            observed_mask = np.ones_like(concepts, dtype=bool)

        params = []
        for i in range(self._n_concepts):
            observed = observed_mask[:, i]
            if not np.any(observed):
                params.append({"w": 1.0, "b": 0.0})
                continue
            z = logits[observed, i : i + 1]
            y = concepts[observed, i].astype(int)
            if np.unique(y).size < 2:
                params.append({"w": 1.0, "b": 0.0})
                continue
            lr = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
            lr.fit(z, y)
            params.append({"w": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})

        self.calibration_params = params

    def _predict_logits(self, dataset: ConceptDatasetSample) -> np.ndarray:
        """Run inference and collect raw concept logits."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        batch_size = int(self._eval_config.get("batch_size", 128))
        num_workers = self._eval_config.get("num_workers", 0)
        pin_memory = self._eval_config.get("pin_memory", False)
        device = torch.device(self._eval_config.get("device", "cpu"))

        loader = dataset.loader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model = self.model.to(device)
        model.eval()

        outputs = []
        with torch.no_grad():
            for batch_X, _, _ in loader:
                batch_X = _prepare_inputs(batch_X, device)
                logits = _extract_logits(model(batch_X))
                outputs.append(logits.detach().cpu().numpy())

        model.cpu()

        if not outputs:
            return np.zeros((0, self._n_concepts or 0), dtype=np.float32)
        return np.vstack(outputs)

    @property
    def n_concepts(self) -> int:
        if self._n_concepts is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return int(self._n_concepts)

    def predict(
        self,
        dataset: ConceptDatasetSample,
        embedding_params: dict | None = None,
        should_calibrate: bool | None = None,
    ) -> np.ndarray:
        """Predict concept probabilities for each sample.

        Returns an ``(N, n_concepts)`` array of probabilities in ``[0, 1]``,
        **not** binary predictions.  This differs from the sklearn convention
        where ``predict()`` returns class labels.

        To get binary concept predictions, threshold at 0.5::

            binary = (cd.predict(dataset) > 0.5).astype(int)

        If calibration was fitted during :meth:`fit`, it is applied by default.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. Please call fit() first."
            )

        logits = self._predict_logits(dataset)

        if should_calibrate is None:
            apply_cal = self.calibration_params is not None
        else:
            apply_cal = should_calibrate
            if apply_cal and self.calibration_params is None:
                raise RuntimeError(
                    "Calibration requested but not fitted. Call fit(..., should_calibrate=True)."
                )

        if apply_cal and self.calibration_params is not None:
            w = np.array(
                [
                    (p.get("w", 1.0) if p is not None else 1.0)
                    for p in self.calibration_params
                ],
                dtype=np.float32,
            )
            b = np.array(
                [
                    (p.get("b", 0.0) if p is not None else 0.0)
                    for p in self.calibration_params
                ],
                dtype=np.float32,
            )
            z_scaled = logits * w.reshape(1, -1) + b.reshape(1, -1)
            return 1.0 / (1.0 + np.exp(-z_scaled))

        return 1.0 / (1.0 + np.exp(-logits))


class FrontEndModel:
    """Logistic-regression classifier mapping binary concepts to labels.

    This is the "downstream classifier" in the CBM architecture: it takes
    thresholded concept predictions (0/1) and produces a final label.  The
    default back-end is :class:`sklearn.linear_model.LogisticRegression`.
    """

    def __init__(self, **kwargs) -> None:
        self.model = None

    def fit(
        self, C: np.ndarray, y: np.ndarray, fit_params: dict | None = None
    ) -> None:
        """Fit the logistic-regression back-end.

        Parameters
        ----------
        C : np.ndarray
            Binary concept array of shape ``(N, n_concepts)``.
        y : np.ndarray
            Label array of shape ``(N,)``.
        fit_params : dict, optional
            Keyword arguments forwarded to
            :class:`~sklearn.linear_model.LogisticRegression`.  Defaults:
            ``solver="lbfgs"``, ``max_iter=1000``, ``C=1.0``.
        """
        lr_params = {
            "random_state": 42,
            "max_iter": 1000,
            "solver": "lbfgs",
            "C": 1.0,
        }

        if fit_params:
            lr_params.update(fit_params)

        self.model = LogisticRegression(**lr_params)

        self.model.fit(C, y)

    def predict(self, C: np.ndarray) -> np.ndarray:
        """Predict labels from concept values.

        Args:
            C: Binary concept array of shape ``(N, n_concepts)``.  Values
                should be 0/1 (thresholded), not raw probabilities.

        Returns:
            Integer label array of shape ``(N,)``.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. Please call fit() first."
            )

        return self.model.predict(C)

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        """Predict label probabilities from concept values.

        Args:
            C: Binary concept array of shape ``(N, n_concepts)``.

        Returns:
            Probability array of shape ``(N, n_classes)``.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. Please call fit() first."
            )

        return self.model.predict_proba(C)


class ConceptBasedModel:
    """Two-stage concept bottleneck model.

    Composes a :class:`ConceptDetector` (X → concept probabilities) with a
    :class:`FrontEndModel` (binary concepts → labels).  When ``should_propagate``
    is ``True``, concept uncertainty is propagated to the label prediction
    via Monte Carlo sampling instead of hard-thresholding at 0.5.

    Parameters
    ----------
    concept_detector : ConceptDetector, optional
        First stage that maps raw inputs to concept probabilities.
    label_predictor : FrontEndModel, optional
        Second stage that maps binary concepts to labels.
    should_propagate : bool
        If ``True``, use MC sampling to propagate concept uncertainty
        through the downstream classifier.
    mc_mode : ``{"auto", "mc", "exact"}``
        ``"auto"`` selects exact enumeration when feasible (≤
        ``mc_exact_threshold`` binary assignments), otherwise MC sampling.
    mc_samples, mc_max_samples, mc_chunk_size, mc_tolerance, mc_exact_threshold
        Control the MC sampling budget and convergence tolerance.
    random_state : int, optional
        Seed for reproducible MC sampling.
    """

    def __init__(
        self,
        concept_detector: ConceptDetector | None = None,
        label_predictor: FrontEndModel | None = None,
        should_propagate: bool = False,
        mc_mode: str = "auto",
        mc_samples: int = 1024,
        mc_max_samples: int = 16384,
        mc_chunk_size: int = 2048,
        mc_tolerance: float = 1e-3,
        random_state: int | None = None,
        mc_exact_threshold: int = 4096,
        **kwargs,
    ) -> None:
        if concept_detector:
            assert isinstance(concept_detector, ConceptDetector), (
                "concept_detector must be an instance of ConceptDetector or its subclass."
            )

        if label_predictor:
            assert isinstance(label_predictor, FrontEndModel), (
                "label_predictor must be an instance of FrontEndModel or FrontEndModelCVXPY."
            )

        self.concept_detector = (
            concept_detector if concept_detector else ConceptDetector(**kwargs)
        )
        self.label_predictor = label_predictor if label_predictor else FrontEndModel()

        self._propagate = should_propagate
        self._concept_possibilities = None
        self._y_proba_all_concepts = None

        # MC configuration
        assert mc_mode in {"auto", "mc", "exact"}
        self._mc_mode = mc_mode
        self._mc_samples = int(mc_samples)
        self._mc_max_samples = int(mc_max_samples)
        self._mc_chunk_size = int(mc_chunk_size)
        self._mc_tol = float(mc_tolerance)
        self._random_state = random_state
        self._mc_exact_threshold = int(mc_exact_threshold)

    @property
    def should_propagate(self) -> bool:
        return self._propagate

    @property
    def concept_possibilities(self) -> np.ndarray | None:
        """
        Return all possible concept combinations.
        """
        return self._concept_possibilities

    @property
    def y_proba_all_concepts(self) -> dict | None:
        """
        Return predicted probabilities for all concept combinations.
        """
        return self._y_proba_all_concepts

    def fit(
        self,
        train_dataset: ConceptDatasetSample,
        valid_dataset: ConceptDatasetSample,
        freeze_backbone: bool = True,
        *,
        concept_fit_params: dict | None = None,
        concept_embed_params: dict | None = None,
        front_fit_params: dict | None = None,
        should_calibrate: bool = False,
        should_log_training: bool = False,
        log_interval: int | None = None,
    ) -> None:
        """
        Fit the concept detector and front-end model.

        Args:
            train_dataset: Training split.
            valid_dataset: Validation split (used for concept detector early stopping/calibration).
            freeze_backbone: If True, skip (re)training the concept detector and only fit the front-end model.
            concept_fit_params: Dict forwarded to ConceptDetector.fit(..., fit_params=...).
            concept_embed_params: Dict forwarded to dataset.embed(...); passed via ConceptDetector.fit(..., embedding_params=...).
            front_fit_params: Dict forwarded to FrontEndModel.fit(..., fit_params=...).
            should_calibrate: If True, perform per-concept calibration after training detector.
        """
        # Ensure concept_fit_params dict exists and inject logging toggles if set
        if concept_fit_params is None:
            concept_fit_params = {}
        concept_fit_params.setdefault("verbose", bool(should_log_training))
        if log_interval is not None:
            concept_fit_params.setdefault("log_interval", int(log_interval))

        if not freeze_backbone:
            self.concept_detector.fit(
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                freeze_backbone=freeze_backbone,
                embedding_params=concept_embed_params,
                fit_params=concept_fit_params,
                should_calibrate=should_calibrate,
                should_log_training=should_log_training,
                log_interval=log_interval,
            )

        C_train = train_dataset.C  # independent training
        y_train = train_dataset.y
        missing_enabled = bool(getattr(train_dataset, "concept_missing", False))
        missing_mask = getattr(train_dataset, "concept_missing_mask", None)
        if missing_enabled:
            if missing_mask is None:
                fill_value = getattr(
                    train_dataset, "concept_missing_fill_value", np.nan
                )
                if isinstance(fill_value, float) and np.isnan(fill_value):
                    missing_mask = np.isnan(C_train)
                else:
                    missing_mask = np.isclose(C_train, fill_value)
            observed_rows = ~missing_mask.any(axis=1)
            if not np.all(observed_rows):
                dropped = int((~observed_rows).sum())
                warnings.warn(
                    f"Dropping {dropped} samples with missing concepts for front-end training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if not np.any(observed_rows):
                raise ValueError(
                    "No fully-observed concept rows available for front-end training."
                )
            C_train = C_train[observed_rows]
            y_train = y_train[observed_rows]

        self.label_predictor.fit(C_train, y_train, fit_params=front_fit_params)

        if self.should_propagate:
            # Only prepare exact propagation tables if we'll use exact mode
            try:
                n_concepts = self.concept_detector.n_concepts
            except AttributeError:
                n_concepts = None
            if n_concepts is not None and self._should_use_exact(n_concepts):
                self._prep_propagation()

    def predict(
        self,
        dataset: ConceptDatasetSample,
        should_propagate: bool | None = None,
    ) -> np.ndarray:
        """
        Predict label for the dataset.
        """
        probas = self.predict_proba(
            dataset,
            should_propagate=should_propagate,
        )
        preds = np.argmax(probas, axis=1)

        return preds

    def predict_proba(
        self,
        dataset: ConceptDatasetSample,
        should_propagate: bool | None = None,
        return_concepts: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict probabilities for the dataset.
        """
        concept_preds = self.concept_detector.predict(dataset)

        # Override object's should_propagate if specified
        should_propagate = (
            self.should_propagate if should_propagate is None else should_propagate
        )

        if should_propagate:
            n_concepts = concept_preds.shape[1]
            if self._should_use_exact(n_concepts):
                y_prob = self._propagate_predict_proba(concept_preds)
            else:
                y_prob = self._propagate_predict_proba_mc(concept_preds)
            return (y_prob, concept_preds) if return_concepts else y_prob

        binary_concept_preds = (concept_preds > 0.5).astype(np.float32)
        pred_y_prob = self.label_predictor.predict_proba(binary_concept_preds)

        out = pred_y_prob if not return_concepts else (pred_y_prob, concept_preds)

        return out

    def _propagate_predict_proba(
        self,
        concept_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Predict probabilities using concept propagation.
        """
        if self._concept_possibilities is None or self._y_proba_all_concepts is None:
            self._prep_propagation()

        # Vectorized propagation over all samples and concept combinations
        # Shapes:
        #   concept_preds: (N, C)
        #   concept_possibilities:  (M, C) with binary {0,1}
        #   y_proba_all_concepts: dict[(C,)-> (1,K)] -> stacked to (M, K)

        # Ensure concept combination order aligns with y_proba matrix rows
        combs = self._concept_possibilities  # (M, C)
        # Stack dict values in the same order as combs
        y_mat = np.vstack(
            [
                np.asarray(self._y_proba_all_concepts[tuple(c)]).reshape(1, -1)
                for c in combs
            ]
        )  # (M, K)

        P = np.asarray(concept_preds, dtype=np.float64)  # (N, C)
        # Clip for numerical stability when taking logs
        P = np.clip(P, 1e-9, 1.0 - 1e-9)

        # Compute log-weights for each sample and concept combination:
        # log w_ij = sum_k [ c_jk * log p_ik + (1-c_jk) * log(1-p_ik) ]
        logP = np.log(P)  # (N, C)
        log1mP = np.log1p(-P)  # (N, C)
        A = combs.T.astype(np.float64)  # (C, M)
        logW = logP @ A + log1mP @ (1.0 - A)  # (N, M)
        W = np.exp(logW)  # (N, M)

        # Aggregate over concept combinations to get class probabilities
        # result: (N, K)
        out = W @ y_mat

        return out

    def _should_use_exact(self, n_concepts: int) -> bool:
        if self._mc_mode == "exact":
            return True
        if self._mc_mode == "mc":
            return False
        # auto: use exact if 2^C <= threshold
        try:
            return (2**n_concepts) <= self._mc_exact_threshold
        except OverflowError:
            return False

    def _propagate_predict_proba_mc(self, concept_preds: np.ndarray) -> np.ndarray:
        """
        Monte Carlo propagation: approximate E_y[ y | concept probabilities ]
        by sampling concept vectors and averaging front-end predictions.
        """
        P = np.asarray(concept_preds, dtype=np.float64)
        P = np.clip(P, 1e-9, 1.0 - 1e-9)
        N, C = P.shape

        # Initialize accumulators
        counts = np.zeros(N, dtype=np.int64)
        sum_acc = None  # will be (N, K)
        sumsq_acc = None  # will be (N, K)
        done = np.zeros(N, dtype=bool)

        # RNG: deterministic only if seed provided
        rng = (
            np.random.default_rng(self._random_state)
            if self._random_state is not None
            else np.random.default_rng()
        )

        target_samples = max(1, self._mc_samples)
        max_samples = max(target_samples, self._mc_max_samples)
        chunk_size = max(1, self._mc_chunk_size)

        pbar = tqdm(total=N, desc="MC concept propagation", unit="samples")
        while True:
            active_idx = np.where(~done)[0]
            if active_idx.size == 0:
                break

            # Determine per-loop chunk size constrained by remaining budget per active example
            remaining = max_samples - counts[active_idx]
            if remaining.min() <= 0:
                # Reached max_samples for some/all active; mark exhausted as done
                done[active_idx[remaining <= 0]] = True
                continue
            s = int(min(chunk_size, remaining.min()))

            # Sample Bernoulli for active examples: shape (A, s, C)
            P_active = P[active_idx]  # (A, C)
            Z = (rng.random((P_active.shape[0], s, C)) < P_active[:, None, :]).astype(
                np.float32
            )
            Z_flat = Z.reshape(-1, C)

            # Deduplicate concept vectors to reduce model calls
            try:
                uniq, inv = np.unique(Z_flat, axis=0, return_inverse=True)
                y_uniq = self.label_predictor.predict_proba(uniq)
                Y_flat = y_uniq[inv]
            except (TypeError, ValueError):
                # Fallback without deduplication (e.g. non-sortable dtypes)
                Y_flat = self.label_predictor.predict_proba(Z_flat)

            # Reshape back to (A, s, K)
            Y = Y_flat.reshape(P_active.shape[0], s, -1)
            if sum_acc is None:
                K = Y.shape[2]
                sum_acc = np.zeros((N, K), dtype=np.float64)
                sumsq_acc = np.zeros((N, K), dtype=np.float64)

            # Update accumulators
            chunk_sum = Y.sum(axis=1)  # (A, K)
            chunk_sumsq = (Y**2).sum(axis=1)  # (A, K)
            sum_acc[active_idx] += chunk_sum
            sumsq_acc[active_idx] += chunk_sumsq
            counts[active_idx] += s

            # Check convergence for active examples
            means = sum_acc[active_idx] / counts[active_idx][:, None]
            vars_ = sumsq_acc[active_idx] / counts[active_idx][:, None] - means**2
            np.maximum(vars_, 0.0, out=vars_)  # numerical safety
            se = np.sqrt(vars_ / counts[active_idx][:, None])

            # Aggregate SE across classes (flexible; default: max over classes)
            # This can be swapped out for other strategies if needed.
            se_agg = np.max(se, axis=1)
            conv_mask = (se_agg <= self._mc_tol) & (
                counts[active_idx] >= target_samples
            )
            newly_done = active_idx[conv_mask]
            done[newly_done] = True
            pbar.update(len(newly_done))

            # Also stop if everyone has at least target_samples and we've hit that
            if np.all(counts >= target_samples) and done.all():
                break

            # If some have reached max_samples after this update, mark done
            newly_maxed = ~done & (counts >= max_samples)
            pbar.update(int(newly_maxed.sum()))
            done |= newly_maxed

        pbar.update(N - pbar.n)  # ensure bar reaches 100%
        pbar.close()

        # Final means as output
        if sum_acc is None:
            # No sampling happened (edge case), fallback to deterministic round
            return self.label_predictor.predict_proba((P > 0.5).astype(np.float32))
        out = sum_acc / counts[:, None]
        return out

    def _calc_concept_probas(self, concept_probas: np.ndarray) -> dict:
        """
        Calculate probabilities for each concept combination
        given concept probabilities (from ConceptDetector).

        Args:
            concept_probas (np.ndarray): Probabilities for each concept (single instance).

        Returns:
            dict: A dictionary where keys are tuples of concept combinations
                  and values are their corresponding probabilities.
        """
        probas = {}
        for c in self.concept_possibilities:
            probas[tuple(c)] = np.prod(
                (concept_probas**c) * (1 - concept_probas) ** (1 - c)
            )

        return probas

    def _gen_concept_possibilities(self):
        """
        Generate all possible concept combinations.
        """
        n_concepts = self.concept_detector.n_concepts
        all_poss = np.array(list(itertools.product([0, 1], repeat=n_concepts)))

        return all_poss

    def _pred_y_proba_concept_possibilities(self) -> dict:
        """
        Predict probabilities for all concept combination.
        """
        all_y_probas = {}

        for c in tqdm(self.concept_possibilities):
            pr = self.label_predictor.predict_proba(c.reshape(1, -1))
            all_y_probas[tuple(c)] = pr

        return all_y_probas

    def _prep_propagation(self):
        """
        Prepare for propagation by generating concept possibilities and
        predicting probabilities for all concept combinations.
        """
        self._concept_possibilities = self._gen_concept_possibilities()
        self._y_proba_all_concepts = self._pred_y_proba_concept_possibilities()
