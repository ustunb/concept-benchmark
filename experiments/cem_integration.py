from __future__ import annotations

__all__ = [
    "CEMDependencyError",
    "CEMSampleAdapterDataset",
    "CEMBenchmarkModel",
    "ECBMBenchmarkModel",
    "ProbCBMBenchmarkModel",
    "compute_ecbm_interpretation_summary",
    "make_cem_loader",
    "require_cem_dependencies",
    "train_cem_model",
    "train_ecbm_model",
    "train_probcbm_model",
]

import copy
import importlib
import inspect
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.utils import determine_device, get_loader_config
from experiments.models import ConceptBasedModel, ConceptDetector, FrontEndModel


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_CEM_CHECKOUT = _REPO_ROOT / "third_party" / "cem"


class CEMDependencyError(ImportError):
    """Raised when the optional official CEM dependencies are unavailable."""


@dataclass(frozen=True)
class _CEMDependencies:
    pl: Any
    ConceptEmbeddingModel: type
    ProbCBM: type
    train_prob_cbm: Any | None = None


@dataclass
class _PredictionCache:
    dataset_id: int | None
    concept_probs: np.ndarray
    label_probs: np.ndarray
    pos_embeddings: torch.Tensor | None = None
    neg_embeddings: torch.Tensor | None = None
    probcbm_pred_embeddings: torch.Tensor | None = None
    probcbm_pred_mean: torch.Tensor | None = None
    probcbm_pred_logsigma: torch.Tensor | None = None
    ecbm_features: torch.Tensor | None = None


def _slice_prediction_cache(
    cache: _PredictionCache,
    row_indices: np.ndarray,
) -> _PredictionCache:
    tensor_index = torch.as_tensor(row_indices, dtype=torch.long)
    return _PredictionCache(
        dataset_id=cache.dataset_id,
        concept_probs=cache.concept_probs[row_indices],
        label_probs=cache.label_probs[row_indices],
        pos_embeddings=(
            None if cache.pos_embeddings is None else cache.pos_embeddings.index_select(0, tensor_index)
        ),
        neg_embeddings=(
            None if cache.neg_embeddings is None else cache.neg_embeddings.index_select(0, tensor_index)
        ),
        probcbm_pred_embeddings=(
            None
            if cache.probcbm_pred_embeddings is None
            else cache.probcbm_pred_embeddings.index_select(0, tensor_index)
        ),
        probcbm_pred_mean=(
            None
            if cache.probcbm_pred_mean is None
            else cache.probcbm_pred_mean.index_select(0, tensor_index)
        ),
        probcbm_pred_logsigma=(
            None
            if cache.probcbm_pred_logsigma is None
            else cache.probcbm_pred_logsigma.index_select(0, tensor_index)
        ),
        ecbm_features=(
            None
            if cache.ecbm_features is None
            else cache.ecbm_features.index_select(0, tensor_index)
        ),
    )


def _ensure_local_cem_checkout_on_path() -> None:
    if (_LOCAL_CEM_CHECKOUT / "cem").is_dir():
        checkout = str(_LOCAL_CEM_CHECKOUT)
        if checkout not in sys.path:
            sys.path.insert(0, checkout)


def _patch_reduce_on_plateau_for_torch_compat() -> None:
    scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    if getattr(scheduler_cls, "_concept_benchmark_compat", False):
        return
    sig = inspect.signature(scheduler_cls.__init__)
    if "verbose" in sig.parameters:
        return

    class _CompatReduceLROnPlateau(scheduler_cls):
        _concept_benchmark_compat = True

        def __init__(self, optimizer, *args, verbose: bool = False, **kwargs):
            super().__init__(optimizer, *args, **kwargs)

    torch.optim.lr_scheduler.ReduceLROnPlateau = _CompatReduceLROnPlateau


def _format_install_message(exc: Exception | None = None) -> str:
    hint = (
        "Optional CEM/ProbCBM support requires the official `cem` package and "
        "`pytorch_lightning`. Install with:\n"
        "  python -m pip install pytorch-lightning 'torchmetrics<1.0'\n"
        "  git clone https://github.com/mateoespinosa/cem.git third_party/cem\n"
        "  python -m pip install -r third_party/cem/requirements.txt\n"
        "or use the repo helper once available.\n"
        "This error is only raised when `cem` or `probcbm` is explicitly requested."
    )
    if exc is None:
        return hint
    return f"{hint}\nOriginal import error: {type(exc).__name__}: {exc}"


def require_cem_dependencies(
    *,
    include_probcbm_training_helper: bool = False,
) -> _CEMDependencies:
    """Import official CEM dependencies lazily and raise a clean error on failure."""

    _ensure_local_cem_checkout_on_path()
    _patch_reduce_on_plateau_for_torch_compat()

    try:
        pl = importlib.import_module("pytorch_lightning")
        ConceptEmbeddingModel = importlib.import_module(
            "cem.models.cem"
        ).ConceptEmbeddingModel
        ProbCBM = importlib.import_module("cem.models.probcbm").ProbCBM
    except Exception as exc:  # pragma: no cover - exercised by targeted unit test
        raise CEMDependencyError(_format_install_message(exc)) from exc

    train_prob_cbm = None
    if include_probcbm_training_helper:
        try:
            train_prob_cbm = importlib.import_module(
                "cem.train.train_prob_cbm"
            ).train_prob_cbm
        except Exception:
            train_prob_cbm = None

    return _CEMDependencies(
        pl=pl,
        ConceptEmbeddingModel=ConceptEmbeddingModel,
        ProbCBM=ProbCBM,
        train_prob_cbm=train_prob_cbm,
    )


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _to_label_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.long).reshape(())
    arr = np.asarray(value)
    if arr.ndim == 0:
        return torch.tensor(int(arr), dtype=torch.long)
    return torch.as_tensor(arr, dtype=torch.long)


class CEMSampleAdapterDataset(Dataset):
    """Thin wrapper that reorders benchmark samples from `(x, c, y)` to `(x, y, c)`."""

    def __init__(self, sample: ConceptDatasetSample) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return len(self.sample)

    def __getitem__(self, idx: int):
        x, c, y = self.sample[idx]
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32, copy=False)
        y = _to_label_tensor(y)
        c = _to_float_tensor(c)
        return x, y, c


def make_cem_loader(
    sample: ConceptDatasetSample,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a dataloader in the tuple order expected by the official package."""

    dataset = CEMSampleAdapterDataset(sample)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class RobotImageBackbone(nn.Module):
    """Small CNN aligned with the repo's robot image baselines."""

    def __init__(self, *, input_size: int, output_dim: int) -> None:
        super().__init__()
        if input_size >= 128:
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
        self.proj = nn.Linear(64 * 4 * 4, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class TabularBackbone(nn.Module):
    """Simple MLP for generic tabular concept benchmarks."""

    def __init__(self, *, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class SudokuTabularBackbone(nn.Module):
    """Group-pooling backbone for sudoku tabular boards.

    Mirrors the structure of ``GroupPoolingConceptSudokuCNN`` so that
    CEM/ProbCBM operate on the same per-row / per-column / per-block
    features the regular CBM consumes. The backbone produces a single
    global feature vector of size ``output_dim`` for CEM's per-concept
    embedding layer.
    """

    _NUM_DIGITS = 10
    _N_GROUPS = 27  # 9 rows + 9 cols + 9 blocks

    def __init__(
        self,
        *,
        output_dim: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(self._NUM_DIGITS, embedding_dim, bias=False)
        self.group_head = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.project = nn.Linear(self._N_GROUPS * hidden_dim, output_dim)

    @staticmethod
    def _pool_groups(x: torch.Tensor, dim: int) -> torch.Tensor:
        mean = x.mean(dim=dim)
        maxv = x.amax(dim=dim)
        return torch.cat([mean, maxv], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(x.long(), self._NUM_DIGITS).float()
        x = self.embedding(x)  # (N, 81, D)
        x = x.view(x.size(0), 9, 9, -1)  # (N, 9, 9, D)

        row_feats = self._pool_groups(x, dim=2)  # (N, 9, 2D)
        col_feats = self._pool_groups(x, dim=1)  # (N, 9, 2D)

        blocks = x.view(x.size(0), 3, 3, 3, 3, x.size(-1))
        blocks = blocks.permute(0, 1, 3, 2, 4, 5).contiguous()
        block_cells = blocks.view(x.size(0), 9, 9, x.size(-1))
        block_feats = self._pool_groups(block_cells, dim=2)  # (N, 9, 2D)

        groups = torch.cat([row_feats, col_feats, block_feats], dim=1)  # (N, 27, 2D)
        per_group = self.group_head(groups)  # (N, 27, hidden_dim)
        flat = per_group.view(per_group.size(0), -1)  # (N, 27*hidden_dim)
        return self.project(flat)


def _default_cem_output_dim(config: Any | None) -> int:
    emb_size = int(getattr(config, "cem_emb_size", 16))
    return max(64, emb_size * 8)


def _infer_backbone_spec(
    sample: ConceptDatasetSample,
    *,
    benchmark: str,
    config: Any | None,
) -> dict[str, Any]:
    data_type = sample.meta.get("data_type", "tabular")

    # Sudoku CBM/CEM/ProbCBM always operate on the tabular (one-hot board)
    # representation, even when the dataset was generated from images via OCR.
    if benchmark == "sudoku":
        return {
            "kind": "sudoku_tabular",
            "default_output_dim": _default_cem_output_dim(config),
        }

    if data_type == "image":
        input_size = int(sample.meta.get("resolution", 32))
        try:
            x0, _, _ = sample[0]
            if isinstance(x0, torch.Tensor) and x0.ndim >= 3:
                input_size = int(x0.shape[-1])
        except Exception:
            pass
        return {
            "kind": "robot_image",
            "input_size": input_size,
            "default_output_dim": _default_cem_output_dim(config),
        }

    x0 = np.asarray(sample.X[0])
    return {
        "kind": "tabular",
        "input_dim": int(x0.reshape(-1).shape[0]),
        "hidden_dim": 128,
        "default_output_dim": _default_cem_output_dim(config),
    }


def _make_backbone_factory(backbone_spec: dict[str, Any]):
    kind = backbone_spec["kind"]
    default_output_dim = int(backbone_spec.get("default_output_dim", 128))

    def factory(output_dim: int | None = None):
        used_output_dim = int(output_dim or default_output_dim)
        if kind == "robot_image":
            return RobotImageBackbone(
                input_size=int(backbone_spec["input_size"]),
                output_dim=used_output_dim,
            )
        if kind == "sudoku_tabular":
            return SudokuTabularBackbone(output_dim=used_output_dim)
        if kind == "tabular":
            return TabularBackbone(
                input_dim=int(backbone_spec["input_dim"]),
                hidden_dim=int(backbone_spec.get("hidden_dim", 128)),
                output_dim=used_output_dim,
            )
        raise ValueError(f"Unsupported backbone kind: {kind!r}")

    return factory


def _to_numpy_task_proba(outputs: torch.Tensor, n_tasks: int) -> np.ndarray:
    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(-1)
    if n_tasks <= 2 and outputs.shape[-1] == 1:
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        return np.concatenate([1.0 - probs, probs], axis=1).astype(np.float32)

    detached = outputs.detach()
    if detached.numel() and bool(
        torch.all((detached >= -1e-6) & (detached <= 1.0 + 1e-6))
    ):
        row_sums = detached.sum(dim=1)
        if bool(torch.all(torch.isfinite(row_sums))) and bool(
            torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3)
        ):
            return detached.cpu().numpy().astype(np.float32)

    return torch.softmax(detached, dim=1).cpu().numpy().astype(np.float32)


def _stack_numpy(chunks: list[np.ndarray], *, cols: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, cols), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _stack_tensors(chunks: list[torch.Tensor] | None) -> torch.Tensor | None:
    if not chunks:
        return None
    return torch.cat(chunks, dim=0).cpu()


def _soft_cross_entropy_from_probs(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    target_probs = torch.clamp(target_probs, min=1e-6)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _safe_binary_logit(probs: torch.Tensor) -> torch.Tensor:
    probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
    return torch.log(probs) - torch.log1p(-probs)


def _safe_class_logit(probs: torch.Tensor) -> torch.Tensor:
    probs = torch.clamp(probs, min=1e-6)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.log(probs)


def _prepare_batch_features(batch_x: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(batch_x, torch.Tensor):
        return batch_x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(batch_x, dtype=torch.float32, device=device)


def _prepare_batch_concepts(batch_c: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(batch_c, torch.Tensor):
        return batch_c.to(device=device, dtype=torch.float32)
    return torch.as_tensor(batch_c, dtype=torch.float32, device=device)


def _prepare_batch_labels(batch_y: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(batch_y, torch.Tensor):
        return batch_y.to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(batch_y, dtype=torch.long, device=device).reshape(-1)


def _device_to_pl_args(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        return {"accelerator": "gpu", "devices": 1}
    # pytorch-lightning <2.0 does not support MPS; fall back to CPU
    return {"accelerator": "cpu", "devices": 1}


def _effective_pl_device(device: torch.device) -> torch.device:
    """Return the device pytorch-lightning will actually use."""
    if device.type == "cuda":
        return device
    # PL <2.0 doesn't support MPS — always CPU
    return torch.device("cpu")


def _build_trainer(
    *,
    pl_module: Any,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> Any:
    callbacks = []
    if patience > 0:
        callbacks.append(
            pl_module.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,
            )
        )
    return pl_module.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=False,
        **_device_to_pl_args(device),
    )


class _OfficialConceptDetectorAdapter(ConceptDetector):
    def __init__(self, owner: "_OfficialBenchmarkModelBase") -> None:
        super().__init__(model=None)
        self._owner = owner
        self._n_concepts = owner.n_concepts

    def predict(self, dataset: ConceptDatasetSample, **kwargs) -> np.ndarray:
        return (self.predict_proba(dataset, **kwargs) >= 0.5).astype(int)

    def predict_proba(self, dataset: ConceptDatasetSample, **kwargs) -> np.ndarray:
        _, concept_probs = self._owner._predict_dataset(dataset, cache=True)
        return concept_probs


class _OfficialFrontEndAdapter(FrontEndModel):
    _kflip_fast_path = False
    supports_aligned_concept_replay = True

    def __init__(self, owner: "_OfficialBenchmarkModelBase") -> None:
        super().__init__()
        self._owner = owner

    def fit(self, C: np.ndarray, y: np.ndarray, fit_params: dict | None = None) -> None:
        raise RuntimeError(
            "Official CEM/ProbCBM frontends are trained jointly with the wrapped model."
        )

    def predict(self, C: np.ndarray) -> np.ndarray:
        return self.predict_proba(C).argmax(axis=1)

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        return self._owner.predict_proba_from_concepts(C)

    def predict_proba_from_concepts(self, C: np.ndarray, **kwargs) -> np.ndarray:
        return self._owner.predict_proba_from_concepts(C, **kwargs)


class _OfficialBenchmarkModelBase(ConceptBasedModel):
    family = "official"
    supports_aligned_concept_replay = True

    def __init__(
        self,
        *,
        official_model: Any,
        benchmark: str,
        concept_names: list[str],
        class_names: list[str],
        backbone_spec: dict[str, Any],
        model_init_kwargs: dict[str, Any],
        eval_config: dict[str, Any],
        training_summary: dict[str, Any] | None = None,
    ) -> None:
        self.official_model = official_model
        self.benchmark = benchmark
        self.concept_names = list(concept_names)
        self.class_names = list(class_names)
        self.backbone_spec = copy.deepcopy(backbone_spec)
        self.model_init_kwargs = copy.deepcopy(model_init_kwargs)
        self.eval_config = copy.deepcopy(eval_config)
        self.training_summary = copy.deepcopy(training_summary or {})
        self._prediction_cache: _PredictionCache | None = None
        self._dependency_error_message = _format_install_message()

        concept_detector = _OfficialConceptDetectorAdapter(self)
        label_predictor = _OfficialFrontEndAdapter(self)
        super().__init__(
            concept_detector=concept_detector,
            label_predictor=label_predictor,
            should_propagate=False,
        )

    @property
    def n_concepts(self) -> int:
        return len(self.concept_names)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def fit(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} is already trained. Use train_{self.family}_model(...)."
        )

    def _require_official_model(self) -> Any:
        if self.official_model is None:
            raise CEMDependencyError(self._dependency_error_message)
        return self.official_model

    def _loader_kwargs(self) -> dict[str, Any]:
        defaults = get_loader_config()
        return {
            "batch_size": int(self.eval_config.get("batch_size", defaults["batch_size"])),
            "num_workers": int(
                self.eval_config.get("num_workers", defaults["num_workers"])
            ),
            "pin_memory": bool(
                self.eval_config.get("pin_memory", defaults["pin_memory"])
            ),
        }

    def _inference_device(self) -> torch.device:
        configured = self.eval_config.get("device")
        if configured is None:
            return _effective_pl_device(determine_device())
        return _effective_pl_device(torch.device(configured))

    def predict_proba(
        self,
        dataset: ConceptDatasetSample,
        should_propagate: bool | None = None,
        return_concepts: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        y_prob, concept_probs = self._predict_dataset(dataset, cache=True)
        return (y_prob, concept_probs) if return_concepts else y_prob

    def predict(
        self,
        dataset: ConceptDatasetSample,
        should_propagate: bool | None = None,
    ) -> np.ndarray:
        return self.predict_proba(dataset, should_propagate=should_propagate).argmax(
            axis=1
        )

    def predict_proba_from_concepts(
        self,
        concepts: np.ndarray,
        *,
        dataset: ConceptDatasetSample | None = None,
        row_indices: np.ndarray | None = None,
        source_indices: np.ndarray | None = None,
        baseline_concepts: np.ndarray | None = None,
        intervention_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if dataset is not None:
            self._predict_dataset(dataset, cache=True)
        cache = self._prediction_cache
        if cache is None:
            raise RuntimeError(
                "No official-model context is cached. Call concept_detector.predict_proba(dataset) "
                "or predict_proba(dataset) before re-evaluating labels from intervened concepts."
            )
        concepts = np.asarray(concepts, dtype=np.float32)
        if concepts.ndim == 1:
            concepts = concepts.reshape(1, -1)
        if concepts.ndim != 2 or concepts.shape[1] != self.n_concepts:
            raise ValueError(
                "Concept matrix must have shape (N, n_concepts), got "
                f"{concepts.shape} for n_concepts={self.n_concepts}."
            )
        if row_indices is not None and source_indices is not None:
            raise ValueError("Pass only one of row_indices or source_indices.")
        if source_indices is not None:
            row_indices = source_indices

        if row_indices is None:
            if concepts.shape != cache.concept_probs.shape:
                raise ValueError(
                    "Row-aligned replay requires row_indices/source_indices when the concept "
                    "matrix does not cover the full cached dataset in the original row order. "
                    f"Got {concepts.shape} and expected {cache.concept_probs.shape}."
                )
            if baseline_concepts is None:
                if not np.allclose(
                    concepts,
                    cache.concept_probs,
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise ValueError(
                        "Full-dataset replay without dataset/row_indices is only safe when "
                        "replaying the cached concept predictions exactly. Pass dataset=..., "
                        "or pass row_indices/source_indices together with baseline_concepts "
                        "for modified concept matrices."
                    )
            else:
                baseline_candidate = np.asarray(baseline_concepts, dtype=np.float32)
                if baseline_candidate.ndim == 1:
                    baseline_candidate = baseline_candidate.reshape(1, -1)
                if baseline_candidate.shape != cache.concept_probs.shape:
                    raise ValueError(
                        "Full-dataset baseline_concepts must match the cached concept matrix "
                        f"shape, got {baseline_candidate.shape} and expected "
                        f"{cache.concept_probs.shape}."
                    )
                if not np.allclose(
                    baseline_candidate,
                    cache.concept_probs,
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise ValueError(
                        "baseline_concepts must align with the cached dataset rows in their "
                        "original order. Pass dataset=... to refresh the cache or pass explicit "
                        "row_indices/source_indices for subset/repeated replay."
                    )
            row_indices = np.arange(cache.concept_probs.shape[0], dtype=int)
        else:
            row_indices = np.asarray(row_indices)
            if row_indices.ndim != 1:
                raise ValueError("row_indices must be a 1D integer array.")
            if row_indices.shape[0] != concepts.shape[0]:
                raise ValueError(
                    "row_indices length must match the number of concept rows, got "
                    f"{row_indices.shape[0]} and {concepts.shape[0]}."
                )
            if not np.issubdtype(row_indices.dtype, np.integer):
                raise ValueError("row_indices must contain integers.")
            if row_indices.size and (
                np.any(row_indices < 0)
                or np.any(row_indices >= cache.concept_probs.shape[0])
            ):
                raise ValueError(
                    "row_indices must refer to rows in the cached dataset, got "
                    f"min={int(row_indices.min())} max={int(row_indices.max())} "
                    f"for cached_rows={cache.concept_probs.shape[0]}."
                )
            row_indices = row_indices.astype(int, copy=False)

        if baseline_concepts is None:
            baseline_concepts = cache.concept_probs[row_indices]
        else:
            baseline_concepts = np.asarray(baseline_concepts, dtype=np.float32)
            if baseline_concepts.ndim == 1:
                baseline_concepts = baseline_concepts.reshape(1, -1)
            if baseline_concepts.shape != concepts.shape:
                raise ValueError(
                    "baseline_concepts must match the concept matrix shape, got "
                    f"{baseline_concepts.shape} and expected {concepts.shape}."
                )

        if intervention_mask is not None:
            intervention_mask = np.asarray(intervention_mask, dtype=bool)
            if intervention_mask.shape != concepts.shape:
                raise ValueError(
                    "intervention_mask must match the concept matrix shape, got "
                    f"{intervention_mask.shape} and expected {concepts.shape}."
                )

        row_cache = _slice_prediction_cache(cache, row_indices)
        return self._predict_from_cached_concepts(
            concepts,
            row_cache,
            baseline_concepts=baseline_concepts,
            intervention_mask=intervention_mask,
        )

    def _predict_dataset(
        self,
        dataset: ConceptDatasetSample,
        *,
        cache: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_prob, concept_probs, prediction_cache = self._run_official_model(dataset)
        if cache:
            self._prediction_cache = prediction_cache
        return y_prob, concept_probs

    def _serialize_common_state(self) -> dict[str, Any]:
        model = self._require_official_model()
        state_dict = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        return {
            "version": 1,
            "family": self.family,
            "benchmark": self.benchmark,
            "concept_names": copy.deepcopy(self.concept_names),
            "class_names": copy.deepcopy(self.class_names),
            "backbone_spec": copy.deepcopy(self.backbone_spec),
            "model_init_kwargs": copy.deepcopy(self.model_init_kwargs),
            "eval_config": copy.deepcopy(self.eval_config),
            "training_summary": copy.deepcopy(self.training_summary),
            "model_state_dict": state_dict,
        }

    def __getstate__(self) -> dict[str, Any]:
        return self._serialize_common_state()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._restore_from_serialized_state(state)

    def _restore_from_serialized_state(self, state: dict[str, Any]) -> None:
        self.family = state["family"]
        self.benchmark = state["benchmark"]
        self.concept_names = list(state["concept_names"])
        self.class_names = list(state["class_names"])
        self.backbone_spec = copy.deepcopy(state["backbone_spec"])
        self.model_init_kwargs = copy.deepcopy(state["model_init_kwargs"])
        self.eval_config = copy.deepcopy(state["eval_config"])
        self.training_summary = copy.deepcopy(state.get("training_summary", {}))
        self._prediction_cache = None
        self._dependency_error_message = _format_install_message()

        try:
            official_model = self._rebuild_model(
                model_init_kwargs=self.model_init_kwargs,
                backbone_spec=self.backbone_spec,
            )
        except Exception as exc:
            if isinstance(exc, (ImportError, ModuleNotFoundError, CEMDependencyError)):
                self._dependency_error_message = _format_install_message(exc)
                official_model = None
            else:
                raise
        if official_model is not None:
            official_model.load_state_dict(state["model_state_dict"])
            official_model.eval()
            official_model.cpu()

        self.official_model = official_model
        concept_detector = _OfficialConceptDetectorAdapter(self)
        label_predictor = _OfficialFrontEndAdapter(self)
        ConceptBasedModel.__init__(
            self,
            concept_detector=concept_detector,
            label_predictor=label_predictor,
            should_propagate=False,
        )

    def _run_official_model(
        self,
        dataset: ConceptDatasetSample,
    ) -> tuple[np.ndarray, np.ndarray, _PredictionCache]:
        raise NotImplementedError

    def _predict_from_cached_concepts(
        self,
        concepts: np.ndarray,
        cache: _PredictionCache,
        *,
        baseline_concepts: np.ndarray,
        intervention_mask: np.ndarray | None,
    ) -> np.ndarray:
        raise NotImplementedError

    def _rebuild_model(
        self,
        *,
        model_init_kwargs: dict[str, Any],
        backbone_spec: dict[str, Any],
    ) -> Any:
        raise NotImplementedError


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
        # Shape: [B, n_concepts, 1, embed_dim] — broadcasts across n_samples
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
    if forced_concept_mask is not None and fixed_logits is not None:
        with torch.no_grad():
            concept_logits.data = torch.where(
                forced_concept_mask,
                fixed_logits,
                concept_logits.data,
            )
            # Re-initialize label logits to zeros for intervention phase
            # (matching original ECBM paper's Phase 2)
            label_logits.data.zero_()

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
            # When intervening, follow the original ECBM paper (Xu et al.):
            # turn OFF xy and xc losses, amplify cy loss so the label
            # prediction is forced to follow the corrected concepts.
            if forced_concept_mask is not None and forced_concept_mask.any():
                loss = 3.0 * loss_cy
            else:
                loss = (
                    model.lambda_xy * loss_xy
                    + model.lambda_xc * loss_xc
                    + model.lambda_cy * loss_cy
                )
            loss.backward()
            if forced_concept_mask is not None and concept_logits.grad is not None:
                concept_logits.grad.masked_fill_(forced_concept_mask, 0.0)
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


def _resolve_loader_config(
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> dict[str, Any]:
    defaults = get_loader_config()
    return {
        "batch_size": int(batch_size),
        "num_workers": int(defaults["num_workers"] if num_workers is None else num_workers),
        "pin_memory": bool(
            defaults["pin_memory"] if pin_memory is None else pin_memory
        )
        and device.type == "cuda",
        "device": str(device),
    }


def _resolve_epochs(config: Any, *, benchmark: str, family: str) -> int:
    attr = f"{family}_max_epochs"
    override = getattr(config, attr, None)
    if override is not None:
        return int(override)
    if benchmark == "sudoku":
        return int(getattr(config, "cs_epochs", getattr(config, "epochs", 20)))
    return int(getattr(config, "epochs", 50))


def _resolve_patience(config: Any, *, benchmark: str) -> int:
    if benchmark == "sudoku":
        return int(getattr(config, "cs_patience", getattr(config, "patience", 5)))
    return int(getattr(config, "patience", 10))


def _resolve_learning_rate(config: Any) -> float:
    return float(getattr(config, "learning_rate", 1e-3))


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
        "train_class_mode": "sequential",
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
                input_shape=np.asarray(train_dataset.X[0]).shape,
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
        trainer = _build_trainer(
            pl_module=deps.pl,
            max_epochs=_resolve_epochs(config, benchmark=benchmark, family="probcbm"),
            patience=_resolve_patience(config, benchmark=benchmark),
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
        if current_val < best_val_loss - 1e-6:
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
        training_summary={
            "max_epochs": int(best_epoch + 1),
            "best_val_loss": best_val_loss,
        },
    )
