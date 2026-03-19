from __future__ import annotations

__all__ = [
    "TrainerResult",
    "ConceptTrainer",
    "DefaultConceptTrainer",
    "train_concept_heads",
]

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from concept_benchmark.data import ConceptDatasetSample


@dataclass
class TrainerResult:
    """Container returned by a concept trainer."""

    model: nn.Module
    history: Optional[Dict[str, Any]] = None
    best_metric: Optional[float] = None


TrainerOutput = Union[nn.Module, Tuple[nn.Module, Dict[str, Any]], TrainerResult]


class ConceptTrainer(Protocol):
    """Protocol for pluggable concept trainers."""

    def __call__(
        self,
        model: nn.Module,
        train_dataset: ConceptDatasetSample,
        valid_dataset: Optional[ConceptDatasetSample],
        *,
        num_concepts: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> TrainerOutput: ...


def _prepare_inputs(x: Any, device: torch.device) -> Any:
    """Move nested tensors/arrays to the target device while preserving structure."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _prepare_inputs(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        prepared = [_prepare_inputs(v, device) for v in x]
        try:
            return torch.stack(prepared, dim=0)
        except (RuntimeError, TypeError):
            return prepared  # heterogeneous shapes/types — keep as list
    return torch.as_tensor(x, device=device)


def _extract_logits(output: Any) -> torch.Tensor:
    """Convert a model forward output into a tensor of logits."""
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Model forward returned an empty sequence.")
        output = output[0]
    if isinstance(output, dict):
        if "logits" in output:
            output = output["logits"]
        else:
            raise TypeError(
                "Cannot extract logits from dict output without 'logits' key."
            )
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    return output


def _build_observation_mask(
    tensor: torch.Tensor,
    *,
    fill_value: Any,
    enabled: bool,
) -> torch.Tensor:
    """Return a boolean mask indicating observed concept entries.

    Args:
        tensor: Concept target tensor for a mini-batch.
        fill_value: Sentinel value stored for missing annotations.
        enabled: Whether missing annotations are present.

    Returns:
        torch.Tensor: Boolean tensor with ``True`` for observed labels.
    """

    if not enabled:
        return torch.ones_like(tensor, dtype=torch.bool)

    if isinstance(fill_value, float) and math.isnan(fill_value):
        return ~torch.isnan(tensor)

    fill_tensor = torch.as_tensor(
        fill_value,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return ~torch.isclose(tensor, fill_tensor)


class DefaultConceptTrainer:
    """Default BCE-with-logits trainer with early stopping on mean validation F1."""

    def __call__(
        self,
        model: nn.Module,
        train_dataset: ConceptDatasetSample,
        valid_dataset: Optional[ConceptDatasetSample],
        *,
        num_concepts: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> TrainerResult:
        """Train ``model`` on concept supervision and return the best checkpoint.

        Args:
            model: Joint concept model emitting ``num_concepts`` logits.
            train_dataset: Dataset used for optimisation.
            valid_dataset: Optional dataset used for early stopping and metric
                tracking. Pass ``None`` to disable early stopping.
            num_concepts: Number of concept targets in the dataset; used for
                computing mean validation F1.
            params: Optional dictionary overriding the default hyperparameters
                (e.g. ``epochs``, ``batch_size``, ``lr``).

        Returns:
            TrainerResult: Object containing the best-scoring model snapshot plus
            training history.

        Raises:
            ValueError: If the model forward pass returns an empty sequence of
                outputs.
        """
        cfg: Dict[str, Any] = {
            "epochs": 10,
            "batch_size": 64,
            "valid_batch_size": None,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "min_delta": 0.0,
            "patience": 5,
            "device": "cpu",
            "num_workers": 0,
            "pin_memory": False,
            "loss_fn": None,
            "optimizer_factory": None,
            "scheduler_factory": None,
            "use_tqdm": True,
            "verbose": False,
            "log_interval": 50,
        }
        if params:
            cfg.update(params)

        device = torch.device(cfg["device"])
        valid_batch_size = cfg["valid_batch_size"] or cfg["batch_size"]

        model = model.to(device)

        default_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        loss_fn = cfg["loss_fn"] if cfg["loss_fn"] is not None else default_loss_fn
        use_masked_loss = cfg["loss_fn"] is None

        train_missing_enabled = (
            bool(getattr(train_dataset, "concept_missing", False))
            and getattr(train_dataset, "concept_missing_mask", None) is not None
        )
        train_fill_value = getattr(
            train_dataset, "concept_missing_fill_value", float("nan")
        )

        valid_missing_enabled = False
        valid_fill_value = float("nan")
        if valid_dataset is not None:
            valid_missing_enabled = (
                bool(getattr(valid_dataset, "concept_missing", False))
                and getattr(valid_dataset, "concept_missing_mask", None) is not None
            )
            valid_fill_value = getattr(
                valid_dataset, "concept_missing_fill_value", float("nan")
            )

        if train_missing_enabled and not use_masked_loss:
            raise ValueError(
                "Custom loss functions must handle concept-missing masks; set fit_params['loss_fn']=None to use the default masked BCE loss."
            )

        if cfg["optimizer_factory"] is not None:
            optimizer = cfg["optimizer_factory"](model.parameters())
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )

        scheduler = None
        if cfg["scheduler_factory"] is not None:
            scheduler = cfg["scheduler_factory"](optimizer)

        train_loader = train_dataset.loader(
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
        )
        valid_loader = (
            valid_dataset.loader(
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=cfg["num_workers"],
                pin_memory=cfg["pin_memory"],
            )
            if valid_dataset is not None
            else None
        )

        best_metric = float("-inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        history: Dict[str, Any] = {
            "train_loss": [],
            "val_f1": [],
        }

        epoch_iter = range(cfg["epochs"])
        if cfg["use_tqdm"]:
            epoch_iter = tqdm(epoch_iter)

        global_step = 0
        for epoch in epoch_iter:
            model.train()
            running_loss = 0.0
            batches = 0
            for batch in train_loader:
                batch_X, batch_C, *_ = batch
                # Training forward/backward pass
                batch_X = _prepare_inputs(batch_X, device)
                if isinstance(batch_C, torch.Tensor):
                    batch_C = batch_C.to(device)
                else:
                    batch_C = torch.as_tensor(batch_C, device=device)
                batch_C = batch_C.float()

                mask_tensor = _build_observation_mask(
                    batch_C,
                    fill_value=train_fill_value,
                    enabled=train_missing_enabled,
                )
                targets = torch.where(mask_tensor, batch_C, torch.zeros_like(batch_C))

                logits = _extract_logits(model(batch_X))

                if use_masked_loss:
                    loss_tensor = loss_fn(logits, targets)
                    mask_float = mask_tensor.to(loss_tensor.dtype)
                    observed = mask_float.sum()
                    if observed <= 0:
                        continue
                    loss = (loss_tensor * mask_float).sum() / observed
                else:
                    loss = loss_fn(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                batches += 1
                global_step += 1

                if cfg["verbose"] and cfg["log_interval"]:
                    if global_step % int(cfg["log_interval"]) == 0:
                        print(
                            f"Epoch {epoch + 1}/{cfg['epochs']} step {global_step}: loss={loss.item():.4f}"
                        )

            avg_train_loss = running_loss / max(1, batches)
            history["train_loss"].append(avg_train_loss)

            if scheduler is not None:
                scheduler.step()

            val_metric = float("nan")
            if valid_loader is not None:
                model.eval()
                preds = []
                targets = []
                masks = []
                with torch.no_grad():
                    for batch in valid_loader:
                        batch_X, batch_C, *_ = batch
                        # Validation forward pass for metrics only
                        batch_X = _prepare_inputs(batch_X, device)
                        if isinstance(batch_C, torch.Tensor):
                            batch_C = batch_C.to(device)
                        else:
                            batch_C = torch.as_tensor(batch_C, device=device)
                        batch_C = batch_C.float()

                        mask_tensor = _build_observation_mask(
                            batch_C,
                            fill_value=valid_fill_value,
                            enabled=valid_missing_enabled,
                        )
                        targets_tensor = torch.where(
                            mask_tensor,
                            batch_C,
                            torch.zeros_like(batch_C),
                        )

                        logits = _extract_logits(model(batch_X))
                        prob = torch.sigmoid(logits).cpu().numpy()
                        preds.append(prob)
                        targets.append(targets_tensor.cpu().numpy())
                        masks.append(mask_tensor.cpu().numpy().astype(bool))

                if preds:
                    pred_arr = np.vstack(preds)
                    tgt_arr = np.vstack(targets)
                    mask_arr = (
                        np.vstack(masks).astype(bool)
                        if masks
                        else np.ones_like(tgt_arr, dtype=bool)
                    )
                    concept_f1s = []
                    for j in range(num_concepts):
                        observed = mask_arr[:, j]
                        if not np.any(observed):
                            continue
                        y_true = tgt_arr[observed, j].astype(int)
                        y_pred = (pred_arr[observed, j] > 0.5).astype(int)
                        if np.unique(y_true).size < 2:
                            continue
                        concept_f1s.append(f1_score(y_true, y_pred))
                    val_metric = (
                        float(np.mean(concept_f1s)) if concept_f1s else float("nan")
                    )
                history["val_f1"].append(val_metric)

                improved = val_metric > best_metric + cfg["min_delta"]
                if improved:
                    best_metric = val_metric
                    patience_counter = 0
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= cfg["patience"]:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        model = model.cpu()
        model.eval()

        best_value = None if best_metric == float("-inf") else best_metric
        return TrainerResult(model=model, history=history, best_metric=best_value)


def train_concept_heads(
    train_dataset: ConceptDatasetSample,
    valid_dataset: Optional[ConceptDatasetSample],
    embedding_model: Optional[nn.Module],
    *,
    input_dim: Optional[int] = None,
    hidden_layer_size: int = 100,
    freeze_backbone: bool = False,
    fit_params: Optional[Dict[str, Any]] = None,
) -> nn.ModuleList:
    params = {
        "epochs": 10,
        "batch_size": 64,
        "valid_batch_size": None,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "min_delta": 0.0,
        "patience": 5,
        "device": "cpu",
        "num_workers": 0,
        "pin_memory": False,
        "loss_fn": None,
        "optimizer_factory": None,
        "scheduler_factory": None,
        "use_tqdm": False,
        "verbose": False,
        "log_interval": 50,
    }
    if fit_params:
        params.update(fit_params)

    device = torch.device(params["device"])
    num_concepts = int(getattr(train_dataset, "n_concepts", train_dataset.C.shape[1]))

    if input_dim is None:
        samp_bs = min(8, max(1, int(getattr(train_dataset, "n", 8))))
        loader = train_dataset.loader(
            batch_size=samp_bs,
            shuffle=False,
            num_workers=int(params["num_workers"]),
            pin_memory=bool(params["pin_memory"]),
        )
        bx, _, *_ = next(iter(loader))
        bx = bx
        if embedding_model is None:
            with torch.no_grad():
                if isinstance(bx, torch.Tensor):
                    emb = bx
                else:
                    emb = torch.as_tensor(bx, dtype=torch.float32)
        else:
            embedding_model = embedding_model.to(device)
            embedding_model.eval()
            with torch.no_grad():
                x = bx
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                out = embedding_model(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                emb = (
                    out
                    if isinstance(out, torch.Tensor)
                    else torch.as_tensor(out, dtype=torch.float32, device=device)
                )
        input_dim = int(emb.shape[1])

    class _EmbedAndLinear(nn.Module):
        def __init__(
            self, enc: Optional[nn.Module], d_in: int, k: int, freeze_enc: bool
        ):
            super().__init__()
            self.enc = enc
            if self.enc is not None:
                self.enc.to(device)
                if freeze_enc:
                    for p in self.enc.parameters():
                        p.requires_grad = False
                    self.enc.eval()
            self.linear = nn.Linear(d_in, k)

        def forward(self, x):
            h = x
            if self.enc is not None:
                h = self.enc(x)
            if isinstance(h, (list, tuple)):
                h = h[0]
            if not isinstance(h, torch.Tensor):
                h = torch.as_tensor(h, dtype=torch.float32)
            return self.linear(h)

    model = _EmbedAndLinear(
        embedding_model, int(input_dim), int(num_concepts), bool(freeze_backbone)
    )

    trainer = DefaultConceptTrainer()
    res = trainer(
        model, train_dataset, valid_dataset, num_concepts=num_concepts, params=params
    )
    joint = res.model if isinstance(res, TrainerResult) else res
    W = joint.linear.weight.detach().cpu().clone()  # (k, d)
    b = joint.linear.bias.detach().cpu().clone()  # (k,)
    heads = nn.ModuleList([nn.Linear(int(input_dim), 1) for _ in range(num_concepts)])
    for j in range(num_concepts):
        heads[j].weight.data.copy_(W[j : j + 1, :])
        heads[j].bias.data.copy_(b[j : j + 1])
    heads.cpu()
    if embedding_model is not None:
        embedding_model.cpu()
    return heads
