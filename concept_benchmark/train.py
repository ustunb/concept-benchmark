from __future__ import annotations

from dataclasses import dataclass
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
    ) -> TrainerOutput:
        ...


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
        except Exception:
            return prepared
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
            raise TypeError("Cannot extract logits from dict output without 'logits' key.")
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    return output


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

        loss_fn = cfg["loss_fn"] if cfg["loss_fn"] is not None else nn.BCEWithLogitsLoss()

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
            for batch_X, batch_C, _ in train_loader:
                # Training forward/backward pass
                batch_X = _prepare_inputs(batch_X, device)
                if isinstance(batch_C, torch.Tensor):
                    batch_C = batch_C.to(device)
                else:
                    batch_C = torch.as_tensor(batch_C, device=device)
                batch_C = batch_C.float()

                logits = _extract_logits(model(batch_X))
                loss = loss_fn(logits, batch_C)

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
                with torch.no_grad():
                    for batch_X, batch_C, _ in valid_loader:
                        # Validation forward pass for metrics only
                        batch_X = _prepare_inputs(batch_X, device)
                        if isinstance(batch_C, torch.Tensor):
                            batch_C = batch_C.to(device)
                        else:
                            batch_C = torch.as_tensor(batch_C, device=device)
                        batch_C = batch_C.float()

                        logits = _extract_logits(model(batch_X))
                        prob = torch.sigmoid(logits).cpu().numpy()
                        preds.append(prob)
                        targets.append(batch_C.cpu().numpy())

                if preds:
                    pred_arr = np.vstack(preds)
                    tgt_arr = np.vstack(targets).astype(int)
                    concept_f1s = []
                    for j in range(num_concepts):
                        try:
                            concept_f1s.append(
                                f1_score(tgt_arr[:, j], (pred_arr[:, j] > 0.5).astype(int))
                            )
                        except ValueError:
                            concept_f1s.append(0.0)
                    val_metric = float(np.mean(concept_f1s))
                history["val_f1"].append(val_metric)

                improved = val_metric > best_metric + cfg["min_delta"]
                if improved:
                    best_metric = val_metric
                    patience_counter = 0
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
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
