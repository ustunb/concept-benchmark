from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from concept_benchmark.data import ConceptDatasetSample


def train_concept_heads(
    train_dataset: ConceptDatasetSample,
    valid_dataset: ConceptDatasetSample,
    embedding_model: Optional[nn.Module],
    *,
    input_dim: Optional[int] = None,
    l1_size: int = 100,
    freeze: bool = False,
    fit_params: Optional[dict] = None,
) -> nn.ModuleList:
    """
    Train per-concept heads with optional finetuning of the embedding model.

    - If `freeze=True`, only head parameters are updated; the embedding model runs in eval mode.
    - If `freeze=False` and an embedding model is provided, both embedding model and heads are optimized with separate LRs (joint training).
    - Loss: mean BCEWithLogits across concepts.
    - Early stopping based on mean validation F1.

    Returns:
        nn.ModuleList: trained list of per-concept heads (each maps embedding -> 1 logit).
    """
    params = {
        "epochs": 10,
        "batch_size": 64,
        "lr_encoder": 1e-5,
        "lr_heads": 1e-3,
        "min_delta": 0.001,
        "patience": 5,
        "device": "cpu",
        "num_workers": 0,
        "pin_memory": False,
        "loss_fn": None,
        # logging controls
        "verbose": False,
        "log_interval": 50,
    }
    if fit_params:
        params.update(fit_params)

    device = params["device"]

    # Infer number of concepts and embedding dimension
    num_concepts = train_dataset.n_concepts

    # If input_dim unknown, infer from a small forward pass
    if input_dim is None:
        samp_bs = min(8, max(1, getattr(train_dataset, "n", 8)))
        samp_loader = train_dataset.loader(
            batch_size=samp_bs,
            shuffle=False,
            num_workers=params["num_workers"],
            pin_memory=params["pin_memory"],
        )
        with torch.no_grad():
            bx, _, _ = next(iter(samp_loader))
            if isinstance(bx, torch.Tensor):
                bx = bx.to(device)
            if embedding_model is None:
                emb = bx
            else:
                embedding_model = embedding_model.to(device)
                embedding_model.eval()
                emb = embedding_model(bx)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
            input_dim = int(emb.shape[1])

    # Heads: independent per-concept MLPs
    heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(input_dim, l1_size), nn.ReLU(), nn.Linear(l1_size, 1)
            )
            for _ in range(num_concepts)
        ]
    )

    # Move modules to device
    if embedding_model is not None:
        embedding_model.to(device)
        if freeze:
            for p in embedding_model.parameters():
                p.requires_grad = False
        embedding_model.eval() if freeze else embedding_model.train()
    heads.to(device)

    # Optimizer parameter groups
    optim_params = []
    if embedding_model is not None and not freeze:
        optim_params.append(
            {"params": embedding_model.parameters(), "lr": params["lr_encoder"]}
        )
    optim_params.append({"params": heads.parameters(), "lr": params["lr_heads"]})
    optimizer = torch.optim.Adam(optim_params)

    loss_fn = params["loss_fn"] if params["loss_fn"] is not None else nn.BCEWithLogitsLoss()

    train_loader = train_dataset.loader(
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
    )
    valid_loader = valid_dataset.loader(
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
    )

    best_val_f1 = -1.0
    patience_counter = 0
    best_heads_state = None
    best_encoder_state = None

    for epoch in tqdm(range(params["epochs"])):
        # Train epoch
        if embedding_model is not None and not freeze:
            embedding_model.train()
        heads.train()
        running_loss = 0.0
        batches = 0
        global_step = 0
        for batch_X, batch_C, _ in train_loader:
            # Move data
            if isinstance(batch_X, torch.Tensor):
                batch_X = batch_X.to(device)
            if isinstance(batch_C, torch.Tensor):
                batch_C = batch_C.to(device)
            batch_C = batch_C.float()

            # Forward
            with torch.set_grad_enabled(True):
                if embedding_model is None:
                    emb = batch_X
                else:
                    emb = embedding_model(batch_X)
                if isinstance(emb, (list, tuple)):
                    emb = emb[0]
                if not isinstance(emb, torch.Tensor):
                    emb = torch.as_tensor(emb, device=device, dtype=torch.float32)

                logits = [head(emb).squeeze(1) for head in heads]
                logits = torch.stack(logits, dim=1)
                loss = loss_fn(logits, batch_C)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logging
            running_loss += float(loss.item())
            batches += 1
            global_step += 1
            if params["verbose"] and params["log_interval"] and (global_step % int(params["log_interval"])) == 0:
                print(
                    f"Epoch {epoch+1}/{params['epochs']} step {global_step}: loss={loss.item():.4f}"
                )

        # Validate
        if embedding_model is not None:
            embedding_model.eval()
        heads.eval()
        val_f1s = []
        with torch.no_grad():
            for batch_X, batch_C, _ in valid_loader:
                if isinstance(batch_X, torch.Tensor):
                    batch_X = batch_X.to(device)
                if isinstance(batch_C, torch.Tensor):
                    batch_C = batch_C.to(device)
                batch_C = batch_C.float()

                emb = batch_X if embedding_model is None else embedding_model(batch_X)
                if isinstance(emb, (list, tuple)):
                    emb = emb[0]
                if not isinstance(emb, torch.Tensor):
                    emb = torch.as_tensor(emb, device=device, dtype=torch.float32)

                logits = [head(emb).squeeze(1) for head in heads]
                logits = torch.stack(logits, dim=1)
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                target = batch_C.cpu().numpy().astype(int)

                # F1 per concept then mean
                concept_f1s = [
                    f1_score(target[:, i], preds[:, i]) for i in range(num_concepts)
                ]
                val_f1s.append(np.mean(concept_f1s))

        avg_train_loss = running_loss / max(1, batches)
        avg_val_f1 = float(np.mean(val_f1s)) if len(val_f1s) else -1.0

        if params["verbose"]:
            print(
                f"Epoch {epoch+1}/{params['epochs']} | train_loss={avg_train_loss:.4f} | val_f1={avg_val_f1:.4f} | best_f1={best_val_f1 if best_val_f1>=0 else 0.0:.4f}"
            )

        if avg_val_f1 > best_val_f1 + params["min_delta"]:
            best_val_f1 = avg_val_f1
            patience_counter = 0
            best_heads_state = {
                k: v.cpu().clone() for k, v in heads.state_dict().items()
            }
            if embedding_model is not None and not freeze:
                best_encoder_state = {
                    k: v.cpu().clone() for k, v in embedding_model.state_dict().items()
                }
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                if params["verbose"]:
                    print(f"Early stopping at val F1={avg_val_f1:.4f}")
                break

    # Load best states
    if best_heads_state is not None:
        heads.load_state_dict(best_heads_state)
    if embedding_model is not None and not freeze and best_encoder_state is not None:
        embedding_model.load_state_dict(best_encoder_state)

    # Ensure on CPU for downstream wrapping/embedding
    heads.cpu()
    if embedding_model is not None:
        embedding_model.cpu()

    return heads
