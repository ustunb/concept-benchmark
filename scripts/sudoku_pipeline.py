"""
sudoku_pipeline.py

Run a single sudoku CBM experiment end-to-end:
  1. Generate a tabular sudoku dataset
  2. Train a concept-based model (CBM)
  3. Train a DNN baseline
  4. Evaluate both and print results

Examples
--------
python scripts/sudoku_pipeline.py --n 3 --n-samples 1000 --seed 42
python scripts/sudoku_pipeline.py --n 3 --n-samples 5000 --max-corrupt 5 --epochs 50
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --- repo path shim (safe if already installed) ---
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from concept_benchmark.synthetic.sudoku import create_sudoku_dataset, default_transform
from concept_benchmark.models import ConceptBasedModel, ConceptDetector
from concept_benchmark.models import (
    GroupPoolingConceptSudokuCNN as SudokuConceptModel,
    SudokuValidatorCNN as DNNSudokuModel,
)


def determine_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, _, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = (outputs.squeeze() > 0.5).long()
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total > 0 else 0


def train_dnn(data, device, epochs=20, patience=5, lr=1e-3):
    """Train a DNN baseline (tabular input -> binary label)."""
    model = DNNSudokuModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader_cfg = {"batch_size": 32, "num_workers": 0, "pin_memory": False}
    train_loader = data.training.loader(shuffle=True, **loader_cfg)
    valid_loader = data.validation.loader(shuffle=False, **loader_cfg)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs), desc="DNN training"):
        model.train()
        for X, _, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X).squeeze(), y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X).squeeze(), y.float()).item()
                n_batches += 1
        avg_val_loss = val_loss / max(n_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"  DNN early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser(description="Run a single sudoku CBM experiment.")
    ap.add_argument("--n", type=int, default=3, help="Block size (3 -> 9x9 board)")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--valid-ratio", type=float, default=0.5)
    ap.add_argument("--max-corrupt", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = determine_device()
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    # ---- 1. Generate dataset ----
    print("\n=== Generating dataset ===")
    data = create_sudoku_dataset(
        n=args.n,
        n_samples=args.n_samples,
        valid_ratio=args.valid_ratio,
        max_corrupt=args.max_corrupt,
        data_type="tabular",
        seed=args.seed,
        transform=default_transform,
    )
    data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=args.seed)
    data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)
    print(f"  samples: {len(data.y)}  |  X shape: {data.X.shape}  |  C shape: {data.C.shape}")

    # ---- 2. Train CBM (concept detector + frontend) ----
    print("\n=== Training CBM ===")
    config = {
        "device": device,
        "batch_size": 32,
        "num_workers": 0 if str(device) == "mps" else 0,
        "pin_memory": False,
    }

    concept_model = SudokuConceptModel()
    cd = ConceptDetector(model=concept_model)
    cbm = ConceptBasedModel(concept_detector=cd, propagate=True)
    cbm.fit(
        train_dataset=data.training,
        valid_dataset=data.validation,
        freeze=False,
        concept_embed_params={"shuffle": False, **config},
        fit_params={
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            **config,
        },
    )

    cbm_preds = cbm.predict(data.test)
    cbm_acc = float(np.mean(cbm_preds == data.test.y))

    # ---- 3. Train DNN baseline ----
    print("\n=== Training DNN baseline ===")
    dnn = train_dnn(data, device, epochs=args.epochs, patience=args.patience, lr=args.lr)

    loader_cfg = {"batch_size": 32, "num_workers": 0, "pin_memory": False}
    dnn_acc = compute_accuracy(dnn, data.test.loader(shuffle=False, **loader_cfg), device)

    # ---- 4. Results ----
    print("\n=== Results ===")
    print(f"  CBM test accuracy: {cbm_acc:.4f}")
    print(f"  DNN test accuracy: {dnn_acc:.4f}")


if __name__ == "__main__":
    main()
