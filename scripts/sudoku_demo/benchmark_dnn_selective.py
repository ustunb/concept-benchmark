import copy
import csv
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import results_dir
from scripts.sudoku_demo.sudoku_models import SudokuValidatorCNN
from scripts.sudoku_demo.utils import (
    DEFAULT_SUDOKU_SETTINGS,
    compute_accuracy,
    determine_device,
    get_dataset_file,
)


def parse_args():
    settings = DEFAULT_SUDOKU_SETTINGS.copy()
    p = ArgumentParser(
        description="Train Sudoku DNN across seeds and record selective metrics."
    )
    p.add_argument("--n", type=int, default=settings["n"])
    p.add_argument("--n-samples", dest="n_samples", type=int, default=settings["n_samples"])
    p.add_argument("--max-corrupt", dest="max_corrupt", type=int, default=settings["max_corrupt"])
    p.add_argument("--epochs", type=int, default=settings["epochs"])
    p.add_argument("--patience", type=int, default=settings.get("patience", 0))
    p.add_argument("--min-delta", dest="min_delta", type=float, default=0.0)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=12)
    p.add_argument("--target-accuracy", dest="target_accuracy", type=float, default=0.9)
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[171, 172, 173, 174, 175],
        help="Training seeds to sweep. Dataset seed stays fixed at DEFAULT_SUDOKU_SETTINGS['seed'].",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(results_dir / "sudoku_dnn_selective_seed_sweep.csv"),
        help="Output CSV path for per-seed metrics.",
    )
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _decision_threshold_sweep(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    if y_true.shape[0] != prob_pos.shape[0]:
        raise ValueError("y_true and prob_pos must have the same length.")

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101, dtype=float)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    best_acc = -1.0
    best_thresholds = []
    for t in thresholds:
        preds = (prob_pos >= t).astype(int)
        acc = float((preds == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_thresholds = [float(t)]
        elif acc == best_acc:
            best_thresholds.append(float(t))

    best_t = 0.5
    if best_thresholds:
        best_t = 0.5 * (min(best_thresholds) + max(best_thresholds))
    return best_t, best_acc


def _selective_accuracy_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    target_acc: float,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float | None]:
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    if y_true.shape[0] != prob_pos.shape[0]:
        raise ValueError("y_true and prob_pos must have the same length.")

    min_prob = np.minimum(prob_pos, 1.0 - prob_pos)
    candidates = np.unique(np.concatenate(([0.0], min_prob)))
    candidates = candidates[(candidates >= 0.0) & (candidates <= 0.5)]
    candidates.sort()

    for t in candidates[::-1]:
        mask = min_prob <= t
        if not np.any(mask):
            continue
        preds = (prob_pos[mask] >= decision_threshold).astype(int)
        acc = float((preds == y_true[mask]).mean())
        if acc >= target_acc:
            coverage = float(mask.mean())
            return float(t), coverage

    return None, None


def _selective_metrics(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    t: float | None,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float]:
    if t is None:
        return None, 0.0

    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    min_prob = np.minimum(prob_pos, 1.0 - prob_pos)
    mask = min_prob <= t
    if not np.any(mask):
        return None, 0.0

    preds = (prob_pos[mask] >= decision_threshold).astype(int)
    acc = float((preds == y_true[mask]).mean())
    coverage = float(mask.mean())
    return acc, coverage


def _dnn_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_y = []
    with torch.no_grad():
        for X, _, y in loader:
            X = X.to(device)
            probs = model(X).squeeze(-1).detach().cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_y)


def build_data(settings: dict):
    dataset_seed = int(DEFAULT_SUDOKU_SETTINGS["seed"])
    dataset_settings = settings.copy()
    dataset_settings["seed"] = dataset_seed
    dataset_settings["data_type"] = "tabular"

    tab_ds_dir = get_dataset_file(**dataset_settings)
    data = load(tab_ds_dir / "sudoku_dataset.pkl")
    data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=dataset_seed)
    data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)
    return data, dataset_seed


def train_one_seed(training_seed: int, data, settings: dict, device):
    set_global_seed(training_seed)

    model = SudokuValidatorCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader_config = {
        "batch_size": settings["batch_size"],
        "num_workers": settings["num_workers"],
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(training_seed),
    }

    train_loader = data.training.loader(shuffle=True, **loader_config)
    valid_loader = data.validation.loader(shuffle=False, **loader_config)
    test_loader = data.test.loader(shuffle=False, **loader_config)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0
    use_early_stopping = settings["patience"] > 0

    for _ in tqdm(range(settings["epochs"]), desc=f"Seed {training_seed}", leave=False):
        model.train()
        for X, _, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                batch_loss = criterion(outputs.squeeze(), y.float())
                val_loss_sum += batch_loss.item()
                val_batches += 1
        avg_val_loss = val_loss_sum / max(val_batches, 1)

        if avg_val_loss < (best_val_loss - settings["min_delta"]):
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if use_early_stopping and epochs_no_improve >= settings["patience"]:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_accuracy = compute_accuracy(model, test_loader, device=device)

    val_probs, val_y = _dnn_probs(model, valid_loader, device)
    decision_threshold, val_acc_swept = _decision_threshold_sweep(val_y, val_probs)
    tau, _ = _selective_accuracy_threshold(
        val_y, val_probs, settings["target_accuracy"], decision_threshold
    )
    test_probs, test_y = _dnn_probs(model, test_loader, device)
    selective_accuracy, coverage = _selective_metrics(
        test_y, test_probs, tau, decision_threshold
    )

    return {
        "seed": training_seed,
        "test_accuracy": float(test_accuracy),
        "val_accuracy_swept": float(val_acc_swept),
        "decision_threshold": float(decision_threshold),
        "tau": None if tau is None else float(tau),
        "coverage": float(coverage),
        "selective_accuracy": None if selective_accuracy is None else float(selective_accuracy),
    }


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "test_accuracy",
        "val_accuracy_swept",
        "decision_threshold",
        "tau",
        "coverage",
        "selective_accuracy",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> None:
    def collect(key: str):
        vals = [r[key] for r in rows if r[key] is not None]
        return np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)

    for key in ["selective_accuracy", "coverage", "tau"]:
        vals = collect(key)
        if vals.size == 0:
            print(f"{key}: no valid values")
        else:
            print(f"{key}: mean={vals.mean():.4f} std={vals.std(ddof=0):.4f} n={vals.size}")


def main():
    args = parse_args()
    settings = vars(args).copy()
    settings["data_type"] = "tabular"

    data, dataset_seed = build_data(settings)
    device = determine_device()
    print(f"Device: {device}")
    print(f"Dataset seed fixed at: {dataset_seed}")
    print(f"Training seeds: {settings['seeds']}")

    rows = []
    for seed in settings["seeds"]:
        row = train_one_seed(seed, data, settings, device)
        rows.append(row)
        print(
            "seed={seed} selective_accuracy={sel} coverage={cov} tau={tau}".format(
                seed=row["seed"],
                sel="None" if row["selective_accuracy"] is None else f"{row['selective_accuracy']:.4f}",
                cov=f"{row['coverage']:.4f}",
                tau="None" if row["tau"] is None else f"{row['tau']:.4f}",
            )
        )

    out_path = Path(settings["out"])
    write_csv(rows, out_path)
    print(f"Wrote results: {out_path}")
    summarize(rows)


if __name__ == "__main__":
    main()
