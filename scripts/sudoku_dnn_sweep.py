#!/usr/bin/env python3
"""
Evaluate CBM-only selective abstention baselines on Sudoku:

- Selective DNN baseline (on CBM label probabilities)
- No-intervention CBM baseline (standard abstention, no concept fixes)
- Standard CBM DNN (no abstention, no interventions)

We support concept missingness **only** for the no-intervention CBM:
- Sweep over mechanisms (none / mcar / mnar) and missingness levels.
- Missingness is applied via eval_common.apply_missingness, using base_concepts.

We track:
- selective accuracy
- coverage
- coverage gain
- simple "work" metrics (all 0 for these baselines)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from concept_benchmark.ext import fileutils
from concept_benchmark.intervention import (
    ConceptualSafeguardsStrategy,
    InterventionConfig,
    NoInterventionCBMStrategy,
)
from concept_benchmark.models import ConceptBasedModel, FrontEndModel, ConceptDetector
from concept_benchmark.paths import results_dir
import utils as big_demo_utils
from eval_common import (
    BASE_DATASET_CONFIGS,
    INTERVENTION_SPLITS,
    ConceptInterventionRunner,
    MetricRecord,
    build_settings,
    default_target_options,
    default_missingness_levels,
    iter_splits,
    write_metrics_csv,
    apply_missingness,
)

ACC_TARGET = 0.90  # target selective accuracy on validation for tau calibration

# Strategy name constants (what will appear in CSV)
DNN_STRATEGY_NAME = "selective_dnn_cbm"
NOINT_STRATEGY_NAME = "no_intervention_cbm"
STANDARD_STRATEGY_NAME = "standard_dnn_cbm"

# -------------------------------------------------------------------------
# Device
# -------------------------------------------------------------------------
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# -------------------------------------------------------------------------
# ConceptSudokuCNN: board -> 27 concept probabilities
# -------------------------------------------------------------------------
class ConceptSudokuCNN(nn.Module):
    def __init__(self, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.row_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.col_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.block_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.block_pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 9, 9) integer Sudoku board digits (0..9).
        Returns: (N, 27) concept probabilities in [0,1].
        """
        x = x.long()
        x = self.embedding(x)  # (N, 9, 9, D)
        x = x.permute(0, 3, 1, 2)  # (N, D, 9, 9)

        x = F.relu(self.conv1(x))
        features = F.relu(self.conv2(x))  # (N, 128, 9, 9)

        # Rows
        row_features = torch.mean(features, dim=3).permute(0, 2, 1)  # (N, 9, 128)
        row_preds = self.row_head(row_features).squeeze(-1)  # (N, 9)

        # Columns
        col_features = torch.mean(features, dim=2).permute(0, 2, 1)  # (N, 9, 128)
        col_preds = self.col_head(col_features).squeeze(-1)  # (N, 9)

        # Blocks
        block_features = self.block_pool(features)  # (N, 128, 3, 3)
        block_features = block_features.view(
            features.size(0), features.size(1), -1
        ).permute(0, 2, 1)  # (N, 9, 128)
        block_preds = self.block_head(block_features).squeeze(-1)  # (N, 9)

        all_preds = torch.cat([row_preds, col_preds, block_preds], dim=1)  # (N, 27)
        return torch.sigmoid(all_preds)


# -------------------------------------------------------------------------
# JSON helpers: boards and evaluation
# -------------------------------------------------------------------------
def _extract_board_generic(rec: dict, keys: Sequence[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in rec and rec[k] is not None:
            arr = np.array(rec[k])
            if arr.size == 81:
                return arr.reshape(9, 9).astype(np.int64)
            if arr.shape == (9, 9):
                return arr.astype(np.int64)
    return None


def load_json_boards(jsonl_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    boards: Dict[str, Dict[str, np.ndarray]] = {}
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_summary"):
                continue
            img = rec.get("img")
            if img is None:
                continue
            key = Path(img).name

            gt = _extract_board_generic(rec, ["gt_board", "gt_grid", "gt"])
            pred = _extract_board_generic(rec, ["pred_board", "pred_grid", "pred_argmax"])

            if key not in boards:
                boards[key] = {}
            if gt is not None:
                boards[key]["gt"] = gt
            if pred is not None:
                boards[key]["pred"] = pred
    return boards


def eval_boards_from_json(boards: Dict[str, Dict[str, np.ndarray]]) -> None:
    total_cells = 0
    correct_cells = 0
    total_boards = 0
    exact_match = 0

    for _, rec in boards.items():
        gt = rec.get("gt")
        pred = rec.get("pred")
        if gt is None or pred is None:
            continue
        if gt.shape != pred.shape:
            continue

        total_boards += 1
        total_cells += gt.size
        correct_cells += int((gt == pred).sum())
        if np.all(gt == pred):
            exact_match += 1

    if total_boards == 0 or total_cells == 0:
        print("[JSON boards] No comparable (gt,pred) pairs; skipping.")
        return

    cell_acc = correct_cells / total_cells
    board_acc = exact_match / total_boards
    print(
        f"[JSON boards] Boards compared: {total_boards}\n"
        f"  Cell-wise accuracy:   {cell_acc:.4f}\n"
        f"  Exact-board accuracy: {board_acc:.4f}"
    )


# -------------------------------------------------------------------------
# Helper to robustly read raw X from a split without triggering broken .X
# -------------------------------------------------------------------------
def _raw_X_from_split(split):
    if hasattr(split, "_X"):
        return getattr(split, "_X")

    d = getattr(split, "__dict__", {})
    if "X" in d:
        return d["X"]

    if hasattr(split, "paths"):
        return getattr(split, "paths")

    if hasattr(split, "images"):
        return getattr(split, "images")

    raise AttributeError(
        f"Could not find raw X data for split of type {type(split)}; "
        "tried _X, __dict__['X'], paths, images."
    )


# -------------------------------------------------------------------------
# ConceptDetector subclass: use ConceptSudokuCNN over boards
# -------------------------------------------------------------------------
class BoardConceptDetector(ConceptDetector):
    def __init__(
        self,
        cnn: ConceptSudokuCNN,
        boards: Dict[str, Dict[str, np.ndarray]],
        device: torch.device,
    ):
        super().__init__(embedding_model=cnn)
        self._cnn = cnn
        self._boards = boards
        self._device = device

    def _get_boards_for_split(self, split, which: str = "pred") -> np.ndarray:
        xs = []
        raw_X = _raw_X_from_split(split)
        for x in raw_X:
            key = Path(str(x)).name
            entry = self._boards.get(key)
            if entry is None or which not in entry:
                raise ValueError(
                    f"[BoardConceptDetector] No {which} board found in JSON for image {key}"
                )
            xs.append(entry[which])
        return np.stack(xs, axis=0)

    def predict_proba(
        self,
        split,
        embed_params: Optional[dict] = None,
        which: str = "pred",
        batch_size: int = 64,
        **kwargs,
    ) -> np.ndarray:
        self._cnn.eval()
        boards_np = self._get_boards_for_split(split, which=which)
        N = boards_np.shape[0]
        out_chunks: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, N, batch_size):
                chunk = boards_np[i : i + batch_size]
                t = torch.from_numpy(chunk).to(self._device)
                out = self._cnn(t)
                out_chunks.append(out.cpu().numpy())

        return np.concatenate(out_chunks, axis=0)

    def predict(
        self,
        split,
        embed_params: Optional[dict] = None,
        which: str = "pred",
        batch_size: int = 64,
        **kwargs,
    ) -> np.ndarray:
        probs = self.predict_proba(
            split,
            embed_params=embed_params,
            which=which,
            batch_size=batch_size,
            **kwargs,
        )
        return (probs > 0.5).astype(np.float32)


# -------------------------------------------------------------------------
# Train CBM (ConceptSudokuCNN + FrontEndModel) on clean concepts
# -------------------------------------------------------------------------
def train_cnn_cbm_from_sidecar(
    *,
    settings: dict,
    dataset,
    boards: Dict[str, Dict[str, np.ndarray]],
    epochs: int = 10,
    verbose: bool = True,
) -> ConceptBasedModel:
    from eval_common import iter_splits  # local import to avoid circulars

    split_map = {name.lower(): split for name, split in iter_splits(dataset)}
    train_split = split_map.get("training") or split_map.get("train")
    val_split = split_map.get("validation") or split_map.get("val")
    test_split = split_map.get("test")

    if train_split is None or val_split is None or test_split is None:
        raise RuntimeError(
            "[CBM train] Dataset must have training, validation, and test splits."
        )

    # Disable noise / missingness for CBM training
    for split in (train_split, val_split, test_split):
        if hasattr(split, "concept_noise"):
            split.concept_noise = False
        if hasattr(split, "concept_missing"):
            split.concept_missing = False

    def _get_base_concepts(split, tag: str) -> np.ndarray:
        if hasattr(split, "base_concepts"):
            Y = np.asarray(split.base_concepts, dtype=np.float32)
        else:
            Y = np.asarray(split.C, dtype=np.float32)

        if not np.isfinite(Y).all():
            raise ValueError(
                f"[CBM train] Non-finite values in {tag} concepts: "
                f"min={np.nanmin(Y)}, max={np.nanmax(Y)}"
            )
        y_min, y_max = float(Y.min()), float(Y.max())
        if y_min < 0.0 or y_max > 1.0:
            print(
                f"[CBM train] WARNING: {tag} concepts outside [0,1] "
                f"(min={y_min:.3f}, max={y_max:.3f}); "
                "mapping to {0,1} via (Y > 0)."
            )
        return (Y > 0).astype(np.float32)

    Y_train = _get_base_concepts(train_split, "train")
    Y_val = _get_base_concepts(val_split, "val")
    Y_test = _get_base_concepts(test_split, "test")

    def boards_for_split(split, which: str) -> np.ndarray:
        arrs = []
        raw_X = _raw_X_from_split(split)
        for x in raw_X:
            key = Path(str(x)).name
            entry = boards.get(key)
            if entry is None or which not in entry:
                raise ValueError(
                    f"[CBM train] No {which} board for {key} in JSON sidecar."
                )
            arrs.append(entry[which])
        return np.stack(arrs, axis=0)

    X_train_gt = boards_for_split(train_split, "gt")
    X_val_gt = boards_for_split(val_split, "gt")
    X_test_gt = boards_for_split(test_split, "gt")

    X_train = X_train_gt
    Y_train_aug = Y_train
    train_gt_n = X_train_gt.shape[0]
    train_pred_n = 0

    has_train_pred = all(
        "pred" in boards.get(Path(str(x)).name, {})
        for x in _raw_X_from_split(train_split)
    )
    if has_train_pred:
        X_train_pred = boards_for_split(train_split, "pred")
        train_pred_n = X_train_pred.shape[0]
        X_train = np.concatenate([X_train_gt, X_train_pred], axis=0)
        Y_train_aug = np.concatenate([Y_train, Y_train], axis=0)

    if verbose:
        if has_train_pred:
            print(
                "[CBM train] Training ConceptSudokuCNN on GT + predicted boards: "
                f"train_total={X_train.shape[0]} (gt={train_gt_n}, pred={train_pred_n}), "
                f"val={X_val_gt.shape[0]}, test={X_test_gt.shape[0]}"
            )
        else:
            print(
                "[CBM train] Training ConceptSudokuCNN on GT boards only: "
                f"train={train_gt_n}, val={X_val_gt.shape[0]}, test={X_test_gt.shape[0]}"
            )

    cnn = ConceptSudokuCNN().to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    def run_epoch(X, Y, train_mode: bool) -> float:
        cnn.train(train_mode)
        N = X.shape[0]
        total_loss = 0.0
        batch_size = 64

        idx = np.random.permutation(N) if train_mode else np.arange(N)
        for i in range(0, N, batch_size):
            j = idx[i : i + batch_size]
            xb = torch.from_numpy(X[j]).to(device)
            yb = torch.from_numpy(Y[j]).to(device)

            with torch.set_grad_enabled(train_mode):
                preds = cnn(xb)
                loss = criterion(preds, yb)

                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            total_loss += float(loss.item()) * len(j)

        return total_loss / N

    for ep in range(epochs):
        train_loss = run_epoch(X_train, Y_train_aug, train_mode=True)
        val_loss = run_epoch(X_val_gt, Y_val, train_mode=False)
        if verbose:
            print(
                f"[CBM train] Epoch {ep+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

    cnn.eval()

    def concept_metrics(X, Y, tag: str):
        with torch.no_grad():
            t = torch.from_numpy(X).to(device)
            preds = cnn(t).cpu().numpy()
        bin_preds = (preds > 0.5).astype(np.float32)
        per_bit = (bin_preds == Y).mean(axis=0)
        mean_acc = float(per_bit.mean())
        print(f"[CBM train] Concept accuracy ({tag} boards): mean={mean_acc:.4f}")
        return preds

    print("[CBM train] Evaluating CNN on GT boards...")
    _ = concept_metrics(X_test_gt, Y_test, tag="GT")

    if all(
        "pred" in boards.get(Path(str(x)).name, {})
        for x in _raw_X_from_split(test_split)
    ):
        X_test_pred = boards_for_split(test_split, "pred")
        print("[CBM train] Evaluating CNN on predicted boards...")
        _ = concept_metrics(X_test_pred, Y_test, tag="PRED")

    fe_model = FrontEndModel()
    fe_model.fit(train_split.C, train_split.y)

    with torch.no_grad():
        t_train_gt = torch.from_numpy(X_train_gt).to(device)
        C_train_gt_probs = cnn(t_train_gt).cpu().numpy()

    y_pred_train = fe_model.predict((C_train_gt_probs > 0.5).astype(np.float32))
    label_acc_train = float((y_pred_train == train_split.y).mean())
    print(f"[CBM train] Label accuracy on train (GT boards): {label_acc_train:.4f}")

    detector = BoardConceptDetector(cnn=cnn, boards=boards, device=device)

    cbm = ConceptBasedModel(
        concept_detector=detector,
        front_end_model=fe_model,
        propagate=True,
    )

    return cbm


# -------------------------------------------------------------------------
# CBM-compatible Selective DNN baseline (no interventions, but abstains)
# -------------------------------------------------------------------------
class CBMSelectiveDNNStrategy(ConceptualSafeguardsStrategy):
    """
    "Selective DNN" baseline operating on the CBM's label probabilities.
    Uses a confidence margin tau to abstain; no concept interventions.
    """

    name = DNN_STRATEGY_NAME

    def propose(self, model, batch, config: InterventionConfig):
        C_pred = batch.C_pred
        n, m = C_pred.shape

        y_prob = model._propagate_predict_proba_mc(C_pred)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = batch.y_true

        tau = float(getattr(config, "tau", 0.1))
        p1 = y_prob[:, 1]
        abstain_mask = (p1 >= tau) & (p1 <= 1.0 - tau)

        non_abstain_mask = ~abstain_mask
        if non_abstain_mask.any() and y_true is not None:
            selective_acc_before = float(
                (y_pred[non_abstain_mask] == y_true[non_abstain_mask]).mean()
            )
        else:
            selective_acc_before = float("nan")

        coverage_before = 1.0 - float(abstain_mask.mean())

        mask = np.zeros((n, m), dtype=bool)
        selected_instances = np.array([], dtype=int)

        details = {
            "selective_acc_before": selective_acc_before,
            "coverage_before": coverage_before,
            # we also set "after" to match, since no interventions happen
            "selective_acc_after": selective_acc_before,
            "coverage_after": coverage_before,
            "work_total_concepts": 0.0,
            "work_total_concepts_on_abstained": 0.0,
            "concepts_checked_per_board": 0.0,
            "cells_checked_per_board": 0.0,
            "concepts_checked_per_abstained_board": 0.0,
            "cells_checked_per_abstained_board": 0.0,
        }

        from concept_benchmark.intervention import StrategyProposal

        return StrategyProposal(
            mask=mask,
            selected_instances=selected_instances,
            details=details,
        )


# -------------------------------------------------------------------------
# Standard CBM DNN baseline (no abstention, no interventions)
# -------------------------------------------------------------------------
class StandardCBMDNNStrategy(ConceptualSafeguardsStrategy):
    """
    Standard CBM classifier baseline:
    - Always predicts (no abstention).
    - No concept interventions.
    """

    name = STANDARD_STRATEGY_NAME

    def propose(self, model, batch, config: InterventionConfig):
        C_pred = batch.C_pred
        n, m = C_pred.shape

        y_prob = model._propagate_predict_proba_mc(C_pred)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = batch.y_true

        if y_true is not None:
            acc = float((y_pred == y_true).mean())
        else:
            acc = float("nan")

        coverage = 1.0  # always predict, never abstain

        mask = np.zeros((n, m), dtype=bool)
        selected_instances = np.array([], dtype=int)

        details = {
            "selective_acc_before": acc,
            "selective_acc_after": acc,
            "coverage_before": coverage,
            "coverage_after": coverage,
            "work_total_concepts": 0.0,
            "work_total_concepts_on_abstained": 0.0,
            "concepts_checked_per_board": 0.0,
            "cells_checked_per_board": 0.0,
            "concepts_checked_per_abstained_board": 0.0,
            "cells_checked_per_abstained_board": 0.0,
        }

        from concept_benchmark.intervention import StrategyProposal

        return StrategyProposal(
            mask=mask,
            selected_instances=selected_instances,
            details=details,
        )


# -------------------------------------------------------------------------
# Helpers for metrics + tau calibration
# -------------------------------------------------------------------------
def _inject_work_metrics_from_result(result, metrics: dict) -> None:
    proposal = getattr(result, "proposal", None)

    if proposal is None:
        for name in ("strategy_proposal",):
            if hasattr(result, name):
                candidate = getattr(result, name)
                if hasattr(candidate, "details") and hasattr(candidate, "mask"):
                    proposal = candidate
                    break

    if proposal is None:
        return

    details = getattr(proposal, "details", None)
    if not isinstance(details, dict):
        return

    for key in [
        "concepts_checked_per_board",
        "cells_checked_per_board",
        "concepts_checked_per_abstained_board",
        "cells_checked_per_abstained_board",
        "work_total_concepts",
        "work_total_concepts_on_abstained",
    ]:
        if key in details and details[key] is not None:
            metrics[key] = float(details[key])


def write_raw_metrics_csv(records: List[MetricRecord], path: Path) -> None:
    if not records:
        print(f"[main] No metric records to write to {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        rows = [asdict(r) for r in records]
    except TypeError:
        rows = [r.__dict__ for r in records]

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if "params" in row and not isinstance(row["params"], str):
                row = dict(row)
                row["params"] = json.dumps(row["params"])
            writer.writerow(row)


def calibrate_tau_on_validation(
    *,
    runner: ConceptInterventionRunner,
    strategy,
    dataset,
    tau_candidates: Sequence[float],
    target_label: str,
    target_value: float,
    settings: dict,
    verbose: bool = False,
    acc_target: float = ACC_TARGET,
) -> Sequence[float]:
    val_split = None
    for split_name, split_data in iter_splits(dataset):
        if split_name.lower().startswith("val"):
            val_split = split_data
            break

    if val_split is None:
        if verbose:
            print(
                f"[tau calibration] data={settings['data_name']} "
                f"target_label={target_label} target_value={target_value} "
                f"strategy={getattr(strategy, 'name', 'strategy')} "
                f"tau_grid={list(tau_candidates)} acc_target={acc_target} "
                "→ no validation split; using tau grid unchanged."
            )
        return list(tau_candidates)

    best_tau = None
    best_sel = -np.inf
    satisfying: List[Tuple[float, float]] = []

    for tau in tau_candidates:
        cfg = InterventionConfig(tau=float(tau))
        setattr(cfg, "budget_frac", 0.0)
        setattr(cfg, "max_concepts_per_instance", None)

        result = runner.run(
            strategy=strategy,
            config=cfg,
            dataset=val_split,
        )

        m = result.strat_metrics

        sel_after = m.get("selective_acc_after", None)
        if sel_after is None:
            sel_after = m.get("selective_acc_before", None)

        if sel_after is None:
            continue

        sel_after = float(sel_after)
        if sel_after > best_sel:
            best_sel = sel_after
            best_tau = float(tau)
        if sel_after >= acc_target:
            satisfying.append((float(tau), sel_after))

    if satisfying:
        satisfying.sort(key=lambda x: x[0])
        chosen_tau = satisfying[0][0]
        chosen_acc = satisfying[0][1]
    else:
        chosen_tau = best_tau
        chosen_acc = best_sel

    if verbose:
        name = getattr(strategy, "name", "strategy")
        if chosen_tau is None:
            print(
                f"[tau calibration] data={settings['data_name']} "
                f"target_label={target_label} target_value={target_value} "
                f"strategy={name} "
                f"tau_grid={list(tau_candidates)} acc_target={acc_target} "
                "→ could not compute selective accuracy for any tau; using tau grid unchanged."
            )
        else:
            status = (
                ">= {:.0f}% target".format(acc_target * 100.0)
                if chosen_acc >= acc_target
                else "(best available < {:.0f}%)".format(acc_target * 100.0)
            )
            print(
                f"[tau calibration] data={settings['data_name']} "
                f"target_label={target_label} target_value={target_value} "
                f"strategy={name} "
                f"tau_grid={list(tau_candidates)} acc_target={acc_target} "
                f"chosen_tau={chosen_tau:.3f} val_selective_acc={chosen_acc:.3f} {status} "
                "(budget_frac=0.0, max_concepts_per_instance=None)"
            )

    if chosen_tau is None:
        return list(tau_candidates)
    return [chosen_tau]


# -------------------------------------------------------------------------
# Progress estimation (approximate)
# -------------------------------------------------------------------------
def _estimate_total_configs(
    *,
    target_labels: Sequence[str],
    tau_values: Sequence[float],
    mechanisms: Sequence[str],
    missing_levels: Sequence[float],
) -> int:
    """
    Approximate number of progress steps:

    - DNN baseline: run once per (target_label, tau) on clean data.
    - Standard DNN: run once per target_label (tau is irrelevant but we still run once).
    - No-intervention: run once per (target_label, mechanism, level, tau).
    """
    missing_counts = 0
    for mech in mechanisms:
        if mech == "none":
            missing_counts += 1  # level=0.0
        else:
            missing_counts += len(missing_levels)

    dnn_configs = len(target_labels) * len(tau_values)
    standard_configs = len(target_labels)  # one per label
    noint_configs = missing_counts * len(target_labels) * len(tau_values)
    return dnn_configs + standard_configs + noint_configs


# -------------------------------------------------------------------------
# CLI + main loop (DNN + standard + no-intervention)
# -------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir
        / "big_demo"
        / "conceptual_safeguards_metrics_dnn_noint.csv",
        help="Destination CSV path for the melted metric table.",
    )
    parser.add_argument(
        "--data-name",
        choices=tuple(BASE_DATASET_CONFIGS.keys()),
        default="multimodal_m_21_image",
        help="Dataset key used for settings + (some) metadata.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path(
            "/home/mds010/concept-benchmark/data/sudoku/multimodal_m_21/image/sudoku_dataset.pkl"
        ),
        help=(
            "Path to an existing dataset pickle *with splits* "
            "(e.g., multimodel_m_21_image.pkl)."
        ),
    )
    parser.add_argument(
        "--target-accuracy-labels",
        choices=tuple(big_demo_utils.DIFFICULTY.keys()),
        nargs="+",
        help="Subset of target accuracy labels (easy/medium/hard).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        help=(
            "Candidate confidence margins. For each target label and each "
            "strategy (selective DNN / no-intervention), "
            "tau is calibrated on the validation split by choosing the smallest "
            "candidate with selective accuracy >= 0.9 if possible."
        ),
    )
    parser.add_argument(
        "--concept-json",
        type=Path,
        default=Path(
            "/home/mds010/concept-benchmark/data/sudoku/multimodal_m_21_image/"
            "ocr_predictions_fast.jsonl"
        ),
        help=(
            "OCR JSONL sidecar with predicted boards and GT boards used to "
            "train ConceptSudokuCNN CBM and for board evaluation."
        ),
    )
    parser.add_argument(
        "--missing-mechanisms",
        nargs="+",
        choices=("none", "mcar", "mnar"),
        default=["none", "mcar", "mnar"],
        help="Concept missingness mechanisms to consider.",
    )
    parser.add_argument(
        "--concept-missing",
        type=float,
        nargs="+",
        help="Override concept missingness levels (defaults to utils.CONCEPT_MISSING).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress reporting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skip information and tau calibration details.",
    )
    parser.add_argument(
        "--cbm-train-epochs",
        type=int,
        default=10,
        help="Number of epochs to train ConceptSudokuCNN CBM.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    tau_values = list(args.tau)
    for tau in tau_values:
        if not (0.0 <= tau <= 0.5):
            raise ValueError("All tau values must lie within [0, 0.5].")

    target_pairs = default_target_options(args.target_accuracy_labels)
    target_labels = [label for label, _ in target_pairs]

    missing_levels = default_missingness_levels(args.concept_missing)

    total = None
    if not args.no_progress:
        total = _estimate_total_configs(
            target_labels=target_labels,
            tau_values=tau_values,
            mechanisms=args.missing_mechanisms,
            missing_levels=missing_levels,
        )
    progress = None if total is None else tqdm(total=total, desc="Configs")

    records: List[MetricRecord] = []
    dnn_strategy = CBMSelectiveDNNStrategy()
    noint_strategy = NoInterventionCBMStrategy()
    standard_strategy = StandardCBMDNNStrategy()

    # For sanity in logs:
    noint_strategy.name = NOINT_STRATEGY_NAME

    if not args.concept_json.is_file():
        raise FileNotFoundError(f"Concept JSON not found: {args.concept_json}")
    boards = load_json_boards(args.concept_json)
    eval_boards_from_json(boards)

    try:
        for target_label, target_value in target_pairs:
            if not args.dataset_file.is_file():
                raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")

            if args.verbose:
                print(f"[main] Loading CLEAN dataset from: {args.dataset_file}")
            dataset_clean = fileutils.load(args.dataset_file)
            dataset_clean.generate_cvindices(seed=42)
            dataset_clean.split("K05N01", fold_num_validation=4, fold_num_test=5)

            clean_settings = build_settings(
                data_name=args.data_name,
                target_accuracy=target_value,
                concept_missing=0.0,
                concept_noise=0.0,
                concept_missing_mech="none",
            )

            if args.verbose:
                print(f"[main] Clean settings for CBM train: {clean_settings}")

            # Train CBM once per target_label on clean concepts
            cbm = train_cnn_cbm_from_sidecar(
                settings=clean_settings,
                dataset=dataset_clean,
                boards=boards,
                epochs=args.cbm_train_epochs,
                verbose=args.verbose,
            )

            runner_clean = ConceptInterventionRunner(model=cbm)

            # Calibrate tau for DNN baseline on CLEAN validation split
            dnn_tau_values = calibrate_tau_on_validation(
                runner=runner_clean,
                strategy=dnn_strategy,
                dataset=dataset_clean,
                tau_candidates=tau_values,
                target_label=target_label,
                target_value=target_value,
                settings=clean_settings,
                verbose=args.verbose,
                acc_target=ACC_TARGET,
            )

            # ---- Selective DNN baseline (clean only, no missingness) ----
            dnn_before = len(records)
            for tau in dnn_tau_values:
                dnn_config = InterventionConfig(tau=float(tau))
                setattr(dnn_config, "budget_frac", 0.0)
                setattr(dnn_config, "max_concepts_per_instance", None)

                for split_name, split_data in iter_splits(dataset_clean):
                    if split_name not in INTERVENTION_SPLITS:
                        continue

                    dnn_result = runner_clean.run(
                        strategy=dnn_strategy,
                        config=dnn_config,
                        dataset=split_data,
                    )

                    m = dnn_result.strat_metrics
                    _inject_work_metrics_from_result(dnn_result, m)

                    # Force 0 work for DNN baseline
                    m.setdefault("concepts_checked_per_board", 0.0)
                    m.setdefault("cells_checked_per_board", 0.0)
                    m.setdefault("concepts_checked_per_abstained_board", 0.0)
                    m.setdefault("cells_checked_per_abstained_board", 0.0)
                    m.setdefault("work_total_concepts", 0.0)
                    m.setdefault("work_total_concepts_on_abstained", 0.0)

                    if args.verbose:
                        print(
                            "[run-dnn] strategy={strategy} "
                            "data={data_name} ({data_type}) "
                            "target_label={target_label} target_value={target_value} "
                            "split={split} tau={tau:.3f} "
                            "mech={mech} missing={missing}".format(
                                strategy=DNN_STRATEGY_NAME,
                                data_name=clean_settings["data_name"],
                                data_type=clean_settings["data_type"],
                                target_label=target_label,
                                target_value=target_value,
                                split=split_name,
                                tau=tau,
                                mech="none",
                                missing=0.0,
                            )
                        )
                        print(
                            "      selective_acc_before={sel_before} "
                            "selective_acc_after={sel_after} "
                            "coverage_before={cov_before} "
                            "coverage_after={cov_after}".format(
                                sel_before=m.get("selective_acc_before"),
                                sel_after=m.get("selective_acc_after"),
                                cov_before=m.get("coverage_before"),
                                cov_after=m.get("coverage_after"),
                            )
                        )

                    for metric_name, metric_value in m.items():
                        if metric_value is None:
                            continue
                        records.append(
                            MetricRecord(
                                strategy=DNN_STRATEGY_NAME,
                                metric=metric_name,
                                value=float(metric_value),
                                split=split_name,
                                data_name=clean_settings["data_name"],
                                data_type=clean_settings["data_type"],
                                concept_missing=0.0,
                                concept_missing_mech="none",
                                target_accuracy_label=target_label,
                                target_accuracy_value=target_value,
                                params={
                                    "tau": float(tau),
                                    "budget_frac": 0.0,
                                    "max_concepts_per_instance": None,
                                },
                            )
                        )

                    sel_after = m.get("selective_acc_after", None)
                    cov_after = m.get("coverage_after", None)
                    cov_before = m.get("coverage_before", cov_after)

                    if sel_after is not None:
                        records.append(
                            MetricRecord(
                                strategy=DNN_STRATEGY_NAME,
                                metric="selective_accuracy",
                                value=float(sel_after),
                                split=split_name,
                                data_name=clean_settings["data_name"],
                                data_type=clean_settings["data_type"],
                                concept_missing=0.0,
                                concept_missing_mech="none",
                                target_accuracy_label=target_label,
                                target_accuracy_value=target_value,
                                params={
                                    "tau": float(tau),
                                    "budget_frac": 0.0,
                                    "max_concepts_per_instance": None,
                                },
                            )
                        )

                    if cov_after is not None:
                        records.append(
                            MetricRecord(
                                strategy=DNN_STRATEGY_NAME,
                                metric="coverage_pct_handled_by_ai",
                                value=float(cov_after) * 100.0,
                                split=split_name,
                                data_name=clean_settings["data_name"],
                                data_type=clean_settings["data_type"],
                                concept_missing=0.0,
                                concept_missing_mech="none",
                                target_accuracy_label=target_label,
                                target_accuracy_value=target_value,
                                params={
                                    "tau": float(tau),
                                    "budget_frac": 0.0,
                                    "max_concepts_per_instance": None,
                                },
                            )
                        )

                    if cov_after is not None and cov_before is not None:
                        cov_gain = (float(cov_after) - float(cov_before)) * 100.0
                        records.append(
                            MetricRecord(
                                strategy=DNN_STRATEGY_NAME,
                                metric="coverage_gain",
                                value=cov_gain,
                                split=split_name,
                                data_name=clean_settings["data_name"],
                                data_type=clean_settings["data_type"],
                                concept_missing=0.0,
                                concept_missing_mech="none",
                                target_accuracy_label=target_label,
                                target_accuracy_value=target_value,
                                params={
                                    "tau": float(tau),
                                    "budget_frac": 0.0,
                                    "max_concepts_per_instance": None,
                                },
                            )
                        )

                if progress is not None:
                    progress.update(1)

            if args.verbose:
                dnn_added = len(records) - dnn_before
                print(
                    f"[main] DNN records added for "
                    f"target_label={target_label}, "
                    f"target_value={target_value}, "
                    f"mech=none, missing=0.0: {dnn_added}"
                )

            # ---- Standard CBM DNN baseline (no abstention, clean only) ----
            standard_before = len(records)
            std_config = InterventionConfig(tau=0.0)
            setattr(std_config, "budget_frac", 0.0)
            setattr(std_config, "max_concepts_per_instance", None)

            for split_name, split_data in iter_splits(dataset_clean):
                if split_name not in INTERVENTION_SPLITS:
                    continue

                std_result = runner_clean.run(
                    strategy=standard_strategy,
                    config=std_config,
                    dataset=split_data,
                )

                m = std_result.strat_metrics
                _inject_work_metrics_from_result(std_result, m)

                # explicitly enforce 0 work
                m.setdefault("concepts_checked_per_board", 0.0)
                m.setdefault("cells_checked_per_board", 0.0)
                m.setdefault("concepts_checked_per_abstained_board", 0.0)
                m.setdefault("cells_checked_per_abstained_board", 0.0)
                m.setdefault("work_total_concepts", 0.0)
                m.setdefault("work_total_concepts_on_abstained", 0.0)

                if args.verbose:
                    print(
                        "[run-standard] strategy={strategy} "
                        "data={data_name} ({data_type}) "
                        "target_label={target_label} target_value={target_value} "
                        "split={split} tau={tau:.3f} "
                        "mech={mech} missing={missing}".format(
                            strategy=STANDARD_STRATEGY_NAME,
                            data_name=clean_settings["data_name"],
                            data_type=clean_settings["data_type"],
                            target_label=target_label,
                            target_value=target_value,
                            split=split_name,
                            tau=0.0,
                            mech="none",
                            missing=0.0,
                        )
                    )
                    print(
                        "      selective_acc_before={sel_before} "
                        "selective_acc_after={sel_after} "
                        "coverage_before={cov_before} "
                        "coverage_after={cov_after}".format(
                            sel_before=m.get("selective_acc_before"),
                            sel_after=m.get("selective_acc_after"),
                            cov_before=m.get("coverage_before"),
                            cov_after=m.get("coverage_after"),
                        )
                    )

                for metric_name, metric_value in m.items():
                    if metric_value is None:
                        continue
                    records.append(
                        MetricRecord(
                            strategy=STANDARD_STRATEGY_NAME,
                            metric=metric_name,
                            value=float(metric_value),
                            split=split_name,
                            data_name=clean_settings["data_name"],
                            data_type=clean_settings["data_type"],
                            concept_missing=0.0,
                            concept_missing_mech="none",
                            target_accuracy_label=target_label,
                            target_accuracy_value=target_value,
                            params={
                                "tau": 0.0,
                                "budget_frac": 0.0,
                                "max_concepts_per_instance": None,
                            },
                        )
                    )

                sel_after = m.get("selective_acc_after", None)
                cov_after = m.get("coverage_after", None)
                cov_before = m.get("coverage_before", cov_after)

                if sel_after is not None:
                    records.append(
                        MetricRecord(
                            strategy=STANDARD_STRATEGY_NAME,
                            metric="selective_accuracy",
                            value=float(sel_after),
                            split=split_name,
                            data_name=clean_settings["data_name"],
                            data_type=clean_settings["data_type"],
                            concept_missing=0.0,
                            concept_missing_mech="none",
                            target_accuracy_label=target_label,
                            target_accuracy_value=target_value,
                            params={
                                "tau": 0.0,
                                "budget_frac": 0.0,
                                "max_concepts_per_instance": None,
                            },
                        )
                    )

                if cov_after is not None:
                    records.append(
                        MetricRecord(
                            strategy=STANDARD_STRATEGY_NAME,
                            metric="coverage_pct_handled_by_ai",
                            value=float(cov_after) * 100.0,
                            split=split_name,
                            data_name=clean_settings["data_name"],
                            data_type=clean_settings["data_type"],
                            concept_missing=0.0,
                            concept_missing_mech="none",
                            target_accuracy_label=target_label,
                            target_accuracy_value=target_value,
                            params={
                                "tau": 0.0,
                                "budget_frac": 0.0,
                                "max_concepts_per_instance": None,
                            },
                        )
                    )

                if cov_after is not None and cov_before is not None:
                    cov_gain = (float(cov_after) - float(cov_before)) * 100.0
                    records.append(
                        MetricRecord(
                            strategy=STANDARD_STRATEGY_NAME,
                            metric="coverage_gain",
                            value=cov_gain,
                            split=split_name,
                            data_name=clean_settings["data_name"],
                            data_type=clean_settings["data_type"],
                            concept_missing=0.0,
                            concept_missing_mech="none",
                            target_accuracy_label=target_label,
                            target_accuracy_value=target_value,
                            params={
                                "tau": 0.0,
                                "budget_frac": 0.0,
                                "max_concepts_per_instance": None,
                            },
                        )
                    )

            if progress is not None:
                progress.update(1)

            if args.verbose:
                standard_added = len(records) - standard_before
                print(
                    f"[main] Standard DNN records added for "
                    f"target_label={target_label}, "
                    f"target_value={target_value}, "
                    f"mech=none, missing=0.0: {standard_added}"
                )

            # ---- No-intervention CBM baseline with missingness sweeps ----
            for mechanism in args.missing_mechanisms:
                if mechanism == "none":
                    levels = (0.0,)
                else:
                    levels = missing_levels

                for concept_missing in levels:
                    # Reload dataset per missingness config
                    if args.verbose:
                        print(
                            f"[main] Loading dataset for no-intervention: {args.dataset_file}"
                        )
                    dataset = fileutils.load(args.dataset_file)
                    dataset.generate_cvindices(seed=42)
                    dataset.split("K05N01", fold_num_validation=4, fold_num_test=5)

                    settings = build_settings(
                        data_name=args.data_name,
                        target_accuracy=target_value,
                        concept_missing=concept_missing,
                        concept_noise=0.0,
                        concept_missing_mech=mechanism,
                    )

                    if args.verbose:
                        print(f"[main] Settings (NOINT): {settings}")

                    apply_missingness(dataset, mechanism, float(concept_missing))

                    if args.verbose:
                        print(
                            f"[main] Missingness config (NOINT): "
                            f"mech={mechanism}, concept_missing={concept_missing}"
                        )

                    runner = ConceptInterventionRunner(model=cbm)

                    noint_tau_values = calibrate_tau_on_validation(
                        runner=runner,
                        strategy=noint_strategy,
                        dataset=dataset,
                        tau_candidates=tau_values,
                        target_label=target_label,
                        target_value=target_value,
                        settings=settings,
                        verbose=args.verbose,
                        acc_target=ACC_TARGET,
                    )

                    if args.verbose:
                        print(
                            f"[main] Tau (noint={noint_tau_values}) "
                            f"for mechanism={mechanism}, missing={concept_missing}"
                        )

                    noint_before = len(records)
                    for tau in noint_tau_values:
                        noint_config = InterventionConfig(tau=float(tau))
                        setattr(noint_config, "budget_frac", 0.0)
                        setattr(noint_config, "max_concepts_per_instance", None)

                        for split_name, split_data in iter_splits(dataset):
                            if split_name not in INTERVENTION_SPLITS:
                                continue

                            noint_result = runner.run(
                                strategy=noint_strategy,
                                config=noint_config,
                                dataset=split_data,
                            )

                            m = noint_result.strat_metrics
                            _inject_work_metrics_from_result(noint_result, m)

                            m.setdefault("concepts_checked_per_board", 0.0)
                            m.setdefault("cells_checked_per_board", 0.0)
                            m.setdefault("concepts_checked_per_abstained_board", 0.0)
                            m.setdefault("cells_checked_per_abstained_board", 0.0)
                            m.setdefault("work_total_concepts", 0.0)
                            m.setdefault("work_total_concepts_on_abstained", 0.0)

                            if args.verbose:
                                print(
                                    "[run-noint] strategy={strategy} "
                                    "data={data_name} ({data_type}) "
                                    "target_label={target_label} target_value={target_value} "
                                    "split={split} tau={tau:.3f} "
                                    "mech={mech} missing={missing}".format(
                                        strategy=NOINT_STRATEGY_NAME,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        target_label=target_label,
                                        target_value=target_value,
                                        split=split_name,
                                        tau=tau,
                                        mech=mechanism,
                                        missing=concept_missing,
                                    )
                                )
                                print(
                                    "      selective_acc_before={sel_before} "
                                    "selective_acc_after={sel_after} "
                                    "coverage_before={cov_before} "
                                    "coverage_after={cov_after}".format(
                                        sel_before=m.get("selective_acc_before"),
                                        sel_after=m.get("selective_acc_after"),
                                        cov_before=m.get("coverage_before"),
                                        cov_after=m.get("coverage_after"),
                                    )
                                )

                            for metric_name, metric_value in m.items():
                                if metric_value is None:
                                    continue
                                records.append(
                                    MetricRecord(
                                        strategy=NOINT_STRATEGY_NAME,
                                        metric=metric_name,
                                        value=float(metric_value),
                                        split=split_name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        concept_missing=float(concept_missing),
                                        concept_missing_mech=mechanism,
                                        target_accuracy_label=target_label,
                                        target_accuracy_value=target_value,
                                        params={
                                            "tau": float(tau),
                                            "budget_frac": 0.0,
                                            "max_concepts_per_instance": None,
                                        },
                                    )
                                )

                            sel_after = m.get("selective_acc_after", None)
                            cov_after = m.get("coverage_after", None)
                            cov_before = m.get("coverage_before", cov_after)

                            if sel_after is not None:
                                records.append(
                                    MetricRecord(
                                        strategy=NOINT_STRATEGY_NAME,
                                        metric="selective_accuracy",
                                        value=float(sel_after),
                                        split=split_name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        concept_missing=float(concept_missing),
                                        concept_missing_mech=mechanism,
                                        target_accuracy_label=target_label,
                                        target_accuracy_value=target_value,
                                        params={
                                            "tau": float(tau),
                                            "budget_frac": 0.0,
                                            "max_concepts_per_instance": None,
                                        },
                                    )
                                )

                            if cov_after is not None:
                                records.append(
                                    MetricRecord(
                                        strategy=NOINT_STRATEGY_NAME,
                                        metric="coverage_pct_handled_by_ai",
                                        value=float(cov_after) * 100.0,
                                        split=split_name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        concept_missing=float(concept_missing),
                                        concept_missing_mech=mechanism,
                                        target_accuracy_label=target_label,
                                        target_accuracy_value=target_value,
                                        params={
                                            "tau": float(tau),
                                            "budget_frac": 0.0,
                                            "max_concepts_per_instance": None,
                                        },
                                    )
                                )

                            if cov_after is not None and cov_before is not None:
                                cov_gain = (float(cov_after) - float(cov_before)) * 100.0
                                records.append(
                                    MetricRecord(
                                        strategy=NOINT_STRATEGY_NAME,
                                        metric="coverage_gain",
                                        value=cov_gain,
                                        split=split_name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        concept_missing=float(concept_missing),
                                        concept_missing_mech=mechanism,
                                        target_accuracy_label=target_label,
                                        target_accuracy_value=target_value,
                                        params={
                                            "tau": float(tau),
                                            "budget_frac": 0.0,
                                            "max_concepts_per_instance": None,
                                        },
                                    )
                                )

                        if progress is not None:
                            progress.update(1)

                    if args.verbose:
                        noint_added = len(records) - noint_before
                        print(
                            f"[main] No-intervention records added for "
                            f"target_label={target_label}, "
                            f"target_value={target_value}, "
                            f"mech={mechanism}, missing={concept_missing}: {noint_added}"
                        )

    finally:
        if progress is not None:
            progress.close()

    n_total = len(records)
    n_dnn = sum(1 for r in records if r.strategy == DNN_STRATEGY_NAME)
    n_std = sum(1 for r in records if r.strategy == STANDARD_STRATEGY_NAME)
    n_noint = sum(1 for r in records if r.strategy == NOINT_STRATEGY_NAME)
    print(
        f"[main] Collected {n_total} metric records "
        f"({n_dnn} {DNN_STRATEGY_NAME}, "
        f"{n_std} {STANDARD_STRATEGY_NAME}, "
        f"{n_noint} {NOINT_STRATEGY_NAME})"
    )

    write_metrics_csv(records, args.output)
    print(f"[main] Wrote aggregated metrics to {args.output}")

    raw_output = args.output.with_name(args.output.stem + "_raw.csv")
    write_raw_metrics_csv(records, raw_output)
    print(f"[main] Wrote raw metrics (DNN + standard + no-intervention) to {raw_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
