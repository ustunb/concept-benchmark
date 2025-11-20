#!/usr/bin/env python3
"""
Evaluate the conceptual safeguards intervention strategy with a CBM
trained from Sudoku boards using ConceptSudokuCNN, plus a Selective DNN
baseline.

Differences from the original big_demo script
---------------------------------------------
- Uses ConceptSudokuCNN as the concept detector:
    X = flattened Sudoku boards (length 81, digits 0..9)
    C = 27 binary concepts (row/col/block)
- Does NOT rely on BASE_DATASET_CONFIGS or build_settings.
  You can pass ANY ConceptDataset pickle via --dataset-file.
- --data-name is now a free-form string, used only for metadata in outputs.
- User-specified budget_frac must be in (0,1]; budget_frac=0.0 is not allowed.
  Internally, tau calibration still uses budget_frac=0.0, but that is not
  exposed as a user setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from concept_benchmark.ext import fileutils
from concept_benchmark.intervention import (
    ConceptualSafeguardsStrategy,
    InterventionConfig,
    StrategyProposal,
)
from concept_benchmark.models import ConceptBasedModel, FrontEndModel, ConceptDetector
from concept_benchmark.paths import results_dir
import utils as big_demo_utils
from eval_common import (
    INTERVENTION_SPLITS,
    ConceptInterventionRunner,
    MetricRecord,
    default_target_options,
    iter_splits,
    write_metrics_csv,
)

ACC_TARGET = 0.90  # target selective accuracy on validation for tau calibration

# -------------------------------------------------------------------------
# Device
# -------------------------------------------------------------------------
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# -------------------------------------------------------------------------
# Concept model: ConceptSudokuCNN (boards -> 27 concepts)
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
        x: (N, 81) integer Sudoku board digits (0..9).
        Returns: (N, 27) concept probabilities in [0,1].
        """
        x = x.long()
        x = self.embedding(x)                                   # (N, 81, D)
        x = x.permute(0, 2, 1).view(-1, x.size(2), 9, 9)        # (N, D, 9, 9)
        x = F.relu(self.conv1(x))
        features = F.relu(self.conv2(x))                        # (N, 128, 9, 9)

        # Rows
        row_features = torch.mean(features, dim=3).permute(0, 2, 1)   # (N, 9, 128)
        row_preds = self.row_head(row_features).squeeze(-1)           # (N, 9)

        # Cols
        col_features = torch.mean(features, dim=2).permute(0, 2, 1)   # (N, 9, 128)
        col_preds = self.col_head(col_features).squeeze(-1)           # (N, 9)

        # Blocks
        block_features = self.block_pool(features)                    # (N, 128, 3, 3)
        block_features = block_features.view(
            features.size(0), features.size(1), -1
        ).permute(0, 2, 1)                                            # (N, 9, 128)
        block_preds = self.block_head(block_features).squeeze(-1)     # (N, 9)

        all_preds = torch.cat([row_preds, col_preds, block_preds], dim=1)  # (N, 27)
        return torch.sigmoid(all_preds)


# -------------------------------------------------------------------------
# ConceptDetector subclass: use ConceptSudokuCNN directly on boards in X
# -------------------------------------------------------------------------
class SudokuBoardConceptDetector(ConceptDetector):
    """
    ConceptDetector that runs ConceptSudokuCNN on split.X, assuming each
    row of X is a flattened 9x9 Sudoku board (length 81, values 0..9).

    - We train ConceptSudokuCNN directly on (X -> C) from the dataset.
    - During interventions we always use split.X as the board.
    """

    def __init__(
        self,
        cnn: ConceptSudokuCNN,
        device: torch.device,
    ):
        super().__init__(embedding_model=cnn)
        self._cnn = cnn
        self._device = device

    def _get_boards_for_split(self, split) -> np.ndarray:
        X = split.X
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        X = X.astype(np.int64)  # digits 0..9, shape (N, 81) or (N, 9, 9)
        if X.ndim == 3 and X.shape[1:] == (9, 9):
            X = X.reshape(X.shape[0], -1)
        if X.shape[1] != 81:
            raise ValueError(
                f"Expected boards of length 81 (flattened 9x9), got X.shape={X.shape}"
            )
        return X

    def predict_proba(
        self,
        split,
        embed_params: Optional[dict] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> np.ndarray:
        self._cnn.eval()
        boards = self._get_boards_for_split(split)  # (N, 81)
        N = boards.shape[0]
        out_chunks: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, N, batch_size):
                chunk = boards[i : i + batch_size]
                t = torch.from_numpy(chunk).to(self._device)
                out = self._cnn(t)  # (B, 27)
                out_chunks.append(out.cpu().numpy())

        return np.concatenate(out_chunks, axis=0)

    def predict(
        self,
        split,
        embed_params: Optional[dict] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> np.ndarray:
        probs = self.predict_proba(
            split,
            embed_params=embed_params,
            batch_size=batch_size,
            **kwargs,
        )
        return (probs > 0.5).astype(np.float32)


# -------------------------------------------------------------------------
# Generic settings builder (no BASE_DATASET_CONFIGS)
# -------------------------------------------------------------------------
def build_generic_settings(dataset, data_name: Optional[str], target_value: float) -> dict:
    """
    Build a minimal settings dict for logging and the few places that
    previously used BASE_DATASET_CONFIGS.

    - data_name: free-form string (or taken from dataset.meta if None)
    - data_type: from dataset.meta['data_type'] or 'tabular'
    - max_corrupt: defaults to dataset.n_concepts
    """
    meta = getattr(dataset, "meta", getattr(getattr(dataset, "_full", None), "meta", {})) or {}
    name = data_name or meta.get("data_name", "custom_dataset")
    data_type = meta.get("data_type", "tabular")
    n_concepts = getattr(dataset, "n_concepts", None)
    if n_concepts is None and hasattr(dataset, "C"):
        n_concepts = dataset.C.shape[1]
    if n_concepts is None:
        raise ValueError("Could not infer n_concepts from dataset.")

    return {
        "data_name": name,
        "data_type": data_type,
        "max_corrupt": int(n_concepts),
        "target_accuracy": float(target_value),
    }


# -------------------------------------------------------------------------
# Ensure base_concepts is present (no noise/missingness here)
# -------------------------------------------------------------------------
def ensure_base_concepts(dataset) -> None:
    """
    Ensure every split object (train/val/test) has a .base_concepts attribute.

    Since we are not modeling concept noise/missingness here, we can safely
    set base_concepts = C.
    """
    for split_name, split in iter_splits(dataset):
        if hasattr(split, "C") and not hasattr(split, "base_concepts"):
            split.base_concepts = split.C.copy()


# -------------------------------------------------------------------------
# Train CBM (ConceptSudokuCNN + FrontEndModel) from boards + dataset.C/y
# -------------------------------------------------------------------------
def train_sudoku_board_cbm(
    *,
    settings: dict,
    dataset,
    epochs: int = 10,
    verbose: bool = True,
) -> ConceptBasedModel:
    train_split = dataset.training
    val_split = getattr(dataset, "validation", None)
    test_split = getattr(dataset, "test", None)

    if val_split is None or test_split is None:
        raise RuntimeError(
            "[CBM train] Dataset must have training, validation, and test splits."
        )

    def _boards_and_concepts(split):
        X = split.X
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        # Allow (N, 81) or (N, 9, 9); always convert to (N, 81) int
        X = X.astype(np.int64)
        if X.ndim == 3 and X.shape[1:] == (9, 9):
            X = X.reshape(X.shape[0], -1)
        if X.shape[1] != 81:
            raise ValueError(
                f"[CBM train] Expected flattened 9x9 boards (length 81), got X.shape={X.shape}"
            )
        C = split.C.astype(np.float32)
        y = split.y.astype(np.int64)
        return X, C, y

    X_train, C_train, y_train = _boards_and_concepts(train_split)
    X_val, C_val, y_val = _boards_and_concepts(val_split)
    X_test, C_test, y_test = _boards_and_concepts(test_split)

    n_concepts = C_train.shape[1]

    if n_concepts != 27:
        raise ValueError(
            f"[CBM train] ConceptSudokuCNN expects 27 concepts (9 rows, 9 cols, 9 blocks), "
            f"but got n_concepts={n_concepts}"
        )

    if verbose:
        print(
            "[CBM train] Training ConceptSudokuCNN on Sudoku boards: "
            f"train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}"
        )

    cnn = ConceptSudokuCNN().to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    def run_epoch(X, C, train_mode: bool) -> float:
        cnn.train(train_mode)
        N = X.shape[0]
        total_loss = 0.0
        batch_size = 128

        idx = np.random.permutation(N) if train_mode else np.arange(N)
        for i in range(0, N, batch_size):
            j = idx[i : i + batch_size]
            xb = torch.from_numpy(X[j]).to(device)   # ints (board)
            cb = torch.from_numpy(C[j]).to(device)   # float concepts

            with torch.set_grad_enabled(train_mode):
                preds = cnn(xb)
                loss = criterion(preds, cb)

                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            total_loss += float(loss.item()) * len(j)

        return total_loss / N

    for ep in range(epochs):
        train_loss = run_epoch(X_train, C_train, train_mode=True)
        val_loss = run_epoch(X_val, C_val, train_mode=False)
        if verbose:
            print(
                f"[CBM train] Epoch {ep+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

    cnn.eval()

    def concept_metrics(X, C_true, tag: str):
        with torch.no_grad():
            t = torch.from_numpy(X).to(device)
            preds = cnn(t).cpu().numpy()
        bin_preds = (preds > 0.5).astype(np.float32)
        per_bit = (bin_preds == C_true).mean(axis=0)
        mean_acc = float(per_bit.mean())
        print(f"[CBM train] Concept accuracy ({tag}): mean={mean_acc:.4f}")
        return preds

    print("[CBM train] Evaluating CNN on validation concepts...")
    _ = concept_metrics(X_val, C_val, tag="val")

    print("[CBM train] Evaluating CNN on test concepts...")
    _ = concept_metrics(X_test, C_test, tag="test")

    # Front-end label model: fit on (predicted concepts on train, y)
    with torch.no_grad():
        t_train = torch.from_numpy(X_train).to(device)
        C_train_probs = cnn(t_train).cpu().numpy()

    C_train_bin = (C_train_probs > 0.5).astype(np.float32)
    fe_model = FrontEndModel()
    fe_model.fit(C_train_bin, y_train)

    y_pred_train = fe_model.predict(C_train_bin)
    label_acc_train = float((y_pred_train == y_train).mean())
    print(f"[CBM train] Label accuracy on train (using predicted concepts): {label_acc_train:.4f}")

    detector = SudokuBoardConceptDetector(cnn=cnn, device=device)

    cbm = ConceptBasedModel(
        concept_detector=detector,
        front_end_model=fe_model,
        propagate=True,
    )

    return cbm


# -------------------------------------------------------------------------
# Greedy Conceptual Safeguards Strategy (Algorithm 1)
# -------------------------------------------------------------------------
class GreedyConceptualSafeguardsStrategy(ConceptualSafeguardsStrategy):
    name = "conceptual_safeguards"

    def propose(self, model, batch, config: InterventionConfig) -> StrategyProposal:
        C_pred = batch.C_pred  # (n, m)
        n, m = C_pred.shape

        y_prob_before = model._propagate_predict_proba_mc(C_pred)
        y_pred_before = np.argmax(y_prob_before, axis=1)
        y_true = batch.y_true

        tau = float(getattr(config, "tau", 0.1))
        p1 = y_prob_before[:, 1]
        abstain_mask = (p1 >= tau) & (p1 <= 1.0 - tau)

        non_abstain_mask = ~abstain_mask
        if non_abstain_mask.any() and y_true is not None:
            selective_acc_before = float(
                (y_pred_before[non_abstain_mask] == y_true[non_abstain_mask]).mean()
            )
        else:
            selective_acc_before = float("nan")

        coverage_before = 1.0 - float(abstain_mask.mean())

        # Simple per-concept "gain" proxy: how uncertain the concept is
        gain = np.minimum(C_pred, 1.0 - C_pred)
        base_mask = np.broadcast_to(abstain_mask[:, None], (n, m))

        mask = self._greedy_concept_selection(
            gain=gain,
            base_mask=base_mask,
            config=config,
        )

        selected_instances = np.where(abstain_mask)[0]

        details = {
            "selective_acc_before": selective_acc_before,
            "coverage_before": coverage_before,
        }

        return StrategyProposal(
            mask=mask,
            selected_instances=selected_instances,
            details=details,
        )

    def _greedy_concept_selection(
        self,
        *,
        gain: np.ndarray,
        base_mask: np.ndarray,
        config: InterventionConfig,
    ) -> np.ndarray:
        n, m = gain.shape
        assert base_mask.shape == (n, m)

        S_mask = np.zeros((n, m), dtype=bool)
        total_slots = n * m

        budget_frac = getattr(config, "budget_frac", None)
        if budget_frac is None:
            B = float("inf")
        else:
            frac = max(0.0, min(1.0, float(budget_frac)))
            B = float(np.floor(frac * total_slots))

        Kmax = getattr(config, "max_concepts_per_instance", None)
        used_per_instance = np.zeros(n, dtype=int)

        candidates = np.argwhere(base_mask)
        if candidates.size == 0 or B <= 0:
            return S_mask

        def _best_remaining():
            best_score = -np.inf
            best_i = -1
            best_k = -1
            for i, k in candidates:
                if S_mask[i, k]:
                    continue
                if Kmax is not None and used_per_instance[i] >= Kmax:
                    continue
                score = gain[i, k]
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_k = k
            return best_i, best_k, best_score

        gamma = 1.0

        while B > 0:
            i_star, k_star, s_star = _best_remaining()
            if i_star < 0:
                break

            S_mask[i_star, k_star] = True
            used_per_instance[i_star] += 1
            B -= gamma
            if B <= 0:
                break

            if Kmax is None:
                if np.all(S_mask.all(axis=1)):
                    break
            else:
                if np.all(used_per_instance >= Kmax):
                    break

        return S_mask


# -------------------------------------------------------------------------
# CBM-compatible Selective DNN baseline
# -------------------------------------------------------------------------
class CBMSelectiveDNNStrategy(ConceptualSafeguardsStrategy):
    """
    "Selective DNN" baseline operating on the CBM's label probabilities.

    - No concept interventions (mask is all False).
    - Uses the same abstention rule as conceptual safeguards:
      binary label probability p1, abstain if p1 in [tau, 1 - tau].
    """

    name = "selective_dnn_cbm"

    def propose(self, model, batch, config: InterventionConfig) -> StrategyProposal:
        C_pred = batch.C_pred  # (n, m)

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

        mask = np.zeros_like(C_pred, dtype=bool)
        selected_instances: np.ndarray = np.array([], dtype=int)

        details = {
            "selective_acc_before": selective_acc_before,
            "coverage_before": coverage_before,
        }

        return StrategyProposal(
            mask=mask,
            selected_instances=selected_instances,
            details=details,
        )


# -------------------------------------------------------------------------
# tau calibration on validation split (uses budget_frac=0 internally)
# -------------------------------------------------------------------------
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
        # Calibration: no concept interventions (budget_frac = 0, K = None)
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
        if chosen_tau is None:
            print(
                f"[tau calibration] data={settings['data_name']} "
                f"target_label={target_label} target_value={target_value} "
                f"strategy={getattr(strategy, 'name', 'strategy')} "
                f"tau_grid={list(tau_candidates)} acc_target={acc_target} "
                "→ could not compute selective accuracy for any tau; using grid unchanged."
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
                f"strategy={getattr(strategy, 'name', 'strategy')} "
                f"tau_grid={list(tau_candidates)} acc_target={acc_target} "
                f"chosen_tau={chosen_tau:.3f} val_selective_acc={chosen_acc:.3f} {status} "
                "(budget_frac=0.0, max_concepts_per_instance=None)"
            )

    if chosen_tau is None:
        return list(tau_candidates)
    return [chosen_tau]


# -------------------------------------------------------------------------
# CLI + helpers
# -------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir / "big_demo" / "conceptual_safeguards_sudoku_tabular_metrics.csv",
        help="Destination CSV path for the melted metric table.",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="multimodal_m_21_tabular",
        help="Arbitrary dataset name used only for metadata/logging.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default = Path("/home/mds010/concept-benchmark/data/sudoku/multimodal_m_21_tabular.pkl"),
        help="Path to an existing ConceptDataset pickle with splits (X = boards, C = concepts).",
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
        default=[0.05, 0.1, 0.2, 0.25, 0.5],
        help=(
            "Candidate confidence margins. For each target label and each "
            "strategy (CBM conceptual safeguards / selective DNN), tau is "
            "calibrated on the validation split by choosing the smallest "
            "candidate with selective accuracy >= 0.9 if possible."
        ),
    )
    parser.add_argument(
        "--budget-frac",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5],
        help=(
            "List of budget fractions in (0,1]. Each is interpreted as a fraction "
            "of all concept slots (n * m) that the conceptual safeguards strategy "
            "is allowed to intervene on. budget_frac=0.0 is not allowed."
        ),
    )
    parser.add_argument(
        "--max-concepts-per-instance",
        type=str,
        nargs="+",
        default=["1", "2", "3", "5", "7", "12", "max_incorrect"],
        help=(
            "List of K values: maximum number of concepts per instance (board) that can "
            "be intervened on. Each element can be an integer (e.g. '1', '2') or the "
            "special token 'max_incorrect', which is resolved to dataset.n_concepts. "
            "Use 'none' / 'None' / '-1' for no per-instance cap."
        ),
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


def _estimate_total_configs(
    *,
    target_labels: Sequence[str],
    tau_values: Sequence[float],
    budget_fracs: Sequence[float],
    K_tokens: Sequence[str],
) -> int:
    return (
        len(target_labels)
        * len(budget_fracs)
        * len(K_tokens)
        * len(tau_values)
    )


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    tau_values = list(args.tau)
    for tau in tau_values:
        if not (0.0 <= tau <= 0.5):
            raise ValueError("All tau values must lie within [0, 0.5].")

    # Enforce valid budget_fracs strictly in (0,1]
    budget_fracs = list(args.budget_frac)
    for bf in budget_fracs:
        if not (0.0 < bf <= 1.0):
            raise ValueError("All budget_frac values must lie within (0, 1]. "
                             "budget_frac=0.0 is not allowed.")

    if not args.dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")

    target_pairs = default_target_options(args.target_accuracy_labels)
    target_labels = [label for label, _ in target_pairs]

    total = None
    if not args.no_progress:
        total = _estimate_total_configs(
            target_labels=target_labels,
            tau_values=tau_values,
            budget_fracs=budget_fracs,
            K_tokens=args.max_concepts_per_instance,
        )
    progress = None if total is None else tqdm(total=total, desc="Configs")

    records: List[MetricRecord] = []
    cs_strategy = GreedyConceptualSafeguardsStrategy()
    dnn_strategy = CBMSelectiveDNNStrategy()

    try:
        for target_label, target_value in target_pairs:
            dataset = fileutils.load(args.dataset_file)

            settings = build_generic_settings(
                dataset=dataset,
                data_name=args.data_name,
                target_value=target_value,
            )

            if args.verbose:
                print(f"[main] Settings: {settings}")
                print(f"[main] Loaded dataset from: {args.dataset_file}")

            ensure_base_concepts(dataset)

            cbm = train_sudoku_board_cbm(
                settings=settings,
                dataset=dataset,
                epochs=args.cbm_train_epochs,
                verbose=args.verbose,
            )

            runner = ConceptInterventionRunner(model=cbm)

            cs_tau_values = calibrate_tau_on_validation(
                runner=runner,
                strategy=cs_strategy,
                dataset=dataset,
                tau_candidates=tau_values,
                target_label=target_label,
                target_value=target_value,
                settings=settings,
                verbose=args.verbose,
                acc_target=ACC_TARGET,
            )

            dnn_tau_values = calibrate_tau_on_validation(
                runner=runner,
                strategy=dnn_strategy,
                dataset=dataset,
                tau_candidates=tau_values,
                target_label=target_label,
                target_value=target_value,
                settings=settings,
                verbose=args.verbose,
                acc_target=ACC_TARGET,
            )

            K_values: List[Optional[int]] = []
            max_incorrect_default = int(settings.get("max_corrupt", getattr(dataset, "n_concepts", 27)))
            for token in args.max_concepts_per_instance:
                if token in ("none", "None", "-1"):
                    K_values.append(None)
                elif token == "max_incorrect":
                    K_values.append(max_incorrect_default)
                else:
                    K_values.append(int(token))

            # ---- Conceptual safeguards runs (budget_frac > 0 only) ----
            for budget_frac in budget_fracs:
                for K in K_values:
                    for tau in cs_tau_values:
                        config = InterventionConfig(tau=float(tau))
                        setattr(config, "budget_frac", budget_frac)
                        setattr(config, "max_concepts_per_instance", K)

                        for split_name, split_data in iter_splits(dataset):
                            if split_name not in INTERVENTION_SPLITS:
                                continue

                            result = runner.run(
                                strategy=cs_strategy,
                                config=config,
                                dataset=split_data,
                            )

                            m = result.strat_metrics

                            if args.verbose:
                                print(
                                    "[run] strategy={strategy} "
                                    "data={data_name} ({data_type}) "
                                    "target_label={target_label} target_value={target_value} "
                                    "split={split} tau={tau:.3f} "
                                    "budget_frac={budget_frac} "
                                    "max_concepts_per_instance={K}".format(
                                        strategy=cs_strategy.name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        target_label=target_label,
                                        target_value=target_value,
                                        split=split_name,
                                        tau=tau,
                                        budget_frac=budget_frac,
                                        K=K,
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
                                        strategy=cs_strategy.name,
                                        metric=metric_name,
                                        value=float(metric_value),
                                        split=split_name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        concept_noise=0.0,
                                        concept_missing=0.0,
                                        concept_missing_mech="none",
                                        target_accuracy_label=target_label,
                                        target_accuracy_value=target_value,
                                        params={
                                            "tau": float(tau),
                                            "budget_frac": budget_frac,
                                            "max_concepts_per_instance": K,
                                        },
                                    )
                                )

                            sel_after = m.get("selective_acc_after", None)
                            cov_before = m.get("coverage_before", None)
                            cov_after = m.get("coverage_after", None)

                            if sel_after is not None:
                                records.append(
                                    MetricRecord(
                                        strategy=cs_strategy.name,
                                        metric="selective_accuracy",
                                        value=float(sel_after),
                                        split=split_name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        concept_noise=0.0,
                                        concept_missing=0.0,
                                        concept_missing_mech="none",
                                        target_accuracy_label=target_label,
                                        target_accuracy_value=target_value,
                                        params={
                                            "tau": float(tau),
                                            "budget_frac": budget_frac,
                                            "max_concepts_per_instance": K,
                                        },
                                    )
                                )

                            if cov_after is not None:
                                records.append(
                                    MetricRecord(
                                        strategy=cs_strategy.name,
                                        metric="coverage_pct_handled_by_ai",
                                        value=float(cov_after) * 100.0,
                                        split=split_name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        concept_noise=0.0,
                                        concept_missing=0.0,
                                        concept_missing_mech="none",
                                        target_accuracy_label=target_label,
                                        target_accuracy_value=target_value,
                                        params={
                                            "tau": float(tau),
                                            "budget_frac": budget_frac,
                                            "max_concepts_per_instance": K,
                                        },
                                    )
                                )

                            if cov_after is not None and cov_before is not None:
                                cov_gain = (float(cov_after) - float(cov_before)) * 100.0
                                records.append(
                                    MetricRecord(
                                        strategy=cs_strategy.name,
                                        metric="coverage_gain",
                                        value=cov_gain,
                                        split=split_name,
                                        data_name=settings["data_name"],
                                        data_type=settings["data_type"],
                                        concept_noise=0.0,
                                        concept_missing=0.0,
                                        concept_missing_mech="none",
                                        target_accuracy_label=target_label,
                                        target_accuracy_value=target_value,
                                        params={
                                            "tau": float(tau),
                                            "budget_frac": budget_frac,
                                            "max_concepts_per_instance": K,
                                        },
                                    )
                                )

                        if progress is not None:
                            progress.update(1)

            # ---- CBM selective DNN baseline (no concept interventions) ----
            for tau in dnn_tau_values:
                dnn_config = InterventionConfig(tau=float(tau))
                setattr(dnn_config, "budget_frac", 0.0)
                setattr(dnn_config, "max_concepts_per_instance", None)

                for split_name, split_data in iter_splits(dataset):
                    if split_name not in INTERVENTION_SPLITS:
                        continue

                    dnn_result = runner.run(
                        strategy=dnn_strategy,
                        config=dnn_config,
                        dataset=split_data,
                    )

                    m = dnn_result.strat_metrics

                    if args.verbose:
                        print(
                            "[run] strategy={strategy} "
                            "data={data_name} ({data_type}) "
                            "target_label={target_label} target_value={target_value} "
                            "split={split} tau={tau:.3f} "
                            "budget_frac=0.0 max_concepts_per_instance=None".format(
                                strategy=dnn_strategy.name,
                                data_name=settings["data_name"],
                                data_type=settings["data_type"],
                                target_label=target_label,
                                target_value=target_value,
                                split=split_name,
                                tau=tau,
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

                    sel_after = m.get("selective_acc_after", None)
                    cov_after = m.get("coverage_after", None)
                    cov_before = m.get("coverage_before", cov_after)

                    if sel_after is not None:
                        records.append(
                            MetricRecord(
                                strategy=dnn_strategy.name,
                                metric="selective_accuracy",
                                value=float(sel_after),
                                split=split_name,
                                data_name=settings["data_name"],
                                data_type=settings["data_type"],
                                concept_noise=0.0,
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
                                strategy=dnn_strategy.name,
                                metric="coverage_pct_handled_by_ai",
                                value=float(cov_after) * 100.0,
                                split=split_name,
                                data_name=settings["data_name"],
                                data_type=settings["data_type"],
                                concept_noise=0.0,
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
                                strategy=dnn_strategy.name,
                                metric="coverage_gain",
                                value=cov_gain,
                                split=split_name,
                                data_name=settings["data_name"],
                                data_type=settings["data_type"],
                                concept_noise=0.0,
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

    finally:
        if progress is not None:
            progress.close()

    write_metrics_csv(records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
