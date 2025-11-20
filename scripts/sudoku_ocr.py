#!/usr/bin/env python3
"""
Sudoku OCR (TinyResNet-only) with fast vs full inference.

- trains TinyResNet
- FAST inference: argmax-only, no solver, no TTA, big chunks
- FULL inference: argmax + solver (same logic as original), supports TTA

Outputs:
- /.../ocr_predictions_fast.jsonl
- /.../ocr_predictions_full.jsonl
- state in /.../ocr_state.json (resumable)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from tqdm.auto import tqdm
import time

# -------------------------------------------------------------------------
# paths
# -------------------------------------------------------------------------
ROOT = Path("/home/mds010/concept-benchmark")
DATA_SUDOKU = ROOT / "data" / "sudoku"
DEBUG_DIR = DATA_SUDOKU / "blue_debug"
PRED_DEBUG_DIR = DATA_SUDOKU / "multi_data_demo_ocr_preds"
BEST_MODEL_PATH = DATA_SUDOKU / "best_digit_model.pt"
BEST_META_PATH = DATA_SUDOKU / "best_digit_model.json"
DEFAULT_LOG_PATH = DATA_SUDOKU / "ocr_multi_data_demo.log"
DEFAULT_STATE_PATH = DATA_SUDOKU / "ocr_state_multi_data_demo.json"


# -------------------------------------------------------------------------
# logging
# -------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sudoku_ocr")
    logger.setLevel(logging.INFO)

    have_file = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path)
        for h in logger.handlers
    )
    if not have_file:
        fh = logging.FileHandler(str(log_path), mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    logger.info(f"[LOG] logging to {log_path}")
    return logger


# -------------------------------------------------------------------------
# state
# -------------------------------------------------------------------------
def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        with state_path.open("r") as f:
            return json.load(f)
    return {}


def save_state(state_path: Path, state: Dict[str, Any]):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w") as f:
        json.dump(state, f, indent=2)


# -------------------------------------------------------------------------
# data utils
# -------------------------------------------------------------------------
def load_sidecars(jsonl_path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def crop_cell(bgr: np.ndarray, r: int, c: int, *, cell_px: int = 40, margin_px: int = 2) -> np.ndarray:
    x0 = margin_px + c * cell_px
    y0 = margin_px + r * cell_px
    x1 = x0 + cell_px
    y1 = y0 + cell_px
    return bgr[y0:y1, x0:x1, :].copy()


def cell_preprocess_28x28(cell_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (28, 28), interpolation=cv2.INTER_AREA)
    g = 255.0 - g
    g = g.astype(np.float32) / 255.0
    return g


class SudokuCellDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        jsonl_path: Path,
        cell_px: int = 40,
        margin_px: int = 2,
        dump_debug: bool = True,
        max_debug: int = 150,
    ):
        self.dataset_dir = dataset_dir
        self.cell_px = cell_px
        self.margin_px = margin_px

        self.records = load_sidecars(jsonl_path)
        self.samples: List[Tuple[Path, int, int, int]] = []
        for rec in self.records:
            img_name = rec["img"]
            board = np.array(rec["board"], dtype=np.int32)
            img_path = dataset_dir / img_name
            assert board.shape == (9, 9)
            for r in range(9):
                for c in range(9):
                    label = int(board[r, c])
                    self.samples.append((img_path, r, c, label))

        if dump_debug:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            self._dump_debug(max_debug)

    def _dump_debug(self, max_debug: int):
        dumped = 0
        for (img_path, r, c, label) in self.samples:
            if dumped >= max_debug:
                break
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            cell = crop_cell(bgr, r, c, cell_px=self.cell_px, margin_px=self.margin_px)
            cell28 = cell_preprocess_28x28(cell)
            cell28_vis = (cell28 * 255.0).astype(np.uint8)
            outp = DEBUG_DIR / f"{img_path.stem}_r{r}_c{c}_d{label}.png"
            cv2.imwrite(str(outp), cell28_vis)
            dumped += 1
        logging.getLogger("sudoku_ocr").info(f"[DEBUG] dumped {dumped} cell crops to {DEBUG_DIR}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, r, c, label = self.samples[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            cell28 = np.zeros((28, 28), dtype=np.float32)
        else:
            cell = crop_cell(bgr, r, c, cell_px=self.cell_px, margin_px=self.margin_px)
            cell28 = cell_preprocess_28x28(cell)
        x = torch.from_numpy(cell28).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class DigitExampleDataset(Dataset):
    """
    Dataset for standalone digit example images in DATA_SUDOKU/digit_examples.

    Expects files like:
        digit_examples/digit_1_hand.png
        digit_examples/digit_3_hand_something.png

    Label is parsed as the integer immediately after 'digit_'.
    """

    def __init__(
        self,
        root: Path,
        dump_debug: bool = False,
        max_debug: int = 150,
    ):
        self.root = root
        self.samples: List[Tuple[Path, int]] = []

        for p in sorted(root.glob("digit_*_hand*.png")):
            stem = p.stem  # e.g., 'digit_3_hand'
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            try:
                digit = int(parts[1])
            except ValueError:
                continue

            if 1 <= digit <= 9:
                self.samples.append((p, digit))

        if dump_debug:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            dumped = 0
            for img_path, digit in self.samples:
                if dumped >= max_debug:
                    break
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                cell28 = cell_preprocess_28x28(bgr)
                cell28_vis = (cell28 * 255.0).astype(np.uint8)
                outp = DEBUG_DIR / f"{img_path.stem}_d{digit}.png"
                cv2.imwrite(str(outp), cell28_vis)
                dumped += 1
            logging.getLogger("sudoku_ocr").info(
                f"[DEBUG] dumped {dumped} digit example crops to {DEBUG_DIR}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, digit = self.samples[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            cell28 = np.zeros((28, 28), dtype=np.float32)
        else:
            cell28 = cell_preprocess_28x28(bgr)
        x = torch.from_numpy(cell28).unsqueeze(0)
        y = torch.tensor(digit, dtype=torch.long)  # 1..9
        return x, y


# -------------------------------------------------------------------------
# model: TinyResNet ONLY
# -------------------------------------------------------------------------
class TinyResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = out + x
        return self.act(out)


class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(True),
        )
        self.block1 = TinyResBlock(32)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = TinyResBlock(32)
        self.pool2 = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        return self.head(x)


# -------------------------------------------------------------------------
# training / eval
# -------------------------------------------------------------------------
def train_resnet_tiny(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    class_weights: torch.Tensor | None,
    logger: logging.Logger,
) -> Dict[str, Any]:
    model = TinyResNet(num_classes=10).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    best_val_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"resnet_tiny train {ep}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            preds = out.argmax(1)
            acc = (preds == yb).float().mean().item()
            pbar.set_postfix(loss=float(loss.item()), acc=acc)

        val_loss, val_acc, _ = eval_model(model, val_loader, device)
        logger.info(f"[VAL] resnet_tiny ep={ep} loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return {
        "best_val_acc": float(best_val_acc),
        "best_state": best_state,
    }


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    total_correct = 0
    n_classes = 10
    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        loss = crit(out, yb)

        total_loss += float(loss.item()) * xb.size(0)
        total += xb.size(0)

        preds = out.argmax(1)
        total_correct += (preds == yb).sum().item()

        for cls in range(n_classes):
            mask = (yb == cls)
            total_per_class[cls] += mask.sum().item()
            correct_per_class[cls] += ((preds == yb) & mask).sum().item()

    avg_loss = total_loss / total
    avg_acc = total_correct / total
    per_class_acc = {}
    for cls in range(n_classes):
        if total_per_class[cls] == 0:
            per_class_acc[cls] = None
        else:
            per_class_acc[cls] = correct_per_class[cls] / total_per_class[cls]
    return avg_loss, avg_acc, per_class_acc


# -------------------------------------------------------------------------
# sudoku + validity
# -------------------------------------------------------------------------
def sudoku_valid(partial: List[List[int]]) -> bool:
    for i in range(9):
        r = [v for v in partial[i] if v]
        if len(r) != len(set(r)):
            return False
        c = [partial[r][i] for r in range(9) if partial[r][i]]
        if len(c) != len(set(c)):
            return False
    for br in range(3):
        for bc in range(3):
            vals = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    v = partial[r][c]
                    if v:
                        vals.append(v)
            if len(vals) != len(set(vals)):
                return False
    return True


def compute_concepts_27(board: np.ndarray) -> List[int]:
    """
    Compute 27 binary concepts from a 9x9 board:
      - 9 row validity bits
      - 9 column validity bits
      - 9 block (3x3) validity bits

    Validity = no duplicate non-zero digits in that row/col/block.
    """
    assert board.shape == (9, 9)
    # rows
    row_valid: List[int] = []
    for i in range(9):
        vals = [int(v) for v in board[i, :] if v != 0]
        row_valid.append(int(len(vals) == len(set(vals))))

    # columns
    col_valid: List[int] = []
    for j in range(9):
        vals = [int(board[i, j]) for i in range(9) if board[i, j] != 0]
        col_valid.append(int(len(vals) == len(set(vals))))

    # 3x3 blocks
    block_valid: List[int] = []
    for br in range(3):
        for bc in range(3):
            vals = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    v = int(board[r, c])
                    if v != 0:
                        vals.append(v)
            block_valid.append(int(len(vals) == len(set(vals))))

    # 27 concepts: rows (0–8), cols (9–17), blocks (18–26)
    return row_valid + col_valid + block_valid


def get_board_valid_gt(rec: Dict[str, Any], gt_board: np.ndarray) -> bool:
    if "valid" in rec:
        return bool(rec["valid"])
    if "is_valid" in rec:
        return bool(rec["is_valid"])
    img_name = str(rec.get("img", "")).lower()
    if img_name.startswith("valid_"):
        return True
    if img_name.startswith("invalid_"):
        return False
    return sudoku_valid(gt_board.tolist())


# === NEW: exact DP to compute concept probability from digit probabilities ===
def valid_prob_no_duplicates(group_probs: np.ndarray) -> float:
    """
    Exact probability that a 9-cell group is valid (no duplicate non-zero digits).

    Args:
        group_probs: shape (9, 10)
            group_probs[i, d] = P(cell i is digit d), d in {0..9}.

    Returns:
        float: P(group is valid)
    """
    assert group_probs.shape == (9, 10)
    n_cells = 9
    n_states = 1 << 9  # 2^9 masks for digits 1..9

    dp_prev = np.zeros(n_states, dtype=np.float64)
    dp_prev[0] = 1.0  # no digits used yet

    for i in range(n_cells):
        p = group_probs[i]  # (10,)
        dp_next = np.zeros_like(dp_prev)
        for mask in range(n_states):
            base = dp_prev[mask]
            if base == 0.0:
                continue

            # digit 0 (blank): always allowed, mask unchanged
            dp_next[mask] += base * float(p[0])

            # digits 1..9: only if that digit not already used
            for d in range(1, 10):
                bit = 1 << (d - 1)
                if mask & bit:
                    continue  # would be a duplicate
                dp_next[mask | bit] += base * float(p[d])

        dp_prev = dp_next

    # all masks correspond to valid configs by construction
    return float(dp_prev.sum())


def compute_concept_probs_27_from_cell_probs(cell_probs: np.ndarray) -> List[float]:
    """
    Given per-cell digit probabilities, compute 27 concept probabilities:
      - 9 row validity probs
      - 9 column validity probs
      - 9 block validity probs

    Args:
        cell_probs: shape (81, 10) or (9, 9, 10)

    Returns:
        List[float] of length 27
    """
    if cell_probs.shape == (81, 10):
        cp = cell_probs.reshape(9, 9, 10)
    elif cell_probs.shape == (9, 9, 10):
        cp = cell_probs
    else:
        raise ValueError(f"expected (81,10) or (9,9,10), got {cell_probs.shape}")

    concept_probs: List[float] = []

    # rows
    for r in range(9):
        group_probs = cp[r, :, :]  # (9,10)
        concept_probs.append(valid_prob_no_duplicates(group_probs))

    # columns
    for c in range(9):
        group_probs = cp[:, c, :]  # (9,10)
        concept_probs.append(valid_prob_no_duplicates(group_probs))

    # 3x3 blocks
    for br in range(3):
        for bc in range(3):
            cells = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    cells.append(cp[r, c, :])
            group_probs = np.stack(cells, axis=0)  # (9,10)
            concept_probs.append(valid_prob_no_duplicates(group_probs))

    assert len(concept_probs) == 27
    return concept_probs


# -------------------------------------------------------------------------
# solver (from original script)
# -------------------------------------------------------------------------
def _candidate_lists_from_topk(top_digits, top_logp, seed_grid=None):
    cand = []
    for r in range(9):
        for c in range(9):
            if seed_grid is not None and seed_grid[r, c] != 0:
                d = int(seed_grid[r, c])
                cand.append((r, c, [(d, 0.0)]))
                continue
            pairs = list(zip(top_digits[r, c].tolist(), top_logp[r, c].tolist()))
            pairs.sort(key=lambda x: -x[1])
            cand.append((r, c, pairs))
    cand.sort(key=lambda x: len(x[2]))
    return cand


def solve_with_probs(top_digits, top_logp, seed_grid=None, max_nodes=200000):
    import heapq

    start = [[0] * 9 for _ in range(9)]
    c0 = _candidate_lists_from_topk(top_digits, top_logp, seed_grid=seed_grid)
    pq = []
    heapq.heappush(pq, (0.0, start, c0))
    visited = 0
    while pq and visited < max_nodes:
        neg_score, grid, cands = heapq.heappop(pq)
        visited += 1
        if not cands:
            return np.array(grid, dtype=int)
        r, c, opts = cands[0]
        for d, lp in opts:
            g2 = [row[:] for row in grid]
            g2[r][c] = d
            if not sudoku_valid(g2):
                continue
            c2 = _candidate_lists_from_topk(top_digits, top_logp, seed_grid=np.array(g2))
            heapq.heappush(pq, (neg_score - lp, g2, c2))
    return None


@torch.no_grad()
def predict_board_logits(
    model: nn.Module,
    img_path: Path,
    *,
    device: str = "cpu",
    tta: int = 0,
    cell_px: int = 40,
    margin_px: int = 2,
) -> torch.Tensor:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(str(img_path))

    cells = []
    for r in range(9):
        for c in range(9):
            cell = crop_cell(bgr, r, c, cell_px=cell_px, margin_px=margin_px)
            x28 = cell_preprocess_28x28(cell)
            cells.append(x28)
    cells = np.stack(cells, axis=0)
    h, w = 28, 28

    if tta <= 1:
        xb = torch.from_numpy(cells).unsqueeze(1).to(device)
        return model(xb)

    variants = [cells]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        shifted_cells = []
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        for i in range(cells.shape[0]):
            shifted = cv2.warpAffine((cells[i] * 255).astype(np.uint8), M, (w, h),
                                     flags=cv2.INTER_NEAREST, borderValue=0)
            shifted_cells.append(shifted.astype(np.float32) / 255.0)
        shifted_cells = np.stack(shifted_cells, axis=0)
        variants.append(shifted_cells)

    all_logits = []
    for var in variants[:tta]:
        xb = torch.from_numpy(var).unsqueeze(1).to(device)
        out = model(xb)
        all_logits.append(out)
    logits = torch.stack(all_logits, dim=0).mean(dim=0)
    return logits


@torch.no_grad()
def predict_board_argmax(
    model: nn.Module,
    img_path: Path,
    *,
    device: str = "cpu",
    tta: int = 0,
) -> np.ndarray:
    logits = predict_board_logits(model, img_path, device=device, tta=tta)
    return logits.argmax(1).cpu().numpy().reshape(9, 9)


@torch.no_grad()
def predict_board_solved(
    model: nn.Module,
    img_path: Path,
    *,
    device: str = "cpu",
    tta: int = 0,
    topk: int = 3,
    conf_gate: float = 0.0,
    solver_max_nodes: int = 200000,
) -> np.ndarray:
    logits = predict_board_logits(model, img_path, device=device, tta=tta)
    logp = torch.log_softmax(logits, dim=1)[:, 1:]
    prob = torch.softmax(logits, dim=1)[:, 1:]

    pmax, pidx = prob.max(dim=1)
    seed = np.zeros((9, 9), dtype=int)
    for i in range(81):
        r, c = divmod(i, 9)
        if pmax[i].item() >= conf_gate:
            seed[r, c] = int(pidx[i].item()) + 1

    K = max(1, int(topk))
    top_logp, top_idx = torch.topk(logp, k=K, dim=1)
    top_digits = (top_idx + 1).cpu().numpy().reshape(9, 9, K)
    top_logp = top_logp.cpu().numpy().reshape(9, 9, K)

    grid = solve_with_probs(top_digits, top_logp, seed_grid=seed, max_nodes=solver_max_nodes)
    if grid is None:
        return (pidx.cpu().numpy().reshape(9, 9) + 1)
    return grid


def overlay_predictions(
    img_path: Path,
    pred_board: np.ndarray,
    *,
    cell_px: int = 40,
    margin_px: int = 2,
    out_dir: Path = PRED_DEBUG_DIR,
    color=(0, 0, 255),
):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return
    out = bgr.copy()
    for r in range(9):
        for c in range(9):
            v = int(pred_board[r, c])
            x0 = margin_px + c * cell_px
            y0 = margin_px + r * cell_px
            txt = "." if v == 0 else str(v)
            cv2.putText(
                out,
                txt,
                (x0 + 6, y0 + cell_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
                cv2.LINE_AA,
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_path.name), out)


# -------------------------------------------------------------------------
# generalized inference pass
# -------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    name: str
    tta: int
    use_solver: bool
    solver_topk: int
    solver_conf: float
    solver_max_nodes: int
    save_previews: bool
    chunk_size: int


def run_inference_pass(
    model: nn.Module,
    records: List[Dict[str, Any]],
    dataset_dir: Path,
    state: Dict[str, Any],
    cfg: InferenceConfig,
    preds_path: Path,
    state_path: Path,
    *,
    device: str,
    logger: logging.Logger,
    force: bool = False,
):
    state_key = f"inference_{cfg.name}"
    inf = state.get(state_key)
    if inf is None:
        inf = {
            "done": False,
            "preds_path": str(preds_path),
            "next_idx": 0,
            "totals": {
                "total_cells_argmax": 0,
                "total_correct_argmax": 0,
                "boards": 0,
                "boards_correct_argmax": 0,
                "board_cls_total_argmax": 0,
                "board_cls_correct_argmax": 0,
            },
            "config": {
                "tta": cfg.tta,
                "use_solver": cfg.use_solver,
                "solver_topk": cfg.solver_topk,
                "solver_conf": cfg.solver_conf,
                "solver_max_nodes": cfg.solver_max_nodes,
            },
        }
        if cfg.use_solver:
            inf["totals"].update(
                {
                    "total_cells_solved": 0,
                    "total_correct_solved": 0,
                    "boards_correct_solved": 0,
                    "board_cls_total_solved": 0,
                    "board_cls_correct_solved": 0,
                }
            )
        state[state_key] = inf

    if force:
        logger.info(f"[INFO] force-inference: resetting state for {state_key}")
        inf["done"] = False
        inf["next_idx"] = 0
        inf["totals"] = {
            "total_cells_argmax": 0,
            "total_correct_argmax": 0,
            "boards": 0,
            "boards_correct_argmax": 0,
            "board_cls_total_argmax": 0,
            "board_cls_correct_argmax": 0,
        }
        if cfg.use_solver:
            inf["totals"].update(
                {
                    "total_cells_solved": 0,
                    "total_correct_solved": 0,
                    "boards_correct_solved": 0,
                    "board_cls_total_solved": 0,
                    "board_cls_correct_solved": 0,
                }
            )
        inf["config"] = {
            "tta": cfg.tta,
            "use_solver": cfg.use_solver,
            "solver_topk": cfg.solver_topk,
            "solver_conf": cfg.solver_conf,
            "solver_max_nodes": cfg.solver_max_nodes,
        }

    old_cfg = inf["config"]
    if (
        old_cfg.get("tta") != cfg.tta
        or old_cfg.get("use_solver") != cfg.use_solver
        or old_cfg.get("solver_topk") != cfg.solver_topk
        or old_cfg.get("solver_conf") != cfg.solver_conf
        or old_cfg.get("solver_max_nodes") != cfg.solver_max_nodes
    ):
        logger.info(f"[WARN] inference config changed for {state_key}; restarting")
        inf["done"] = False
        inf["next_idx"] = 0
        inf["totals"] = {
            "total_cells_argmax": 0,
            "total_correct_argmax": 0,
            "boards": 0,
            "boards_correct_argmax": 0,
            "board_cls_total_argmax": 0,
            "board_cls_correct_argmax": 0,
        }
        if cfg.use_solver:
            inf["totals"].update(
                {
                    "total_cells_solved": 0,
                    "total_correct_solved": 0,
                    "boards_correct_solved": 0,
                    "board_cls_total_solved": 0,
                    "board_cls_correct_solved": 0,
                }
            )
        inf["config"] = {
            "tta": cfg.tta,
            "use_solver": cfg.use_solver,
            "solver_topk": cfg.solver_topk,
            "solver_conf": cfg.solver_conf,
            "solver_max_nodes": cfg.solver_max_nodes,
        }

    start_idx = inf["next_idx"]
    totals = inf["totals"]

    if inf["done"]:
        logger.info(f"[INFO] {state_key} already done, skipping")
        return

    preds_path.parent.mkdir(parents=True, exist_ok=True)

    n_records = len(records)
    t0 = time.perf_counter()

    # Open the file for the whole pass so it can't be half-closed mid-loop
    with preds_path.open("a") as fw:
        pbar = tqdm(total=n_records, desc=state_key, initial=start_idx)

        i = start_idx
        while i < n_records:
            end = min(i + cfg.chunk_size, n_records)
            for rec in records[i:end]:
                img_name = rec["img"]
                gt_board = np.array(rec["board"], dtype=np.int64)
                img_path = dataset_dir / img_name

                # count boards so board-level accuracy is correct
                totals["boards"] += 1

                gt_valid = get_board_valid_gt(rec, gt_board)

                # --- logits + per-cell probabilities (used for argmax + concepts) ---
                logits = predict_board_logits(
                    model,
                    img_path,
                    device=device,
                    tta=cfg.tta,
                )  # (81, 10)
                probs = torch.softmax(logits, dim=1)  # (81, 10)

                # argmax board
                pred_argmax = probs.argmax(1).cpu().numpy().reshape(9, 9)
                eq_arg = (pred_argmax == gt_board)
                totals["total_cells_argmax"] += int(eq_arg.size)
                totals["total_correct_argmax"] += int(eq_arg.sum())
                if bool(eq_arg.all()):
                    totals["boards_correct_argmax"] += 1

                pred_valid_arg = sudoku_valid(pred_argmax.tolist())
                totals["board_cls_total_argmax"] += 1
                if pred_valid_arg == gt_valid:
                    totals["board_cls_correct_argmax"] += 1

                # concept probabilities from digit probabilities
                cell_probs_np = probs.cpu().numpy()  # (81, 10)
                pred_concepts_probs_27 = compute_concept_probs_27_from_cell_probs(cell_probs_np)

                pred_solved = None
                pred_valid_solved = None
                pred_concepts_solved_27 = None
                pred_concepts_solved_probs: List[float] | None = None

                if cfg.use_solver:
                    pred_solved = predict_board_solved(
                        model,
                        img_path,
                        device=device,
                        tta=cfg.tta,
                        topk=cfg.solver_topk,
                        conf_gate=cfg.solver_conf,
                        solver_max_nodes=cfg.solver_max_nodes,
                    )
                    eq_sol = (pred_solved == gt_board)
                    totals["total_cells_solved"] += int(eq_sol.size)
                    totals["total_correct_solved"] += int(eq_sol.sum())
                    if bool(eq_sol.all()):
                        totals["boards_correct_solved"] += 1

                    pred_valid_solved = sudoku_valid(pred_solved.tolist())
                    totals["board_cls_total_solved"] += 1
                    if pred_valid_solved == gt_valid:
                        totals["board_cls_correct_solved"] += 1

                    pred_concepts_solved_27 = compute_concepts_27(pred_solved)
                    # reuse same probability vector (comes from digit distribution)
                    pred_concepts_solved_probs = pred_concepts_probs_27

                # --- concept labels (27): row/col/block validity ---
                gt_concepts_27 = compute_concepts_27(gt_board)
                pred_concepts_argmax_27 = compute_concepts_27(pred_argmax)

                if cfg.save_previews:
                    overlay_predictions(
                        img_path,
                        pred_argmax,
                        out_dir=PRED_DEBUG_DIR,
                        color=(0, 0, 255),
                    )
                    if pred_solved is not None:
                        overlay_predictions(
                            img_path,
                            pred_solved,
                            out_dir=PRED_DEBUG_DIR / "solver",
                            color=(0, 180, 0),
                        )

                out_rec = {
                    "img": img_name,
                    "gt": gt_board.tolist(),
                    "gt_valid": gt_valid,

                    # board-level predictions
                    "pred_argmax": pred_argmax.tolist(),
                    "pred_valid_argmax": pred_valid_arg,
                    "pred_solved": None if pred_solved is None else pred_solved.tolist(),
                    "pred_valid_solved": pred_valid_solved,

                    # concept-level ground truth (27: rows, cols, blocks)
                    "gt_concepts": gt_concepts_27,

                    # ARGMAX concepts: final predictions + probabilities
                    "pred_concepts_argmax": pred_concepts_argmax_27,
                    "pred_concepts_argmax_probs": [float(p) for p in pred_concepts_probs_27],

                    # SOLVER concepts (if solver is used): final predictions + probabilities
                    "pred_concepts_solved": pred_concepts_solved_27,
                    "pred_concepts_solved_probs": None
                    if pred_concepts_solved_probs is None
                    else [float(p) for p in pred_concepts_solved_probs],
                }
                fw.write(json.dumps(out_rec) + "\n")

                pbar.update(1)

            i = end
            inf["next_idx"] = i
            save_state(state_path, state)
            logger.info(f"[INFO] {state_key}: processed {i}/{n_records}")

        pbar.close()

    # after the with-block, fw is closed cleanly
    inf["done"] = True
    save_state(state_path, state)
    t1 = time.perf_counter()

    # ================== METRICS ==================
    logger.info("")
    logger.info(f"=========== {state_key} METRICS ===========")

    # ARGMAX metrics
    cell_acc_arg = (
        totals["total_correct_argmax"] / totals["total_cells_argmax"]
        if totals["total_cells_argmax"]
        else 0.0
    )
    board_acc_arg = (
        totals["boards_correct_argmax"] / totals["boards"]
        if totals["boards"]
        else 0.0
    )
    cls_acc_arg = None
    if totals["board_cls_total_argmax"]:
        cls_acc_arg = totals["board_cls_correct_argmax"] / totals["board_cls_total_argmax"]

    logger.info(
        f"[{cfg.name}] ARGMAX cell acc: {cell_acc_arg:.4f} "
        f"({totals['total_correct_argmax']}/{totals['total_cells_argmax']})"
    )
    logger.info(
        f"[{cfg.name}] ARGMAX board acc: {board_acc_arg:.4f} "
        f"({totals['boards_correct_argmax']}/{totals['boards']})"
    )
    if cls_acc_arg is not None:
        logger.info(
            f"[{cfg.name}] ARGMAX board VALIDITY cls acc: {cls_acc_arg:.4f} "
            f"({totals['board_cls_correct_argmax']}/{totals['board_cls_total_argmax']})"
        )

    # SOLVER metrics (optional)
    cell_acc_sol = None
    board_acc_sol = None
    cls_acc_sol = None

    if cfg.use_solver and totals.get("total_cells_solved", 0):
        cell_acc_sol = totals["total_correct_solved"] / totals["total_cells_solved"]
        board_acc_sol = (
            totals["boards_correct_solved"] / totals["boards"]
            if totals["boards"]
            else 0.0
        )
        logger.info(
            f"[{cfg.name}] SOLVER cell acc: {cell_acc_sol:.4f} "
            f"({totals['total_correct_solved']}/{totals['total_cells_solved']})"
        )
        logger.info(
            f"[{cfg.name}] SOLVER board acc: {board_acc_sol:.4f} "
            f"({totals['boards_correct_solved']}/{totals['boards']})"
        )
        if totals.get("board_cls_total_solved", 0):
            cls_acc_sol = totals["board_cls_correct_solved"] / totals["board_cls_total_solved"]
            logger.info(
                f"[{cfg.name}] SOLVER board VALIDITY cls acc: {cls_acc_sol:.4f} "
                f"({totals['board_cls_correct_solved']}/{totals['board_cls_total_solved']})"
            )

    logger.info(f"[{cfg.name}] elapsed: {t1 - t0:.2f}s")

    summary_rec = {
        "summary": True,
        "inference_name": cfg.name,
        "totals": totals,
        "metrics": {
            "cell_acc_argmax": cell_acc_arg,
            "board_acc_argmax": board_acc_arg,
            "board_valid_cls_acc_argmax": cls_acc_arg,
            "cell_acc_solver": cell_acc_sol,
            "board_acc_solver": board_acc_sol,
            "board_valid_cls_acc_solver": cls_acc_sol,
            "elapsed_seconds": t1 - t0,
        },
    }
    with preds_path.open("a") as fw_sum:
        fw_sum.write(json.dumps(summary_rec) + "\n")



# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(DATA_SUDOKU / "multimodal_m_21_image"))
    ap.add_argument("--jsonl", default=str(DATA_SUDOKU / "multimodal_m_21_image" / "ocr_preprocessing.jsonl"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--previews", action="store_true", help="save PNG overlays (off by default)")
    ap.add_argument("--log-file", default=str(DEFAULT_LOG_PATH))
    ap.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))

    # inference toggles
    ap.add_argument("--force-fast", action="store_true", help="force rerun fast inference")
    ap.add_argument("--force-full", action="store_true", help="force rerun full inference")

    # full inference options (solver)
    ap.add_argument("--solver-topk", type=int, default=3)
    ap.add_argument("--solver-conf", type=float, default=0.0)
    ap.add_argument("--solver-max-nodes", type=int, default=200000)
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=25)

    args = ap.parse_args()

    logger = setup_logging(Path(args.log_file))
    state_path = Path(args.state_file)
    state = load_state(state_path)

    dataset_dir = Path(args.dataset_dir)
    jsonl_path = Path(args.jsonl)

    logger.info(f"[INFO] loading dataset from {dataset_dir} using {jsonl_path}")
    sudoku_ds = SudokuCellDataset(dataset_dir, jsonl_path, dump_debug=True)

    digit_examples_dir = DATA_SUDOKU / "digit_examples"
    if digit_examples_dir.exists():
        logger.info(f"[INFO] loading digit examples from {digit_examples_dir}")
        digit_ds = DigitExampleDataset(digit_examples_dir, dump_debug=False)
        full_ds = ConcatDataset([sudoku_ds, digit_ds])
        logger.info(
            f"[INFO] combined dataset size: sudoku={len(sudoku_ds)}, "
            f"digit_examples={len(digit_ds)}, total={len(full_ds)}"
        )
    else:
        full_ds = sudoku_ds
        logger.info("[INFO] no digit_examples directory found; using Sudoku cells only")

    # split
    n_total = len(full_ds)
    n_val = max(600, int(0.12 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    logger.info(f"[INFO] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    # class weights
    counts = np.zeros(10, dtype=np.int64)
    for i in range(min(8000, len(train_ds))):
        _, y = train_ds[i]
        counts[int(y)] += 1
    logger.info("[INFO] class counts (train sample): %s", counts.tolist())
    counts = np.where(counts == 0, 1, counts)
    inv = 1.0 / counts
    inv = inv / inv.sum() * 10.0
    class_weights = torch.tensor(inv, dtype=torch.float32, device=args.device)

    device = args.device

    # train TinyResNet
    logger.info("\n==============================")
    logger.info("[INFO] training model: resnet_tiny (forced)")
    logger.info("==============================")
    res = train_resnet_tiny(
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        class_weights=class_weights,
        logger=logger,
    )
    torch.save(res["best_state"], BEST_MODEL_PATH)
    with BEST_META_PATH.open("w") as f:
        json.dump({"name": "resnet_tiny", "val_acc": float(res["best_val_acc"])}, f, indent=2)
    state["trained_models"] = {"resnet_tiny": {"best_val_acc": res["best_val_acc"]}}
    state["best_overall"] = {"name": "resnet_tiny", "val_acc": res["best_val_acc"]}
    save_state(state_path, state)

    logger.info("\n================ BEST MODEL ================")
    logger.info("name: resnet_tiny")
    logger.info(f"val acc: {res['best_val_acc']:.4f}")
    logger.info(f"[INFO] best model weights at {BEST_MODEL_PATH}")

    # load model back
    best_model = TinyResNet(num_classes=10).to(device)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    best_model.eval()

    # board records
    records = load_sidecars(jsonl_path)

    # FAST pass
    fast_cfg = InferenceConfig(
        name="fast",
        tta=0,
        use_solver=False,
        solver_topk=0,
        solver_conf=0.0,
        solver_max_nodes=0,
        save_previews=False,
        chunk_size=100,
    )
    fast_preds_path = dataset_dir / "ocr_predictions_fast.jsonl"
    run_inference_pass(
        best_model,
        records,
        dataset_dir,
        state,
        fast_cfg,
        preds_path=fast_preds_path,
        state_path=state_path,
        device=device,
        logger=logger,
        force=args.force_fast,
    )

    # FULL pass (long, with solver)
    full_cfg = InferenceConfig(
        name="full",
        tta=args.tta,
        use_solver=True,
        solver_topk=args.solver_topk,
        solver_conf=args.solver_conf,
        solver_max_nodes=args.solver_max_nodes,
        save_previews=args.previews,
        chunk_size=args.chunk_size,
    )
    full_preds_path = dataset_dir / "ocr_predictions_full.jsonl"
    run_inference_pass(
        best_model,
        records,
        dataset_dir,
        state,
        full_cfg,
        preds_path=full_preds_path,
        state_path=state_path,
        device=device,
        logger=logger,
        force=args.force_full,
    )

    logger.info("[INFO] done.")


if __name__ == "__main__":
    main()
