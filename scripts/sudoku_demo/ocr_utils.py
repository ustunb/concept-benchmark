from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from concept_benchmark.paths import data_dir

# Base data location reused by both the generator and trainer.
DATA_SUDOKU = data_dir / "sudoku"
DEBUG_DIR = DATA_SUDOKU / "demo_ocr_debug"


def load_sidecars(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Read OCR sidecar rows into memory."""
    recs: List[Dict[str, Any]] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def crop_cell(bgr: np.ndarray, r: int, c: int, *, cell_px: int = 50, margin_px: int = 2) -> np.ndarray:
    x0 = margin_px + c * cell_px
    y0 = margin_px + r * cell_px
    x1 = x0 + cell_px
    y1 = y0 + cell_px
    return bgr[y0:y1, x0:x1, :].copy()


def cell_preprocess_28x28(cell_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (28, 28), interpolation=cv2.INTER_AREA)
    g = 255.0 - g  # invert to white-on-black
    g = g.astype(np.float32) / 255.0
    return g


class SudokuCellDataset(Dataset):
    """
    Dataset that breaks each Sudoku board image into 81 labeled cell crops.
    Labels come from the 9x9 board stored in the OCR sidecar JSONL.
    """

    def __init__(
        self,
        dataset_dir: Path,
        jsonl_path: Path,
        cell_px: int = 50,
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
        logging.getLogger("sudoku_ocr_demo").info(f"[DEBUG] dumped {dumped} cell crops to {DEBUG_DIR}")

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


class TinyResBlock(nn.Module):
    def __init__(self, ch: int):
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
    """Small ResNet used for digit classification."""

    def __init__(self, num_classes: int = 10):
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


def compute_class_weights(dataset: Dataset, max_samples: int = 8000, device: str = "cpu") -> torch.Tensor:
    counts = np.zeros(10, dtype=np.int64)
    take = min(max_samples, len(dataset))
    for i in range(take):
        _, y = dataset[i]
        counts[int(y)] += 1
    counts = np.where(counts == 0, 1, counts)  # avoid divide by zero
    inv = 1.0 / counts
    inv = inv / inv.sum() * 10.0
    return torch.tensor(inv, dtype=torch.float32, device=device)
