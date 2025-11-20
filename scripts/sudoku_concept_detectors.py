#!/usr/bin/env python3
import sys, os, random, json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader

# make repo importable
sys.path.append(os.getcwd())
from concept_benchmark.ext import fileutils


# ==================== setup ====================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== dataset ====================
class SudokuImageConceptDataset(Dataset):
    """
    One split from your *_images.pkl.

    Expects:
      split.X = list of image paths (relative or absolute)
      split.C = (N, 27) concept labels
      split.y = (N,) board-valid labels
    """
    def __init__(self, images_root: Path, split, img_size: int = 288):
        self.images_root = images_root
        self.paths = list(split.X)
        self.C = np.asarray(split.C, dtype=np.int64)
        self.y = np.asarray(split.y, dtype=np.int64)
        self.img_size = img_size

        assert self.C.ndim == 2 and self.C.shape[1] == 27, "expected (N,27) concepts"

    def __len__(self):
        return len(self.paths)

    def _resolve(self, p):
        p = Path(p)
        if p.is_absolute():
            return p
        return self.images_root / p

    def __getitem__(self, idx):
        img_path = self._resolve(self.paths[idx])
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            # missing image → black image
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            bgr = cv2.resize(bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))   # (3,H,W)
        x = torch.from_numpy(img).float()
        c = torch.from_numpy(self.C[idx]).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, c, y, str(img_path)


# ==================== models ====================
class ImageSudokuConceptCNN(nn.Module):
    """
    Image (3xHxW) -> 27 concepts
    We mirror your original idea: produce a 9x9 feature map, then
    row/col/block heads.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),      # 144
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),      # 72
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            # go to 9x9 so we can reuse row/col/block logic
            nn.AdaptiveAvgPool2d((9, 9)),  # (128, 9, 9)
        )

        self.row_head = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.col_head = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.block_head = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.block_pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        # x: (N,3,H,W)
        feats = self.backbone(x)                 # (N,128,9,9)

        # rows
        row_feats = feats.mean(dim=3).permute(0, 2, 1)  # (N,9,128)
        row_preds = self.row_head(row_feats).squeeze(-1)

        # cols
        col_feats = feats.mean(dim=2).permute(0, 2, 1)  # (N,9,128)
        col_preds = self.col_head(col_feats).squeeze(-1)

        # blocks
        block_feats = self.block_pool(feats)            # (N,128,3,3)
        block_feats = block_feats.view(feats.size(0), 128, 9).permute(0, 2, 1)  # (N,9,128)
        block_preds = self.block_head(block_feats).squeeze(-1)

        all_preds = torch.cat([row_preds, col_preds, block_preds], dim=1)  # (N,27)
        return torch.sigmoid(all_preds)


class ImageSudokuDNN(nn.Module):
    """
    End-to-end DNN baseline for Sudoku board validity (binary).
    Input: (N, 3, H, W) images.
    """
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),      # 144
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),      # 72
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((9, 9)),  # (128,9,9)
        )
        self.norm = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # logit
        )

    def forward(self, x):
        # x: (N,3,H,W)
        x = self.backbone(x)              # (N,128,9,9)
        x = self.norm(x)
        x = self.dropout(x)
        x = x.mean(dim=(2, 3))           # global avg pool -> (N,128)
        logit = self.head(x).squeeze(-1) # (N,)
        return logit


# --- small helper to train/eval image DNN + model utils ---
def train_image_dnn(train_loader, val_loader, test_loader,
                    epochs=12, lr=1e-3, seed=42):
    set_seed(seed)
    model = ImageSudokuDNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    # training
    for _ in tqdm(range(epochs), leave=False, desc="Train image DNN"):
        model.train()
        for imgs, _, y, _ in train_loader:
            imgs = imgs.to(device)
            yb = y.float().to(device)  # (B,)
            opt.zero_grad()
            logit = model(imgs)
            loss = crit(logit, yb)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def acc(loader):
        model.eval()
        all_pred = []
        all_true = []
        for imgs, _, y, _ in loader:
            imgs = imgs.to(device)
            yb = y.float()
            logit = model(imgs)
            prob = torch.sigmoid(logit).cpu()
            pred = (prob > 0.5).float()
            all_pred.append(pred)
            all_true.append(yb)
        pred_cat = torch.cat(all_pred, dim=0)
        true_cat = torch.cat(all_true, dim=0)
        return float((pred_cat == true_cat).float().mean())

    train_acc = acc(train_loader)
    valid_acc = acc(val_loader)
    test_acc  = acc(test_loader)
    return model, train_acc, valid_acc, test_acc


# ==================== metrics ====================
@torch.no_grad()
def eval_per_concept(model, loader):
    model.eval()
    all_eq = []
    for imgs, C_labels, _, _ in loader:
        imgs = imgs.to(device)
        C_labels = C_labels.to(device)
        out = (model(imgs) > 0.5).float()
        eq = (out == C_labels).float().cpu().numpy()
        all_eq.append(eq)
    if not all_eq:
        return None
    arr = np.concatenate(all_eq, axis=0)  # (N,27)
    return arr.mean(axis=0)               # (27,)


# ==================== main ====================
def main():
    set_seed(42)

    # 1) load your PKL
    pkl_path = Path("~/concept-benchmark/data/sudoku/multi_data_demo_images.pkl").expanduser()
    obj = fileutils.load(str(pkl_path))
    # your file is probably {'image': ConceptDataset(...)}
    if isinstance(obj, dict):
        ds = obj["image"]
    else:
        ds = obj

    # where images live – if paths in X are relative, this makes them work
    images_root = pkl_path.parent

    # 2) build real image datasets/loaders (uses X paths)
    train_ds = SudokuImageConceptDataset(images_root, ds.training, img_size=288)
    val_ds   = SudokuImageConceptDataset(images_root, ds.validation, img_size=288)
    test_ds  = SudokuImageConceptDataset(images_root, ds.test, img_size=288)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)

    # 3) concept model (CNN)
    model = ImageSudokuConceptCNN().to(device)
    crit = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4) train CNN
    epochs = 10
    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train image concept {ep+1}/{epochs}")
        for imgs, C_labels, _, _ in pbar:
            imgs = imgs.to(device)
            C_labels = C_labels.to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, C_labels)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

    # 5) compute per-concept accuracies on val/test (CNN)
    val_per_concept  = eval_per_concept(model, val_loader)
    test_per_concept = eval_per_concept(model, test_loader)

    # 6) train image-based DNN baseline on board validity (images only, no meta)
    dnn_model, dnn_train_acc, dnn_val_acc, dnn_test_acc = train_image_dnn(
        train_loader, val_loader, test_loader,
        epochs=12, lr=1e-3, seed=42
    )
    dnn_model.to(device)

    # 7) dump JSONL: all examples + summary with CNN concept stats + DNN board stats
    out_path = Path("./sudoku_image_concepts.jsonl")
    with out_path.open("w") as f, torch.no_grad():
        model.eval()
        dnn_model.eval()

        def dump_split(split_name, loader):
            idx_base = 0
            for imgs, C_labels, y, paths in loader:
                imgs = imgs.to(device)

                # CNN concept predictions
                concept_probs = model(imgs).cpu().numpy()   # (B,27)
                concept_preds = (concept_probs > 0.5).astype(int)

                C_np = C_labels.numpy()
                y_np = y.numpy()

                # DNN board-valid predictions
                dnn_logits = dnn_model(imgs)                # (B,)
                dnn_probs  = torch.sigmoid(dnn_logits).cpu().numpy()  # (B,)
                dnn_preds  = (dnn_probs > 0.5).astype(int)            # (B,)

                batch_size = imgs.size(0)
                for i in range(batch_size):
                    rec = {
                        "split": split_name,
                        "index": int(idx_base + i),
                        "img_path": paths[i],

                        # CNN concept outputs
                        "concept_probs": concept_probs[i].tolist(),
                        "true_concepts": C_np[i].tolist(),
                        "pred_concepts": concept_preds[i].tolist(),

                        # CNN-derived board validity (all concepts == 1)
                        "true_board_valid": int(y_np[i]),
                        "pred_board_valid": int(int(concept_preds[i].prod() == 1)),

                        # DNN board-valid outputs (what you asked for)
                        "dnn_board_valid_prob": float(dnn_probs[i]),  # 1) probability board is valid
                        "dnn_board_valid_pred": int(dnn_preds[i]),    # 2) final 0/1 classification
                    }
                    f.write(json.dumps(rec) + "\n")

                idx_base += batch_size

        dump_split("training",   train_loader)
        dump_split("validation", val_loader)
        dump_split("test",       test_loader)

        # clear CNN vs DNN stats in summary
        summary = {
            "_summary": True,

            # CNN (concept model) concept accuracies
            "cnn_validation_per_concept_accuracy":
                None if val_per_concept is None else val_per_concept.tolist(),
            "cnn_validation_mean_concept_accuracy":
                None if val_per_concept is None else float(val_per_concept.mean()),
            "cnn_test_per_concept_accuracy":
                None if test_per_concept is None else test_per_concept.tolist(),
            "cnn_test_mean_concept_accuracy":
                None if test_per_concept is None else float(test_per_concept.mean()),

            # DNN (image baseline) board-valid accuracies
            "dnn_train_board_accuracy": float(dnn_train_acc),
            "dnn_validation_board_accuracy": float(dnn_val_acc),
            "dnn_test_board_accuracy": float(dnn_test_acc),
        }
        f.write(json.dumps(summary) + "\n")

    print(f"wrote per-example predictions and per-concept accuracies to {out_path}")


if __name__ == "__main__":
    main()
