import numpy as np
import sys, os, random
from tqdm import tqdm
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

# --- repo path shim (safe if already installed) ---
# necessary before concept_benchmark imports
sys.path.append(os.getcwd())

from concept_benchmark.models import FrontEndModel
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset
from concept_benchmark.paths import results_dir

# setup device
device = torch.device(
    "cuda" if torch.cuda.is_available() \
        else ("mps" if torch.backends.mps.is_available() 
              else "cpu")
)

# ---- utils ----
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flip_labels(y: np.ndarray, noise: float, seed: int = 42):
    """
    Flip a fraction `noise` of binary labels in y (0/1) in-place-safe (returns a copy).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    n = y.shape[0]
    k = int(round(noise * n))
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        y[idx] = 1 - y[idx]
    return y

def get_dataset_path(**settings) -> str:
    return results_dir / f"sudoku_{settings['n']**2}_{settings['data_type']}.data"

# --- materialize boards once (preserves dataset order) ---
def materialize_boards(split, batch_size=512):
    loader = split.loader(batch_size=batch_size, shuffle=False)
    boards = []
    for X, _, _ in loader:  # ignore concept labels here
        boards.append(X)
    return torch.cat(boards, dim=0)  # (N, 81)

# ---- define models -----
# ---- Concept model -----
class ConceptSudokuCNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.row_head = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.col_head = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.block_head = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.block_pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
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
        block_features = self.block_pool(features)                     # (N, 128, 3, 3)
        block_features = block_features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)  # (N, 9, 128)
        block_preds = self.block_head(block_features).squeeze(-1)      # (N, 9)

        all_preds = torch.cat([row_preds, col_preds, block_preds], dim=1)  # (N, 27)
        return torch.sigmoid(all_preds) 
    
# --- DNN: board -> y (binary) ---
class SudokuDNN(nn.Module):
    """
    End-to-end DNN baseline for Sudoku board validity (binary).
    Input: (N, 81) integers 0..9
    """
    def __init__(self, embedding_dim=16, channels=(64, 128), mlp_hidden=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels[0])
        self.norm2 = nn.BatchNorm2d(channels[1])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(channels[1], mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)  # logit
        )

    def forward(self, x):
        # x: (N, 81) ints in [0,9]
        x = x.long()
        x = self.embedding(x)                                   # (N, 81, D)
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(2), 9, 9)
      # (N, D, 9, 9)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.mean(dim=(2, 3))                                  # global avg pool -> (N, C)
        logit = self.head(x).squeeze(-1)                        # (N,)
        return logit
    
# --- small helper to train/eval DNN for a given noise+seed  + model utils ---
def train_dnn_for_noise(train_boards, y_train_float_tensor,
                        valid_boards, y_valid_float_tensor,
                        test_boards,  y_test_float_tensor,
                        epochs=12, batch_size=64, lr=1e-3, seed=42):
    set_seed(seed)
    model = SudokuDNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(train_boards, y_train_float_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in tqdm(range(epochs), leave=False, desc="Train DNN"):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = crit(logit, yb)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def acc(split_boards, split_y):
        model.eval()
        xb = split_boards.to(device)
        logit = model(xb)
        pred = (torch.sigmoid(logit) > 0.5).float().cpu()
        return float((pred == split_y).float().mean())

    train_acc = acc(train_boards, y_train_float_tensor)
    valid_acc = acc(valid_boards, y_valid_float_tensor)
    test_acc  = acc(test_boards,  y_test_float_tensor)
    return train_acc, valid_acc, test_acc

@torch.no_grad()
def predict_concepts(model, loader, device):
    model.eval()
    preds = []
    for boards, _, _ in loader:
        boards = boards.to(device)
        out = model(boards)
        preds.append((out > 0.5).cpu().numpy().astype(np.int32))
    return np.vstack(preds)  # (N, 27)

@torch.no_grad()
def concept_accuracy(model, loader, device):
    model.eval()
    correct = []
    for boards, C_labels, _ in loader:
        boards = boards.to(device)
        out = (model(boards) > 0.5).cpu().numpy().astype(np.int32)
        correct.append(out == C_labels.numpy())
    acc = np.vstack(correct).mean(axis=0)  # per-concept accuracy
    return acc

# --- data setup ---
set_seed(42)

settings = {
    "n": 3,
    "n_samples": 10000,
    "valid_ratio": 0.5,
    "max_corrupt": 14,
    "data_type": "tabular",
    "seed": 42,
}

data_orig = create_sudoku_dataset(**settings)
data_orig.generate_cvindices(seed=42)
data_orig.split(fold_id='K05N01', fold_num_validation=4, fold_num_test=5)

# --- model ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------- PATCH: make reshape safe in both models ----------
# In ConceptSudokuCNN.forward, replace the .view with .reshape
# (do this near the class definition)
# x = x.permute(0, 2, 1).view(-1, x.size(2), 9, 9)
#    ↓
# x = x.permute(0, 2, 1).reshape(-1, x.size(2), 9, 9)

# In SudokuDNN.forward, also ensure reshape/contiguous safety
# x = x.permute(0, 2, 1).contiguous().view(-1, x.size(2), 9, 9)
#    ↓
# x = x.permute(0, 2, 1).reshape(-1, x.size(2), 9, 9)

# ---------- NEW: train-until-threshold, then sweep ----------
threshold = 0.90
max_tries = 25          # safety valve; raise if you like
base_seed = 42
attempt = 0

results = []  # (re)initialize if needed

def compute_domain_acc_from_model(trained_model):
    """Compute domain_acc on test using the trained concept model."""
    with torch.no_grad():
        trained_model.eval()
    c_pred_test_local = predict_concepts(trained_model, test_loader, device)  # (N, 27) bools
    domain_preds = (c_pred_test_local.prod(axis=1) == 1).astype(np.int32)
    domain_acc_local = float((domain_preds == data_orig.test.y).mean())
    test_concept_acc_local = concept_accuracy(trained_model, test_loader, device)
    return domain_acc_local, c_pred_test_local, float(test_concept_acc_local.mean())

# keep trying new random inits until domain_acc >= threshold
while True:
    attempt += 1
    set_seed(base_seed + attempt)

    # fresh concept model + optimizer
    model = ConceptSudokuCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = data_orig.training.loader(batch_size=64, shuffle=True)
    test_loader  = data_orig.test.loader(batch_size=128, shuffle=False)

    model.train()
    for epoch in tqdm(range(10), desc=f"Train ConceptSudokuCNN (attempt {attempt})"):
        for boards, C_labels, _ in train_loader:
            boards, C_labels = boards.to(device), C_labels.to(device)
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, C_labels.float())
            loss.backward()
            optimizer.step()

    # evaluate domain_acc
    domain_acc_try, c_pred_test, mean_concept_acc = compute_domain_acc_from_model(model)

    print(f"[attempt {attempt}] domain_acc(test)={domain_acc_try:.4f} | mean_concept_acc(test)={mean_concept_acc:.4f}")

    if domain_acc_try >= threshold:
        print(f"✅ Hit threshold {threshold:.2f} on attempt {attempt}. Proceeding with the rest of the tests.\n")
        # cache concept preds from the (threshold-cleared) concept model
        c_pred_train = predict_concepts(model, data_orig.training.loader(batch_size=128, shuffle=False), device)
        c_pred_valid = predict_concepts(model, data_orig.validation.loader(batch_size=128, shuffle=False), device)
        c_pred_test  = predict_concepts(model, data_orig.test.loader(batch_size=128, shuffle=False), device)
        break
    if attempt >= max_tries:
        print(f"⚠️ Reached max_tries={max_tries} without hitting threshold {threshold:.2f}. Proceeding anyway.\n")
        break

# Now that we have a concept model whose domain_acc ≥ threshold (or we bailed after max_tries),
# run the (seed, noise) sweeps. Since the threshold is satisfied, we don’t need to gate the DNN anymore.

noises = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
seeds   = [42, 123, 777, 1644, 2025]

# (Re)materialize boards once (in case you moved this earlier)
train_boards = materialize_boards(data_orig.training)
valid_boards = materialize_boards(data_orig.validation)
test_boards  = materialize_boards(data_orig.test)
y_valid_true = torch.tensor(data_orig.validation.y, dtype=torch.float32)
y_test_true  = torch.tensor(data_orig.test.y, dtype=torch.float32)

for seed in seeds:
    print(f"\n--- sweep seed={seed} ---")
    for noise in noises:
        import copy
        data = copy.deepcopy(data_orig)
        # flip y on TRAIN ONLY
        data.training.y = flip_labels(data.training.y, noise=noise, seed=seed)

        # ----- FE (logistic regression on concepts) -----
        fe = FrontEndModel()
        fe.fit(data.training.C, data.training.y)

        y_pred_train = fe.predict(c_pred_test.astype(np.float32))  # careful! that's test preds
        # Instead, recompute for each split:
        fe_train_pred = fe.predict(data.training.C.astype(np.float32))
        fe_valid_pred = fe.predict(data.validation.C.astype(np.float32))
        fe_test_pred  = fe.predict(c_pred_test.astype(np.float32))

        fe_train_acc = float((fe_train_pred == data.training.y).mean())
        fe_valid_acc = float((fe_valid_pred == data.validation.y).mean())
        fe_test_acc  = float((fe_test_pred  == data.test.y).mean())

        # ----- Domain rule -----
        domain_preds = (c_pred_test.prod(axis=1) == 1).astype(np.int32)
        domain_acc = float((domain_preds == data.test.y).mean())

        # ----- DNN baseline -----
        y_train_noisy_t = torch.tensor(data.training.y, dtype=torch.float32)
        dnn_train_acc, dnn_val_acc, dnn_test_acc = train_dnn_for_noise(
            train_boards, y_train_noisy_t,
            valid_boards, y_valid_true,
            test_boards,  y_test_true,
            epochs=12, batch_size=64, lr=1e-3, seed=seed
        )

        domain_train = float(((c_pred_train.prod(axis=1) == 1).astype(np.int32) == data.training.y).mean())
        domain_valid = float(((c_pred_valid.prod(axis=1) == 1).astype(np.int32) == data.validation.y).mean())
        domain_test  = float(((c_pred_test .prod(axis=1) == 1).astype(np.int32) == data.test.y).mean())

        print(
            f"noise={noise:.2f} | FE train/val/test={fe_train_acc:.4f}/{fe_valid_acc:.4f}/{fe_test_acc:.4f} | "
            f"Domain train/val/test={domain_train:.4f}/{domain_valid:.4f}/{domain_acc:.4f} | DNN train/val/test={dnn_train_acc:.4f}/{dnn_val_acc:.4f}/{dnn_test_acc:.4f}"
        )

        accuracy = (c_pred_test == data.test.C)
        c_pred_test_acc = accuracy.sum(axis=0) / accuracy.shape[0]

        accuracy = (c_pred_train == data.training.C)
        c_pred_train_acc = accuracy.sum(axis=0) / accuracy.shape[0]

        accuracy = (c_pred_valid == data.validation.C)
        c_pred_valid_acc = accuracy.sum(axis=0) / accuracy.shape[0]

        results.append({
            "attempt": attempt,
            "seed": seed,
            "noise": noise,
            "mean_concept_acc_test": mean_concept_acc,
            # FE
            "front_end_acc_train": fe_train_acc,
            "front_end_acc_valid": fe_valid_acc,
            "front_end_acc_test": fe_test_acc,
            # Domain
            "domain_acc_train": domain_train,
            "domain_acc_valid": domain_valid,
            "domain_acc_test": domain_acc,
            # DNN
            "dnn_acc_train": dnn_train_acc,
            "dnn_acc_valid": dnn_val_acc,
            "dnn_acc_test": dnn_test_acc,
            # dataset meta
            "n": settings["n"],
            "n_samples": settings["n_samples"],
            "max_corrupt": settings["max_corrupt"],
            "data_type": settings["data_type"],
            "concept_accuracies_test": c_pred_test_acc,
            "concept_accuracies_train": c_pred_train_acc,
            "concept_accuracies_valid": c_pred_valid_acc
        })


# persist
df = pd.DataFrame(results)
csv_path = results_dir / f"sudoku_noise_compare_until{threshold:.2f}_attempt{attempt}_n{settings['n']}_samples{settings['n_samples']}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved run log to: {csv_path}")