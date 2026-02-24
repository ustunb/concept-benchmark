import copy
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from psutil import Process
from tqdm import tqdm
from scripts.sudoku_demo.utils import (
    DEFAULT_SUDOKU_SETTINGS,
    determine_device,
    get_dataset_file,
    get_model_file,
    compute_accuracy
)
from scripts.sudoku_demo.sudoku_models import SudokuValidatorCNN as DNNSudokuModel

from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.paths import data_dir


settings = DEFAULT_SUDOKU_SETTINGS.copy()

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--max_corrupt', type=int, default=settings['max_corrupt'])
    p.add_argument('--seed', type=int, default=settings['seed'])
    p.add_argument('--epochs', type=int, default=20)
    args, _ = p.parse_known_args()
    settings.update(vars(args))

training_seed = int(settings["seed"])
dataset_seed = int(DEFAULT_SUDOKU_SETTINGS["seed"])
dataset_settings = settings.copy()
dataset_settings["seed"] = dataset_seed

torch.manual_seed(training_seed)
torch.cuda.manual_seed_all(training_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = DNNSudokuModel()

settings['data_type'] = 'tabular'
dataset_settings['data_type'] = 'tabular'
tab_ds_dir = get_dataset_file(**dataset_settings)
data = load(tab_ds_dir / "sudoku_dataset.pkl")
data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=dataset_seed)
data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

# --- Training Run ---
device = determine_device()
print(f"Using device: {device}")
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def seed_worker(worker_id):
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

loader_config = {
    'batch_size': 32,
    'num_workers': 0 if device.type == 'mps' else 12,
    'pin_memory': False if device.type == 'mps' else True,
    'worker_init_fn': seed_worker,
    'generator': torch.Generator().manual_seed(training_seed)
}

train_loader = data.training.loader(shuffle=True, **loader_config)
valid_loader = data.validation.loader(shuffle=False, **loader_config)
test_loader = data.test.loader(shuffle=False, **loader_config)

# Move model once to device
model.to(device)

# Early stopping state
best_val_loss = float('inf')
best_state_dict = None
epochs_no_improve = 0
use_early_stopping = settings.get('patience', 0) > 0
min_delta = settings.get('min_delta', 0.0)

for epoch in tqdm(range(settings['epochs']), desc="Epochs"):
    # Training
    model.train()
    for X, _, y in train_loader:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y.float())
        loss.backward()
        optimizer.step()

    # Validation
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

    # Early stopping check
    if avg_val_loss < (best_val_loss - min_delta):
        best_val_loss = avg_val_loss
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if use_early_stopping and epochs_no_improve >= settings['patience']:
            print(f"Early stopping at epoch {epoch + 1} with best val loss {best_val_loss:.6f}")
            break

# Restore best model (if we tracked one)
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# compute print accuracy stats
train_accuracy = compute_accuracy(model, train_loader, device=device)
valid_accuracy = compute_accuracy(model, valid_loader, device=device)
test_accuracy = compute_accuracy(model, test_loader, device=device)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# save the model weights
weights = best_state_dict if best_state_dict is not None else model.state_dict()
save(weights, get_model_file(model_class="dnn", **settings), overwrite=True)
