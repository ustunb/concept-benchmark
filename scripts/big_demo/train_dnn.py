import os
from psutil import Process
from argparse import ArgumentParser
import copy
import torch
import torch.nn as nn
from tqdm import tqdm

from demo_models import SudokuValidatorCNN, RobotClassifierCNN
from utils import get_dataset_file, get_model_file, determine_device

from concept_benchmark.ext.fileutils import save, load

settings = {
    'data_name': 'sudoku',
    'data_type': 'tabular',
    'n': 3,
    'max_corrupt': 21,
    'concept_noise': 0.0,  # doesn't matter but need for dataset loading
    'target_accuracy': 1.0,
    'epochs': 10,
    # Early stopping settings
    'patience': 3,      # set to 0 to disable early stopping
    'min_delta': 0.0    # required improvement in val loss to reset patience
}

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--data_name', type=str, choices=['sudoku', 'robot'], default=settings['data_name'])
    p.add_argument('--data_type', type=str, choices=['image', 'text'], default=settings['data_type'])
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--target_accuracy', type=float, default=settings['target_accuracy'])
    p.add_argument('--epochs', type=int, default=settings['epochs'])
    p.add_argument('--patience', type=int, default=settings['patience'], help='Early stopping patience; 0 disables early stopping')
    p.add_argument('--min_delta', type=float, default=settings['min_delta'], help='Minimum improvement in validation loss to reset patience')
    args, _ = p.parse_known_args()
    settings.update(vars(args))

if settings['data_name'] == 'sudoku':
    model = SudokuValidatorCNN()
elif settings['data_name'] == 'robot':
    model = RobotClassifierCNN()
else:
    raise ValueError(f"Unknown dataset name: {settings['data_name']}")

data = load(get_dataset_file(**settings))
data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

# --- Training Run ---
device = determine_device()
print(f"Using device: {device}")
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loader_config = {
    'batch_size': 32,
    'num_workers': 12,
    'pin_memory': True
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
def compute_accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, _, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = (outputs.squeeze() > 0.5).long()  # Thresholding at 0.5
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total > 0 else 0

train_accuracy = compute_accuracy(train_loader)
valid_accuracy = compute_accuracy(valid_loader)
test_accuracy = compute_accuracy(test_loader)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# save the model weights
weights = best_state_dict if best_state_dict is not None else model.state_dict()
save(weights, get_model_file(model_type="dnn", **settings), overwrite=True)
