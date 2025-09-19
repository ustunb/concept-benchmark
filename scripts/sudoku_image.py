import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision.io import read_image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset, image_transform

device = torch.device(
    "cuda" if torch.cuda.is_available() \
        else ("mps" if torch.backends.mps.is_available()
              else "cpu")
)

def transform(*args, **kwargs):
    return image_transform(*args, font_size=18, **kwargs) # increase font size

settings = {
    "dataset_name": "test",
    "n": 3,
    "n_samples": 1000,
    "valid_ratio": 0.5,
    "max_corrupt": 81,
    "data_type": "image",
    "transform": transform,
    "model_type": "vit"
}

data = create_sudoku_dataset(**settings)
data.generate_cvindices(seed=42)
data.split("K05N01", fold_num_validation=4, fold_num_test=5)

with open(results_dir / "sudoku_data.pkl", "wb") as f: # reload later
    pickle.dump(data, f)


class ImageDataset(Dataset):
    def __init__(self, img_files, board, C, y):
        self.board, self.C, self.y = board, C, y
        self.transform = transforms.Compose([
            transforms.Resize((225, 225)), # allows (9, 9) grid, each with shape (25, 25)
            transforms.Grayscale(num_output_channels=1),
            transforms.ConvertImageDtype(torch.float32),
        ])
        self.X = [self.transform(read_image(f)) for f in img_files]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.board[idx], self.C[idx], self.y[idx]

# Load in cases where we don't want to regenerate data repeatedly
with open(results_dir / "sudoku_data.pkl", "rb") as f:
    data = pickle.load(f)

def get_loader(data, split):
    """Custom loader."""
    data = getattr(data, split)
    loader = DataLoader(
        ImageDataset(
            data.X,
            data.meta["boards"][data.indices],
            data.C.astype(np.float32),
            data.y
        ),
        batch_size=128, shuffle=True
    )
    return loader

train_loader = get_loader(data, "training")
val_loader = get_loader(data, "validation")
test_loader = get_loader(data, "test")

class SudokuCNN(nn.Module):
    def __init__(self):

        super().__init__()

        # extract 64 features per 25x25 patch
        self.patch_conv = nn.Conv2d(1, 64, 25, stride=25)

        # classifier after flattening convolution product
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 9)
        )

    def forward(self, x):
        # extract 64 features per 25x25 patch
        features = self.patch_conv(x) # (B, 64, 9, 9)

        # reshape
        features = features.view(-1, 64, 81) # (B, 64, 81)
        features = features.permute(0, 2, 1) # (B, 81, 64)

        # apply classifier to each patch
        logits = self.classifier(features) # (B, 81, 9)

        return logits


print("Training: prediction on 9x9 digits")

model = SudokuCNN()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

n_epochs = 200

for epoch in range(n_epochs):

    for X, board, concepts, y in train_loader:

        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, 9), (board - 1).long().view(-1))

        loss.backward()
        opt.step()
        opt.zero_grad()

    if epoch % 10 != 0:
        continue

    with torch.no_grad():
        loss = 0
        acc = 0
        n = 0
        for X, board, concepts, y in val_loader:

            logits = model(X)

            loss += F.cross_entropy(logits.view(-1, 9), (board - 1).long().view(-1))
            acc += (torch.argmax(logits, dim=-1) == (board.reshape(-1, 81) - 1)).sum() / board.numel()
            n += len(X)

        print(f"epoch {epoch} \t val_loss: {loss / len(val_loader)} \t val_acc: {acc / len(val_loader)}")

torch.save(model, results_dir / "sudoku_digit_detector.pt")

class ConceptSudokuCNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=64):
        super(ConceptSudokuCNN, self).__init__()

        # :brain: 1. Shared Backbone (Feature Extractor)
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # :dart: 2. Prediction Heads
        # Each head is a small neural network that processes aggregated features.
        # The input to each head's MLP will be the number of channels from the last conv layer (128).

        # Head for predicting Row validity
        self.row_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Head for predicting Column validity
        self.col_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Head for predicting Block validity
        self.block_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # A pooling layer to aggregate features for the 3x3 blocks
        self.block_pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        # Ensure input is long type for embedding layer
        x = x.long()

        # --- Shared Backbone ---
        x = self.embedding(x) # (N, 81, D_embed)
        x = x.permute(0, 2, 1).view(-1, x.size(2), 9, 9) # (N, D_embed, 9, 9)
        x = F.relu(self.conv1(x))
        features = F.relu(self.conv2(x)) # (N, 128, 9, 9) - These are the shared features

        # --- Row Predictions ---
        # Aggregate features along each row (mean across the width dimension)
        row_features = torch.mean(features, dim=3) # (N, 128, 9)
        row_features = row_features.permute(0, 2, 1) # (N, 9, 128)
        row_preds = self.row_head(row_features).squeeze(-1) # (N, 9)

        # --- Column Predictions ---
        # Aggregate features along each column (mean across the height dimension)
        col_features = torch.mean(features, dim=2) # (N, 128, 9)
        col_features = col_features.permute(0, 2, 1) # (N, 9, 128)
        col_preds = self.col_head(col_features).squeeze(-1) # (N, 9)

        # --- Block Predictions ---
        # Pool features in each 3x3 block
        block_features = self.block_pool(features) # (N, 128, 3, 3)
        # Flatten the 3x3 grid to get 9 block vectors
        block_features = block_features.view(features.size(0), features.size(1), -1) # (N, 128, 9)
        block_features = block_features.permute(0, 2, 1) # (N, 9, 128)
        block_preds = self.block_head(block_features).squeeze(-1) # (N, 9)

        # Concatenate all predictions
        all_preds = torch.cat([row_preds, col_preds, block_preds], dim=1) # (N, 27)

        return torch.sigmoid(all_preds)


print("Training: prediction on concepts")
sudoku_cnn = torch.load(results_dir / "sudoku_digit_detector.pt", weights_only=False)

model = ConceptSudokuCNN().to(device)
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in tqdm(range(200)):
    for X, board, concepts, y in train_loader:

        board_pred = sudoku_cnn(X).argmax(dim=2)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(board_pred.to(device))
        loss = criterion(outputs, concepts.float().to(device))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

concepts_all = []
concepts_pred_all = []
for X, board, concepts, y in test_loader:
    with torch.no_grad():
        board_pred = sudoku_cnn(X).argmax(dim=2)
        outputs = model(board_pred.to(device))
        concepts_pred_all.append((outputs > 0.5).to(int).to('cpu'))
        concepts_all.append(concepts)
conf_matrix = confusion_matrix(
    torch.vstack(concepts_all).flatten(), torch.vstack(concepts_pred_all).flatten()
)
print("test concept confusion matrix: \n", conf_matrix)

