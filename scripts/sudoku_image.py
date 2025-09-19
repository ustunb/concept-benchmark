import os
import sys
sys.path.append(os.getcwd())
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
    "max_corrupt": 20,
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


# Models
class SudokuCNN(nn.Module):
    def __init__(self):

        super().__init__()

        # extract one feature per 25x25 patch
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


class SudokuConceptDector(nn.Module):
    def __init__(self, sudoku_cnn):

        super().__init__()

        # pre-trained digit classifier
        self.sudoku_cnn = sudoku_cnn

        # concept detector head
        n_features = int(81 * 9)

        self.concept_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 364),
            nn.ReLU(),
            nn.Linear(364, 91),
            nn.ReLU(),
            nn.Linear(91, 27)
        )

    def forward(self, X):
        digits = self.sudoku_cnn(X)
        # probs = F.softmax(digits, dim=-1)  # (B, 81, 9)
        return self.concept_head(digits)

print("Training: prediction on concepts")

sudoku_cnn = torch.load(results_dir / "sudoku_digit_detector.pt", weights_only=False)
model = SudokuConceptDector(sudoku_cnn)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
n_epochs = 200

for epoch in range(n_epochs):

    for X, board, concepts, y in train_loader:
        logits = model(X)
        loss = loss_fn(logits, concepts)

        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():

        loss = 0
        acc = 0
        conf_matrix = np.zeros(4)

        for X, board, concepts, y in val_loader:

            logits = model(X)
            c_pred = torch.sigmoid(logits) > 0.5
            y_pred = (c_pred).all(dim=1) # AND

            loss += loss_fn(logits, concepts)
            acc += (c_pred == concepts).to(torch.float16).mean()
            conf_matrix += confusion_matrix(y, y_pred).ravel()


        if epoch % 10 != 0:
            continue

        perf = [f"{i}: {j}" for i, j in zip(["tn", "fp", "fn", "tp"], confusion_matrix(y, y_pred).ravel())]
        loss = round(float(loss / len(val_loader)), 3)
        acc = round(float(acc / len(val_loader)), 3)

        print(f"epoch {epoch} \t val_loss: {loss:.3f} \t val_acc: {acc:.3f}", "\t", "\t".join(perf))
