import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification


class ImageDS(Dataset):
    def __init__(self, X_paths, y, transform):
        self.X = [str(p) for p in X_paths]
        self.y = np.asarray(y, dtype=int)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = Image.open(self.X[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(self.y[i], dtype=torch.long)
        return img, y


def train_eval_image(paths_tr, y_tr, paths_te, y_te, epochs, batch_size, lr, device, seed, tf, input_size=32):
    """
    Train and evaluate using RobotClassifierCNN.

    Args:
        input_size: Image resolution (default 32 to match your "medium" setting)
    """
    from concept_benchmark.models import RobotClassifierCNN

    generator = torch.Generator()
    generator.manual_seed(seed)

    ds_tr = ImageDS(paths_tr, y_tr, tf)
    ds_te = ImageDS(paths_te, y_te, tf)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=generator)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    # Use your custom CNN model
    model = RobotClassifierCNN(num_classes=2, input_size=input_size)
    model.to(device)

    # Use CrossEntropyLoss for 2-class classification
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in dl_tr:
            xb = xb.to(device)  # Direct tensor, no dict unpacking needed
            yb = yb.to(device)

            out = model(xb)
            loss = criterion(out, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

    model.eval()
    correct = 0
    total = 0
    preds = []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=-1)
            preds.append(pred.cpu().numpy())
            correct += (pred == yb).sum().item()
            total += yb.numel()


    acc = correct / total if total > 0 else 0.0
    return float(acc), model, preds
