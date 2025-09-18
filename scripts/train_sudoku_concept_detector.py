import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from concept_benchmark.ext.fileutils import load
from concept_benchmark.models import FrontEndModel
from concept_benchmark.paths import results_dir


def get_dataset_path(**settings) -> str:
    return results_dir / f"sudoku_{settings['n']**2}_{settings['data_type']}.data"

settings = {
    "n": 3,
    "n_samples": 5000,
    "valid_ratio": 0.5,
    "max_corrupt": 21,
    "data_type": "tabular",
    "seed": 42,
}

data = load(get_dataset_path(**settings))
data.split(fold_id='K05N01', fold_num_validation=4, fold_num_test=5)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class ConceptSudokuCNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=64):
        super(ConceptSudokuCNN, self).__init__()
        
        # 🧠 1. Shared Backbone (Feature Extractor)
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 🎯 2. Prediction Heads
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

model = ConceptSudokuCNN()
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = data.training.loader(batch_size=32, shuffle=True)

for epoch in tqdm(range(10)): 
    for boards, labels, _ in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        model.to(device)
        boards, labels = boards.to(device), labels.to(device)
        outputs = model(boards)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

# print accuracies for each concept
def evaluate_model(model, data_loader):
    """returns accuracy of the model on the given data_loader"""
    with torch.no_grad():
        model.eval()
        preds = []
        for boards, labels, _ in data_loader:
            boards, labels = boards.to(device), labels.to(device)
            outputs = model(boards)
            predicted = (outputs > 0.5)
            batch_correct = (predicted == labels).cpu().numpy()
            preds.append(batch_correct)

    preds = np.vstack(preds)
    accuracy = preds.mean(axis=0)  # Mean accuracy for each concept
    return accuracy

def predict_model(model, data_loader):
    """returns predictions of the model on the given data_loader"""
    with torch.no_grad():
        model.eval()
        preds = []
        for boards, labels, _ in data_loader:
            boards, labels = boards.to(device), labels.to(device)
            outputs = model(boards)
            predicted = (outputs > 0.5).cpu().numpy()
            preds.append(predicted)
    
    preds = np.vstack(preds)
    return preds


train_loader = data.training.loader(batch_size=32, shuffle=False)
valid_loader = data.validation.loader(batch_size=32, shuffle=False)
test_loader = data.test.loader(batch_size=32, shuffle=False)
train_accuracy = evaluate_model(model, train_loader)
valid_accuracy = evaluate_model(model, valid_loader)
test_accuracy = evaluate_model(model, test_loader)

# Concept accuracies
fe = FrontEndModel()
fe.fit(data.training.C, data.training.y)

train_c_pred = predict_model(model, train_loader)
valid_c_pred = predict_model(model, valid_loader)
test_c_pred = predict_model(model, test_loader)

data_embed = data.embed(model, device=device)

train_y_pred = fe.predict(train_c_pred)
valid_y_pred = fe.predict(valid_c_pred)
test_y_pred = fe.predict(test_c_pred)

train_acc = (train_y_pred == data.training.y).mean()
valid_acc = (valid_y_pred == data.validation.y).mean()
test_acc = (test_y_pred == data.test.y).mean()