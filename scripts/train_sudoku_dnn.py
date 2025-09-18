from concept_benchmark.ext.fileutils import load
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class SudokuValidatorCNN(nn.Module):
    def __init__(self, embedding_dim=16):
        """
        Initializes the Sudoku Validator model.
        
        Args:
            embedding_dim (int): The size of the vector for each number embedding.
        """
        super(SudokuValidatorCNN, self).__init__()
        
        # 🧠 The Embedding Layer
        # We have 10 possible tokens: 0 (for padding/empty) and 1-9 for numbers.
        # It will map each number to a dense vector of size `embedding_dim`.
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        
        # The first Conv2d layer will take the embedding dimension as its input channels.
        self.conv1 = nn.Conv2d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # A fully connected network for the final classification.
        # The input size is 128 (channels) * 9 (height) * 9 (width).
        self.fc1 = nn.Linear(128 * 9 * 9, 256)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 81).
                                The board should be flattened.
        """
        # Ensure input is long type for embedding layer
        x = x.long()
        
        # 1. Apply embedding
        # Input: (batch_size, 81)
        # Output: (batch_size, 81, embedding_dim)
        x = self.embedding(x)
        
        # 2. Reshape for CNN
        # PyTorch CNNs expect input in (N, C, H, W) format.
        # We need to permute the dimensions.
        # (batch_size, 81, embedding_dim) -> (batch_size, embedding_dim, 81)
        x = x.permute(0, 2, 1)
        # -> (batch_size, embedding_dim, 9, 9)
        x = x.view(-1, x.size(1), 9, 9)
        
        # 3. Pass through convolutional layers
        # Output of conv1: (batch_size, 64, 9, 9)
        # Output of conv2: (batch_size, 128, 9, 9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # 4. Flatten for the fully connected layers
        # Output: (batch_size, 128 * 9 * 9)
        x = torch.flatten(x, 1)
        
        # 5. Pass through dense layers for classification
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        
        # 6. Apply sigmoid to get a probability
        return torch.sigmoid(x)

# --- Example Usage ---
# Create an instance of the model
model = SudokuValidatorCNN(embedding_dim=16)

# Create a dummy batch of 4 Sudoku boards (flattened to 81 elements)
# In a real scenario, these would be your actual board data.
dummy_boards = torch.randint(1, 10, (4, 81))

# Get the model's prediction
predictions = model(dummy_boards)

print("Model Architecture:\n", model)
print("\n---")
print(f"Input shape: {dummy_boards.shape}")
print(f"Output predictions shape: {predictions.shape}")
print(f"Example predictions (probabilities):\n{predictions.detach().numpy()}")

model = SudokuValidatorCNN()
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = data.training.loader(batch_size=32, shuffle=True)

for epoch in tqdm(range(10)): 
    for boards, _, labels in train_loader:
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


# print accuracies
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for boards, _, labels in data_loader:
            boards, labels = boards.to(device), labels.to(device)
            outputs = model(boards)
            predicted = (outputs.squeeze() >= 0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

valid_loader = data.validation.loader(batch_size=32, shuffle=False)
test_loader = data.test.loader(batch_size=32, shuffle=False)
train_accuracy = evaluate_model(model, train_loader)
valid_accuracy = evaluate_model(model, valid_loader)
test_accuracy = evaluate_model(model, test_loader)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Validation Accuracy: {valid_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")