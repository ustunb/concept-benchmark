import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SpecializedHeadSudokuCNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=32, channels=128):
        super(SpecializedHeadSudokuCNN, self).__init__()
        
        # 🧠 1. Shared Backbone
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        
        # 🎯 2. Specialized Prediction Heads
        # We create a separate MLP head for each of the 27 concepts.
        # An nn.ModuleList is the proper way to hold a list of layers.
        
        self.row_heads = nn.ModuleList([
            self._create_head(input_dim=channels * 9, hidden_dim=hidden_dim) for _ in range(9)
        ])
        
        self.col_heads = nn.ModuleList([
            self._create_head(input_dim=channels * 9, hidden_dim=hidden_dim) for _ in range(9)
        ])
        
        self.block_heads = nn.ModuleList([
            self._create_head(input_dim=channels * 9, hidden_dim=hidden_dim) for _ in range(9)
        ])

    def _create_head(self, input_dim, hidden_dim):
        """Helper function to create a single head."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.long()
        
        # --- Shared Backbone ---
        x = self.embedding(x)
        x = x.permute(0, 2, 1).view(-1, x.size(2), 9, 9)
        features = F.relu(self.conv2(F.relu(self.conv1(x)))) # (N, 128, 9, 9)
        
        batch_size = features.size(0)
        all_preds = []

        # --- Process each concept with its dedicated head ---
        
        # Row Predictions
        for i in range(9):
            row_slice = features[:, :, i, :] # (N, 128, 9)
            row_flat = row_slice.reshape(batch_size, -1) # Flatten the slice
            pred = self.row_heads[i](row_flat)
            all_preds.append(pred)
            
        # Column Predictions
        for i in range(9):
            col_slice = features[:, :, :, i] # (N, 128, 9)
            col_flat = col_slice.reshape(batch_size, -1)
            pred = self.col_heads[i](col_flat)
            all_preds.append(pred)
            
        # Block Predictions
        for i in range(9):
            # Calculate top-left corner of the 3x3 block
            row_start = (i // 3) * 3
            col_start = (i % 3) * 3
            block_slice = features[:, :, row_start:row_start+3, col_start:col_start+3] # (N, 128, 3, 3)
            block_flat = block_slice.reshape(batch_size, -1)
            pred = self.block_heads[i](block_flat)
            all_preds.append(pred)
            
        # Concatenate all 27 predictions
        logits = torch.cat(all_preds, dim=1) # (N, 27)
        
        return logits

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
        logits = torch.cat([row_preds, col_preds, block_preds], dim=1) # (N, 27)
        
        return logits

class RobotClassifierCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(RobotClassifierCNN, self).__init__()
        
        # --- Feature Extractor ---
        # Input images are assumed to be 3-channel RGB
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves the dimensions
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves the dimensions again
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Classifier Head ---
        # The input size to the linear layer depends on the input image size.
        # Let's assume input images are 224 x 224 pixels.
        # After 3 pooling layers, the size becomes 224 -> 112 -> 56 -> 28
        # So the feature map size is 64 (channels) * 28 * 28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass through the feature extractor
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the feature maps for the classifier
        x = torch.flatten(x, 1)
        
        # Pass through the classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # For binary classification, we apply a sigmoid function to the output
        return torch.sigmoid(x)

class RobotConceptClassifier(nn.Module):
    def __init__(self, num_concepts: int):
        super(RobotConceptClassifier, self).__init__()
        
        # 1) Shared CNN Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 224 -> 112 -> 56 -> 28 after 3 MaxPool(2,2)
        feature_size = 64 * 28 * 28

        # 2) One head per concept (order matches input labels), wrapped in nn.Sequential
        self.heads = nn.ModuleList([
            nn.Linear(feature_size, 1)
            for _ in range(num_concepts)
        ])

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        # Concatenate per-concept logits into shape (N, num_concepts)
        logits = torch.cat([head(features) for head in self.heads], dim=1)
        return logits