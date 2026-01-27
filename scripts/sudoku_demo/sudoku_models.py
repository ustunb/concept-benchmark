import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SudokuValidatorCNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=128):
        super(SudokuValidatorCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = x.permute(0, 2, 1).view(-1, x.size(2), 9, 9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = self.mlp(x)
        return torch.sigmoid(x)


class SpecializedHeadSudokuCNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=128):
        super(SpecializedHeadSudokuCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.row_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.col_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.block_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.block_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.final = nn.Sequential(
            nn.Linear(27, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = x.permute(0, 2, 1).view(-1, x.size(2), 9, 9)
        x = F.relu(self.conv1(x))
        features = F.relu(self.conv2(x))

        row_features = torch.mean(features, dim=3)
        row_features = row_features.permute(0, 2, 1)
        row_preds = self.row_head(row_features).squeeze(-1)

        col_features = torch.mean(features, dim=2)
        col_features = col_features.permute(0, 2, 1)
        col_preds = self.col_head(col_features).squeeze(-1)

        block_features = self.block_pool(features)
        block_features = block_features.view(features.size(0), features.size(1), -1)
        block_features = block_features.permute(0, 2, 1)
        block_preds = self.block_head(block_features).squeeze(-1)

        validity_logits = torch.cat([row_preds, col_preds, block_preds], dim=1)
        return torch.sigmoid(self.final(validity_logits))


class SpecializedHeadConceptSudokuCNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=32, channels=128):
        super(SpecializedHeadConceptSudokuCNN, self).__init__()
        
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
        concatenated_preds = torch.cat(all_preds, dim=1) # (N, 27)

        return concatenated_preds


class GroupPoolingConceptSudokuCNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=64):
        super(GroupPoolingConceptSudokuCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _pool_groups(self, x, dim):
        mean = x.mean(dim=dim)
        maxv = x.amax(dim=dim)
        return torch.cat([mean, maxv], dim=-1)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)  # (N, 81, D)
        x = x.view(x.size(0), 9, 9, -1)  # (N, 9, 9, D)

        row_feats = self._pool_groups(x, dim=2)  # (N, 9, 2D)
        col_feats = self._pool_groups(x, dim=1)  # (N, 9, 2D)

        blocks = x.view(x.size(0), 3, 3, 3, 3, x.size(-1))
        blocks = blocks.permute(0, 1, 3, 2, 4, 5).contiguous()
        block_cells = blocks.view(x.size(0), 9, 9, x.size(-1))
        block_feats = self._pool_groups(block_cells, dim=2)  # (N, 9, 2D)

        row_logits = self.head(row_feats).squeeze(-1)
        col_logits = self.head(col_feats).squeeze(-1)
        block_logits = self.head(block_feats).squeeze(-1)

        logits = torch.cat([row_logits, col_logits, block_logits], dim=1)
        return logits
