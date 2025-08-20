import numpy as np
import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTModel


class RobotConceptViT(nn.Module):
    """Custom ViT model with additional layers for concept prediction in the robot dataset"""

    def __init__(self, num_classes=1000, hidden_dim=512, dropout=0.1):
        super(RobotConceptViT, self).__init__()

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # Freeze the ViT backbone
        for par in self.vit.parameters():
             par.requires_grad = False

        # Get the hidden size from ViT (768 for base model)
        vit_hidden_size = self.vit.config.hidden_size

        # Custom fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vit_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, pixel_values):
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values, interpolate_pos_encoding=True)

        # Use the [CLS] token representation (first token)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0]  # [CLS] token

        logits = self.classifier(cls_token)

        return logits


class RobotConceptTrainer:
    """Training class for robot concept detection"""

    def __init__(self, model, device, seed=123):
        """
        Args:
            model: RobotConceptViT model
            device: torch device (cpu/mps/cuda)
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.seed = seed

    def create_dataloaders(self, dataset, batch_size=16, val_split=0.2, num_workers=0):
        """
        Create train/val dataloaders from RobotImageDataset
        Split by robots, not by copies to avoid data leakage
        """
        unique_robot_ids = dataset.robot_ids
        n_robots = len(unique_robot_ids)

        # Split robots into train/val
        val_size = int(n_robots * val_split)
        train_size = n_robots - val_size

        # Randomly select robots for train/val
        torch.manual_seed(self.seed)
        indices = torch.randperm(n_robots).numpy()

        print(unique_robot_ids)
        train_robot_ids = unique_robot_ids.iloc[indices[:train_size]]
        val_robot_ids = unique_robot_ids.iloc[indices[train_size:]]

        # Create boolean masks for train/val samples
        train_mask = dataset.robot_ids.isin(train_robot_ids)
        val_mask = dataset.robot_ids.isin(val_robot_ids)

        train_indices = dataset.robot_ids[train_mask].index.tolist()
        val_indices = dataset.robot_ids[val_mask].index.tolist()

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        print(f"Train robots: {len(train_robot_ids)}, Train samples: {len(train_dataset)}")
        print(f"Val robots: {len(val_robot_ids)}, Val samples: {len(val_dataset)}")

        return train_loader, val_loader

    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [Train]')

        for batch_idx, (images, targets, _) in enumerate(progress_bar):
            images, targets = images.to(self.device), targets.to(self.device).float()

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})

        return total_loss / num_batches

    def validate_epoch(self, val_loader, criterion, epoch, feature_names):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1} [Val]')

        with torch.no_grad():
            for images, targets, _ in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device).float()

                outputs = self.model(images)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                # Get predictions (apply sigmoid for multi-label)
                predictions = torch.sigmoid(outputs) > 0.5

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        concept_accuracies = []
        for i in range(len(feature_names)):
            acc = (all_predictions[:, i] == all_targets[:, i]).mean()
            concept_accuracies.append(acc)

        avg_accuracy = np.mean(concept_accuracies)
        avg_loss = total_loss / len(val_loader)

        return avg_loss, avg_accuracy, concept_accuracies

    def train(self, dataset, epochs=50, batch_size=16, learning_rate=1e-4,
              val_split=0.2, save_best=True, model_save_path="robot_concept_model.pth"):
        """
        Main training loop

        Args:
            dataset: RobotImageDataset
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            val_split: Fraction of robots to use for validation
            save_best: Whether to save the best model
            model_save_path: Path to save the best model
        """

        # Get feature names for logging
        feature_names = dataset.get_concept_names()
        print(f"Training model to predict: {feature_names}")

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(
            dataset, batch_size, val_split
        )

        # Setup training
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")

        epoch = 0

        while epoch <= epochs and best_val_acc < 1.0:
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion, epoch)

            # Validate
            val_loss, val_acc, concept_accs = self.validate_epoch(val_loader, criterion, epoch, feature_names)

            # Step scheduler
            scheduler.step()

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Print results
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")

            concept_acc_str = ', '.join([f"{feat}:{acc:.3f}" for feat, acc in zip(feature_names, concept_accs)])
            print(f"  Concept Accuracies: {concept_acc_str}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'feature_names': feature_names,
                    'model_params': {'num_classes': len(feature_names), 'hidden_dim': 512, 'dropout': 0.1}
                }, model_save_path)
                print(f"  -> New best model saved! (Val Acc: {val_acc:.4f})")

            epoch += 1

            print("-" * 50)

        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")

        return self.train_losses, self.val_losses, self.val_accuracies
