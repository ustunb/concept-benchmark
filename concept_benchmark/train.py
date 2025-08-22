import torch
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional

from concept_benchmark.data import ConceptDatasetSample


class TorchSKLearnWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper to make a PyTorch model compatible with sklearn API
    """

    def __init__(self, model: nn.Module):
        self.model = model
        super().__init__()

    # Required to make compatible with CalibratedClassifierCV
    def __sklearn_tags__(self):
        tags = super(TorchSKLearnWrapper, self).__sklearn_tags__()
        tags.estimator_type = "classifier"

        return tags

    def fit(self, X, y):
        # The model is assumed to be pre-trained.
        # ConceptDetectors are binary classifiers
        self.classes_ = np.array([0, 1])  # Required for CalibratedClassifierCV
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            probs = torch.sigmoid(self.model(X_tensor)).numpy()
            # Ensure the output is 2D
            if len(probs.shape) == 1:
                probs = probs.reshape(-1, 1)
            return np.hstack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def train_concept_layer(
    train_dataset: ConceptDatasetSample,
    valid_dataset: ConceptDatasetSample,
    concept_idx: int,
    fit_params: Optional[dict] = None,
    input_dim: Optional[int] = None,
    l1_size: Optional[int] = 100,
) -> nn.Module:
    """
    A helper function to train a single concept layer.

    Args:
        train_dataset (ConceptDatasetSample): training dataset.
        valid_dataset (ConceptDatasetSample): validation dataset.
        concept_idx (int): index of the concept to train.
        fit_params (Optional[dict]): Additional parameters for training.
        input_dim (Optional[int]): Input dimension for the model.
        l1_size (Optional[int]): Size of the first linear layer.

    Returns:
        nn.Module: The trained model.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, l1_size), nn.ReLU(), nn.Linear(l1_size, 1)
    )
    params = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 10,
        "min_delta": 0.01,
        "patience": 5,
    }
    if fit_params:
        params.update(fit_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=params["batch_size"], shuffle=False
    )

    best_val_f1 = -1
    patience_counter = 0

    model.train()
    for _ in tqdm(range(params["epochs"])):
        train_losses, train_f1s = [], []
        for batch_X, batch_C, batch_y in train_loader:
            batch_C_i = batch_C[:, concept_idx]  # Get the specific concept column
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = loss_fn(outputs, batch_C_i)
            loss.backward()
            optimizer.step()

            f1 = f1_score(
                batch_C_i.cpu().numpy(), (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            )
            train_f1s.append(f1)
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_train_f1 = np.mean(train_f1s)

        model.eval()
        val_losses, val_f1s = [], []
        with torch.no_grad():
            for batch_X, batch_C, batch_y in valid_loader:
                batch_C_i = batch_C[:, concept_idx]
                val_outputs = model(batch_X).squeeze()
                val_loss = loss_fn(val_outputs, batch_C_i)
                val_losses.append(val_loss.item())

                val_f1 = f1_score(
                    batch_C_i.cpu().numpy(),
                    (torch.sigmoid(val_outputs) > 0.5).cpu().numpy(),
                )
                val_f1s.append(val_f1)

        avg_val_loss = np.mean(val_losses)
        avg_val_f1 = np.mean(val_f1s)

        # Early stopping
        if avg_val_f1 > best_val_f1 + params["min_delta"]:
            best_val_f1 = avg_val_f1
            patience_counter = 0
            # Optionally save best model
            # torch.save(self.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= params["patience"]:
            print(f"Early stopping at epoch {_ + 1}")
            print(
                f"Epoch {_ + 1}/{params['epochs']} - Train Loss: {avg_train_loss:.4f}, "
                f"Train F1: {avg_train_f1:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {avg_val_f1:.4f}"
            )
            break

    return model


def train_calib_concept_layer(
    train_dataset: ConceptDatasetSample,
    valid_dataset: ConceptDatasetSample,
    concept_idx: int,
    fit_params: Optional[dict] = None,
    input_dim: Optional[int] = None,
    l1_size: Optional[int] = 100,
) -> CalibratedClassifierCV:
    """
    A helper function to train and then calibrate a single concept layer.
    """
    model = train_concept_layer(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        concept_idx=concept_idx,
        fit_params=fit_params,
        input_dim=input_dim,
        l1_size=l1_size,
    )

    sklearn_model = TorchSKLearnWrapper(model)
    sklearn_model.fit(train_dataset.X, train_dataset.C[:, concept_idx])

    calibrated_model = CalibratedClassifierCV(
        sklearn_model, method="sigmoid", cv="prefit"
    )
    calibrated_model.fit(valid_dataset.X, valid_dataset.C[:, concept_idx])

    return calibrated_model


# def train_classical_worker(
#     train_dataset: ConceptDatasetSample,
#     valid_dataset: ConceptDatasetSample,
#     concept_idx: int,
#     fit_params: Optional[dict] = None,
#     input_dim: Optional[int] = None,
#     l1_size: Optional[int] = 100,
# ):
#     """
#     A worker function to train a classical concept layer.
#     Used in parallel processing.
#     """
#     return train_concept_layer(
#         train_dataset=train_dataset,
#         valid_dataset=valid_dataset,
#         concept_idx=concept_idx,
#         fit_params=fit_params,
#         input_dim=input_dim,
#         l1_size=l1_size,
#     )

# def train_calibrated_worker(
#     train_dataset: ConceptDatasetSample,
#     valid_dataset: ConceptDatasetSample,
#     concept_idx: int,
#     fit_params: Optional[dict] = None,
#     input_dim: Optional[int] = None,
#     l1_size: Optional[int] = 100,
# ):
#     """
#     A worker function to train and calibrate a concept layer.
#     Used in parallel processing.
#     """
#     return train_calib_concept_layer(
#         train_dataset=train_dataset,
#         valid_dataset=valid_dataset,
#         concept_idx=concept_idx,
#         fit_params=fit_params,
#         input_dim=input_dim,
#         l1_size=l1_size,
#     )
