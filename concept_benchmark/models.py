from abc import ABC, abstractmethod
import torch
from concept_benchmark.data import ConceptDataset
import torch.nn as nn
from torch.utils.data import DataLoader

class ConceptDetector(object):
    pass

class FrontEndModel(object):
    pass

class ConceptBasedModel(nn.Module):
    """
    Abstract base class for concept-based models.
    """

    def __init__(
        self, 
        concept_detector: ConceptDetector, 
        frontend_model: FrontEndModel
    ):
        """
        Initialize the model with a concept detector and a frontend model.
        
        Parameters:
        - concept_detector: An instance of ConceptDetector.
        - frontend_model: An instance of FrontEndModel.
        """
        self.concept_detector = concept_detector
        self.frontend_model = frontend_model

    def train(
        self, 
        dataset: ConceptDataset
    ):
        """
        Fit the model to the data.
        
        Parameters:
        - X: Features
        - C: Concepts
        - Y: Labels
        """
        pass

    def predict(
        self, 
        X: torch.Tensor = None, 
        loader: DataLoader = None
    ) -> torch.Tensor:
        """
        Predict using the model.
        
        Parameters:
        - X: Features
        
        Returns:
        - Predictions
        """
        assert X is not None or loader is not None, \
            "Either X or loader must be provided"
        
        if loader is not None:
            predictions = []
            for batch in loader:
                x_batch = batch[0]
                with torch.no_grad():
                    preds = self(x_batch)
                predictions.append(preds)
            return torch.cat(predictions, dim=0)
        else:
            with torch.no_grad():
                return self(X)