from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
import itertools

from joblib import Parallel, delayed
from typing import Optional, List
from sklearn.linear_model import LogisticRegression

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.train import train_concept_layer, train_calib_concept_layer


class ConceptDetector(ABC):
    """
    Abstract base class for concept detectors.
    """

    def __init__(
        self,
        embedding_model: Optional[nn.Module] = None,
        concept_layers: Optional[List] = None,
    ) -> None:
        """
        Initialize the concept detector with an optional embedding model and concept layers.

        :param embedding_model: Optional PyTorch model for embedding.
        :param concept_layers: Optional list of concept layers (PyTorch models).
        """
        self.embedding_model = embedding_model
        self.concept_layers = concept_layers
        self.train_fn = (
            train_concept_layer
            if isinstance(self, ClassicalConceptDetector)
            else train_calib_concept_layer
        )

    def fit(
        self,
        train_dataset: ConceptDatasetSample,
        valid_dataset: ConceptDatasetSample,
        freeze: bool = True,
        embed_params: Optional[dict] = None,
        fit_params: Optional[dict] = None,
        l1_size: Optional[int] = 100,
        n_jobs: Optional[int] = -1,
        **kwargs,
    ) -> None:
        """
        Fit the concept detector for each concept in the dataset.

        Args:
            train_dataset (ConceptDatasetSample): Training dataset.
            valid_dataset (ConceptDatasetSample): Validation dataset.
            freeze (bool): Whether to freeze the embedding model parameters.
            embed_params (Optional[dict]): Parameters for embedding.
            fit_params (Optional[dict]): Parameters for fitting the concept layers.
            l1_size (Optional[int]): Size of the first linear layer in the model.
                The other dimension is determined by the size of the embedded dataset.
            n_jobs (Optional[int]): Number of parallel jobs to run. -1 for all available cores (default).
        """
        if self.embedding_model and freeze:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

        if self.embedding_model:
            embed_train = train_dataset.embed(
                self.embedding_model, **(embed_params or {})
            )
            embed_valid = valid_dataset.embed(
                self.embedding_model, **(embed_params or {})
            )
        else:
            embed_train = train_dataset
            embed_valid = valid_dataset

        input_dim = embed_train.X.shape[1]
        num_concepts = embed_train.n_concepts

        self.concept_layers = Parallel(n_jobs=n_jobs)(
            delayed(self.train_fn)(
                train_dataset=embed_train,
                valid_dataset=embed_valid,
                concept_idx=i,
                fit_params=fit_params,
                input_dim=input_dim,
                l1_size=l1_size,
            )
            for i in range(num_concepts)
        )

    @abstractmethod
    def predict(self, dataset: ConceptDatasetSample) -> np.ndarray:
        """
        Predict concepts for the dataset.
        """
        raise NotImplementedError

    @property
    def n_concepts(self) -> int:
        """
        Return the number of concepts.
        """
        if self.concept_layers is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        return len(self.concept_layers)


class ClassicalConceptDetector(ConceptDetector):
    """
    A concept detector with uncalibrated concept layers.
    """

    def fit(
        self,
        train_dataset: ConceptDatasetSample,
        valid_dataset: ConceptDatasetSample,
        freeze: bool = True,
        embed_params: Optional[dict] = None,
        fit_params: Optional[dict] = None,
    ) -> None:
        super().fit(train_dataset, valid_dataset, freeze, embed_params, fit_params)
        self.concept_layers = nn.ModuleList(self.concept_layers)

    def predict(
        self,
        dataset: ConceptDatasetSample,
        embed_params: Optional[dict] = None,
    ) -> np.ndarray:
        if self.concept_layers is None:
            raise RuntimeError(
                "Model has not been fitted yet. Please call fit() first."
            )

        embedded_dataset = (
            dataset.embed(self.embedding_model, **(embed_params or {}))
            if self.embedding_model
            else dataset
        )

        with torch.no_grad():
            X_tensor = torch.from_numpy(embedded_dataset.X).float()
            predictions = [
                torch.sigmoid(model(X_tensor)) for model in self.concept_layers
            ]

        return torch.cat(predictions, dim=1).numpy()


class CalibratedConceptDetector(ConceptDetector):
    """
    A concept detector with calibrated concept layers.
    """

    def predict(
        self,
        dataset: ConceptDatasetSample,
        emebed_params: Optional[dict] = None,
    ) -> np.ndarray:
        if self.concept_layers is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        embedded_dataset = (
            dataset.embed(self.embedding_model, **(emebed_params or {}))
            if self.embedding_model
            else dataset
        )

        predictions = [
            model.predict_proba(embedded_dataset.X)[:, 1]
            for model in self.concept_layers
        ]

        return np.array(predictions).T


class FrontEndModel(object):
    def __init__(self, **kwargs) -> None:
        """
        Initialize the front-end model.
        """
        self.model = None

    def fit(
        self, C: np.ndarray, y: np.ndarray, fit_params: Optional[dict] = None
    ) -> None:
        """
        Fit the front-end model to the dataset.
        """
        lr_params = {
            "random_state": 42,
            "max_iter": 1000,
            "solver": "lbfgs",
            "penalty": "l2",
            "C": 1.0,
            "n_jobs": -1,
        }

        if fit_params:
            lr_params.update(fit_params)

        self.model = LogisticRegression(**lr_params)

        self.model.fit(C, y)

    def predict(self, C: np.ndarray) -> np.ndarray:
        """
        Predict label given concepts.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. Please call fit() first."
            )

        return self.model.predict(C)

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        """
        Predict label probabilities given concepts.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet. Please call fit() first."
            )

        return self.model.predict_proba(C)


# Consider inheriting BaseEstimator and ClassifierMixin?
# TODO: add monte carlo sampling propagation
class ConceptBasedModel(object):
    """
    A model that uses concept-based predictions.
    """

    def __init__(
        self,
        concept_detector: Optional[ConceptDetector] = None,
        front_end_model: Optional[FrontEndModel] = None,
        propagate: bool = False,
        **kwargs,
    ) -> None:
        if concept_detector:
            assert isinstance(concept_detector, ConceptDetector), (
                "concept_detector must be an instance of ConceptDetector or its subclass."
            )

        if front_end_model:
            assert isinstance(front_end_model, FrontEndModel), (
                "front_end_model must be an instance of FrontEndModel."
            )

        self.concept_detector = (
            concept_detector
            if concept_detector
            else CalibratedConceptDetector(**kwargs)
        )
        self.front_end_model = front_end_model if front_end_model else FrontEndModel()

        self._propagate = propagate
        self._concept_poss = None
        self._y_proba_all_concepts = None

    @property
    def propagate(self) -> bool:
        return self._propagate

    @property
    def concept_poss(self) -> Optional[np.ndarray]:
        """
        Return all possible concept combinations.
        """
        return self._concept_poss

    @property
    def y_proba_all_concepts(self) -> Optional[dict]:
        """
        Return predicted probabilities for all concept combinations.
        """
        return self._y_proba_all_concepts

    # TODO: separate out fit params for concept detector and front-end model
    def fit(
        self,
        train_dataset: ConceptDatasetSample,
        valid_dataset: ConceptDatasetSample,
        **kwargs,
    ) -> None:
        """
        Fit the concept detector and front-end model.
        """
        self.concept_detector.fit(train_dataset, valid_dataset, **kwargs)

        C_train = train_dataset.C  # independent training
        y_train = train_dataset.y
        # self.front_end_model.fit(C_train, y_train, **kwargs)
        self.front_end_model.fit(C_train, y_train)

        if self.propagate:
            self._prep_propagation()

    def predict(
        self,
        dataset: ConceptDatasetSample,
        propagate: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Predict label for the dataset.
        """
        probas = self.predict_proba(
            dataset,
            propagate=propagate,
        )
        preds = np.argmax(probas, axis=1)

        return preds

    def predict_proba(
        self,
        dataset: ConceptDatasetSample,
        propagate: Optional[bool] = None,
        return_concepts: bool = False,
    ) -> np.ndarray:
        """
        Predict probabilities for the dataset.
        """
        concept_preds = self.concept_detector.predict(dataset)

        # Override object's propagate if specified
        propagate = self.propagate if propagate is None else propagate

        if propagate:
            return self._propagate_predict_proba(concept_preds)

        pred_y_prob = self.front_end_model.predict_proba(concept_preds)

        out = pred_y_prob if not return_concepts else (pred_y_prob, concept_preds)

        return out

    # TODO: figure out if can be more efficient by vectorizing operations
    def _propagate_predict_proba(
        self,
        concept_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Predict probabilities using concept propagation.
        """
        if self._concept_poss is None or self._y_proba_all_concepts is None:
            print("Preparing for propagation...")
            self._prep_propagation()

        proba_lst = []
        for c in concept_preds:
            concept_probas = self._calc_concept_probas(c)
            y_probas_concepts = [
                self.y_proba_all_concepts[tuple(k)] * v
                for k, v in concept_probas.items()
            ]
            proba = np.sum(y_probas_concepts, axis=0)
            proba_lst.append(proba)

        out = np.array(proba_lst)

        return out

    def _calc_concept_probas(self, concept_probas: np.ndarray) -> dict:
        """
        Calculate probabilities for each concept combination
        given concept probabilities (from ConceptDetector).

        Args:
            concept_probas (np.ndarray): Probabilities for each concept (single instance).

        Returns:
            dict: A dictionary where keys are tuples of concept combinations
                  and values are their corresponding probabilities.
        """
        probas = {}
        for c in self.concept_poss:
            probas[tuple(c)] = np.prod(
                (concept_probas**c) * (1 - concept_probas) ** (1 - c)
            )

        return probas

    def _gen_concept_possibilities(self):
        """
        Generate all possible concept combinations.
        """
        n_concepts = self.concept_detector.n_concepts
        all_poss = np.array(list(itertools.product([0, 1], repeat=n_concepts)))

        return all_poss

    def _pred_y_proba_concept_poss(self) -> dict:
        """
        Predict probabilities for all concept combination.
        """
        all_y_probas = {
            tuple(c): pr
            for c, pr in zip(
                self.concept_poss, self.front_end_model.predict_proba(self.concept_poss)
            )
        }

        return all_y_probas

    def _prep_propagation(self):
        """
        Prepare for propagation by generating concept possibilities and
        predicting probabilities for all concept combinations.
        """
        self._concept_poss = self._gen_concept_possibilities()
        self._y_proba_all_concepts = self._pred_y_proba_concept_poss()
