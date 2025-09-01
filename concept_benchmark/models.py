import itertools
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.train import (
    train_concept_heads,
)


class ConceptDetector(object):
    """
    Concept detector with optional calibration.
    Trains per-concept heads and can apply Platt scaling at inference.
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
        self.calibration_params: Optional[List[Optional[dict]]] = None

    def fit(
        self,
        train_dataset: ConceptDatasetSample,
        valid_dataset: ConceptDatasetSample,
        freeze: bool = True,
        embed_params: Optional[dict] = None,
        fit_params: Optional[dict] = None,
        l1_size: Optional[int] = 100,
        n_jobs: Optional[int] = -1,
        calibrate: bool = False,
        log_training: bool = False,
        log_interval: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Fit the concept detector, and optionally fit per-concept calibration.

        Args:
            train_dataset (ConceptDatasetSample): Training dataset.
            valid_dataset (ConceptDatasetSample): Validation dataset.
            freeze (bool): Whether to freeze the embedding model parameters.
            embed_params (Optional[dict]): Parameters for embedding.
            fit_params (Optional[dict]): Parameters for fitting the concept layers.
            l1_size (Optional[int]): Size of the first linear layer in the model.
                The other dimension is determined by the size of the embedded dataset.
            n_jobs (Optional[int]): Kept for API compatibility (unused).
            calibrate (bool): If True, fit Platt scaling (w, b) per concept on validation logits.
        """
        # Train per-concept heads with optional encoder finetuning
        # Propagate logging toggles into fit_params
        if fit_params is None:
            fit_params = {}
        if "verbose" not in fit_params:
            fit_params["verbose"] = bool(log_training)
        if log_interval is not None and "log_interval" not in fit_params:
            fit_params["log_interval"] = int(log_interval)

        heads = train_concept_heads(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            embedding_model=self.embedding_model,
            input_dim=None,
            l1_size=l1_size or 100,
            freeze=freeze,
            fit_params=fit_params,
        )
        self.concept_layers = nn.ModuleList(heads)

        # Optionally fit calibration using validation logits
        if calibrate:
            self.calibrate(valid_dataset, embed_params=embed_params)
        else:
            self.calibration_params = None

    def calibrate(
        self,
        valid_dataset: ConceptDatasetSample,
        embed_params: Optional[dict] = None,
    ) -> None:
        """
        Fit Platt scaling parameters (w, b) per concept on validation logits.

        Args:
            valid_dataset: Validation split used to fit calibration.
            embed_params: Optional kwargs passed to dataset.embed when using an embedding model.
        """
        if self.concept_layers is None:
            raise RuntimeError("Must call fit(...) before calibrating.")

        embedded_valid = (
            valid_dataset.embed(self.embedding_model, **(embed_params or {}))
            if self.embedding_model
            else valid_dataset
        )
        with torch.no_grad():
            X_t = torch.from_numpy(embedded_valid.X).float()
            logits_list = [m(X_t).squeeze(1) for m in self.concept_layers]
            logits = torch.stack(logits_list, dim=1).numpy()

        params = []
        for i in range(embedded_valid.n_concepts):
            z = logits[:, i:i+1]
            y = embedded_valid.C[:, i].astype(int)
            if np.unique(y).size < 2:
                params.append({"w": 1.0, "b": 0.0})
                continue
            lr = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
            lr.fit(z, y)
            params.append({"w": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})

        self.calibration_params = params

    @property
    def n_concepts(self) -> int:
        """
        Return the number of concepts.
        """
        if self.concept_layers is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        return len(self.concept_layers)

    def predict(
        self,
        dataset: ConceptDatasetSample,
        embed_params: Optional[dict] = None,
        calibrate: Optional[bool] = None,
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
            X_t = torch.from_numpy(embedded_dataset.X).float()
            logits_list = [m(X_t).squeeze(1) for m in self.concept_layers]
            logits = torch.stack(logits_list, dim=1).numpy()

        # Determine whether to apply calibration
        if calibrate is None:
            apply_cal = self.calibration_params is not None
        else:
            apply_cal = calibrate
            if apply_cal and self.calibration_params is None:
                raise RuntimeError(
                    "Calibration requested but not fitted. Call fit(..., calibrate=True)."
                )

        if apply_cal and self.calibration_params is not None:
            w = np.array([
                (p.get("w", 1.0) if p is not None else 1.0) for p in self.calibration_params
            ], dtype=np.float32)
            b = np.array([
                (p.get("b", 0.0) if p is not None else 0.0) for p in self.calibration_params
            ], dtype=np.float32)
            z_scaled = logits * w.reshape(1, -1) + b.reshape(1, -1)
            return 1.0 / (1.0 + np.exp(-z_scaled))

        return 1.0 / (1.0 + np.exp(-logits))


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
            concept_detector if concept_detector else ConceptDetector(**kwargs)
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
        freeze: bool = True,
        *,
        concept_fit_params: Optional[dict] = None,
        concept_embed_params: Optional[dict] = None,
        front_fit_params: Optional[dict] = None,
        calibrate: bool = False,
        log_training: bool = False,
        log_interval: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Fit the concept detector and front-end model.

        Args:
            train_dataset: Training split.
            valid_dataset: Validation split (used for concept detector early stopping/calibration).
            freeze: If True, skip (re)training the concept detector and only fit the front-end model.
            concept_fit_params: Dict forwarded to ConceptDetector.fit(..., fit_params=...).
            concept_embed_params: Dict forwarded to dataset.embed(...); passed via ConceptDetector.fit(..., embed_params=...).
            front_fit_params: Dict forwarded to FrontEndModel.fit(..., fit_params=...).
            calibrate: If True, perform per-concept calibration after training detector.

        Backward compatibility:
            - If kwargs contains 'fit_params', it is treated as concept_fit_params.
            - If kwargs contains 'embed_params', it is treated as concept_embed_params.
            - If kwargs contains 'calibrate', it overrides the calibrate flag.
        """
        # Backward-compat: map legacy kwargs
        if concept_fit_params is None and "fit_params" in kwargs:
            concept_fit_params = kwargs.pop("fit_params")
        if concept_embed_params is None and "embed_params" in kwargs:
            concept_embed_params = kwargs.pop("embed_params")
        if "calibrate" in kwargs:
            calibrate = kwargs.pop("calibrate")

        # Ensure concept_fit_params dict exists and inject logging toggles if set
        if concept_fit_params is None:
            concept_fit_params = {}
        concept_fit_params.setdefault("verbose", bool(log_training))
        if log_interval is not None:
            concept_fit_params.setdefault("log_interval", int(log_interval))

        if not freeze:
            self.concept_detector.fit(
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                freeze=freeze,
                embed_params=concept_embed_params,
                fit_params=concept_fit_params,
                calibrate=calibrate,
                log_training=log_training,
                log_interval=log_interval,
            )

        C_train = train_dataset.C  # independent training
        y_train = train_dataset.y
        self.front_end_model.fit(C_train, y_train, fit_params=front_fit_params)

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
