import itertools
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

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
        # Monte Carlo propagation configuration
        mc_mode: str = "auto",  # 'auto' | 'mc' | 'exact'
        mc_samples: int = 1024,
        mc_max_samples: int = 16384,
        mc_chunk_size: int = 2048,
        mc_tol: float = 1e-3,
        random_state: Optional[int] = None,
        mc_exact_threshold: int = 4096,
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

        # MC configuration
        assert mc_mode in {"auto", "mc", "exact"}
        self._mc_mode = mc_mode
        self._mc_samples = int(mc_samples)
        self._mc_max_samples = int(mc_max_samples)
        self._mc_chunk_size = int(mc_chunk_size)
        self._mc_tol = float(mc_tol)
        self._random_state = random_state
        self._mc_exact_threshold = int(mc_exact_threshold)

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
            # Only prepare exact propagation tables if we'll use exact mode
            try:
                n_concepts = self.concept_detector.n_concepts
            except Exception:
                n_concepts = None
            if n_concepts is not None and self._should_use_exact(n_concepts):
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
            n_concepts = concept_preds.shape[1]
            if self._should_use_exact(n_concepts):
                return self._propagate_predict_proba(concept_preds)
            else:
                return self._propagate_predict_proba_mc(concept_preds)

        binary_concept_preds = (concept_preds > 0.5).astype(np.float32)
        pred_y_prob = self.front_end_model.predict_proba(binary_concept_preds)

        out = pred_y_prob if not return_concepts else (pred_y_prob, concept_preds)

        return out

    def _propagate_predict_proba(
        self,
        concept_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Predict probabilities using concept propagation.
        """
        print("Using concept propagation...")
        if self._concept_poss is None or self._y_proba_all_concepts is None:
            print("Preparing for propagation...")
            self._prep_propagation()

        # Vectorized propagation over all samples and concept combinations
        # Shapes:
        #   concept_preds: (N, C)
        #   concept_poss:  (M, C) with binary {0,1}
        #   y_proba_all_concepts: dict[(C,)-> (1,K)] -> stacked to (M, K)

        # Ensure concept combination order aligns with y_proba matrix rows
        combs = self._concept_poss  # (M, C)
        # Stack dict values in the same order as combs
        y_mat = np.vstack([
            np.asarray(self._y_proba_all_concepts[tuple(c)]).reshape(1, -1)
            for c in combs
        ])  # (M, K)

        P = np.asarray(concept_preds, dtype=np.float64)  # (N, C)
        # Clip for numerical stability when taking logs
        P = np.clip(P, 1e-9, 1.0 - 1e-9)

        # Compute log-weights for each sample and concept combination:
        # log w_ij = sum_k [ c_jk * log p_ik + (1-c_jk) * log(1-p_ik) ]
        logP = np.log(P)                 # (N, C)
        log1mP = np.log1p(-P)            # (N, C)
        A = combs.T.astype(np.float64)   # (C, M)
        logW = logP @ A + log1mP @ (1.0 - A)  # (N, M)
        W = np.exp(logW)                 # (N, M)

        # Aggregate over concept combinations to get class probabilities
        # result: (N, K)
        out = W @ y_mat

        return out

    def _should_use_exact(self, n_concepts: int) -> bool:
        if self._mc_mode == "exact":
            return True
        if self._mc_mode == "mc":
            return False
        # auto: use exact if 2^C <= threshold
        try:
            return (2 ** n_concepts) <= self._mc_exact_threshold
        except OverflowError:
            return False

    def _propagate_predict_proba_mc(self, concept_preds: np.ndarray) -> np.ndarray:
        """
        Monte Carlo propagation: approximate E_y[ y | concept probabilities ]
        by sampling concept vectors and averaging front-end predictions.
        """
        print("Using MC concept propagation...")
        P = np.asarray(concept_preds, dtype=np.float64)
        P = np.clip(P, 1e-9, 1.0 - 1e-9)
        N, C = P.shape

        # Initialize accumulators
        counts = np.zeros(N, dtype=np.int64)
        sum_acc = None  # will be (N, K)
        sumsq_acc = None  # will be (N, K)
        done = np.zeros(N, dtype=bool)

        # RNG: deterministic only if seed provided
        rng = (np.random.default_rng(self._random_state)
               if self._random_state is not None else np.random.default_rng())

        target_samples = max(1, self._mc_samples)
        max_samples = max(target_samples, self._mc_max_samples)
        chunk_size = max(1, self._mc_chunk_size)

        while True:
            active_idx = np.where(~done)[0]
            if active_idx.size == 0:
                break

            # Determine per-loop chunk size constrained by remaining budget per active example
            remaining = max_samples - counts[active_idx]
            if remaining.min() <= 0:
                # Reached max_samples for some/all active; mark exhausted as done
                done[active_idx[remaining <= 0]] = True
                continue
            s = int(min(chunk_size, remaining.min()))

            # Sample Bernoulli for active examples: shape (A, s, C)
            P_active = P[active_idx]  # (A, C)
            Z = (rng.random((P_active.shape[0], s, C)) < P_active[:, None, :]).astype(np.float32)
            Z_flat = Z.reshape(-1, C)

            # Deduplicate concept vectors to reduce model calls
            try:
                uniq, inv = np.unique(Z_flat, axis=0, return_inverse=True)
                y_uniq = self.front_end_model.predict_proba(uniq)
                Y_flat = y_uniq[inv]
            except Exception:
                # Fallback without deduplication
                Y_flat = self.front_end_model.predict_proba(Z_flat)

            # Reshape back to (A, s, K)
            Y = Y_flat.reshape(P_active.shape[0], s, -1)
            if sum_acc is None:
                K = Y.shape[2]
                sum_acc = np.zeros((N, K), dtype=np.float64)
                sumsq_acc = np.zeros((N, K), dtype=np.float64)

            # Update accumulators
            chunk_sum = Y.sum(axis=1)        # (A, K)
            chunk_sumsq = (Y ** 2).sum(axis=1)  # (A, K)
            sum_acc[active_idx] += chunk_sum
            sumsq_acc[active_idx] += chunk_sumsq
            counts[active_idx] += s

            # Check convergence for active examples
            means = sum_acc[active_idx] / counts[active_idx][:, None]
            vars_ = sumsq_acc[active_idx] / counts[active_idx][:, None] - means ** 2
            np.maximum(vars_, 0.0, out=vars_)  # numerical safety
            se = np.sqrt(vars_ / counts[active_idx][:, None])

            # Aggregate SE across classes (flexible; default: max over classes)
            # This can be swapped out for other strategies if needed.
            se_agg = np.max(se, axis=1)
            conv_mask = (se_agg <= self._mc_tol) & (counts[active_idx] >= target_samples)
            done[active_idx[conv_mask]] = True

            # Also stop if everyone has at least target_samples and we've hit that
            if np.all(counts >= target_samples) and done.all():
                break

            # If some have reached max_samples after this update, mark done
            done |= counts >= max_samples

        # Final means as output
        if sum_acc is None:
            # No sampling happened (edge case), fallback to deterministic round
            return self.front_end_model.predict_proba((P > 0.5).astype(np.float32))
        out = sum_acc / counts[:, None]
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
        all_y_probas = {}
        
        for c in tqdm(self.concept_poss):
            pr = self.front_end_model.predict_proba(c.reshape(1, -1))
            all_y_probas[tuple(c)] = pr

        return all_y_probas

    def _prep_propagation(self):
        """
        Prepare for propagation by generating concept possibilities and
        predicting probabilities for all concept combinations.
        """
        self._concept_poss = self._gen_concept_possibilities()
        self._y_proba_all_concepts = self._pred_y_proba_concept_poss()
