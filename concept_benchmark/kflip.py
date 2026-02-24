from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from concept_benchmark.models import ConceptBasedModel
from concept_benchmark.intervention import (
    InterventionStrategy,
    InterventionBatch,
    InterventionConfig,
    StrategyProposal,
    InterventionError,
)


class KFlipInterventionStrategy(InterventionStrategy):
    """
    K-flip probability strategy.

    Config mapping:
      - k is taken from config.max_concepts_per_instance (must be > 0).
      - flip threshold uses config.score_threshold in [0,1].
      - Budgets and selection use the standard helper:
          * instance filtering: _select_instances(...)
          * concept and per-instance caps: _apply_ordering(...)

    Notes:
      - Uses 'hard' others mode: non-intervened concepts are binarized (p>=0.5).
        This matches the runner's path for non-safeguard strategies.
    """

    def __init__(
        self,
        *,
        batch_size: int = 8192,
        limit_subsets: Optional[int] = None,
        exact_k: bool = False,
    ) -> None:
        super().__init__(name="kflip")
        self.batch_size = int(batch_size)
        self.limit_subsets = limit_subsets
        self.exact_k = exact_k

    def propose(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> StrategyProposal:
        # --- inputs and guards
        n_samples, n_concepts = batch.C_pred.shape
        k = config.max_concepts_per_instance
        if k is None or int(k) <= 0:
            raise InterventionError(
                "KFlip requires config.max_concepts_per_instance (k) to be a positive integer."
            )
        k = int(min(k, n_concepts))
        threshold = float(config.score_threshold)

        # concept probabilities and baseline concept vectors
        P = np.clip(batch.C_pred.astype(np.float64), 1e-9, 1.0 - 1e-9)  # (N,C)
        base_Z = (P >= 0.5).astype(np.float32)                           # 'hard' mode

        # baseline labels
        base_probs = model.front_end_model.predict_proba(base_Z)         # (N,K)
        base_lbl = base_probs.argmax(axis=1)
        n_classes = int(base_probs.shape[1])

        # enumerate candidate subsets
        if self.exact_k:
            all_subsets = list(itertools.combinations(range(n_concepts), k))
        else:
            all_subsets = []
            for j in range(1, k+1):
                all_subsets.extend(itertools.combinations(range(n_concepts), j))
        if self.limit_subsets is not None and self.limit_subsets < len(all_subsets):
            # simple heuristic: closeness to 0.5 weighted by |coef|
            try:
                coef = getattr(getattr(model.front_end_model, "model", model.front_end_model), "coef_", None)
                if coef is None:
                    w = np.ones((1, n_concepts), dtype=np.float32)
                else:
                    w = np.abs(coef)
                    if w.ndim == 1:
                        w = w[None, :]
                    w = np.max(w, axis=0, keepdims=True)
            except (AttributeError, IndexError, ValueError):
                w = np.ones((1, n_concepts), dtype=np.float32)
            feat_score = np.mean((0.5 - np.abs(P - 0.5)) * w, axis=0)

            subset_scores = []
            for subset in all_subsets:
                avg_score = np.mean([feat_score[i] for i in subset])
                subset_scores.append((avg_score, subset))

            # Take top scoring subsets
            subset_scores.sort(key=lambda x: x[0], reverse=True)
            all_subsets = [subset for _, subset in subset_scores[:self.limit_subsets]]

        flip_prob = np.zeros(n_samples, dtype=np.float64)
        best_subset: List[Tuple[int, ...]] = [tuple() for _ in range(n_samples)]
        best_label = np.full(n_samples, -1, dtype=int)

        # batching
        rows_per_eval = max(1, self.batch_size)

        # evaluate all subsets
        for subset in tqdm(all_subsets):
            subset = np.asarray(subset, dtype=int)
            subset_size = len(subset)

            # Create assignment grid for the particular subset size
            assign = np.array(list(itertools.product([0.0, 1.0], repeat=subset_size)), dtype=np.float32)
            A = int(assign.shape[0])
            samples_per_chunk = max(1, rows_per_eval // A)

            for s in range(0, n_samples, samples_per_chunk):
                m = min(samples_per_chunk, n_samples - s)

                pS = P[s: s + m][:, subset]  # (m, subset_size)
                w_assign = np.prod(
                    np.where(assign[None, :, :] == 1.0, pS[:, None, :], 1.0 - pS[:, None, :]),
                    axis=2,
                )  # (m, A)

                Z_chunk = np.repeat(base_Z[s: s + m], A, axis=0)  # (m*A, C)
                AS = np.tile(assign, (m, 1))  # (m*A, subset_size)
                Z_chunk[:, subset] = AS

                Y = model.front_end_model.predict_proba(Z_chunk)    # (m*A, j)
                Y_lbl = Y.argmax(axis=1).reshape(m, A)              # (m, A)

                flip_mask = (Y_lbl != base_lbl[s : s + m][:, None])  # (m, A)
                mass = (w_assign * flip_mask).sum(axis=1)            # (m,)

                # track destination label mass (optional)
                cls_mass = (Y.reshape(m, A, n_classes) * (w_assign * flip_mask)[:, :, None]).sum(axis=1)
                lbl_star = cls_mass.argmax(axis=1)

                cur = flip_prob[s : s + m]
                improve = mass > cur
                if np.any(improve):
                    flip_prob[s : s + m][improve] = mass[improve]
                    best_label[s : s + m][improve] = lbl_star[improve]
                    for rel in np.nonzero(improve)[0]:
                        best_subset[s + rel] = tuple(int(x) for x in subset.tolist())

        # candidate instances: exceed threshold (+ optional abstention filter)
        y_prob_now = base_probs
        pred_now = base_lbl
        conf = y_prob_now[np.arange(n_samples), pred_now]
        if config.select_only_abstained and config.tau is not None:
            abstain_mask = (conf >= config.tau) & (conf <= 1.0 - config.tau)
            candidate_ids = np.nonzero((flip_prob >= threshold) & abstain_mask)[0]
        else:
            candidate_ids = np.nonzero(flip_prob >= threshold)[0]

        selected = self._select_instances(candidate_ids, config, rng=config.rng)

        # build mask and enforce budgets via helper
        mask = np.zeros_like(batch.C_pred, dtype=bool)
        total_applied = 0
        for idx in selected:
            order = list(best_subset[idx]) if best_subset[idx] else []
            if not order:
                continue
            total_applied = self._apply_ordering(
                mask,
                order,
                [int(idx)],
                config=config,
                start_total=total_applied,
            )

        details: Dict[str, object] = {
            "flip_prob": flip_prob,
            "best_subset": best_subset,
            "threshold": threshold,
            "k": k,
            "selected_by_threshold": np.array(candidate_ids, dtype=int),
            "best_label": best_label,
        }

        return StrategyProposal(
            mask=mask,
            ordering_used=None,
            selected_instances=np.asarray(selected, dtype=int),
            details=details,
        )
