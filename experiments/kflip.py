from __future__ import annotations

__all__ = ["KFlipInterventionStrategy"]

import itertools

import numpy as np
from tqdm import tqdm

from experiments.models import ConceptBasedModel
from experiments.intervention import (
    InterventionStrategy,
    InterventionBatch,
    InterventionConfig,
    StrategyProposal,
    InterventionError,
    predict_label_proba_from_concepts,
)


class KFlipInterventionStrategy(InterventionStrategy):
    """Intervention strategy based on concept-flip probability.

    For each instance the strategy enumerates candidate subsets of up to
    *k* concepts (from ``config.max_concepts_per_instance``).  For every
    subset it computes the probability that replacing those concepts with
    their ground-truth values would change the downstream label.  The
    subset with the highest flip probability is selected, and instances
    whose best flip probability exceeds ``config.score_threshold`` become
    intervention candidates.

    Parameters
    ----------
    batch_size : int
        Chunk size for the general-path matrix operations (controls peak
        memory in the non-fast-path branch).
    limit_subsets : int, optional
        When set, caps the number of candidate subsets evaluated per
        instance.  Subsets are ranked by a heuristic that weights concept
        closeness to 0.5 by the absolute logistic-regression coefficient.
    use_exact_k : bool
        If ``True``, only evaluate subsets of exactly size *k*.  If
        ``False`` (default), evaluate all subsets of sizes 1 through *k*.

    Config mapping
    --------------
    - ``config.max_concepts_per_instance`` → *k* (must be > 0)
    - ``config.score_threshold`` → minimum flip probability to intervene
    - Budgets and instance selection use the standard
      ``_select_instances`` / ``_apply_ordering`` helpers.

    Notes
    -----
    Non-intervened concepts are binarized at 0.5 ("hard" mode), matching
    the runner's path for non-safeguard strategies.
    """

    def __init__(
        self,
        *,
        batch_size: int = 8192,
        limit_subsets: int | None = None,
        use_exact_k: bool = False,
    ) -> None:
        super().__init__(name="kflip")
        self.batch_size = int(batch_size)
        self.limit_subsets = limit_subsets
        self.use_exact_k = use_exact_k

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
        supports_aligned = bool(
            getattr(model, "supports_aligned_concept_replay", False)
            or getattr(
                getattr(model, "label_predictor", None),
                "supports_aligned_concept_replay",
                False,
            )
        )
        base_cont = batch.C_pred.astype(np.float32)
        base_Z = (P >= 0.5).astype(np.float32)  # 'hard' mode

        # baseline labels
        base_probs = predict_label_proba_from_concepts(
            model,
            base_cont if supports_aligned else base_Z,
            row_indices=np.arange(n_samples, dtype=int) if supports_aligned else None,
            baseline_concepts=base_cont if supports_aligned else None,
        )  # (N,K)
        base_lbl = base_probs.argmax(axis=1)
        n_classes = int(base_probs.shape[1])

        # enumerate candidate subsets
        if self.use_exact_k:
            all_subsets = list(itertools.combinations(range(n_concepts), k))
        else:
            all_subsets = []
            for j in range(1, k + 1):
                all_subsets.extend(itertools.combinations(range(n_concepts), j))
        if self.limit_subsets is not None and self.limit_subsets < len(all_subsets):
            # simple heuristic: closeness to 0.5 weighted by |coef|
            try:
                coef = getattr(
                    getattr(model.label_predictor, "model", model.label_predictor),
                    "coef_",
                    None,
                )
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
            all_subsets = [subset for _, subset in subset_scores[: self.limit_subsets]]

        flip_prob = np.zeros(n_samples, dtype=np.float64)
        best_subset: list[tuple[int, ...]] = [tuple() for _ in range(n_samples)]
        best_label = np.full(n_samples, -1, dtype=int)

        # Cache assignment grids by subset size (reused across subsets)
        _assign_cache: dict[int, np.ndarray] = {}

        # Try to extract logistic regression weights for the fast path.
        # Instead of building (N*A, C) arrays and calling predict_proba,
        # we precompute the base logit and update only the subset columns
        # via broadcasting: logit[i,a] = base_logit[i] - sub_logit[i] + assign_logit[a]
        _fast_w: np.ndarray | None = None
        _fast_b: float | None = None
        try:
            _lr = getattr(model.label_predictor, "model", model.label_predictor)
            _coef = getattr(_lr, "coef_", None)
            _inter = getattr(_lr, "intercept_", None)
            if (
                _coef is not None
                and _inter is not None
                and getattr(model.label_predictor, "_kflip_fast_path", True)
            ):
                _coef = np.asarray(_coef)
                if _coef.shape[0] == 1:  # binary classification
                    assert n_classes == 2, (
                        f"Fast path assumes binary classification but got {n_classes} classes"
                    )
                    _fast_w = _coef[0].astype(np.float64)
                    _fast_b = float(np.asarray(_inter).flat[0])
        except (AttributeError, IndexError, TypeError):
            pass

        if _fast_w is not None:
            # Memory: O(N * 2^|S|) per subset, where |S| <= k.
            # Fast path: exploit logistic regression linearity.
            # logit = Z @ w + b, label = (logit >= 0)
            # For a subset S with assignment a:
            #   logit_new = (base_logit - base_Z[:,S] @ w[S]) + a @ w[S]
            # This replaces N*A*C matmuls with N*|S| + A*|S| + N*A broadcasting.
            base_Z_f64 = base_Z.astype(np.float64)
            base_logit = base_Z_f64 @ _fast_w + _fast_b  # (N,)

            for subset in tqdm(all_subsets):
                subset_arr = np.asarray(subset, dtype=int)
                ss = len(subset_arr)

                if ss not in _assign_cache:
                    _assign_cache[ss] = np.array(
                        list(itertools.product([0.0, 1.0], repeat=ss)),
                        dtype=np.float64,
                    )
                assign = _assign_cache[ss]  # (A, ss)

                w_sub = _fast_w[subset_arr]  # (ss,)
                remaining = base_logit - base_Z_f64[:, subset_arr] @ w_sub  # (N,)
                assign_logit = assign @ w_sub  # (A,)
                all_logit = remaining[:, None] + assign_logit[None, :]  # (N, A)

                all_lbl = (all_logit >= 0).astype(int)  # (N, A)
                flip_mask = all_lbl != base_lbl[:, None]  # (N, A)

                # Probability weights: P(assignment | concept probs)
                pS = P[:, subset_arr]  # (N, ss)
                w_assign = np.prod(
                    np.where(
                        assign[None, :, :] == 1.0, pS[:, None, :], 1.0 - pS[:, None, :]
                    ),
                    axis=2,
                )  # (N, A)

                mass = (w_assign * flip_mask).sum(axis=1)  # (N,)

                # Destination label mass
                all_prob1 = 1.0 / (1.0 + np.exp(-all_logit))  # (N, A)
                weighted = w_assign * flip_mask
                cls1 = (all_prob1 * weighted).sum(axis=1)  # (N,)
                cls0 = ((1.0 - all_prob1) * weighted).sum(axis=1)  # (N,)
                lbl_star = (cls1 >= cls0).astype(int)  # (N,)

                improve = mass > flip_prob
                if np.any(improve):
                    flip_prob[improve] = mass[improve]
                    best_label[improve] = lbl_star[improve]
                    for rel in np.nonzero(improve)[0]:
                        best_subset[rel] = tuple(int(x) for x in subset_arr.tolist())
        else:
            # General path: call predict_proba on expanded arrays.
            # Memory: O(batch_size * C) per chunk, chunked to stay within batch_size.
            rows_per_eval = max(1, self.batch_size)

            for subset in tqdm(all_subsets):
                subset = np.asarray(subset, dtype=int)
                subset_size = len(subset)

                if subset_size not in _assign_cache:
                    _assign_cache[subset_size] = np.array(
                        list(itertools.product([0.0, 1.0], repeat=subset_size)),
                        dtype=np.float32,
                    )
                assign = _assign_cache[subset_size]
                A = int(assign.shape[0])
                samples_per_chunk = max(1, rows_per_eval // A)

                for s in range(0, n_samples, samples_per_chunk):
                    m = min(samples_per_chunk, n_samples - s)

                    pS = P[s : s + m][:, subset]  # (m, subset_size)
                    w_assign = np.prod(
                        np.where(
                            assign[None, :, :] == 1.0,
                            pS[:, None, :],
                            1.0 - pS[:, None, :],
                        ),
                        axis=2,
                    )  # (m, A)

                    base_chunk = base_cont[s : s + m] if supports_aligned else base_Z[s : s + m]
                    Z_chunk = np.repeat(base_chunk, A, axis=0)  # (m*A, C)
                    AS = np.tile(assign, (m, 1))  # (m*A, subset_size)
                    Z_chunk[:, subset] = AS

                    repeated_rows = (
                        np.repeat(np.arange(s, s + m, dtype=int), A)
                        if supports_aligned
                        else None
                    )
                    repeated_baseline = (
                        np.repeat(base_cont[s : s + m], A, axis=0)
                        if supports_aligned
                        else None
                    )
                    intervention_mask = None
                    if supports_aligned:
                        intervention_mask = np.zeros_like(Z_chunk, dtype=bool)
                        intervention_mask[:, subset] = True

                    Y = predict_label_proba_from_concepts(
                        model,
                        Z_chunk,
                        row_indices=repeated_rows,
                        baseline_concepts=repeated_baseline,
                        intervention_mask=intervention_mask,
                    )  # (m*A, j)
                    Y_lbl = Y.argmax(axis=1).reshape(m, A)  # (m, A)

                    flip_mask = Y_lbl != base_lbl[s : s + m][:, None]  # (m, A)
                    mass = (w_assign * flip_mask).sum(axis=1)  # (m,)

                    cls_mass = (
                        Y.reshape(m, A, n_classes) * (w_assign * flip_mask)[:, :, None]
                    ).sum(axis=1)
                    lbl_star = cls_mass.argmax(axis=1)

                    cur = flip_prob[s : s + m]
                    improve = mass > cur
                    if np.any(improve):
                        flip_prob[s : s + m][improve] = mass[improve]
                        best_label[s : s + m][improve] = lbl_star[improve]
                        for rel in np.nonzero(improve)[0]:
                            best_subset[s + rel] = tuple(
                                int(x) for x in subset.tolist()
                            )

        # candidate instances: exceed threshold (+ optional abstention filter)
        y_prob_now = base_probs
        pred_now = base_lbl
        conf = y_prob_now[np.arange(n_samples), pred_now]
        if config.select_only_abstained and config.abstention_threshold is not None:
            abstain_mask = (conf >= config.abstention_threshold) & (
                conf <= 1.0 - config.abstention_threshold
            )
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

        details: dict[str, object] = {
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
