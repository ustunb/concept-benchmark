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

    For each instance the strategy looks for the subset S of at most *k* concepts (``config.per_instance_budget``)
    with the highest flip probability: the probability, under the model's own concept probabilities, that setting
    the concepts in S to their true values changes the predicted label (ties go to the smaller subset). Instances
    whose chosen subset has a flip probability of at least ``config.score_threshold`` become intervention
    candidates. Adding a concept rarely lowers the flip probability, so in practice the subset uses the whole
    budget.

    Search. With at most ``greedy_above`` concepts, every subset of up to *k* concepts is searched. With more,
    the subsets cannot be enumerated: each instance starts from its best single concept and adds the concept
    with the largest gain while the flip probability rises. An instance whose label no single concept can
    flip is then left alone, even if several concepts together could flip it.

    Flip probability. It sums over all ``2^|S|`` value combinations of the subset's concepts while there are at most
    ``n_samples`` of them (``|S| <= 12`` by default) and is otherwise estimated from ``n_samples`` combinations drawn
    from the concept probabilities. The same draws serve every subset, and a subset is re-estimated from them
    before it is compared with its extensions, so a concept's gain compares like with like.

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
        If ``True``, only evaluate subsets of exactly size *k*. If ``False`` (default), use the policy above.
    greedy_above : int
        With more concepts than this, subsets grow greedily from the best single concept.
    n_samples : int
        Value combinations enumerated (subsets with at most this many) or drawn (larger subsets).

    Config mapping
    --------------
    - ``config.per_instance_budget`` → *k* (must be > 0)
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
        greedy_above: int = 12,
        n_samples: int = 4096,
    ) -> None:
        super().__init__(name="kflip")
        self.batch_size = int(batch_size)
        self.limit_subsets = limit_subsets
        self.use_exact_k = use_exact_k
        self.greedy_above = int(greedy_above)
        self.n_samples = int(n_samples)

    def propose(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> StrategyProposal:
        n_samples, n_concepts = batch.C_pred.shape
        k = config.per_instance_budget
        if k is None or int(k) <= 0:
            raise InterventionError(
                "KFlip requires config.per_instance_budget (k) to be a positive integer."
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
        source_rows = (
            np.asarray(batch.instance_ids, dtype=int)
            if batch.instance_ids is not None
            else np.arange(n_samples, dtype=int)
        )
        base_cont = batch.C_pred.astype(np.float32)
        base_Z = (P >= 0.5).astype(np.float32)  # 'hard' mode

        base_probs = predict_label_proba_from_concepts(
            model,
            base_cont if supports_aligned else base_Z,
            row_indices=source_rows if supports_aligned else None,
            baseline_concepts=base_cont if supports_aligned else None,
        )  # (N,K)
        base_lbl = base_probs.argmax(axis=1)
        n_classes = int(base_probs.shape[1])

        is_greedy = not self.use_exact_k and n_concepts > self.greedy_above
        if self.use_exact_k:
            all_subsets = list(itertools.combinations(range(n_concepts), k))
        else:
            all_subsets = [(c,) for c in range(n_concepts)]
            if not is_greedy:
                for size in range(2, k + 1):
                    all_subsets.extend(itertools.combinations(range(n_concepts), size))
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

            subset_scores.sort(key=lambda x: x[0], reverse=True)
            all_subsets = [subset for _, subset in subset_scores[: self.limit_subsets]]

        flip_prob = np.zeros(
            n_samples, dtype=np.float64
        )  # flip probability of the chosen subset
        best_subset: list[tuple[int, ...]] = [tuple() for _ in range(n_samples)]
        best_label = np.full(n_samples, -1, dtype=int)

        # Cache assignment grids by subset size (reused across subsets)
        _assign_cache: dict[int, np.ndarray] = {}

        def assignments(size: int) -> np.ndarray:
            if size not in _assign_cache:
                _assign_cache[size] = np.array(
                    list(itertools.product([0.0, 1.0], repeat=size)), dtype=np.float64
                )
            return _assign_cache[size]

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
            base_Z_f64 = base_Z.astype(np.float64)
            base_logit = base_Z_f64 @ _fast_w + _fast_b  # (N,)

        _draws: list[np.ndarray] = []

        def draws() -> np.ndarray:
            """Uniform draws shared by every subset: sample t sets concept j to 1 for row i if draws[t, j] < P[i, j]."""
            if not _draws:
                seed = getattr(config, "random_state", None)
                _draws.append(
                    np.random.default_rng(0 if seed is None else seed).random(
                        (self.n_samples, n_concepts)
                    )
                )
            return _draws[0]

        def is_sampled(size: int) -> bool:
            return 2**size > self.n_samples

        def subset_mass(
            rows: np.ndarray, subset: tuple[int, ...], sampled: bool = False
        ) -> tuple[np.ndarray, np.ndarray]:
            """Flip probability of intervening on `subset`, and the likelier new label, for the given rows.

            Exact (all value combinations, weighted by their probability) unless `sampled`.
            """
            subset_arr = np.asarray(subset, dtype=int)
            if sampled:
                A = self.n_samples
            else:
                assign = assignments(len(subset_arr))  # (A, ss)
                A = int(assign.shape[0])
            mass = np.empty(len(rows), dtype=np.float64)
            lbl_star = np.empty(len(rows), dtype=int)
            if _fast_w is not None:
                # Fast path: exploit logistic regression linearity.
                # logit = Z @ w + b, label = (logit >= 0). For a subset S with assignment a:
                #   logit_new = (base_logit - base_Z[:,S] @ w[S]) + a @ w[S]
                w_sub = _fast_w[subset_arr]  # (ss,)
                step = max(1, (1 << 21) // (A * len(subset_arr)))
            else:
                # General path: call predict_proba on expanded arrays, chunked to stay within batch_size.
                step = max(1, max(1, self.batch_size) // A)
            for s in range(0, len(rows), step):
                r = rows[s : s + step]
                m = len(r)
                pS = P[r][:, subset_arr]  # (m, ss)
                if sampled:
                    values = (draws()[None, :, subset_arr] < pS[:, None, :]).astype(
                        np.float64
                    )  # (m, A, ss)
                    w_assign = np.full((m, A), 1.0 / A)
                else:
                    values = np.broadcast_to(
                        assign[None, :, :], (m, A, len(subset_arr))
                    )
                    # Probability weights: P(assignment | concept probs)
                    w_assign = np.prod(
                        np.where(
                            assign[None, :, :] == 1.0,
                            pS[:, None, :],
                            1.0 - pS[:, None, :],
                        ),
                        axis=2,
                    )  # (m, A)
                if _fast_w is not None:
                    remaining = (
                        base_logit[r] - base_Z_f64[r][:, subset_arr] @ w_sub
                    )  # (m,)
                    all_logit = remaining[:, None] + values @ w_sub  # (m, A)
                    flip_mask = (all_logit > 0).astype(int) != base_lbl[r][
                        :, None
                    ]  # (m, A)
                    weighted = w_assign * flip_mask
                    all_prob1 = 1.0 / (1.0 + np.exp(-all_logit))  # (m, A)
                    cls1 = (all_prob1 * weighted).sum(axis=1)
                    cls0 = ((1.0 - all_prob1) * weighted).sum(axis=1)
                    lbl_star[s : s + m] = (cls1 >= cls0).astype(int)
                else:
                    base_chunk = base_cont[r] if supports_aligned else base_Z[r]
                    Z_chunk = np.repeat(base_chunk, A, axis=0)  # (m*A, C)
                    Z_chunk[:, subset_arr] = values.reshape(
                        m * A, len(subset_arr)
                    ).astype(np.float32)
                    repeated_rows = (
                        np.repeat(source_rows[r], A) if supports_aligned else None
                    )
                    repeated_baseline = (
                        np.repeat(base_cont[r], A, axis=0) if supports_aligned else None
                    )
                    intervention_mask = None
                    if supports_aligned:
                        intervention_mask = np.zeros_like(Z_chunk, dtype=bool)
                        intervention_mask[:, subset_arr] = True
                    Y = predict_label_proba_from_concepts(
                        model,
                        Z_chunk,
                        row_indices=repeated_rows,
                        baseline_concepts=repeated_baseline,
                        intervention_mask=intervention_mask,
                    )  # (m*A, j)
                    flip_mask = (
                        Y.argmax(axis=1).reshape(m, A) != base_lbl[r][:, None]
                    )  # (m, A)
                    weighted = w_assign * flip_mask
                    cls_mass = (Y.reshape(m, A, n_classes) * weighted[:, :, None]).sum(
                        axis=1
                    )
                    lbl_star[s : s + m] = cls_mass.argmax(axis=1)
                mass[s : s + m] = weighted.sum(axis=1)
            return mass, lbl_star

        every_row = np.arange(n_samples)
        for subset in tqdm(
            all_subsets
        ):  # smaller subsets first, so ties go to the smaller subset
            # a subset with more value combinations than `n_samples` (over 12 concepts) is estimated from the draws
            mass, lbl_star = subset_mass(
                every_row, subset, sampled=is_sampled(len(subset))
            )
            improve = mass > flip_prob
            if np.any(improve):
                flip_prob[improve] = mass[improve]
                best_label[improve] = lbl_star[improve]
                for idx in np.nonzero(improve)[0]:
                    best_subset[int(idx)] = tuple(int(x) for x in subset)

        # greedy growth from the best single concept: add the concept with the largest gain while it is positive
        active = (
            [i for i in range(n_samples) if len(best_subset[i]) == 1 and k > 1]
            if is_greedy
            else []
        )
        while active:
            groups: dict[tuple[int, ...], list[int]] = {}
            for i in active:
                groups.setdefault(best_subset[i], []).append(i)
            active = []
            for subset, members in groups.items():
                rows = np.asarray(members, dtype=int)
                sampled = is_sampled(len(subset) + 1)
                # with sampling, the current subset is re-estimated from the same draws as its extensions
                current = (
                    subset_mass(rows, subset, sampled=True)[0]
                    if sampled
                    else flip_prob[rows]
                )
                grown_mass = np.full(len(rows), -np.inf)
                grown_label = np.full(len(rows), -1, dtype=int)
                grown_concept = np.full(len(rows), -1, dtype=int)
                for j in range(n_concepts):
                    if j in subset:
                        continue
                    mass, lbl_star = subset_mass(
                        rows, tuple(sorted(subset + (j,))), sampled=sampled
                    )
                    better = mass > grown_mass
                    grown_mass[better] = mass[better]
                    grown_label[better] = lbl_star[better]
                    grown_concept[better] = j
                accept = (grown_concept >= 0) & (grown_mass > current)
                for pos in np.nonzero(accept)[0]:
                    i = int(rows[pos])
                    best_subset[i] = tuple(sorted(subset + (int(grown_concept[pos]),)))
                    flip_prob[i] = grown_mass[pos]
                    best_label[i] = grown_label[pos]
                    if len(best_subset[i]) < k:
                        active.append(i)

        # candidate instances: exceed threshold (+ optional abstention filter)
        y_prob_now = base_probs
        pred_now = base_lbl
        conf = y_prob_now[np.arange(n_samples), pred_now]
        # a row without a subset (no concept can flip it) is never a candidate, even at threshold 0
        has_subset = np.array([len(subset) > 0 for subset in best_subset], dtype=bool)
        candidate = has_subset & (flip_prob >= threshold)
        if config.select_only_abstained and config.abstention_threshold is not None:
            abstain_mask = (conf >= config.abstention_threshold) & (
                conf <= 1.0 - config.abstention_threshold
            )
            candidate &= abstain_mask
        candidate_ids = np.nonzero(candidate)[0]

        selected = self._select_instances(candidate_ids, config, rng=config.rng)

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
