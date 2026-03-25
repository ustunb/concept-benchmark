"""Utilities for running concept interventions at test time.

This module provides a small framework to experiment with different
intervention strategies on concept bottleneck style models.  The overall
workflow is:

1.  Build an :class:`InterventionBatch` that contains the concept
    predictions, ground-truth concepts, and (optionally) labels.
2.  Pick an :class:`InterventionStrategy` implementation and configure it via
    :class:`InterventionConfig`.
3.  Use :class:`ConceptInterventionRunner` to (optionally) prep the strategy on
    a validation batch and then apply it to a test batch.  The runner takes
    care of re-evaluating the downstream model after concepts are overwritten
    and returns an :class:`InterventionResult` with all relevant artefacts.

The initial set of strategies implemented mirrors common approaches in the
concept bottleneck literature:

* Conceptual safeguards (only intervene on abstentions, overwriting concepts
  with ground truth).
* Ordered interventions using the ranking from the original concept bottleneck
  paper ("method 1").
* Random interventions that can act globally or per-instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import inspect
import math
from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
from tqdm import tqdm

from concept_benchmark.data import ConceptDatasetSample
from experiments.models import ConceptBasedModel


class InterventionError(RuntimeError):
    """Raised when a strategy or configuration cannot be executed."""


def predict_label_proba_from_concepts(
    model: ConceptBasedModel,
    concepts: np.ndarray,
    *,
    dataset: ConceptDatasetSample | None = None,
    row_indices: np.ndarray | None = None,
    source_indices: np.ndarray | None = None,
    baseline_concepts: np.ndarray | None = None,
    intervention_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predict label probabilities from concept values with optional model override."""

    replay_target = None
    if hasattr(model, "predict_proba_from_concepts"):
        replay_target = model.predict_proba_from_concepts
    elif hasattr(getattr(model, "label_predictor", None), "predict_proba_from_concepts"):
        replay_target = model.label_predictor.predict_proba_from_concepts

    if replay_target is not None:
        replay_kwargs = {
            "dataset": dataset,
            "row_indices": row_indices,
            "source_indices": source_indices,
            "baseline_concepts": baseline_concepts,
            "intervention_mask": intervention_mask,
        }
        replay_kwargs = {k: v for k, v in replay_kwargs.items() if v is not None}
        sig = inspect.signature(replay_target)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        ):
            return replay_target(concepts, **replay_kwargs)
        supported = {
            key: value for key, value in replay_kwargs.items() if key in sig.parameters
        }
        return replay_target(concepts, **supported)
    return model.label_predictor.predict_proba(concepts)


def _supports_aligned_concept_replay(model: ConceptBasedModel) -> bool:
    if getattr(model, "supports_aligned_concept_replay", False):
        return True
    return bool(
        getattr(getattr(model, "label_predictor", None), "supports_aligned_concept_replay", False)
    )


@dataclass
class InterventionBatch:
    """Container for a batch of concept predictions and associated metadata."""

    C_pred: np.ndarray
    C_true: np.ndarray
    y_true: np.ndarray | None = None
    instance_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.C_pred.shape != self.C_true.shape:
            raise ValueError(
                "C_pred and C_true must have identical shape, got"
                f" {self.C_pred.shape} and {self.C_true.shape}."
            )
        if self.instance_ids is None:
            self.instance_ids = np.arange(self.C_pred.shape[0])
        if self.instance_ids.shape[0] != self.C_pred.shape[0]:
            raise ValueError("Length of instance_ids must match the number of samples.")

    @property
    def n_samples(self) -> int:
        return self.C_pred.shape[0]

    @property
    def n_concepts(self) -> int:
        return self.C_pred.shape[1]


@dataclass
class InterventionConfig:
    """Configuration shared by all strategies.

    Different strategies use different subsets of these fields:

    * **KFlipInterventionStrategy** — ``max_concepts_per_instance`` (the *k*
      budget) and ``score_threshold`` (minimum flip probability).
    * **ConceptualSafeguardsStrategy** — ``abstention_threshold`` (abstention margin),
      ``concept_order``, and budget fields.
    * **OrderedCBMStrategy** — ``concept_order`` (or learns one via
      ``prepare``), budget fields, and ``abstention_threshold`` when
      ``select_only_abstained`` is set.
    * **RandomInterventionStrategy** — budget fields and
      ``use_per_instance_ordering``.

    Attributes:
        abstention_threshold: Confidence margin used by conceptual safeguards
            to detect abstentions.  Interventions trigger when the predicted
            class confidence lies within
            ``[abstention_threshold, 1 - abstention_threshold]``.
        concept_budget: Maximum number of concept corrections allowed across
            the entire batch.  Integers are treated as counts; floats in
            ``(0, 1]`` are interpreted as fractions of all possible
            interventions (``n_samples * n_concepts``).
        instance_budget: Maximum number of instances that may receive at least
            one correction.  Same semantics as ``concept_budget`` but with
            ``n_samples`` as the reference population.
        max_concepts_per_instance: Optional cap on the number of concepts that
            may be corrected within a single instance.
        random_state: Seed controlling reproducibility for strategies that rely
            on stochastic choices.
        concept_order: Optional precomputed concept ordering.  If supplied,
            strategies that support custom orderings will prefer this over any
            internally generated ranking.
        shuffle_candidates: If ``True``, candidate instances are shuffled before
            applying budgets.  Useful when ``instance_budget`` is smaller than
            the number of eligible samples and sampling should be random rather
            than deterministic.
        select_only_abstained: When ``True``, strategies should restrict
            themselves to instances that the current downstream model flags as
            abstentions (based on ``abstention_threshold``).  Strategies may
            choose to ignore this flag when abstention logic is not applicable.
        use_per_instance_ordering: Controls whether strategies should generate
            independent concept orders per instance (``True``) or apply a single
            shared order across the batch (``False``).
        allow_repeated: Permit multiple interventions on the same concept within
            repeated calls.  The current implementations ignore this flag but it
            is kept for forward compatibility.
        score_threshold: Minimum score required for an instance to be considered
            for intervention.  Used by KFlip (flip probability) and
            ScoreIntervention.
        intervention_noise_rate: Fraction of concept ground-truth values to
            flip before overwriting, simulating noisy interventions.
    """

    abstention_threshold: float | None = None
    concept_budget: float | None = None
    instance_budget: float | None = None
    max_concepts_per_instance: int | None = None
    random_state: int | None = None
    concept_order: Sequence[int] | None = None
    shuffle_candidates: bool = False
    select_only_abstained: bool = False
    use_per_instance_ordering: bool = True
    allow_repeated: bool = False
    score_threshold: float = 0.2
    intervention_noise_rate: float = 0.0
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.abstention_threshold is not None and not (
            0.0 <= self.abstention_threshold <= 0.5
        ):
            raise ValueError("abstention_threshold must lie within [0, 0.5].")
        if not (0.0 <= self.score_threshold <= 1.0):
            raise ValueError("score_threshold must lie within [0, 1].")
        self._rng = np.random.default_rng(self.random_state)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def resolve_concept_budget(self, total: int) -> int:
        return self._resolve_budget(self.concept_budget, total)

    def resolve_instance_budget(self, total: int) -> int:
        return self._resolve_budget(self.instance_budget, total)

    def per_instance_limit(self, n_concepts: int) -> int:
        if self.max_concepts_per_instance is None:
            return n_concepts
        return max(0, min(n_concepts, int(self.max_concepts_per_instance)))

    @staticmethod
    def _resolve_budget(budget: float | None, total: int) -> int:
        if budget is None:
            return total
        if isinstance(budget, float):
            if not (0.0 <= budget <= 1.0):
                raise ValueError("Fractional budgets must lie within [0, 1].")
            return min(total, max(0, math.ceil(budget * total)))
        count = int(budget)
        if count < 0:
            raise ValueError("Budgets must be non-negative.")
        return min(total, count)


@dataclass
class StrategyProposal:
    """Mask proposed by a strategy alongside optional metadata."""

    mask: np.ndarray
    ordering_used: np.ndarray | None = None
    selected_instances: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionResult:
    """Outcome of running an intervention strategy on a batch."""

    C_pred: np.ndarray
    C_intervened: np.ndarray
    mask: np.ndarray
    y_prob_before: np.ndarray
    y_prob_after: np.ndarray
    y_pred_after: np.ndarray
    proposal: StrategyProposal
    strategy_metrics: dict[str, Any]


class InterventionStrategy:
    """Base class for all intervention strategies."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._state: dict[str, Any] = {}

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def prepare(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> None:
        """Optional hook executed on a validation batch prior to inference."""
        self._state.clear()

    def propose(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> StrategyProposal:
        raise NotImplementedError

    @staticmethod
    def _select_instances(
        candidates: np.ndarray,
        config: InterventionConfig,
        *,
        rng: np.random.Generator,
        shuffle_override: bool | None = None,
    ) -> np.ndarray:
        if candidates.size == 0:
            return candidates
        shuffle = (
            config.shuffle_candidates if shuffle_override is None else shuffle_override
        )
        if shuffle:
            shuffled = candidates.copy()
            rng.shuffle(shuffled)
            candidates = shuffled
        limit = config.resolve_instance_budget(candidates.size)
        return candidates[:limit]

    @staticmethod
    def _apply_ordering(
        mask: np.ndarray,
        order: Sequence[int],
        instances: Sequence[int],
        *,
        config: InterventionConfig,
        start_total: int = 0,
    ) -> int:
        n_concepts = mask.shape[1]
        per_instance_limit = config.per_instance_limit(n_concepts)
        total_limit = config.resolve_concept_budget(mask.size)
        total_applied = start_total

        for idx in instances:
            applied_here = 0
            for concept in order:
                if concept < 0 or concept >= n_concepts:
                    continue
                if mask[idx, concept]:
                    continue
                if applied_here >= per_instance_limit:
                    break
                if total_applied >= total_limit:
                    return total_applied
                mask[idx, concept] = True
                applied_here += 1
                total_applied += 1
        return total_applied


class ConceptualSafeguardsStrategy(InterventionStrategy):
    """Intervene on abstaining predictions by overwriting selected concepts."""

    def __init__(self) -> None:
        super().__init__(name="conceptual_safeguards")

    @staticmethod
    def _instance_uncertainty_scores(C_pred: np.ndarray) -> np.ndarray:
        distances = np.abs(C_pred - 0.5)
        uncertainties = 0.5 - distances
        scores = np.nanmean(uncertainties, axis=1)
        return np.where(np.isnan(scores), -np.inf, scores)

    def _select_instances_by_uncertainty(
        self,
        candidate_ids: np.ndarray,
        scores: np.ndarray,
        *,
        config: InterventionConfig,
    ) -> np.ndarray:
        if candidate_ids.size == 0:
            return candidate_ids
        if config.shuffle_candidates:
            tie_breaker = config.rng.random(candidate_ids.size)
            order = np.lexsort((tie_breaker, -scores))
        else:
            order = np.argsort(-scores)
        ranked = candidate_ids[order]
        limit = config.resolve_instance_budget(ranked.size)
        return ranked[:limit]

    def propose(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> StrategyProposal:
        if config.abstention_threshold is None:
            raise InterventionError(
                "Conceptual safeguards require abstention_threshold to be specified in the config."
            )

        supports_aligned = _supports_aligned_concept_replay(model)
        if supports_aligned:
            y_prob = predict_label_proba_from_concepts(
                model,
                batch.C_pred,
                row_indices=np.arange(batch.n_samples, dtype=int),
                baseline_concepts=batch.C_pred,
            )
        elif hasattr(model, "predict_proba_from_concepts"):
            y_prob = model.predict_proba_from_concepts(batch.C_pred)
        else:
            y_prob = model._propagate_predict_proba_mc(batch.C_pred)
        predicted = np.argmax(y_prob, axis=1)
        confidences = y_prob[np.arange(batch.n_samples), predicted]
        abstain_mask = (confidences >= config.abstention_threshold) & (
            confidences <= 1.0 - config.abstention_threshold
        )

        non_abstained = ~abstain_mask
        if batch.y_true is not None:
            selective_acc_before = (
                float((predicted[non_abstained] == batch.y_true[non_abstained]).mean())
                if non_abstained.any()
                else float("nan")
            )
        else:
            selective_acc_before = float("nan")

        candidate_ids = np.nonzero(abstain_mask)[0]
        candidate_scores = self._instance_uncertainty_scores(
            batch.C_pred[candidate_ids]
        )
        selected = self._select_instances_by_uncertainty(
            candidate_ids,
            candidate_scores,
            config=config,
        )

        order = (
            np.asarray(config.concept_order)
            if config.concept_order is not None
            else np.arange(batch.n_concepts)
        )
        mask = np.zeros_like(batch.C_pred, dtype=bool)
        self._apply_ordering(mask, order, selected, config=config)
        return StrategyProposal(
            mask=mask,
            ordering_used=order,
            selected_instances=selected,
            details={
                "selective_acc_before": selective_acc_before,
                "candidate_uncertainty_scores": candidate_scores,
                "candidate_ids": candidate_ids,
            },
        )


class OrderedCBMStrategy(InterventionStrategy):
    """Reproduce the ordered intervention strategy from the original CBM paper."""

    def __init__(self) -> None:
        super().__init__(name="ordered_cbm")

    def prepare(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> None:
        super().prepare(model, batch, config)
        if batch.y_true is None:
            raise InterventionError(
                "Ordered CBM strategy requires ground-truth labels in the validation batch."
            )

        n_samples, n_concepts = batch.C_pred.shape
        supports_aligned = _supports_aligned_concept_replay(model)
        row_indices = np.arange(n_samples, dtype=int)
        y_prob_orig = predict_label_proba_from_concepts(
            model,
            batch.C_pred,
            row_indices=row_indices if supports_aligned else None,
            baseline_concepts=batch.C_pred if supports_aligned else None,
        )
        y_pred_orig = np.argmax(y_prob_orig, axis=1)
        y_error_orig = (y_pred_orig != batch.y_true).astype(int)

        error_deltas = np.zeros((n_samples, n_concepts))
        for concept_idx in range(n_concepts):
            C_intervened = batch.C_pred.copy()
            C_intervened[:, concept_idx] = batch.C_true[:, concept_idx]
            intervention_mask = None
            if supports_aligned:
                intervention_mask = np.zeros_like(batch.C_pred, dtype=bool)
                intervention_mask[:, concept_idx] = True
            y_prob_tti = predict_label_proba_from_concepts(
                model,
                C_intervened,
                row_indices=row_indices if supports_aligned else None,
                baseline_concepts=batch.C_pred if supports_aligned else None,
                intervention_mask=intervention_mask,
            )
            y_pred_tti = np.argmax(y_prob_tti, axis=1)
            y_error_tti = (y_pred_tti != batch.y_true).astype(int)
            error_deltas[:, concept_idx] = y_error_tti - y_error_orig

        best_concepts = np.argmin(error_deltas, axis=1)
        counts = np.bincount(best_concepts, minlength=n_concepts)
        ordering = np.argsort(-counts)
        self.state["ordering"] = ordering
        self.state["error_deltas"] = error_deltas

    def propose(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> StrategyProposal:
        if config.concept_order is not None:
            order = np.asarray(config.concept_order)
        else:
            if "ordering" not in self.state:
                raise InterventionError(
                    "Ordered CBM strategy requires prepare(...) to be called first or"
                    " an explicit concept_order in the config."
                )
            order = np.asarray(self.state["ordering"])

        supports_aligned = _supports_aligned_concept_replay(model)
        if config.select_only_abstained and config.abstention_threshold is not None:
            y_prob = predict_label_proba_from_concepts(
                model,
                batch.C_pred,
                row_indices=np.arange(batch.n_samples, dtype=int)
                if supports_aligned
                else None,
                baseline_concepts=batch.C_pred if supports_aligned else None,
            )
            predicted = np.argmax(y_prob, axis=1)
            confidences = y_prob[np.arange(batch.n_samples), predicted]
            abstain_mask = (confidences >= config.abstention_threshold) & (
                confidences <= 1.0 - config.abstention_threshold
            )
            candidate_ids = np.nonzero(abstain_mask)[0]
        else:
            candidate_ids = np.arange(batch.n_samples)

        selected = self._select_instances(
            candidate_ids,
            config,
            rng=config.rng,
        )
        mask = np.zeros_like(batch.C_pred, dtype=bool)
        self._apply_ordering(mask, order, selected, config=config)
        details: dict[str, Any] = {}
        if "error_deltas" in self.state:
            details["validation_error_deltas"] = self.state["error_deltas"]
        return StrategyProposal(
            mask=mask, ordering_used=order, selected_instances=selected, details=details
        )


class RandomInterventionStrategy(InterventionStrategy):
    """Select concepts to intervene on at random."""

    def __init__(self) -> None:
        super().__init__(name="random_intervention")

    def propose(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> StrategyProposal:
        supports_aligned = _supports_aligned_concept_replay(model)
        if config.select_only_abstained and config.abstention_threshold is not None:
            y_prob = predict_label_proba_from_concepts(
                model,
                batch.C_pred,
                row_indices=np.arange(batch.n_samples, dtype=int)
                if supports_aligned
                else None,
                baseline_concepts=batch.C_pred if supports_aligned else None,
            )
            predicted = np.argmax(y_prob, axis=1)
            confidences = y_prob[np.arange(batch.n_samples), predicted]
            abstain_mask = (confidences >= config.abstention_threshold) & (
                confidences <= 1.0 - config.abstention_threshold
            )
            candidate_ids = np.nonzero(abstain_mask)[0]
        else:
            candidate_ids = np.arange(batch.n_samples)

        selected = self._select_instances(
            candidate_ids,
            config,
            rng=config.rng,
            shuffle_override=True,
        )

        mask = np.zeros_like(batch.C_pred, dtype=bool)
        n_concepts = batch.n_concepts
        total_limit = config.resolve_concept_budget(mask.size)
        total_applied = 0

        if config.use_per_instance_ordering:
            for idx in selected:
                order = config.rng.permutation(n_concepts)
                total_applied = self._apply_ordering(
                    mask,
                    order,
                    [idx],
                    config=config,
                    start_total=total_applied,
                )
                if total_applied >= total_limit:
                    break
            ordering_used = None
        else:
            order = config.rng.permutation(n_concepts)
            total_applied = self._apply_ordering(
                mask,
                order,
                selected,
                config=config,
                start_total=total_applied,
            )
            ordering_used = order

        return StrategyProposal(
            mask=mask, ordering_used=ordering_used, selected_instances=selected
        )


class ScoreIntervention(InterventionStrategy):
    """Intervene on fixed-size concept subsets exceeding a score threshold."""

    def __init__(self) -> None:
        super().__init__(name="score_intervention")

    def propose(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> StrategyProposal:
        n_samples, n_concepts = batch.C_pred.shape
        mask = np.zeros_like(batch.C_pred, dtype=bool)
        supports_aligned = _supports_aligned_concept_replay(model)

        concepts_before = (
            batch.C_pred if supports_aligned else (batch.C_pred >= 0.5).astype(int)
        )
        p_before = predict_label_proba_from_concepts(
            model,
            concepts_before,
            row_indices=np.arange(n_samples, dtype=int) if supports_aligned else None,
            baseline_concepts=batch.C_pred if supports_aligned else None,
        )
        y_pred_before = np.argmax(p_before, axis=1)

        scores = np.zeros(n_samples, dtype=float)
        selected = np.array([], dtype=int)

        m = config.max_concepts_per_instance
        threshold = config.score_threshold

        if (m is None) or (m > n_concepts) or (m <= 0):
            warnings.warn(
                "max_concepts_per_instance is None, <= 0, or larger than the number of concepts; defaulting to all concepts.",
                RuntimeWarning,
                stacklevel=2,
            )
            m = n_concepts

        combination_arrays = [
            np.asarray(combo, dtype=int)
            for combo in itertools.combinations(range(n_concepts), m)
        ]

        best_subset_arrays: list[np.ndarray | None] = [None] * n_samples

        # Vectorized: iterate over combos (C(n,m)), batch all N samples per call.
        # Reduces from N × C(n,m) to C(n,m) predict_proba calls.
        best_combo_idx = np.full(n_samples, -1, dtype=int)

        for ci, combo_indices in enumerate(
            tqdm(combination_arrays, desc="Computing intervention scores")
        ):
            all_flipped = concepts_before.copy()
            all_flipped[:, combo_indices] = 1 - all_flipped[:, combo_indices]
            intervention_mask = None
            if supports_aligned:
                intervention_mask = np.zeros_like(all_flipped, dtype=bool)
                intervention_mask[:, combo_indices] = True
            p_after = predict_label_proba_from_concepts(
                model,
                all_flipped,
                row_indices=np.arange(n_samples, dtype=int)
                if supports_aligned
                else None,
                baseline_concepts=batch.C_pred if supports_aligned else None,
                intervention_mask=intervention_mask,
            )
            scores_combo = np.max(np.abs(p_after - p_before), axis=1)
            improve = scores_combo > scores
            scores[improve] = scores_combo[improve]
            best_combo_idx[improve] = ci

        for idx in range(n_samples):
            ci = best_combo_idx[idx]
            if ci >= 0:
                best_subset_arrays[idx] = combination_arrays[ci]

        selected_indices: list[int] = []
        for idx, subset_array in enumerate(best_subset_arrays):
            if subset_array is None:
                continue
            if scores[idx] > threshold:
                mask[idx, subset_array] = True
                selected_indices.append(idx)

        if selected_indices:
            selected = np.asarray(selected_indices, dtype=int)
        else:
            selected = np.array([], dtype=int)

        details = self._build_details(
            model=model,
            mask=mask,
            batch=batch,
            p_before=p_before,
            y_pred_before=y_pred_before,
            scores=scores,
            selected=selected,
            threshold=threshold,
            m=m,
        )

        return StrategyProposal(
            mask=mask,
            ordering_used=None,
            selected_instances=selected,
            details=details,
        )

    @staticmethod
    def _build_details(
        *,
        model: ConceptBasedModel,
        mask: np.ndarray,
        batch: InterventionBatch,
        p_before: np.ndarray,
        y_pred_before: np.ndarray,
        scores: np.ndarray,
        selected: np.ndarray,
        threshold: float,
        m: int,
    ) -> dict[str, Any]:
        supports_aligned = _supports_aligned_concept_replay(model)
        overall_acc_before: float | None = None
        acc_non_intervened_before: float | None = None
        confusion_selected_before: np.ndarray | None = None
        confusion_selected_after: np.ndarray | None = None
        confusion_non_selected_before: np.ndarray | None = None
        confusion_non_selected_after: np.ndarray | None = None

        def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            n_classes = p_before.shape[1]
            indices = y_true.astype(int) * n_classes + y_pred.astype(int)
            return np.bincount(indices, minlength=n_classes * n_classes).reshape(
                n_classes, n_classes
            )

        if batch.y_true is not None:
            overall_acc_before = float(np.mean(y_pred_before == batch.y_true))
            non_selected = np.setdiff1d(
                np.arange(batch.n_samples), selected, assume_unique=True
            )
            if non_selected.size:
                acc_non_intervened_before = float(
                    np.mean(y_pred_before[non_selected] == batch.y_true[non_selected])
                )
            else:
                acc_non_intervened_before = math.nan

            if selected.size:
                confusion_selected_before = compute_confusion(
                    batch.y_true[selected], y_pred_before[selected]
                )
            else:
                confusion_selected_before = np.zeros(
                    (p_before.shape[1], p_before.shape[1]), dtype=int
                )

            if non_selected.size:
                confusion_non_selected_before = compute_confusion(
                    batch.y_true[non_selected], y_pred_before[non_selected]
                )
            else:
                confusion_non_selected_before = np.zeros(
                    (p_before.shape[1], p_before.shape[1]), dtype=int
                )

        overwrite_mask = mask & ~np.isnan(batch.C_true)
        C_intervened = np.where(overwrite_mask, batch.C_true, batch.C_pred)
        concepts_after = (
            C_intervened if supports_aligned else (C_intervened >= 0.5).astype(int)
        )
        p_after = predict_label_proba_from_concepts(
            model,
            concepts_after,
            row_indices=np.arange(batch.n_samples, dtype=int) if supports_aligned else None,
            baseline_concepts=batch.C_pred if supports_aligned else None,
            intervention_mask=overwrite_mask if supports_aligned else None,
        )
        y_pred_after = np.argmax(p_after, axis=1)

        overall_acc_after: float | None = None
        if batch.y_true is not None:
            overall_acc_after = float(np.mean(y_pred_after == batch.y_true))
            if selected.size:
                confusion_selected_after = compute_confusion(
                    batch.y_true[selected], y_pred_after[selected]
                )
            else:
                confusion_selected_after = np.zeros(
                    (p_before.shape[1], p_before.shape[1]), dtype=int
                )

            non_selected = np.setdiff1d(
                np.arange(batch.n_samples), selected, assume_unique=True
            )
            if non_selected.size:
                confusion_non_selected_after = compute_confusion(
                    batch.y_true[non_selected], y_pred_after[non_selected]
                )
            else:
                confusion_non_selected_after = np.zeros(
                    (p_before.shape[1], p_before.shape[1]), dtype=int
                )

        return {
            "scores": scores,
            "threshold": threshold,
            "m": m,
            "overall_acc_before": overall_acc_before,
            "acc_non_intervened_before": acc_non_intervened_before,
            "overall_acc_after": overall_acc_after,
            "confusion_selected_before": confusion_selected_before,
            "confusion_selected_after": confusion_selected_after,
            "confusion_non_selected_before": confusion_non_selected_before,
            "confusion_non_selected_after": confusion_non_selected_after,
        }


class ConceptInterventionRunner:
    """Utility that coordinates intervention preparation and execution."""

    def __init__(self, model: ConceptBasedModel) -> None:
        self.model = model

    def prepare(
        self,
        strategy: InterventionStrategy,
        config: InterventionConfig,
        validation_dataset: ConceptDatasetSample,
        *,
        concept_proba: np.ndarray | None = None,
        concept_true: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        instance_ids: np.ndarray | None = None,
    ) -> None:
        batch = self._build_batch(
            dataset=validation_dataset,
            concept_proba=concept_proba,
            concept_true=concept_true,
            labels=labels,
            instance_ids=instance_ids,
        )
        strategy.prepare(self.model, batch, config)

    def run(
        self,
        strategy: InterventionStrategy,
        config: InterventionConfig,
        dataset: ConceptDatasetSample,
        *,
        concept_proba: np.ndarray | None = None,
        concept_true: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        instance_ids: np.ndarray | None = None,
    ) -> InterventionResult:

        # NOTE: config.intervention_noise_rate is not consumed here — intervention
        # noise is applied post-hoc by the pipeline (see _test_interventions in
        # robot_pipeline.py).
        if concept_true is None:
            concept_true = dataset.base_concepts

        batch = self._build_batch(
            dataset=dataset,
            concept_proba=concept_proba,
            concept_true=concept_true,
            labels=labels,
            instance_ids=instance_ids,
        )

        # Propose interventions based on the strategy.
        proposal = strategy.propose(self.model, batch, config)
        if proposal.mask.shape != batch.C_pred.shape:
            raise InterventionError(
                "Strategy returned a mask with shape"
                f" {proposal.mask.shape}, expected {batch.C_pred.shape}."
            )

        overwrite_mask = proposal.mask & ~np.isnan(batch.C_true)
        C_intervened = np.where(overwrite_mask, batch.C_true, batch.C_pred)
        supports_aligned = _supports_aligned_concept_replay(self.model)
        row_indices = np.arange(batch.n_samples, dtype=int)

        if supports_aligned:
            y_prob_before = predict_label_proba_from_concepts(
                self.model,
                batch.C_pred,
                dataset=dataset,
                row_indices=row_indices,
                baseline_concepts=batch.C_pred,
            )
            y_prob_after = predict_label_proba_from_concepts(
                self.model,
                C_intervened,
                row_indices=row_indices,
                baseline_concepts=batch.C_pred,
                intervention_mask=overwrite_mask,
            )
        elif isinstance(strategy, ConceptualSafeguardsStrategy):
            if hasattr(self.model, "predict_proba_from_concepts"):
                predict_proba_fn = self.model.predict_proba_from_concepts
            else:
                predict_proba_fn = self.model._propagate_predict_proba_mc
            y_prob_before = predict_proba_fn(batch.C_pred)
            y_prob_after = predict_proba_fn(C_intervened)
        else:
            concepts_before = batch.C_pred >= 0.5
            concepts_after = C_intervened >= 0.5
            y_prob_before = predict_label_proba_from_concepts(self.model, concepts_before)
            y_prob_after = predict_label_proba_from_concepts(self.model, concepts_after)
        y_pred_after = np.argmax(y_prob_after, axis=1)

        # Conceptual Safeguards results
        strategy_metrics = {}
        if isinstance(strategy, ConceptualSafeguardsStrategy):
            strategy_metrics["selective_acc_before"] = proposal.details.get(
                "selective_acc_before", None
            )
            strategy_metrics["coverage_before"] = (
                1 - (proposal.selected_instances.size / batch.n_samples)
                if batch.n_samples > 0
                else 0.0
            )

            abstain_post = (y_prob_after[:, 1] >= config.abstention_threshold) & (
                y_prob_after[:, 1] <= 1.0 - config.abstention_threshold
            )

            if batch.y_true is not None:
                strategy_metrics["selective_acc_after"] = (
                    (y_pred_after[~abstain_post] == batch.y_true[~abstain_post]).mean()
                    if (~abstain_post).any()
                    else -np.inf
                )
            else:
                strategy_metrics["selective_acc_after"] = float("nan")
            strategy_metrics["coverage_after"] = 1 - abstain_post.mean()

        elif isinstance(strategy, ScoreIntervention):
            strategy_metrics["overall_acc_before"] = proposal.details.get(
                "overall_acc_before", None
            )
            strategy_metrics["overall_acc_after"] = proposal.details.get(
                "overall_acc_after", None
            )
            strategy_metrics["acc_non_intervened_before"] = proposal.details.get(
                "acc_non_intervened_before", None
            )
            strategy_metrics["intervened"] = (
                proposal.selected_instances.size / batch.n_samples
                if batch.n_samples > 0
                else 0.0
            )

        return InterventionResult(
            C_pred=batch.C_pred,
            C_intervened=C_intervened,
            mask=proposal.mask,
            y_prob_before=y_prob_before,
            y_prob_after=y_prob_after,
            y_pred_after=y_pred_after,
            proposal=proposal,
            strategy_metrics=strategy_metrics,
        )

    def _build_batch(
        self,
        dataset: ConceptDatasetSample,
        *,
        concept_proba: np.ndarray | None,
        concept_true: np.ndarray | None,
        labels: np.ndarray | None,
        instance_ids: np.ndarray | None,
    ) -> InterventionBatch:
        C_pred = concept_proba
        if C_pred is None:
            if self.model.concept_detector is None:
                raise InterventionError(
                    "Cannot obtain concept predictions: model.concept_detector is None."
                )
            C_pred = self.model.concept_detector.predict_proba(dataset)

        C_true = concept_true if concept_true is not None else dataset.C
        if C_true.shape != C_pred.shape:
            raise InterventionError(
                "Concept ground truth and predictions must have the same shape."
            )
        y_true = labels if labels is not None else getattr(dataset, "y", None)
        return InterventionBatch(
            C_pred=C_pred,
            C_true=C_true,
            y_true=y_true,
            instance_ids=instance_ids,
        )


__all__ = [
    "ConceptInterventionRunner",
    "ConceptualSafeguardsStrategy",
    "InterventionBatch",
    "InterventionConfig",
    "InterventionError",
    "InterventionResult",
    "OrderedCBMStrategy",
    "ScoreIntervention",
    "RandomInterventionStrategy",
    "StrategyProposal",
]
