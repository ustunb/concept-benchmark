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
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.models import ConceptBasedModel


class InterventionError(RuntimeError):
    """Raised when a strategy or configuration cannot be executed."""


@dataclass
class InterventionBatch:
    """Container for a batch of concept predictions and associated metadata."""

    C_pred: np.ndarray
    C_true: np.ndarray
    y_true: Optional[np.ndarray] = None
    instance_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.C_pred.shape != self.C_true.shape:
            raise ValueError(
                "C_pred and C_true must have identical shape, got"
                f" {self.C_pred.shape} and {self.C_true.shape}."
            )
        if self.instance_ids is None:
            self.instance_ids = np.arange(self.C_pred.shape[0])
        if self.instance_ids.shape[0] != self.C_pred.shape[0]:
            raise ValueError(
                "Length of instance_ids must match the number of samples."
            )

    @property
    def n_samples(self) -> int:
        return self.C_pred.shape[0]

    @property
    def n_concepts(self) -> int:
        return self.C_pred.shape[1]


@dataclass
class InterventionConfig:
    """Configuration shared by all strategies.

    Attributes:
        tau: Confidence margin used by conceptual safeguards to detect
            abstentions.  Interventions trigger when the predicted class
            confidence lies within ``[tau, 1 - tau]``.
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
            abstentions (based on ``tau``).  Strategies may choose to ignore
            this flag when abstention logic is not applicable.
        per_instance_ordering: Controls whether strategies should generate
            independent concept orders per instance (``True``) or apply a single
            shared order across the batch (``False``).
        allow_repeated: Permit multiple interventions on the same concept within
            repeated calls.  The current implementations ignore this flag but it
            is kept for forward compatibility.
    """

    tau: Optional[float] = None
    concept_budget: Optional[float] = None
    instance_budget: Optional[float] = None
    max_concepts_per_instance: Optional[int] = None
    random_state: Optional[int] = None
    concept_order: Optional[Sequence[int]] = None
    shuffle_candidates: bool = False
    select_only_abstained: bool = False
    per_instance_ordering: bool = True
    allow_repeated: bool = False
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tau is not None and not (0.0 <= self.tau <= 0.5):
            raise ValueError("tau must lie within [0, 0.5].")
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
    def _resolve_budget(budget: Optional[float], total: int) -> int:
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
    ordering_used: Optional[np.ndarray] = None
    selected_instances: Optional[np.ndarray] = None
    details: Dict[str, Any] = field(default_factory=dict)


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
    selective_acc_before: Optional[float] = None
    selective_acc_after: Optional[float] = None
    coverage_before: Optional[float] = None
    coverage_after: Optional[float] = None


class InterventionStrategy:
    """Base class for all intervention strategies."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._state: Dict[str, Any] = {}

    @property
    def state(self) -> Dict[str, Any]:
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
        shuffle_override: Optional[bool] = None,
    ) -> np.ndarray:
        if candidates.size == 0:
            return candidates
        shuffle = config.shuffle_candidates if shuffle_override is None else shuffle_override
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

    def propose(
        self,
        model: ConceptBasedModel,
        batch: InterventionBatch,
        config: InterventionConfig,
    ) -> StrategyProposal:
        if config.tau is None:
            raise InterventionError(
                "Conceptual safeguards require tau to be specified in the config."
            )

        y_prob = model._propagate_predict_proba_mc(batch.C_pred)
        predicted = np.argmax(y_prob, axis=1)
        selective_acc_before = (predicted == batch.y_true).mean()
        confidences = y_prob[np.arange(batch.n_samples), predicted]
        abstain_mask = (confidences >= config.tau) & (confidences <= 1.0 - config.tau)
        candidate_ids = np.nonzero(abstain_mask)[0]
        selected = self._select_instances(
            candidate_ids,
            config,
            rng=config.rng,
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
            details={"selective_acc_before": selective_acc_before}
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
        y_prob_orig = model.front_end_model.predict_proba(batch.C_pred)
        y_pred_orig = np.argmax(y_prob_orig, axis=1)
        y_error_orig = (y_pred_orig != batch.y_true).astype(int)

        error_deltas = np.zeros((n_samples, n_concepts))
        for concept_idx in range(n_concepts):
            C_intervened = batch.C_pred.copy()
            C_intervened[:, concept_idx] = batch.C_true[:, concept_idx]
            y_prob_tti = model.front_end_model.predict_proba(C_intervened)
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

        if config.select_only_abstained and config.tau is not None:
            y_prob = model.front_end_model.predict_proba(batch.C_pred)
            predicted = np.argmax(y_prob, axis=1)
            confidences = y_prob[np.arange(batch.n_samples), predicted]
            abstain_mask = (confidences >= config.tau) & (confidences <= 1.0 - config.tau)
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
        details: Dict[str, Any] = {}
        if "error_deltas" in self.state:
            details["validation_error_deltas"] = self.state["error_deltas"]
        return StrategyProposal(mask=mask, ordering_used=order, selected_instances=selected, details=details)


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
        if config.select_only_abstained and config.tau is not None:
            y_prob = model.front_end_model.predict_proba(batch.C_pred)
            predicted = np.argmax(y_prob, axis=1)
            confidences = y_prob[np.arange(batch.n_samples), predicted]
            abstain_mask = (confidences >= config.tau) & (confidences <= 1.0 - config.tau)
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

        if config.per_instance_ordering:
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

        return StrategyProposal(mask=mask, ordering_used=ordering_used, selected_instances=selected)


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
        concept_proba: Optional[np.ndarray] = None,
        concept_true: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None,
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
        concept_proba: Optional[np.ndarray] = None,
        concept_true: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None,
    ) -> InterventionResult:
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

        # TODO: see if we can get rid of explicit >= 0.5
        # Determine how to get predictions from the downstream model.
        # Conceptual safeguards operate on concept probabilities, others on binarized concepts.
        if isinstance(strategy, ConceptualSafeguardsStrategy):
            predict_proba_fn = self.model._propagate_predict_proba_mc
            concepts_before = batch.C_pred
        else:
            predict_proba_fn = self.model.front_end_model.predict_proba
            concepts_before = batch.C_pred >= 0.5

        # Evaluate the model before and after the intervention.
        y_prob_before = predict_proba_fn(concepts_before)
        C_intervened = np.where(proposal.mask, batch.C_true, batch.C_pred)
        concepts_after = C_intervened if isinstance(strategy, ConceptualSafeguardsStrategy) else (C_intervened >= 0.5)
        y_prob_after = predict_proba_fn(concepts_after)
        y_pred_after = np.argmax(y_prob_after, axis=1)

        # Conceptual Safeguards results
        cs_results = {}
        if isinstance(strategy, ConceptualSafeguardsStrategy):
            cs_results['selective_acc_before'] = proposal.details.get("selective_acc_before", None)
            cs_results['coverage_before'] = proposal.selected_instances.size / batch.n_samples if batch.n_samples > 0 else 0.0

            abstain_post = (y_prob_after.max(axis=1) >= config.tau) & (y_prob_after.max(axis=1) <= 1.0 - config.tau)

            cs_results['selective_acc_after'] = (y_pred_after[~abstain_post] == batch.y_true[~abstain_post]).mean() if (~abstain_post).any() else -np.inf
            cs_results['coverage_after'] = 1 - abstain_post.mean()

        return InterventionResult(
            C_pred=batch.C_pred,
            C_intervened=C_intervened,
            mask=proposal.mask,
            y_prob_before=y_prob_before,
            y_prob_after=y_prob_after,
            y_pred_after=y_pred_after,
            proposal=proposal,
            **cs_results,
        )

    def _build_batch(
        self,
        dataset: ConceptDatasetSample,
        *,
        concept_proba: Optional[np.ndarray],
        concept_true: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        instance_ids: Optional[np.ndarray],
    ) -> InterventionBatch:
        C_pred = concept_proba
        if C_pred is None:
            if self.model.concept_detector is None:
                raise InterventionError(
                    "Cannot obtain concept predictions: model.concept_detector is None."
                )
            C_pred = self.model.concept_detector.predict(dataset)

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
    "RandomInterventionStrategy",
    "StrategyProposal",
]
