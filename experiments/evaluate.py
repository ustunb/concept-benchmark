"""The evaluation blocks of the benchmarks: train a model, intervene at a set of budgets, get the results table.

The tables these functions return are what the plots in ``concept_benchmark.evaluation`` read and what the
pipelines write; the pipelines are built from these functions.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from concept_benchmark.evaluation import (
    abstention_threshold,
    intervention_metrics,
    selective_at,
)
from concept_benchmark.utils import (
    determine_device,
    get_loader_config,
    set_deterministic_seed,
)
from experiments.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionConfig,
    predict_label_proba_from_concepts,
)
from experiments.kflip import KFlipInterventionStrategy
from experiments.models import (
    ConceptBasedModel,
    ConceptDetector,
    RobotConceptClassifier,
)

ROBOT_BUDGET_COLUMNS = [
    "budget",
    "accuracy",
    "predictions_intervened_on",
    "predictions_changed",
    "total_concept_confirmations",
    "total_concept_edits_made",
]


# ── Training ──────────────────────────────────────────────────────────


def train_cbm(
    train,
    validation,
    *,
    detector=None,
    input_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int,
    loader_config: dict | None = None,
    should_propagate: bool = False,
    should_calibrate: bool = False,
) -> ConceptBasedModel:
    """Train the paper's CBM: a concept detector and a logistic label predictor on the true concepts.

    The detector is the robot CNN unless `detector`, a module class or factory, is given (e.g.
    ``GroupPoolingConceptSudokuCNN``); it is built after seeding, so that the run is repeatable.
    With `should_calibrate`, each concept's probabilities are Platt-scaled on `validation` after training,
    so that the label probabilities that abstention relies on are calibrated (:func:`calibrate_concepts`).
    With `should_propagate`, the label predictor reads the detector's probabilities instead of its 0/1 calls
    (the sudoku models of the paper). `patience` 0 disables early stopping. `seed` fixes the weights, the
    batches and the model's own sampling.
    """
    set_deterministic_seed(seed)
    loader_config = {
        "device": determine_device(),
        **(loader_config or get_loader_config()),
    }
    torch.manual_seed(seed)
    model = (
        detector()
        if detector is not None
        else RobotConceptClassifier(
            num_concepts=train.n_concepts, input_size=input_size
        )
    )
    cbm = ConceptBasedModel(
        concept_detector=ConceptDetector(model=model),
        should_propagate=should_propagate,
        random_state=seed,
    )
    cbm.fit(
        train_dataset=train,
        valid_dataset=validation,
        freeze_backbone=False,
        concept_embed_params={"shuffle": False, **loader_config},
        concept_fit_params={
            "epochs": epochs,
            "lr": lr,
            "patience": patience if patience > 0 else epochs,
            **loader_config,
        },
        should_calibrate=should_calibrate,
    )
    return cbm


def calibrate_concepts(model: ConceptBasedModel, validation) -> ConceptBasedModel:
    """Platt-scale each concept's probabilities on `validation` (a saved model can be calibrated afterwards).

    Conceptual safeguards abstain on the propagated label probability, which is only as calibrated as the
    concept probabilities it is built from; an underconfident detector makes every valid sudoku board look
    uncertain. The scaling is fitted on the validation split, applied to every later prediction, and
    changes nothing about the detector's weights.
    """
    model.concept_detector.calibrate(validation)
    return model


def train_dnn(
    model,
    train,
    validation,
    *,
    device=None,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    loader_config: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    seed: int | None = None,
) -> nn.Module:
    """Train a binary classifier (one sigmoid output) with early stopping on the validation loss.

    `model` is a module, or a class or factory that builds one; with `seed`, the model is built after
    seeding so that the run is repeatable. Adam with learning rate `lr` unless an `optimizer` is given.
    Returns the model with the weights of its best epoch.
    """
    if seed is not None:
        set_deterministic_seed(seed)
        torch.manual_seed(seed)
    if not isinstance(model, nn.Module):
        model = model()
    device = device or determine_device()
    loader_config = {
        k: v for k, v in (loader_config or get_loader_config()).items() if k != "device"
    }
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = train.loader(shuffle=True, **loader_config)
    valid_loader = validation.loader(shuffle=False, **loader_config)

    best_val_loss, best_state_dict, epochs_no_improve = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for X, _, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X).squeeze(-1), y.float())
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device), y.to(device)
                val_loss_sum += criterion(model(X).squeeze(-1), y.float()).item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)
        if val_loss < best_val_loss:
            best_val_loss, best_state_dict, epochs_no_improve = (
                val_loss,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


def predict_proba_positive(model, dataset, *, device=None) -> np.ndarray:
    """P(positive) of a trained DNN (sigmoid output) or of a ConceptBasedModel on every instance of `dataset`."""
    if isinstance(model, ConceptBasedModel):
        proba = np.asarray(model.predict_proba(dataset))
        return proba[:, 1] if proba.ndim == 2 else proba
    device = device or determine_device()
    model.to(device).eval()
    loader = dataset.loader(shuffle=False, **get_loader_config())
    with torch.no_grad():
        return np.concatenate(
            [model(X.to(device)).squeeze(-1).cpu().numpy() for X, _, _ in loader]
        )


def predict_labels(model, dataset, *, device=None) -> np.ndarray:
    """0/1 labels of a trained DNN or ConceptBasedModel on every instance of `dataset`."""
    return (predict_proba_positive(model, dataset, device=device) >= 0.5).astype(int)


def coverage_at_target(
    model, validation, test, target_accuracy: float
) -> tuple[float, float]:
    """Selective accuracy and coverage on `test` of a model that abstains until it reaches `target_accuracy`.

    The abstention threshold is fitted on `validation` as in the paper; the model predicts the positive
    class above 0.5. Returns ``(nan, 0.0)`` when no threshold reaches the target.
    """
    val_prob, val_y = (
        predict_proba_positive(model, validation),
        np.asarray(validation.y).astype(int),
    )
    threshold, _ = abstention_threshold(val_y, val_prob, target_accuracy)
    if threshold is None:
        return float("nan"), 0.0
    return selective_at(
        np.asarray(test.y).astype(int),
        predict_proba_positive(model, test),
        threshold,
        0.5,
    )


# ── Decision support: interventions with the paper's policy ───────────


def encode_revealed_concepts(C, mask, low, high):
    """Set revealed concepts to `high` (present) or `low` (absent); the rest keep their predicted values."""
    C_encoded = np.asarray(C, dtype=float).copy()
    high = np.broadcast_to(np.asarray(high, dtype=float), C_encoded.shape)
    low = np.broadcast_to(np.asarray(low, dtype=float), C_encoded.shape)
    is_present = mask & (C_encoded >= 0.5)
    is_absent = mask & (C_encoded < 0.5)
    C_encoded[is_present] = high[is_present]
    C_encoded[is_absent] = low[is_absent]
    return C_encoded


def predict_after_intervention(
    cbm,
    label_predictor,
    C_after,
    C_before,
    mask,
    *,
    supports_aligned,
    encoding,
    low_values,
    high_values,
):
    """Label probabilities once the concepts in `mask` hold the intervener's answers.

    Models with their own concept replay (CEM, ProbCBM, ECBM) receive the answers directly. Otherwise
    `encoding` decides what the label predictor reads: `binary` thresholds every concept, `percentile` sets
    revealed concepts to the 5th/95th percentile of their training values (label-free CBMs), and
    `binary_revealed` sets revealed concepts to 0/1; the last two leave the other concepts continuous.
    """
    if supports_aligned:
        return predict_label_proba_from_concepts(
            cbm,
            C_after,
            row_indices=np.arange(C_after.shape[0], dtype=int),
            baseline_concepts=C_before,
            intervention_mask=mask,
        )
    if encoding == "percentile" and low_values is not None:
        return label_predictor.predict_proba(
            encode_revealed_concepts(C_after, mask, low_values, high_values)
        )
    if encoding == "binary_revealed":
        return label_predictor.predict_proba(
            encode_revealed_concepts(C_after, mask, 0.0, 1.0)
        )
    return label_predictor.predict_proba((C_after >= 0.5).astype(int))


def intervene(
    runner: ConceptInterventionRunner,
    test,
    concept_proba: np.ndarray,
    budget: int,
    *,
    rng: np.random.Generator,
    seed: int,
    intervention_accuracy: float = 1.0,
    score_threshold: float = 0.2,
    strategy: str = "up_to_k",
    encoding: str = "binary",
    low_values=None,
    high_values=None,
):
    """Correct up to `budget` concepts per instance with the paper's policy and an intervener of the given accuracy.

    KFlip asks about the concepts whose correction is most likely to change the label; the intervener answers
    every one of them, and is wrong with probability ``1 - intervention_accuracy`` (draws from `rng`, which
    keeps its state across budgets). Returns the runner's result with the answers and the new predictions.
    """
    n_concepts = concept_proba.shape[1]
    config = InterventionConfig(
        per_instance_budget=budget,
        random_state=seed,
        score_threshold=score_threshold,
        intervention_noise_rate=1.0 - intervention_accuracy,
    )
    policy = KFlipInterventionStrategy(
        # k=max corrects every concept of the rows KFlip selects, with the same threshold as k<max
        use_exact_k=(strategy == "exactly_k") or int(budget) >= n_concepts,
    )
    result = runner.run(
        strategy=policy,
        config=config,
        dataset=test,
        concept_proba=concept_proba,
        labels=test.y.astype(int),
    )
    C_true = test.C.astype(np.float32)
    C_after = result.C_intervened.copy()
    mistakes = result.mask & (rng.random(C_after.shape) < 1.0 - intervention_accuracy)
    C_after[mistakes] = 1.0 - C_true[mistakes]
    result.C_intervened = C_after
    cbm = runner.model
    label_predictor = cbm.label_predictor
    result.y_prob_after = predict_after_intervention(
        cbm,
        label_predictor,
        result.C_intervened,
        result.C_pred,
        result.mask,
        supports_aligned=bool(
            getattr(cbm, "supports_aligned_concept_replay", False)
            or getattr(label_predictor, "supports_aligned_concept_replay", False)
        ),
        encoding=encoding,
        low_values=low_values,
        high_values=high_values,
    )
    result.y_pred_after = np.argmax(result.y_prob_after, axis=1)
    return result


def intervention_table(
    cbm: ConceptBasedModel,
    test,
    budgets=(1, 3, "max"),
    *,
    seed: int,
    intervention_accuracy: float = 1.0,
    score_threshold: float = 0.2,
    strategy: str = "up_to_k",
    encoding: str = "binary",
    low_values=None,
    high_values=None,
) -> pd.DataFrame:
    """Accuracy of a CBM with 0, then each budget of corrected concepts per instance: the decision-support table.

    One row per budget (``"max"`` is every concept; budgets are at least 1) with the columns the robot
    pipeline writes: ``budget``, ``accuracy``, ``predictions_intervened_on``, ``predictions_changed``,
    ``total_concept_confirmations``, ``total_concept_edits_made``. Pass it to ``plot_intervention_curve``.
    The ``percentile`` encoding needs `low_values` and `high_values`, one per concept.
    """
    if encoding == "percentile" and (low_values is None or high_values is None):
        raise ValueError("encoding='percentile' needs low_values and high_values")
    torch.manual_seed(seed)
    concept_proba = np.asarray(cbm.concept_detector.predict_proba(test))
    accuracy_before = float((cbm.predict(test) == test.y.astype(int)).mean())
    runner = ConceptInterventionRunner(cbm)
    rng = np.random.default_rng(seed)
    rows = [{"budget": 0, "accuracy": accuracy_before}]
    for k in _budgets(budgets, test.n_concepts):
        result = intervene(
            runner,
            test,
            concept_proba,
            k,
            rng=rng,
            seed=seed,
            intervention_accuracy=intervention_accuracy,
            score_threshold=score_threshold,
            strategy=strategy,
            encoding=encoding,
            low_values=low_values,
            high_values=high_values,
        )
        rows.append(
            {
                "budget": k,
                **intervention_metrics(
                    result.mask,
                    result.C_pred,
                    result.C_intervened,
                    result.y_prob_before,
                    result.y_prob_after,
                    test.y,
                    accuracy_before,
                ),
            }
        )
    table = pd.DataFrame(rows).reindex(columns=ROBOT_BUDGET_COLUMNS).fillna(0)
    return table.astype({column: int for column in ROBOT_BUDGET_COLUMNS[2:]})


# ── Automation: selective classification with checks ─────────────────


def automation_table(
    model: ConceptBasedModel,
    validation,
    test,
    budgets=(1, 3, "max"),
    *,
    target_accuracy: float,
    seed: int,
    concept_groups: int | None = None,
) -> pd.DataFrame:
    """Coverage and the checks it costs with 0, then each budget of concept checks per instance: the automation table.

    The paper's protocol (conceptual safeguards): the model predicts a valid board above 0.5 and abstains
    on ``t <= p <= 1 - t``, with the threshold ``t`` fitted once on the validation predictions (the label
    probabilities the strategy itself works with) so that the kept predictions reach `target_accuracy`;
    a human then checks up to k concepts of each board the model abstains on, and the same gate is applied
    to the probabilities after the checks.
    One row per budget with the columns the sudoku pipeline writes (``coverage_after``,
    ``selective_accuracy_after``, ``total_concept_checks``, ...). Pass it to ``plot_automation``.
    With `concept_groups` (concepts per group, e.g. 9 for sudoku rows/columns/blocks), the checks are also
    counted per group. When no threshold reaches the target, the model cannot abstain its way to it: the
    rows then carry its raw accuracy, no checks, and coverage 0.
    """
    runner = ConceptInterventionRunner(model)
    policy = ConceptualSafeguardsStrategy()
    probe = InterventionConfig(
        abstention_threshold=0.0, per_instance_budget=0, random_state=seed
    )
    val_k0 = runner.run(
        policy, probe, validation
    )  # k=0: the probabilities before any check
    val_prob = val_k0.y_prob_after[:, 1]
    val_y = np.asarray(validation.y).astype(int)
    threshold, _ = abstention_threshold(val_y, val_prob, target_accuracy)
    y_test = np.asarray(test.y).astype(int)
    all_budgets = _budgets(budgets, test.n_concepts)
    test_k0 = runner.run(policy, probe, test)
    if threshold is None:
        raw_accuracy = float(
            ((test_k0.y_prob_after[:, 1] >= 0.5).astype(int) == y_test).mean()
        )
        return pd.DataFrame(
            [
                {
                    "budget": k,
                    "accuracy": raw_accuracy,
                    "predictions_intervened_on": 0,
                    "total_concept_checks": 0,
                    **_group_checks(
                        np.zeros((1, test.n_concepts), dtype=bool), concept_groups
                    ),
                    "total_concept_edits_made": 0,
                    "selective_accuracy_after": float("nan"),
                    "coverage_after": 0.0,
                }
                for k in [0, *all_budgets]
            ]
        )

    accuracy_0, coverage_0 = selective_at(
        y_test, test_k0.y_prob_after[:, 1], threshold, 0.5
    )
    rows = [
        {
            "budget": 0,
            "accuracy": float((test_k0.y_pred_after == y_test).mean()),
            "predictions_intervened_on": 0,
            "total_concept_checks": 0,
            **_group_checks(np.zeros_like(test_k0.mask), concept_groups),
            "total_concept_edits_made": 0,
            "selective_accuracy_after": accuracy_0,
            "coverage_after": coverage_0,
            "abstention_threshold": threshold,
        }
    ]
    for k in all_budgets:
        config = InterventionConfig(
            abstention_threshold=threshold, per_instance_budget=k, random_state=seed
        )
        result = runner.run(policy, config, test, y_prob_baseline=test_k0.y_prob_after)
        accuracy_k, coverage_k = selective_at(
            y_test, result.y_prob_after[:, 1], threshold, 0.5
        )
        edits = (result.C_pred >= 0.5) != (result.C_intervened >= 0.5)
        rows.append(
            {
                "budget": k,
                "accuracy": float((result.y_pred_after == y_test).mean()),
                "predictions_intervened_on": int(np.sum(np.any(result.mask, axis=1))),
                "total_concept_checks": int(np.sum(result.mask)),
                **_group_checks(result.mask, concept_groups),
                "total_concept_edits_made": int(np.sum(edits)),
                "selective_accuracy_after": accuracy_k,
                "coverage_after": coverage_k,
                "abstention_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def _budgets(budgets, n_concepts: int) -> list[int]:
    """Budgets as concept counts: ``"max"`` is every concept, larger numbers are capped, 0 is not a budget."""
    counts = [n_concepts if b == "max" else min(int(b), n_concepts) for b in budgets]
    if any(k < 1 for k in counts):
        raise ValueError(
            "budgets are at least 1; the table always starts with the row without interventions"
        )
    return counts


def _group_checks(mask: np.ndarray, concept_groups: int | None) -> dict[str, int]:
    """Checks per group of `concept_groups` consecutive concepts (sudoku: rows, columns, blocks)."""
    if concept_groups is None:
        return {}
    names = ("row_checks", "col_checks", "block_checks")
    return {
        name: int(mask[:, i * concept_groups : (i + 1) * concept_groups].sum())
        for i, name in enumerate(names)
    }
