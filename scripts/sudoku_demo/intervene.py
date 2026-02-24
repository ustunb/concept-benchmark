# %%
import os
import pandas as pd
import numpy as np
import torch
from argparse import ArgumentParser

from psutil import Process
from utils import (
    DEFAULT_SUDOKU_SETTINGS,
    determine_device,
    get_dataset_file,
    get_model_file,
    compute_accuracy
)

from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import data_dir, results_dir
from concept_benchmark.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionConfig,
)
from scripts.sudoku_demo.sudoku_models import SpecializedHeadSudokuCNN, SudokuValidatorCNN
from scripts.sudoku_demo.utils import AndConceptFrontEndModel

THRESH = [0.2, 0.4, 0.6, 0.8]
METRIC_COLS = [
    "accuracy",
    "predictions_intervened_on",
    "total_concept_checks",
    "total_concept_edits_made",
]
device = determine_device()
settings = DEFAULT_SUDOKU_SETTINGS.copy()

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--n-samples', dest='n_samples', type=int, default=settings['n_samples'])
    p.add_argument('--max_corrupt', type=int, default=settings['max_corrupt'])
    p.add_argument('--concept_missing', type=float, default=0)
    p.add_argument('--concept_missing_mech', type=str, choices=['none', 'mcar', 'mnar'], default='none')
    p.add_argument('--seed', type=int, default=settings['seed'])
    p.add_argument('--target-accuracy', type=float, default=0.9)
    p.add_argument('--decision-threshold', type=float, default=0.5)
    args, _ = p.parse_known_args()
    settings.update(vars(args))

loader_config = {
    'batch_size': 32,
    'num_workers': 0 if device.type == 'mps' else 12,
    'pin_memory': False if device.type == 'mps' else True
}

# load training data (for validation set)
sudoku_data_dir = get_dataset_file(data_type="image", **settings)
data = load(sudoku_data_dir / "ocr_inferred_full_dataset.pkl")
data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=settings['seed'])
data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

def _selective_accuracy_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    target_acc: float,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float | None]:
    """
    Find the largest t in [0, 0.5] such that samples with P(Y=1) in [0, t] U [1-t, 1]
    achieve accuracy >= target_acc. Returns (t, coverage), or (None, None)
    if no t satisfies the target.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)

    if y_true.shape[0] != prob_pos.shape[0]:
        raise ValueError("y_true and prob_pos must have the same length.")

    min_prob = np.minimum(prob_pos, 1.0 - prob_pos)
    candidates = np.unique(np.concatenate(([0.0], min_prob)))
    candidates = candidates[(candidates >= 0.0) & (candidates <= 0.5)]
    candidates.sort()

    for t in candidates[::-1]:
        mask = min_prob <= t
        if not np.any(mask):
            continue
        preds = (prob_pos[mask] >= decision_threshold).astype(int)
        acc = float((preds == y_true[mask]).mean())
        if acc >= target_acc:
            coverage = float(mask.mean())
            return float(t), coverage

    return None, None


def _decision_threshold_sweep(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Sweep decision thresholds to maximize accuracy.
    Returns (best_threshold, best_accuracy).
    """
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    if y_true.shape[0] != prob_pos.shape[0]:
        raise ValueError("y_true and prob_pos must have the same length.")

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101, dtype=float)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    best_t = 0.5
    best_acc = -1.0
    best_thresholds = []
    for t in thresholds:
        preds = (prob_pos >= t).astype(int)
        acc = float((preds == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_thresholds = [float(t)]
        elif acc == best_acc:
            best_thresholds.append(float(t))

    if best_thresholds:
        best_t = 0.5 * (min(best_thresholds) + max(best_thresholds))
    return best_t, best_acc


def _selective_metrics(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    t: float | None,
    decision_threshold: float = 0.5,
) -> tuple[float | None, float]:
    """
    Apply threshold t to compute selective accuracy and coverage.
    If t is None or selects no samples, returns (None, 0.0).
    """
    if t is None:
        return None, 0.0

    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float).reshape(-1)
    min_prob = np.minimum(prob_pos, 1.0 - prob_pos)
    mask = min_prob <= t
    if not np.any(mask):
        return None, 0.0

    preds = (prob_pos[mask] >= decision_threshold).astype(int)
    acc = float((preds == y_true[mask]).mean())
    coverage = float(mask.mean())
    return acc, coverage


def _dnn_val_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_y = []
    with torch.no_grad():
        for X, _, y in loader:
            X = X.to(device)
            probs = model(X).squeeze(-1).detach().cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_y)


def _cs_val_probs(model, dataset):
    probas = model.predict_proba(dataset)
    if probas.ndim == 1:
        prob_pos = probas
    else:
        prob_pos = probas[:, 1]
    y_true = np.asarray(dataset.y)
    return prob_pos, y_true


# DNN
dnn_weights = load(get_model_file(data_type="tabular", model_class="dnn", **settings))
dnn = SudokuValidatorCNN()
dnn.load_state_dict(dnn_weights)
dnn.to(device)
test_loader = data.test.loader(shuffle=False, **loader_config)
test_accuracy = compute_accuracy(dnn, test_loader, device)

# Example threshold selection on validation split
val_loader = data.validation.loader(shuffle=False, **loader_config)
target_accuracy = settings.get("target_accuracy", 0.99)
dnn_probs, dnn_y = _dnn_val_probs(dnn, val_loader, device)
decision_threshold, dnn_val_acc = _decision_threshold_sweep(dnn_y, dnn_probs)
dnn_t, dnn_cov = _selective_accuracy_threshold(
    dnn_y, dnn_probs, target_accuracy, decision_threshold
)
test_probs, test_y = _dnn_val_probs(dnn, test_loader, device)
dnn_sel_acc, dnn_sel_cov = _selective_metrics(
    test_y, test_probs, dnn_t, decision_threshold
)

# CS
cs = load(get_model_file(data_type="tabular", model_class="cs", **settings))
cs._random_state = 171  # need for reproducible mc results
# cs.front_end_model = AndConceptFrontEndModel()
cs_probs, cs_y = _cs_val_probs(cs, data.validation)
decision_threshold, cs_val_acc = _decision_threshold_sweep(cs_y, cs_probs)
cs_t, cs_cov = _selective_accuracy_threshold( cs_y, cs_probs, target_accuracy, decision_threshold
)
cs_test_probs, cs_test_y = _cs_val_probs(cs, data.test)
cs_sel_acc, cs_sel_cov = _selective_metrics(
    cs_test_y, cs_test_probs, cs_t, decision_threshold
)

# CS Intervention Budget 1
if cs_t is None:
    raise ValueError("Could not find a tau for conceptual safeguards at the target accuracy.")

cs_runner = ConceptInterventionRunner(cs)
cs_strategy = ConceptualSafeguardsStrategy()

def _run_cs_intervention(budget: int) -> dict:
    config = InterventionConfig(
        tau=cs_t,
        max_concepts_per_instance=budget,
        random_state=settings["seed"],
    )
    result = cs_runner.run(cs_strategy, config, data.test)
    acc_intervened = float((result.y_pred_after == data.test.y).mean())
    predictions_intervened_on = int(np.sum(np.any(result.mask, axis=1)))
    total_concept_checks = int(np.sum(result.mask))
    pred_binary = (result.C_pred >= 0.5).astype(int)
    final_binary = (result.C_intervened >= 0.5).astype(int)
    total_concept_edits_made = int(np.sum(pred_binary != final_binary))
    selective_acc_after = result.strat_metrics.get("selective_acc_after", None)
    coverage_after = result.strat_metrics.get("coverage_after", None)
    return {
        "accuracy": acc_intervened,
        "predictions_intervened_on": predictions_intervened_on,
        "total_concept_checks": total_concept_checks,
        "total_concept_edits_made": total_concept_edits_made,
        "selective_accuracy_after": selective_acc_after,
        "coverage_after": coverage_after,
    }

cs_budget_1 = _run_cs_intervention(1)

# CS Intervention Budget 3
cs_budget_3 = _run_cs_intervention(3)

# CS Intervention Budget Max
cs_budget_max = _run_cs_intervention(27)
no_interv = {
    'budget': 0,
    'accuracy': cs_sel_acc,
    'predictions_intervened_on': 0,
    'total_concept_checks': 0,
    'total_concept_edits_made': 0,
    'selective_accuracy_after': cs_sel_acc,
    'coverage_after': cs_sel_cov,
}
cs_intervention_df = pd.DataFrame(
    [
        no_interv,
        {"budget": 1, **cs_budget_1},
        {"budget": 3, **cs_budget_3},
        {"budget": 27, **cs_budget_max},
    ]
)

tot = 27 * settings['n_samples']

cs_intervention_df['work_reduced'] = 1 - ((tot * (1 - cs_intervention_df['coverage_after']) + cs_intervention_df['total_concept_checks'])) / (tot)

# --- Print & save results ---
print(f"\nDNN test accuracy: {test_accuracy:.4f}")
print(f"DNN selective accuracy: {dnn_sel_acc}, coverage: {dnn_sel_cov:.4f}")
print(f"CS  selective accuracy: {cs_sel_acc:.4f}, coverage: {cs_sel_cov:.4f}")
print(f"\n{cs_intervention_df.to_string(index=False)}")

out_path = results_dir / "sudoku_intervention_results.csv"
cs_intervention_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")