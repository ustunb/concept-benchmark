import numpy as np

def calc_metric(pred_probs, y_true, tau=0.5):
    selected_y = pred_probs.copy()
    selected_y[pred_probs < tau] = 0
    selected_y[(pred_probs >= tau) & (pred_probs <= 1 - tau)] = -10 # abstain
    selected_y[pred_probs > 1 - tau] = 1

    abstain = selected_y == -10

    if not np.any(~abstain):
        coverage = 0.0
        selective_accuracy = 1.0
    else:
        coverage = (~abstain).mean()
        selective_accuracy = (selected_y[~abstain] == y_true[~abstain]).mean()

    return {
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "tau": tau
    }