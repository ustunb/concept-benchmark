import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize

def _apply_missing(C, mode, rate, rng, y=None):
    if mode == "complete" or rate <= 0:
        return C
    C = C.copy().astype(np.float32)
    n, k = C.shape
    if mode == "mcar":
        M = rng.random((n, k)) < rate
    elif mode == "mar":
        if y is None:
            y = np.zeros(n, dtype=int)
        p1 = min(1.0, rate * 1.5)
        p0 = max(0.0, rate * 0.5)
        p = np.where(y.reshape(-1, 1) == 1, p1, p0)
        M = rng.random((n, k)) < p
    elif mode == "mnar":
        p = rate * (0.5 + 0.5 * C.astype(np.float32))
        M = rng.random((n, k)) < p
    else:
        M = np.zeros_like(C, dtype=bool)
    C[M] = -1.0
    return C


def _apply_label_noise(sample, noise_rate, seed):
    if noise_rate <= 0:
        return sample
    rng = np.random.default_rng(int(seed) + 4242)
    y = sample.y.astype(int).copy()
    flip_mask = rng.random(y.shape[0]) < float(noise_rate)
    y[flip_mask] = 1 - y[flip_mask]  # Flip labels

    return sample.__class__(
        parent=sample.parent, X=sample.X, C=sample.C, y=y, meta=sample.meta,
        transform=sample.transform, concept_transform=sample.concept_transform,
        target_transform=sample.target_transform, base_dir=getattr(sample, 'base_dir', None)
    )


def _rate_tag(r):
    v = int(round(float(r) * 100))
    return f"{v:03d}"


def _get_foot_shape_pred(pred_row, concept_names):
    """Helper to extract foot_shape prediction consistently"""
    if 'foot_shape' in concept_names:
        return int(pred_row[concept_names.index('foot_shape')])

    # Check subtypes - if ANY pointy subtype is 1, return 1 (pointy), else 0 (flat)
    pointy_types = [c for c in concept_names if 'foot_shape_pointy' in c]
    for ptype in pointy_types:
        if pred_row[concept_names.index(ptype)] == 1:
            return 1  # pointy
    return 0  # flat


def _get_concept_accuracies(h_test, H_tr, test, train):
    # get concept accuracy
    concept_names = test.concepts
    per_concept_acc = {}
    train_per_concept_acc = {}
    for i, concept_name in enumerate(concept_names):
        true_labels = test.C[:, i]
        train_true_labels = train.C[:, i]
        train_labels = H_tr[:, i]
        pred_labels = h_test[:, i]

        accuracy = float((pred_labels == true_labels).mean())
        train_accuracy = float((train_labels == train_true_labels).mean())
        train_per_concept_acc[concept_name] = train_accuracy
        per_concept_acc[concept_name] = accuracy
        print(pred_labels)
        print(f"{concept_name}: {accuracy:.4f}")
    return per_concept_acc, train_per_concept_acc


def _get_confusion_matrix(subtype_concepts, missing_concepts, fe, h_test, prob_test, test):
    all_preds = []
    original_probs = fe.predict_proba(h_test)
    original_preds = original_probs.argmax(1)
    for i in range(len(test.y)):
        true_label = int(test.y[i])
        pred_label = int(original_preds[i])

        row_data = {
            'sample_idx': i,
            # Ground truth from UC
            'foot_shape': int(test.meta['UC'][i, test.meta['unfiltered_concepts'].index('foot_shape')]),
            "foot_shape_subtype_string": test.meta['catalog_df'].iloc[test.meta['df_indices'][i]]['foot_shape_subtype'],
            'predicted': pred_label,
            'true_label': true_label,
        }
        # get what each detector predicts for this case:
        for j, concept in enumerate(test.concepts):
            row_data[f"{concept}_pred"] = int(float(prob_test[i, j]) > 0.5)
            row_data[f"{concept}"] = int(test.C[i, j])
        all_preds.append(row_data)

    # for the existing subtype detectors (.startswith(foot_shape_pointy) or .startswith(foot_shape_flat) in test.concepts)
    # check how often in predicted 1 when the subtype string was any of the subconcepts in drop_concepts (again, .startsiwth)
    # store as percentage of total cases where that subconcept was present
    all_preds = pd.DataFrame(all_preds)
    predicted_classes = subtype_concepts + ['other']
    all_concepts = sorted(subtype_concepts + missing_concepts)
    all_concepts = [c for c in all_concepts if "foot_shape_" in c]

    confusion_matrix = {true_subtype: {pred_class: 0 for pred_class in predicted_classes}
                        for true_subtype in all_concepts}

    for idx, row in all_preds.iterrows():
        true_subtype = row['foot_shape_subtype_string']
        foot_shape = "pointy" if row['foot_shape'] else "flat"
        true_type = f"foot_shape_{foot_shape}_{true_subtype}"

        # Find which detector(s) fired (predicted subtype)
        fired_detectors = [det for det in subtype_concepts if row[f"{det}_pred"] == 1]

        if len(fired_detectors) == 0:
            # No detector fired -> predict "other"
            fired_detectors = ['other']
        for ps in fired_detectors:
            confusion_matrix[true_type][ps] += 1

    # make a pd dataframe for better visualization
    confusion_df = pd.DataFrame(confusion_matrix).T
    # print
    print("\nConfusion Matrix for Foot Shape Subtype Detectors:")
    print(confusion_df.to_string())
    return all_preds, confusion_df


def _get_accuracies_per_subconcept(all_preds, missing_concepts, subtype_concepts):
    per_concept_acc2 = {}
    for concept in sorted(subtype_concepts + missing_concepts):
        foot_type = "pointy" if "pointy" in concept else "flat"
        foot_subtype = concept.replace('foot_shape_', '').replace(foot_type + "_", "")
        concept_rows = all_preds[(all_preds['foot_shape_subtype_string'] == foot_subtype) & (
                    all_preds['foot_shape'] == (1 if foot_type == "pointy" else 0))]
        if len(concept_rows) > 0:
            accuracy = float((concept_rows['predicted'] == concept_rows['true_label']).mean())
            per_concept_acc2[concept] = round(accuracy, 4)
        else:
            per_concept_acc2[concept] = None

    print(per_concept_acc2)
    return per_concept_acc2


def find_params_for_target_probabilities(feature_names, target_probs, initial_guess=None):
    """
    Find logit_weights, intercept, and scalar to match target probabilities.

    Args:
        feature_names: list of feature names, e.g., ['mouth_type', 'foot_shape']
        target_probs: dict mapping feature combinations (tuples) to desired P(glorp)
                     e.g., {(0, 0): 0.05, (1, 0): 0.50, (0, 1): 0.95, (1, 1): 0.99}
                     Tuples follow the order of feature_names
        initial_guess: optional dict with 'weights', 'intercept', 'scalar'

    Returns:
        dict with optimized 'logit_weights', 'intercept', 'scalar', and 'optimization_error'

    Example:
        result = find_params_for_target_probabilities(
            feature_names=['mouth_type', 'foot_shape'],
            target_probs={
                (0, 0): 0.02,   # mouth=0, foot=0 → 2% glorp
                (1, 0): 0.50,   # mouth=1, foot=0 → 50% glorp
                (0, 1): 0.50,   # mouth=0, foot=1 → 50% glorp
                (1, 1): 0.98,   # mouth=1, foot=1 → 98% glorp
            }
        )

        # Use in settings:
        settings["logit_weights"] = result['logit_weights']
        settings["logit_intercept"] = result['intercept']
        settings["logit_scalar"] = result['scalar']
    """
    n_features = len(feature_names)

    # Initialize parameters
    if initial_guess is None:
        initial_weights = [1.0] * n_features
        initial_intercept = 0.5
        initial_scalar = 4.0
    else:
        initial_weights = [initial_guess['weights'].get(name, 1.0) for name in feature_names]
        initial_intercept = initial_guess.get('intercept', 0.5)
        initial_scalar = initial_guess.get('scalar', 4.0)

    # Pack into optimization vector: [weight1, weight2, ..., intercept, scalar]
    x0 = initial_weights + [initial_intercept, initial_scalar]

    def objective(x):
        """Compute MSE between predicted and target probabilities"""
        weights = x[:n_features]
        intercept = x[n_features]
        scalar = x[n_features + 1]

        # Compute predicted probabilities for all target combinations
        error = 0
        for combo_tuple, target_prob in target_probs.items():
            # Compute logit for this combination
            logit_sum = sum(w * val for w, val in zip(weights, combo_tuple))
            logit_val = scalar * (logit_sum - intercept)
            pred_prob = expit(logit_val)

            # Add squared error
            error += (pred_prob - target_prob) ** 2

        return error

    # Optimize with bounds
    # Weights: [0.1, 20], Intercept: [-10, 10], Scalar: [0.1, 10]
    bounds = [(0.1, 20.0)] * n_features + [(-10.0, 10.0), (0.1, 10.0)]

    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    # Extract optimized parameters
    opt_weights = res.x[:n_features]
    opt_intercept = res.x[n_features]
    opt_scalar = res.x[n_features + 1]

    # Create logit_weights dictionary
    logit_weights_dict = {name: float(w) for name, w in zip(feature_names, opt_weights)}

    # Verify results
    verification = {}
    for combo_tuple, target_prob in target_probs.items():
        logit_sum = sum(logit_weights_dict[name] * val
                        for name, val in zip(feature_names, combo_tuple))
        logit_val = opt_scalar * (logit_sum - opt_intercept)
        pred_prob = expit(logit_val)
        verification[combo_tuple] = {
            'target': target_prob,
            'achieved': pred_prob,
            'error': abs(pred_prob - target_prob)
        }

    return {
        'logit_weights': logit_weights_dict,
        'logit_intercept': float(opt_intercept),
        'logit_scalar': float(opt_scalar),
        'optimization_error': float(res.fun),
        'verification': verification
    }


# # Example usage
# if __name__ == "__main__":
#     result = find_params_for_target_probabilities(
#         feature_names=['mouth_type', 'foot_shape'],
#         target_probs={
#             (0, 0): 0.01,  # Neither → 1% glorp
#             (1, 0): 0.50,  # Mouth only → 50% glorp
#             (0, 1): 0.50,  # Foot only → 50% glorp
#             (1, 1): 0.99,  # Both → 99% glorp
#         }
#     )