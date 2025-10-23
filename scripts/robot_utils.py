import numpy as np
import pandas as pd

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
