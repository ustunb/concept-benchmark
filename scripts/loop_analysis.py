import json
import pandas as pd
from pathlib import Path
import numpy as np
from concept_benchmark.paths import results_dir

# ==============================================================================
# LOAD ALL RESULTS INTO A SIMPLE TABLE
# ==============================================================================

results = []

for folder in Path(results_dir / 'robots').glob('loop*/'):
    json_file = folder / 'metrics_cbm_detected_robots_image_stochastic_complete__skewint-acc90_seed555.json'
    csv_file = folder / 'confusion.csv'

    if not json_file.exists():
        continue

    with open(json_file) as f:
        data = json.load(f)

    # Extract training subtypes
    training_subtypes = [k.replace('foot_shape_', '')
                         for k in data['concept_accuracies'].keys()
                         if k.startswith('foot_shape_')]

    # Sort for consistent comparison
    training_subtypes_sorted = sorted(training_subtypes)

    # Initialize confusion metrics
    pointy_as_flat_count = 0
    flat_as_pointy_count = 0
    total_confusion_analyzed = 0

    # Load and analyze confusion matrix if available
    if csv_file.exists():
        try:
            conf_df = pd.read_csv(csv_file, index_col=0)

            # For each row (true subtype)
            for true_subtype in conf_df.index:
                # Skip if this was in training
                clean_name = true_subtype.replace('foot_shape_', '')
                if clean_name in training_subtypes:
                    continue

                total_preds = conf_df.loc[true_subtype].sum()
                if total_preds == 0:
                    continue

                total_confusion_analyzed += 1

                # Check if true subtype is pointy or flat
                is_true_pointy = 'pointy' in true_subtype

                # Count predictions to opposite category
                for pred_col in conf_df.columns:
                    pred_count = conf_df.loc[true_subtype, pred_col]

                    if pred_col == 'other':
                        continue

                    is_pred_pointy = 'pointy' in pred_col

                    # Pointy predicted as flat
                    if is_true_pointy and not is_pred_pointy:
                        pointy_as_flat_count += pred_count

                    # Flat predicted as pointy
                    elif not is_true_pointy and is_pred_pointy:
                        flat_as_pointy_count += pred_count
        except Exception as e:
            print(f"Warning: Could not parse confusion matrix for {folder.name}: {e}")

    record = {
        'training_set': ', '.join(training_subtypes_sorted),
        'n_subtypes': len(training_subtypes),
        'n_pointy': sum(1 for s in training_subtypes if 'pointy' in s),
        'n_flat': sum(1 for s in training_subtypes if 'flat' in s),

        # Performance metrics
        'cbm_acc': data['cbm_acc_detected'],
        'oracle_gap': data['cbm_acc_oracle'] - data['cbm_acc_detected'],
        'concept_acc_mean': data['concept_det_acc_mean'],

        # Confusion metrics
        'pointy_as_flat': pointy_as_flat_count,
        'flat_as_pointy': flat_as_pointy_count,
        'cross_confusion_total': pointy_as_flat_count + flat_as_pointy_count,

        # Per-subtype accuracy stats
        'min_subtype_acc': min(data['model_accuracies_per_concept'].values()),
        'max_subtype_acc': max(data['model_accuracies_per_concept'].values()),
        'acc_std': np.std(list(data['model_accuracies_per_concept'].values())),
    }

    results.append(record)

df = pd.DataFrame(results)

# Sort by accuracy (descending)
df = df.sort_values('cbm_acc', ascending=False)

print(f"Loaded {len(df)} experiments\n")

# ==============================================================================
# SAVE TO CSV
# ==============================================================================

# Round for cleaner display
df_export = df.copy()
df_export['cbm_acc'] = df_export['cbm_acc'].round(4)
df_export['oracle_gap'] = df_export['oracle_gap'].round(4)
df_export['concept_acc_mean'] = df_export['concept_acc_mean'].round(4)
df_export['min_subtype_acc'] = df_export['min_subtype_acc'].round(4)
df_export['max_subtype_acc'] = df_export['max_subtype_acc'].round(4)
df_export['acc_std'] = df_export['acc_std'].round(4)

df_export.to_csv('cbm_configurations.csv', index=False)
print(f"Saved full table to: cbm_configurations.csv")
print("Open in Excel/Sheets to sort and filter!\n")