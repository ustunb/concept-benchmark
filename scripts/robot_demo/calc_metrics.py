import pandas as pd
from itertools import product

from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import results_dir
from scripts.robot_demo.utils import (
    DEFAULT_ROBOT_SETTINGS,
    INPUT_MAP,
    MISSING_PROP,
    get_dataset_file,
    get_model_file,
    get_results_file,
    determine_device,
    compute_accuracy,
    RobotClassifierCNN
)

device = determine_device()
settings = DEFAULT_ROBOT_SETTINGS.copy()
data = load(get_dataset_file(**settings))
loader_config = {
    'batch_size': 32,
    'num_workers': 0 if str(device) == 'mps' else 12,
    'pin_memory': False if str(device) == 'mps' else True,
}

acc_rows = []

# DNN accuracy
dnn_weights = load(get_model_file(model_class="dnn", **settings))
dnn = RobotClassifierCNN(input_size=INPUT_MAP[settings['size']]).to(device)
dnn.load_state_dict(dnn_weights)
test_loader = data.test.loader(shuffle=False, **loader_config)
dnn_accuracy = compute_accuracy(dnn, test_loader, device)
acc_rows.append(["ideal", 0.0, "none", "dnn", "accuracy", dnn_accuracy])
acc_rows.append(["subconcept", 0.0, "none", "dnn", "accuracy", dnn_accuracy])

COLS = [
    "accuracy",
    "predictions_intervened_on",
    "predictions_changed",
    "total_concept_confirmations",
]

interv_lst = []
# CBM metrics
for subconcept, (missing, missing_mech) \
    in product(
        [False, True],
        [
            (0.0, "none"),
            (MISSING_PROP, "mcar"),
            (MISSING_PROP, "mnar"),
        ]
    ):

    print(f"Processing CBM - subconcept: {subconcept}, missing: {missing}, mech: {missing_mech}")
    settings['subconcept'] = subconcept
    settings['concept_missing'] = missing
    settings['concept_missing_mech'] = missing_mech
    # raw accuracy
    cbm = load(get_model_file(model_class="cbm", **settings))
    cbm_acc = (cbm.predict(data.test) == data.test.y).mean().item()
    data_name = "subconcept" if subconcept else "ideal"
    acc_rows.append([data_name, missing, missing_mech, "cbm_no_int", "accuracy", cbm_acc])

    # intervention metrics
    metrics = pd.read_csv(get_results_file(model_class="cbm", **settings))
    metrics = metrics.melt(
        id_vars=["data_name", "concept_missing", "concept_missing_mech", "budget", "threshold"],
        value_vars=COLS,
        var_name="metric",
        value_name="value"
    )
    metrics['model'] = "cbm_with_int_" + metrics['budget'].astype(str)
    interv_lst.append(metrics)
    
acc_df = pd.DataFrame(acc_rows, columns=["data_name", "concept_missing", "concept_missing_mech", "model", "metric", "value"])
interv_df = pd.concat(interv_lst, ignore_index=True).reset_index(drop=True)

final_df = pd.concat([acc_df, interv_df], ignore_index=True).reset_index(drop=True)
final_df.to_csv(results_dir / 'robot_demo_results.csv', index=False)