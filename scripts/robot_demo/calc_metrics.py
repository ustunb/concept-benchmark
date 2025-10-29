import pandas as pd

from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import results_dir
from scripts.robot_demo.utils import (
    DEFAULT_ROBOT_SETTINGS,
    INPUT_MAP,
    get_dataset_file,
    get_model_file,
    get_results_file,
    determine_device,
    compute_accuracy,
    RobotClassifierCNN
)

device = determine_device()
settings = DEFAULT_ROBOT_SETTINGS.copy()
METRICS = {
    "accuracy": "accuracy",
    "monitored": "predictions_intervened_on",
    "work_done": "total_concept_edits_made"
}
data = load(get_dataset_file(**settings))
loader_config = {
    'batch_size': 32,
    'num_workers': 12,
    'pin_memory': True
}

acc_rows = []

# DNN accuracy
dnn_weights = load(get_model_file(model_class="dnn", **settings))
dnn = RobotClassifierCNN(input_size=INPUT_MAP[settings['size']]).to(device)
dnn.load_state_dict(dnn_weights)
test_loader = data.test.loader(shuffle=False, **loader_config)
dnn_accuracy = compute_accuracy(dnn, test_loader, device)
acc_rows.append(["dnn", "accuracy", dnn_accuracy])

COLS = [
    "accuracy",
    "predictions_intervened_on",
    "total_concept_checks",
    "total_concept_edits_made",
]

interv_lst = []
# CBM metrics
for subconcept in [True, False]:
    settings['subconcept'] = subconcept
    # raw accuracy
    cbm = load(get_model_file(model_class="cbm", **settings))
    cbm_acc = (cbm.predict(data.test) == data.test.y).mean().item()
    model_str = "subconcept_cbm" if subconcept else "cbm"
    acc_rows.append([model_str + "_no_int", "accuracy", cbm_acc])

    # intervention metrics
    metrics = pd.read_csv(get_results_file(model_class="cbm", **settings))
    metrics = metrics.melt(
        id_vars=["budget", "threshold"],
        value_vars=COLS,
        var_name="metric",
        value_name="value"
    )
    metrics['model'] = model_str + "_with_int"
    metrics = metrics[metrics['metric'].isin(METRICS.values())]
    interv_lst.append(metrics)
    
acc_df = pd.DataFrame(acc_rows, columns=["model", "metric", "value"])
interv_df = pd.concat(interv_lst, ignore_index=True).reset_index(drop=True)

final_df = pd.concat([acc_df, interv_df], ignore_index=True).reset_index(drop=True)
final_df.to_csv(results_dir / 'robot_demo_results.csv', index=False)