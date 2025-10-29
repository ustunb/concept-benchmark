import pandas as pd

from concept_benchmark.ext.fileutils import load
from scripts.robot_demo.utils import (
    DEFAULT_ROBOT_SETTINGS,
    get_dataset_file,
    get_model_file,
    get_results_file,
    determine_device
)
from scripts.robot_demo.train_dnn import compute_accuracy

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
dnn = load(get_model_file(model_class="dnn", **settings))
test_loader = data.test.loader(shuffle=False, **loader_config)
dnn_accuracy = compute_accuracy(dnn, test_loader)
acc_rows.append(["dnn", "accuracy", dnn_accuracy])


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
    metrics = pd.read_csv(get_results_file(model_class="cbm", **settings)).drop(columns=['setup'])
    metrics['model'] = model_str + "_with_int"
    metrics = metrics[metrics['metric'].isin(METRICS.values())]
    interv_lst.append(metrics)
    
metrics_df = pd.DataFrame(acc_rows, columns=["model", "metric", "value"])
interv_df = pd.concat(interv_lst, ignore_index=True).reset_index(drop=True)

final_df = pd.concat([metrics_df, interv_df], ignore_index=True).reset_index(drop=True)
final_df.to_csv('robot_demo_results.csv', index=False)