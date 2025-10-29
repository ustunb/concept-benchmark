import os
import pandas as pd
from argparse import ArgumentParser

from psutil import Process
from utils import (
    DEFAULT_ROBOT_SETTINGS,
    INTERVENTION_SETTINGS,
    determine_device,
    get_dataset_file,
    get_model_file,
    get_results_file
)

from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.paths import results_dir

from scripts.robot_image_training import test_interventions

THRESH = [0.2, 0.4, 0.6, 0.8]
METRIC_COLS = [
    "accuracy",
    "predictions_intervened_on",
    "total_concept_checks",
    "total_concept_edits_made",
]
device = determine_device()
settings = DEFAULT_ROBOT_SETTINGS.copy()

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--data_type', type=str, choices=['image', 'text'], default=settings['data_type'])
    p.add_argument('--subconcept', action='store_true') 
    p.add_argument('--seed', type=int, default=settings['seed'])
    args, _ = p.parse_known_args()
    settings.update(vars(args))

INTERVENTION_SETTINGS.update({"seed": settings['seed']})

data = load(get_dataset_file(**settings))
model = load(get_model_file(model_class="cbm", **settings))

c_preds = model.concept_detector.predict(data.test)
acc = (model.predict(data.test) == data.test.y).mean().item()

df_lst = []
COLS = ['budget', 'threshold'] + METRIC_COLS
for t in THRESH:
    INTERVENTION_SETTINGS.update({"intervention_threshold": t})
    b, a, r = test_interventions(
        prob_test=c_preds,
        sttngs=INTERVENTION_SETTINGS,
        acc_det=acc,
        fe=model.front_end_model,
        test=data.test,
    )
    df_lst.append(
        (
            pd.DataFrame(r)
            .T
            .assign(budget=INTERVENTION_SETTINGS['budget'])
            .assign(threshold=t)
            .reset_index(drop=True)
            [COLS]
        )
    )

results_df = pd.concat(df_lst, axis=0).reset_index(drop=True)
results_df.melt(
    id_vars=["budget", "threshold"],
    value_vars=METRIC_COLS,
    var_name="metric",
    value_name="value"
)
results_df.to_csv(results_dir / get_results_file(**settings), index=False)
