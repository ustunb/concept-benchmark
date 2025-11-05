import os
import pandas as pd
from argparse import ArgumentParser

from psutil import Process
from utils import (
    DEFAULT_ROBOT_SETTINGS,
    determine_device,
    get_dataset_file,
    get_model_file,
    get_results_file
)

from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import results_dir

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

data = load(get_dataset_file(**settings))
model = load(get_model_file(model_class="cbm", **settings))

c_preds = model.concept_detector.predict(data.test)
acc = (model.predict(data.test) == data.test.y).mean().item()
