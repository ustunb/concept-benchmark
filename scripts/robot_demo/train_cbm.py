import os
from argparse import ArgumentParser

import numpy as np
import torch
from psutil import Process
from utils import (
    DEFAULT_ROBOT_SETTINGS,
    INPUT_MAP,
    determine_device,
    get_dataset_file,
    get_model_file,
)

from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.models import (
    ConceptBasedModel,
    ConceptDetector,
    RobotConceptClassifier,
)

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

config = {
    'device': device,
    'batch_size': 32,
    'num_workers': 0 if device == 'mps' else 12,
    'pin_memory': False if device == 'mps' else True,
}
torch.manual_seed(int(settings["seed"]))

cd = ConceptDetector(
    model=RobotConceptClassifier(
        num_concepts=data.training.n_concepts,
        input_size=INPUT_MAP[settings['size']]
    )
)
cbm = ConceptBasedModel(concept_detector=cd)
cbm.fit(
    train_dataset=data.training,
    valid_dataset=data.validation,
    freeze=False,
    concept_embed_params={'shuffle': False, **config},
    fit_params={"epochs": 50, 'lr': 1e-3, "patience": 10, **config}
)

# evaluate on test set
test_pred = cbm.predict(data.test)
print("Test Accuracy:", np.mean(test_pred == data.test.y))

# save model
save(cbm, get_model_file(model_class="cbm", **settings), overwrite=True)