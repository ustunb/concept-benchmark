from argparse import ArgumentParser
import os
import numpy as np
from psutil import Process
from torchvision import transforms

from concept_benchmark.ext.fileutils import save
from concept_benchmark.synthetic.robot import create_synthetic_dataset
from utils import (
    get_dataset_file, 
    DEFAULT_ROBOT_SETTINGS,
    SUBCONCEPT_DROP
)

from scripts.dataset_skewing import create_skewed_splits_full

settings = DEFAULT_ROBOT_SETTINGS.copy()

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--data_type', type=str, choices=['image', 'text'], default=settings['data_type'])
    p.add_argument('--subconcept', action='store_true') 
    p.add_argument('--draw', action='store_true')
    p.add_argument('--seed', type=int, default=settings['seed'])
    args, _ = p.parse_known_args()
    settings.update(vars(args))

    if args.subconcept:
        settings['drop_concepts'] = SUBCONCEPT_DROP

data = create_synthetic_dataset(**settings)
tf = transforms.Compose([transforms.ToTensor()])
data.transform = tf
data.generate_cvindices(seed=int(settings["seed"]))

rng = np.random.default_rng(int(settings["seed"]))
sk_data = create_skewed_splits_full(dataset=data, rng=rng, **settings)
save(sk_data, get_dataset_file(**settings), overwrite=True)