from argparse import ArgumentParser
import os
from psutil import Process
from torchvision import transforms

from concept_benchmark.ext.fileutils import save
from concept_benchmark.synthetic.robot import create_synthetic_dataset
from utils import get_dataset_file, DEFAULT_ROBOT_SETTINGS

settings = DEFAULT_ROBOT_SETTINGS.copy()

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--data_type', type=str, choices=['image', 'text'], default=settings['data_type'])
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--concept_noise', type=float, default=settings['concept_noise'])
    p.add_argument('--target_accuracy', type=float, default=settings['target_accuracy'])
    # need flag for concept noise
    args, _ = p.parse_known_args()

settings.update(vars(args))

data = create_synthetic_dataset(**settings)
data.generate_cvindices(strata=data.y, total_folds_for_cv=[5])
IMG_SIZE = 224
tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
data.transform = tf

save(data, get_dataset_file(**settings), overwrite=True)