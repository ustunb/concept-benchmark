import os
from psutil import Process
from argparse import ArgumentParser

from concept_benchmark.models import FrontEndModel
from utils import get_dataset_file, get_model_file

from concept_benchmark.ext.fileutils import save, load

settings = {
    'data_name': 'sudoku',
    'data_type': 'tabular',
    'n': 3,
    'max_corrupt': 21,
    'concept_noise': 0.0,  
    'concept_missing': 0.2,
    'concept_missing_mech': 'mcar',
    'target_accuracy': 0.8,
}

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--data_name', type=str, choices=['sudoku', 'robot'], default=settings['data_name'])
    p.add_argument('--data_type', type=str, choices=['image', 'text'], default=settings['data_type'])
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--concept_noise', type=float, default=settings['concept_noise'])
    p.add_argument('--concept_missing', type=float, default=settings['concept_missing'])
    p.add_argument('--concept_missing_mech', type=str, choices=['none', 'mcar', 'mnar'], default=settings['concept_missing_mech'])
    p.add_argument('--target_accuracy', type=float, default=settings['target_accuracy'])
    args, _ = p.parse_known_args()
    settings.update(vars(args))

data = load(get_dataset_file(**settings))
data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

model = FrontEndModel()
model.fit(data.training.C, data.training.y)

train_accuracy = (model.predict(data.training.C) == data.training.y).mean()
val_accuracy = (model.predict(data.validation.C) == data.validation.y).mean()
test_accuracy = (model.predict(data.test.C) == data.test.y).mean()
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# save the model
save(model, get_model_file(model_type="fe", **settings), overwrite=True)