import os
from psutil import Process
from argparse import ArgumentParser

from concept_benchmark.models import ConceptDetector
from utils import get_dataset_file, get_model_file, determine_device, create_skewed_splits
from demo_models import RobotConceptClassifier, ConceptSudokuCNN

from concept_benchmark.ext.fileutils import save, load

settings = {
    'data_name': 'robot',
    'data_type': 'image',
    'n': 1,
    'concept_noise': 0.0,
    'concept_missing': 0.0,
    'concept_missing_mech': 'none',
    'target_accuracy': 1.0, # doesn't matter but need for dataset loading
    'epochs': 50,
    'patience': 10,
}

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--data_name', type=str, choices=['sudoku', 'robot'], default=settings['data_name'])
    p.add_argument('--data_type', type=str, choices=['tabular', 'image', 'text'], default=settings['data_type'])
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--concept_noise', type=float, default=settings['concept_noise'])
    p.add_argument('--concept_missing', type=float, default=settings['concept_missing'])
    p.add_argument('--concept_missing_mech', type=str, choices=['none', 'mcar', 'mnar'], default=settings['concept_missing_mech'])
    p.add_argument('--epochs', type=int, default=settings['epochs'])
    p.add_argument('--patience', type=int, default=settings['patience'])
    args, _ = p.parse_known_args()
    settings.update(vars(args))

data = load(get_dataset_file(**settings))
data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

tr, val, test = create_skewed_splits(
    data,
    skew_specs=
    [{'concepts': {'body_shape': 0, 'foot_shape': 1, 'has_antennae': 1}, 'min_fraction': 0.3},
{'concepts': {'mouth_type': 0, 'foot_shape': 1, 'has_antennae': 1}, 'min_fraction': 0.3},
{'concepts': {'body_shape': 0, 'mouth_type': 0, 'has_antennae': 1}, 'min_fraction': 0.3},
{'concepts': {'body_shape': 1, 'mouth_type': 1, 'has_antennae': 0}, 'min_fraction': 0.3},
{'concepts': {'body_shape': 1, 'foot_shape': 0, 'has_antennae': 0}, 'min_fraction': 0.3},
{'concepts': {'foot_shape': 0, 'mouth_type': 1, 'has_antennae': 0}, 'min_fraction': 0.3}])

if settings['concept_missing_mech'] != 'none':
    if settings['concept_missing'] <= 0.0:
        raise ValueError("concept_missing must be > 0 when concept_missing_mech is not 'none'")
    data.sample_concept_missingness(p=settings['concept_missing'], mechanism=settings['concept_missing_mech'])
    data.training.concept_missing = True
    data.validation.concept_missing = True
    
if settings['data_name'] == 'sudoku':
    model = ConceptDetector(model=ConceptSudokuCNN())
elif settings['data_name'] == 'robot':
    model = ConceptDetector(model=RobotConceptClassifier(num_concepts=data.C.shape[1]))
    # model = ConceptDetector(model=RobotViTConceptClassifier(num_concepts=data.C.shape[1]))
else:
    raise ValueError(f"Unknown data_name: {settings['data_name']}")

device = determine_device()

config = {
    'device': device,
    'batch_size': 32,
    'num_workers': 0 if device.type == 'mps' else 12,
    'pin_memory': False if device.type == 'mps' else True,
}

model.fit(
    # train_dataset=data.training,
    train_dataset=tr,
    valid_dataset=val,
    embed_params={'shuffle': False, **config}, 
    fit_params={
        'epochs': settings['epochs'], 
        'lr': 1e-3, 
        'patience': settings['patience'], 
        **config
    }
)

train_accuracy = ((model.predict(tr) > 0.5) == tr.C).mean(axis=0)
valid_accuracy = ((model.predict(val) > 0.5) == val.C).mean(axis=0)
test_accuracy = ((model.predict(test) > 0.5) == test.C).mean(axis=0)

print(f"Train Concept Accuracy: {train_accuracy.mean() * 100:.2f}% ± {train_accuracy.std() * 100:.2f}%")
print(f"Validation Concept Accuracy: {valid_accuracy.mean() * 100:.2f}% ± {valid_accuracy.std() * 100:.2f}%")
print(f"Test Concept Accuracy: {test_accuracy.mean() * 100:.2f}% ± {test_accuracy.std() * 100:.2f}%")

# save the model
# save(model, get_model_file(model_type="cd", **settings), overwrite=True)