import os
from argparse import ArgumentParser

import numpy as np
import torch
from psutil import Process

from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.paths import data_dir
from concept_benchmark.models import (
    ConceptBasedModel,
    ConceptDetector,
)

from scripts.sudoku_demo.utils import (
    DEFAULT_SUDOKU_SETTINGS,
    determine_device,
    get_dataset_file,
    get_model_file,)

from scripts.sudoku_demo.sudoku_models import GroupPoolingConceptSudokuCNN as SudokuConceptModel


device = determine_device()
settings = DEFAULT_SUDOKU_SETTINGS.copy()

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--max_corrupt', type=int, default=settings['max_corrupt'])
    p.add_argument('--concept_missing', type=float, default=0)
    p.add_argument('--concept_missing_mech', type=str, choices=['none', 'mcar', 'mnar'], default='none')
    p.add_argument('--seed', type=int, default=settings['seed'])
    args, _ = p.parse_known_args()
    settings.update(vars(args))

data = load(get_dataset_file(data_type="tabular", **settings) / "sudoku_dataset.pkl")
# sudoku_dir = data_dir / "sudoku"
# data = load(sudoku_dir / "demo_ocr_m_21_tabular" / "sudoku_dataset.pkl")
data.generate_cvindices(strata=data.y, total_folds_for_cv=[5], seed=settings['seed'])
data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

if settings['concept_missing_mech'] != 'none':
    if settings['concept_missing'] <= 0.0:
        raise ValueError("concept_missing must be > 0 when concept_missing_mech is not 'none'")
    data.sample_concept_missingness(
        p=settings['concept_missing'], 
        mechanism=settings['concept_missing_mech'],
        rng=np.random.default_rng(settings['seed'])
    )
    data.training.concept_missing = True
    data.validation.concept_missing = True

config = {
    'device': device,
    'batch_size': 32,
    'num_workers': 0 if device.type == 'mps' else 12,
    'pin_memory': False if device.type == 'mps' else True,
}
torch.manual_seed(int(settings["seed"]))

model = SudokuConceptModel()

cd = ConceptDetector(model=model)
cbm = ConceptBasedModel(concept_detector=cd, propagate=True)
cbm.fit(
    train_dataset=data.training,
    valid_dataset=data.validation,
    freeze=False,
    concept_embed_params={'shuffle': False, **config},
    fit_params={"epochs": 100, 'lr': 1e-3, "patience": 20, **config}
)

# evaluate on test set
test_pred = cbm.predict(data.test)
print("Test Accuracy:", np.mean(test_pred == data.test.y))

# save model
save(cbm, get_model_file(data_type="tabular", model_class="cs", **settings), overwrite=True)
