from argparse import ArgumentParser
import os
from psutil import Process

from concept_benchmark.ext.fileutils import save, load
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset, image_transform
from scripts.sudoku_demo.utils import get_dataset_file, DEFAULT_SUDOKU_SETTINGS

settings = DEFAULT_SUDOKU_SETTINGS.copy()

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--max_corrupt', type=int, default=settings['max_corrupt'])
    p.add_argument('--seed', type=int, default=settings['seed'])
    args, _ = p.parse_known_args()

    settings.update(vars(args))

# for t in ["tabular", "image", "handwriting"]:
for t in ["tabular", "image"]:
    settings['data_type'] = t

    data = create_sudoku_dataset(**settings)
    data.generate_cvindices(strata=data.y, total_folds_for_cv=[5])
    data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

    dataset_dir = get_dataset_file(**settings)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save(data, dataset_dir / "sudoku_dataset.pkl", overwrite=True)
