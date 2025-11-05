from argparse import ArgumentParser
import os
from psutil import Process

from concept_benchmark.ext.fileutils import save
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset
from scripts.sudoku_demo.utils import get_dataset_file, DEFAULT_SUDOKU_SETTINGS

settings = DEFAULT_SUDOKU_SETTINGS.copy()

if Process(pid=os.getppid()).name() not in ("node"):
    p = ArgumentParser()
    p.add_argument('--data_type', type=str, choices=['tabular', 'image', 'handwriting'], default=settings['data_type'])
    p.add_argument('--n', type=int, default=settings['n'])
    p.add_argument('--max_corrupt', type=int, default=settings['max_corrupt'])
    args, _ = p.parse_known_args()

settings.update(vars(args))

data = create_sudoku_dataset(**settings)
data.generate_cvindices(strata=data.y, total_folds_for_cv=[5])

save(data, get_dataset_file(**settings), overwrite=True)