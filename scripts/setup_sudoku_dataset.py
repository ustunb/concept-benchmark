from concept_benchmark.paths import results_dir
from concept_benchmark.ext.fileutils import save
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset

def get_dataset_path(**settings) -> str:
    return results_dir / f"sudoku_{settings['n']**2}_{settings['data_type']}.data"

settings = {
    "n": 3,
    "n_samples": 5000,
    "valid_ratio": 0.5,
    "max_corrupt": 21,
    "data_type": "tabular",
    "seed": 42,
}

data = create_sudoku_dataset(**settings)
data.generate_cvindices(strata=data.y, total_folds_for_cv=[5])
save(data, get_dataset_path(**settings), overwrite=True)