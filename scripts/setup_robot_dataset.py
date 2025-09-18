from concept_benchmark.paths import results_dir
from concept_benchmark.ext.fileutils import save
from concept_benchmark.synthetic.robot import create_synthetic_dataset

def get_dataset_path(**settings) -> str:
    return results_dir / f"robot_{settings['data_type']}.data"

settings = {
    'samples_per_instance': 1,
    'draw': False,
    'output_directory': results_dir / 'robots',
    'concepts': {
        'head_shape': ['square', 'round'],
        'body_shape': ['square', 'round'],
        'has_knees': ['false', 'true'],
        'has_elbows': ['false', 'true'],
        'has_antennae': ['false', 'true'],
        'ears_shape': ['square', 'triangle'],
        'mouth_type': ['closed', 'open'],
        'hand_shape': ['round_circle', 'round_oval', 'round_oval2',
                        'edgy_triangle', 'edgy_square', 'edgy_trapezoid'],
        'foot_shape': ['flat_4sided', 'flat_5sided', 'flat_lshaped',
                        'pointy_3sided', 'pointy_4sided', 'pointy_6sided'],
    },
    'spurious_features': ['has_elbows', 'hand_shape'],  # features that do not appear in the catalog + color
    'model': "'glorp' if (int(row['body_shape']=='square') + int(row['foot_shape']=='pointy') - 2 >= 0) else 'drent'",
    'model_type': 'deterministic', 
    'size': 'large',  
    'color_mode': 'color',  
    'data_type': 'image'
}

data = create_synthetic_dataset(**settings)
data.generate_cvindices(strata=data.y, total_folds_for_cv=[5])
save(data, get_dataset_path(**settings), overwrite=True)