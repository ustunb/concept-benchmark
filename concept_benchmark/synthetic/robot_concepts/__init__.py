from importlib import import_module as _import_module

# Expose text generator
textgen = _import_module(__name__ + ".textgen")
create_robot_text_dataset = textgen.create_robot_text_dataset
__all__ = ["textgen", "create_robot_text_dataset"]
from .train_text_concept_detector import train_concept_detector_text_nn, NNConceptDetectorResult

# Optionally expose heavy stuff (safe if deps are missing)
try:
    from .main import RobotConceptDataset, create_synthetic_dataset, train_robot_concept_model
    __all__ += ["RobotConceptDataset", "create_synthetic_dataset", "train_robot_concept_model"]
except Exception:
    pass
