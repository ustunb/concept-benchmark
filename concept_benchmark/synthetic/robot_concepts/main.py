from __future__ import annotations

import torch
import numpy as np
from PIL import Image

from .catalog import generate_robot_catalog, RobotDistribution
from .concept_detector import RobotConceptViT, RobotConceptTrainer
from concept_benchmark.data import ConceptDataset
from .utils import unlist0, model_to_logistic
from transformers import ViTImageProcessor, AutoImageProcessor
from pathlib import Path
from scipy.special import expit

# Check if MPS (Metal Performance Shaders) is available for M1
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")


class RobotConceptDataset(ConceptDataset):
    """Custom ConceptDataset for robot classification with concept detection"""

    def __init__(self, X, C, y, meta, robot_ids, catalog_df, trained_concept_detector=None):
        super().__init__(X, C, y, meta, base_dir=Path(meta.get('image_dir', '.')))

        self.concept_detector = trained_concept_detector
        self.robot_ids = robot_ids
        self.catalog_df = catalog_df

        self.resolution = meta.get('resolution', 224)
        self.color_mode = meta.get('color_mode', 'color')
        self.labeling_function = meta.get('labeling_function', '')
        self.meta = meta

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.y)

    def get_concept_names(self):
        """Return the names of the concepts in the dataset"""
        return self.meta.get('concepts', [])

    def attach_concept_detector(self, model):
        """
        Attach a trained concept detector to the dataset
        """
        self.concept_detector = model

    def attach_image_processor(self, processor):
        """
        Attach an image processor to the dataset
        """
        self.processor = processor

    def __getitem__(self, idx):
        image, C_idx, y_idx = self._full.__getitem__(idx)
        try:
            processor = self.processor
        except AttributeError:
            raise ValueError("Image processor not attached. Use attach_image_processor() to set it.")
        inputs = processor(images=image, return_tensors="pt")
        image = inputs['pixel_values'].squeeze(0)  # Remove batch dimension

        return image, C_idx, y_idx


def create_synthetic_dataset(**kwargs):
    """
    Create synthetic robot dataset that returns ConceptDataset

    Args:
        **kwargs: Parameters for robot generation (same as your existing params)

    Returns:
        RobotConceptDataset object
    """
    num_combinations = int(np.prod([len(v) for v in kwargs['concepts'].values()]))
    kwargs['num_robots'] = kwargs.get('num_robots', num_combinations) * kwargs.get('samples_per_instance', 1)
    kwargs['resolution'] = 600 if kwargs.get('size', 'large') == 'large' else 32
    kwargs['irrelevant_features'] = kwargs.get('spurious_features', [])

    catalog_df = generate_robot_catalog(kwargs)
    rdist = RobotDistribution(df=catalog_df)
    df = rdist.df

    # Specify true labels
    if kwargs.get('model_type', 'deterministic') == 'deterministic':
        glorp_model_true = lambda row: eval(unlist0(kwargs['model']))
    elif kwargs.get('model_type', 'deterministic') == 'stochastic':
        glorp_model_true = lambda row: eval(model_to_logistic(kwargs['model']))
    else:
        raise ValueError("Invalid model_type. Use 'deterministic' or 'stochastic'.")

    df[rdist.outcome_name] = df.apply(glorp_model_true, axis=1)
    catalog_df[rdist.outcome_name] = catalog_df.apply(glorp_model_true, axis=1)

    if kwargs.get('model_type', 'deterministic') == 'deterministic':
        # change "glorp" to 1 and "drent" to 0
        catalog_df[rdist.outcome_name] = catalog_df[rdist.outcome_name].apply(
            lambda x: 1 if x == 'glorp' else 0)

    if kwargs.get('verbose', 'False'):
        print("Catalog DataFrame:")
        print(catalog_df.to_string(index=False))

    full_dataset = rdist.to_dataset()

    # X: Image paths (stored as strings)
    image_dir = kwargs.get('output_directory', '.static/images')
    X = np.array([row['png_filename'] for _, row in catalog_df.iterrows()])

    # C: Concept matrix
    feature_names = full_dataset.feature_names
    C = full_dataset.X
    # change -1s to 0s in C
    C[C == -1] = 0

    # y: Labels pr P(y=1|x)
    y = catalog_df[full_dataset.outcome_name].values

    if kwargs.get('verbose', 'False'):
        print("Dataset for Training:")
        print(X)
        print(C)
        print(y)

    # Meta: metadata for ConceptDataset
    meta = {
        "classes": ["drent", "glorp"],
        "concepts": feature_names,
        "data_type": "image",
        "image_dir": image_dir,
        "resolution": kwargs.get('resolution', 224),
        "color_mode": kwargs.get('color_mode', 'color'),
        "labeling_function": kwargs.get('model', ''),
        "num_robots": kwargs.get('num_robots', 48)
    }

    robot_dataset = RobotConceptDataset(X, C, y, meta, robot_ids=catalog_df['id'], catalog_df=catalog_df,
                                        trained_concept_detector=None)

    # Train concept detector if requested
    if kwargs.get('train_concept_detector', False):
        model = train_robot_concept_model(
            dataset=robot_dataset,
            resolution=kwargs.get('resolution', 224),
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 16),
        )
        robot_dataset.attach_concept_detector(model)

    return robot_dataset


def train_robot_concept_model(dataset, resolution, epochs, batch_size):
    """
    Create synthetic robot dataset with trained concept detector

    Args:
        params: Dictionary with all parameters

    Returns:
        catalog_df, trained_model
    """
    # Train concept model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224',
                                                  size={"height": resolution, "width": resolution})

    dataset.attach_image_processor(processor)

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Features: {dataset.get_concept_names()}")

    # Initialize model
    model = RobotConceptViT(num_classes=len(dataset.get_concept_names()))

    # Initialize trainer
    trainer = RobotConceptTrainer(model, device)

    # Train
    trainer.train(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-4,
        val_split=0.2
    )

    return model

if __name__ == '__main__':
    params = {
        'samples_per_instance': 1, # how many times to repeat each robot with changed colors (irrelavant feature); max 108
        'draw': True,
        'output_directory': './robot_images',
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
        'spurious_features': ['has_elbows', 'hand_shape'], # features that do not appear in the catalog + color
        'model': "'glorp' if (int(row['body_shape']=='square') + int(row['foot_shape']=='pointy') - 2 >= 0) else 'drent'",
        'model_type': 'deterministic',  # 'deterministic', 'stochastic'
        'size': 'large', # 'small', 'large'
        'color_mode': 'color',  # 'greyscale', 'color'
        'train_concept_detector': False,
        'epochs': 50,
        'verbose': True
    }

    robot_dataset = create_synthetic_dataset(**params)