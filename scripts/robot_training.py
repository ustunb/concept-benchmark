import torch
from torchvision import transforms
from transformers import ViTModel

from concept_benchmark.data import ConceptDataset
from concept_benchmark.models import ClassicalConceptDetector
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot_concepts.main import create_synthetic_dataset

params = {
    "samples_per_instance": 1,  # how many times to repeat each robot with changed colors (irrelavant feature); max 108
    "draw": False,
    "output_directory": results_dir / "robots",
    "concepts": {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": ["false", "true"],
        "has_elbows": ["false", "true"],
        "has_antennae": ["false", "true"],
        "ears_shape": ["square", "triangle"],
        "mouth_type": ["closed", "open"],
        "hand_shape": [
            "round_circle",
            "round_oval",
            "round_oval2",
            "edgy_triangle",
            "edgy_square",
            "edgy_trapezoid",
        ],
        "foot_shape": [
            "flat_4sided",
            "flat_5sided",
            "flat_lshaped",
            "pointy_3sided",
            "pointy_4sided",
            "pointy_6sided",
        ],
    },
    "spurious_features": [
        "has_elbows",
        "hand_shape",
    ],  # features that do not appear in the catalog + color
    "model": "'glorp' if (int(row['body_shape']=='square') + int(row['foot_shape']=='pointy') - 2 >= 0) else 'drent'",
    "model_type": "deterministic",  # 'deterministic', 'stochastic'
    "size": "large",  # 'small', 'large'
    "color_mode": "color",  # 'greyscale', 'color'
    "train_concept_detector": False,
    "epochs": 50,
    "verbose": True,
}


robot_data = create_synthetic_dataset(**params)


IMG_SIZE = 224  # use 384 if you switch to a 384 ViT

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

data = ConceptDataset(
    X=robot_data.X,
    C=robot_data.C,
    y=robot_data.y,
    meta={
        "data_type": "image",
        "concepts": robot_data.meta["concepts"],
        "classes": robot_data.meta["classes"],
    },
    base_dir=results_dir / "robots",
    transform_x=tf,
)

data.generate_cvindices(seed=42)
data.split("K05N01", fold_num_validation=4, fold_num_test=5)

vit = ViTModel.from_pretrained("google/vit-base-patch16-224")


class ViTWrapper(torch.nn.Module):
    def __init__(self, vit_model):
        super(__class__, self).__init__()
        self.vit = vit_model

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]  # Use the CLS token representation


embeded_data = data.embed(model=ViTWrapper(vit), device="mps")

model = ClassicalConceptDetector()
model.fit(data.training, data.validation)