import torch
from torchvision import transforms
from transformers import ViTModel

from concept_benchmark.models import ConceptDetector
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot import create_robot_image_dataset

device = torch.device(
    "cuda" if torch.cuda.is_available() \
        else ("mps" if torch.backends.mps.is_available()
              else "cpu")
)

params = {
    "samples_per_instance": 1,  # how many times to repeat each robot with changed colors (irrelavant feature); max 108
    "draw": False,
    "output_directory": results_dir / "robots",
    "concepts": {
        "head_shape": ["square", "round"],
        "body_shape": ["square", "round"],
        "has_knees": [True, False],
        "has_elbows": [True, False],
        "has_antennae": [True, False],
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

data = create_robot_image_dataset(**params)
data.transform = tf

data.generate_cvindices(seed=42)
data.split("K05N01", fold_num_validation=4, fold_num_test=5)

class ViTWrapper(torch.nn.Module):
    def __init__(self, model=None):
        super(__class__, self).__init__()
        self.vit = model if model else \
            ViTModel.from_pretrained("google/vit-base-patch16-224")

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]  # Use the CLS token representation


model = ConceptDetector(
    embedding_model=ViTWrapper(),
)
model.fit(
    data.training,
    data.validation,
    freeze=True,
    embed_params={"device": device},
    fit_params={"epochs": 10, "device": "cpu"}
)

c_pred = model.predict(data.test, embed_params={"device": device}) > 0.5
accuracy = (c_pred == data.test.C)
accuracy_per_concept = accuracy.sum(axis=0) / accuracy.shape[0]
print("Concept-wise accuracy:", accuracy_per_concept)