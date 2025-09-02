import subprocess
import rich
import numpy as np
import matplotlib.pyplot as plt
from rich.panel import Panel
import sys
import os
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from transformers import ViTModel
from torchvision.models import ResNet

# --- repo path shim (safe if already installed) ---
sys.path.append(os.getcwd())

from concept_benchmark.models import ConceptDetector
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.sudoku import create_sudoku_dataset, image_transform

device = torch.device(
    "cuda" if torch.cuda.is_available() \
        else ("mps" if torch.backends.mps.is_available() 
              else "cpu")
)

sys.path.append(os.getcwd())
from concept_benchmark.paths import results_dir
from concept_benchmark.ext import fileutils
from concept_benchmark.models import ConceptBasedModel
from concept_benchmark.metrics import calc_metric

class ViTWrapper(torch.nn.Module):
    def __init__(self, model=None):
        super(__class__, self).__init__()
        self.vit = model if model else \
            ViTModel.from_pretrained("google/vit-base-patch16-224")

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]  # Use the CLS token representation

settings = {
    "dataset_name": "test",
    "n": 3,
    "n_samples": 1000,
    "valid_ratio": 0.5,
    "max_corrupt": 3,
    "data_type": "image",
    "transform": image_transform,
    "model_type": "vit"
}

if settings["model_type"] == "vit_384":
    IMG_SIZE = 384 
else:
    IMG_SIZE = 224 # use 384 if you switch to a 384 ViT

data = create_sudoku_dataset(**settings) 
data.generate_cvindices(seed=42)
data.split("K05N01", fold_num_validation=4, fold_num_test=5)

embed_model = ViTWrapper().to(device)
model = ConceptDetector(
    embedding_model=embed_model
)

model.fit(
    data.training,
    data.validation, 
    freeze=True, 
    embed_params={"device": device},
    fit_params={"epochs": 10, "device": "cpu"}
)
#TODO: save fit

c_pred = model.predict(data.test, embed_params={"device": device}) > 0.5
accuracy = (c_pred == data.test.C)
accuracy_per_concept = accuracy.sum(axis=0) / accuracy.shape[0]
print("Concept-wise accuracy:", accuracy_per_concept)