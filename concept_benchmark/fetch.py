"""Fetch supported datasets."""
import os
from typing import Optional
import requests
import tarfile
from torchvision.io import decode_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from concept_benchmark.paths import DATA_DIR

LINKS = {
    "CUB":  {
        "url": "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
        "file_name": "CUB_200_2011.tgz"
    }
    # todo: add more datasets
    # todo: add option to redirect to huggingface datasets once we upload
}

# todo: add a base class once there is a shared pattern among dataset classes
class CUBDataset:

    def __init__(self, overwrite: Optional[bool]=False, training: Optional[bool]=True):
        """Initialize object."""
        self.dataset = "CUB"
        assert self.dataset in LINKS.keys(), "unsupported dataset"
        self.raw_path = DATA_DIR / LINKS[self.dataset]["file_name"]
        self.data_path = DATA_DIR / self.dataset
        self.overwrite = overwrite
        self.training = training # allow different transforms for train vs test

    def fetch(self):
        """Fetch raw data from url."""
        if not self.raw_path.exists() or self.overwrite:
            # fetch and write
            response = requests.get(LINKS[self.dataset]["file_name"])
            with open(self.raw_path, "wb") as f:
                f.write(response.content)

    def extract(self):
        """Extract data from raw file."""
        if self.data_path.exists() and not self.overwrite:
            return # data previously extracted
        if self.dataset == "CUB":
            with tarfile.open(self.raw_path, "r:gz") as tar:
                tar.extractall(self.data_path, filter='data')

    @property
    def transform(self):
        """Adapted from Koh et al, 2020."""
        resol = 299
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        if self.training:
            transformer = transforms.Compose([
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer
                ])
        else:
            transformer = transforms.Compose([
                transforms.CenterCrop(resol),
                transforms.ToTensor(),
                normalizer
            ])
        return transformer

    def batch_load(self, batch_size=32):
        """Load and transform raw data."""
        dataset = datasets.ImageFolder(root=self.data_path / "CUB_200_2011" / "images", transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for X, y in dataloader:
            yield X, y
