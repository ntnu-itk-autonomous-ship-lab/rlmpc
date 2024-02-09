"""
    datasets.py

    Summary:
        Contains classes for pytorch datasets.

    Author: Trym Tengesdal
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PerceptionImageDataset(Dataset):
    """Class for perception image dataset from the colav-environment."""

    def __init__(self, npy_file: Path, data_dir: Path, transform=transforms.ToTensor):
        self.data_dir = data_dir
        self.transform = transform
        self.data = np.load(data_dir / npy_file, allow_pickle=True).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name =

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
