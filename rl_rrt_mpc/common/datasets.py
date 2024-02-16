"""
    datasets.py

    Summary:
        Contains classes for pytorch datasets.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PerceptionImageDataset(Dataset):
    """Class for perception image dataset from the colav-environment."""

    def __init__(self, npy_file: Path, data_dir: Path, transform=transforms.ToTensor) -> None:
        """Initializes the dataset.
        Args:
            - npy_file (Path): The path to the npy file.
            - data_dir (Path): The path to the data directory in which the numpy file is found.
            - transform (torchvision.transforms): The transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = np.load(data_dir / npy_file, allow_pickle=True).astype(np.int8)
        self.n_samples, self.n_envs, self.n_channels, self.height, self.width = self.data.shape

    def get_datainfo(self) -> Tuple[int, int, int, int, int]:
        """Returns the data information."""
        return self.n_samples, self.n_envs, self.n_channels, self.height, self.width

    def show_random_image(self):
        """Shows a random image from the dataset."""
        idx = np.random.randint(0, self.n_samples)
        env_idx = np.random.randint(0, self.n_envs)
        plt.imshow(self.data[idx, env_idx, 0, :, :])
        plt.show()

    def __len__(self):
        return self.n_samples * self.n_envs

    def __getitem__(self, idx: Tuple[int, int]) -> torch.Tensor:
        assert idx[0] < self.n_samples and idx[1] < self.n_envs, "Index out of range."

        sample = self.data[idx[0], idx[1], :, :, :]

        if self.transform:
            sample = self.transform(sample)
        return sample
