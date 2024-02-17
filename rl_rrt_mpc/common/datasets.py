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
from torchvision.transforms import v2 as transforms_v2


class UnNormalize(transforms_v2.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1.0 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


class PerceptionImageDataset(Dataset):
    """Class for perception image dataset from the colav-environment."""

    # consider augmenting dataset by flipping and rotating images, cropping, color manipulaton,

    def __init__(
        self,
        npy_file: Path,
        data_dir: Path,
        transform=transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
            ]
        ),
    ):
        """Initializes the dataset.
        Args:
            - npy_file (Path): The path to the npy file.
            - data_dir (Path): The path to the data directory in which the numpy file is found.
            - transform (transforms_v2): The transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = np.load(data_dir / npy_file, allow_pickle=True).astype(np.uint8)
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        assert idx < self.n_samples * self.n_envs, "Index out of range"
        env_idx = idx % self.n_envs
        sample_idx = idx // self.n_envs
        sample = torch.from_numpy(self.data[sample_idx, env_idx, :, :, :])

        if self.transform:
            sample = self.transform(sample)
        return sample
