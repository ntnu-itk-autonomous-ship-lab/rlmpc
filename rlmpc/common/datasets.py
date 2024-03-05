"""
    datasets.py

    Summary:
        Contains classes for pytorch datasets.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Optional, Tuple

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
        data_npy_file: str,
        data_dir: Path,
        mask_npy_file: Optional[str] = None,
        transform=transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
            ]
        ),
    ):
        """Initializes the dataset.
        Args:
            - data_npy_file (str): The name of the npy file containing the data.
            - mask_npy_file (str): The name of the npy file containing the segmentation masks.
            - data_dir (Path): The path to the data directory in which the numpy file is found.
            - transform (transforms_v2...): The transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = np.load(data_dir / data_npy_file, mmap_mode="r", allow_pickle=True).astype(np.uint8)
        # self.data = self.data[3:13, 0, :, :, :]  # disregard 3 first.
        self.masks = None
        if mask_npy_file is not None:
            self.masks = np.load(data_dir / mask_npy_file, mmap_mode="r", allow_pickle=True).astype(np.uint8)
            # self.masks = self.masks[3:13, 0, :, :, :]  # disregard 3 first.

        if len(self.data.shape) == 4:
            self.data = np.expand_dims(self.data, axis=0)
            self.masks = np.expand_dims(self.masks, axis=0) if self.masks is not None else None

        self.n_samples, self.n_envs, self.n_channels, self.height, self.width = self.data.shape

        self.unnormalize_transform = transforms_v2.Compose(
            [UnNormalize(mean=[0.5], std=[0.5]), transforms_v2.ToDtype(torch.uint8, scale=True)]
        )

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
        sample = torch.from_numpy(self.data[sample_idx, env_idx, :, :, :].copy().astype(np.uint8))
        if self.masks is not None:
            mask = torch.from_numpy(self.masks[sample_idx, env_idx, :, :, :].copy())
        else:
            mask = 255 * torch.ones_like(sample, dtype=torch.uint8)

        if self.transform:
            both_images = torch.cat((sample.unsqueeze(0), mask.unsqueeze(0)), 0)
            transformed = self.transform(both_images)
            sample = transformed[0]
            mask = transformed[1]
        assert not torch.isinf(sample).any(), "Sample contains inf"
        # print(f"Sample min: {sample.min()}, max: {sample.max()}")
        return sample, mask


# class PartialDataset(Dataset):
# """
# * Description: custom `Dataset` module for processing `.npy` files (N, C, H, W) (N > 1) grouped by date
# - i.e. mini-batched .npy file stored by date
# - Therefore, the number of samples, 'N', is different from each other...
# """
# def __init__(self, read_path, date, transform=None):
#     """
#     * Arguments:
#     - read_path (string): path of `.npy` files
#     - data (string): date(yymmdd) as a file name
#     - transform (callable, optional): optional transform to be applied on a sample
#     """
#     self.transform = transform
#     self.path = read_path
#     self.date = date

#     self.data = self.read_memmap(f'{os.path.join(self.path, self.date)}.npy')

# def read_memmap(self, file_name):
#     """
#     * Descripton: read `np.memmap` file from the directory

#     * Argument:
#     - file_name (string): path of '.npy' and '.npy.conf' files

#     * Output:
#     - whole data loaded in a memory-efficient manner (np.memmap)
#     """
#     with open(file_name + '.conf', 'r') as file:
#         memmap_configs = json.load(file)
#         return np.memmap(file_name, mode='r+', shape=tuple(memmap_configs['shape']), dtype=memmap_configs['dtype'])

# def __getitem__(self, index):
#     """
#     * Description: function for indexing samples

#     * Argument:
#     - index (int): index of the sample

#     * Output:
#     - input data, output data (torch.Tensor, torch.Tensor)
#     - (batch_size, 4 (Mask(0 - background, 1 - foreground) / input1 / input2 / input3), height, width), (batch_size, output, height, width)
#     """

#     mask = torch.Tensor(self.data[index, 0, :, :]).reshape(1, PATCH_HEIGHT, PATCH_WIDTH)
#     inputs = torch.Tensor(self.data[index, 2:4, :, :])
#     output = torch.Tensor(self.data[index, 1, :, :]).reshape(1, PATCH_HEIGHT, PATCH_WIDTH)

#     if self.transform is not None:
#         inputs = self.transform(inputs)

#     inputs = np.concatenate([mask, inputs], axis=0)
#     return (inputs, output)

# def __len__(self):
#     """
#     * Description: fucntion for noticing the length of dataset

#     * Output:
#     - length (int)
#     """
#     return self.data.shape[0]
