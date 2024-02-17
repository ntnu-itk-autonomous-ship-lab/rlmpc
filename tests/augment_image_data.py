from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.datasets as rl_ds
import torch
from torchvision.transforms import v2 as transforms_v2

# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


IMAGE_DATADIR = Path("/home/doctor/Desktop/machine_learning/data/vae/")
# IMAGE_DATADIR = Path("/Users/trtengesdal/Desktop/machine_learning/data/vae/training")
assert IMAGE_DATADIR.exists(), f"Directory {IMAGE_DATADIR} does not exist."


def plot_image(axes: plt.Axes, img: np.ndarray, title: str) -> None:
    """Plots an image."""
    axes.imshow(img, aspect="equal")
    axes.set_title(title)
    axes.axes.get_xaxis().set_visible(False)
    axes.axes.get_yaxis().set_visible(False)
    plt.tight_layout()


if __name__ == "__main__":

    NPY_FILE = "perception_images_rogaland_random_everything_vecenv_test.npy"

    dataset = rl_ds.PerceptionImageDataset(NPY_FILE, IMAGE_DATADIR)
    rotated_dataset = rl_ds.PerceptionImageDataset(
        NPY_FILE,
        IMAGE_DATADIR,
        transform=transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
                transforms_v2.RandomRotation(45),
                transforms_v2.ToDtype(torch.float16, scale=True),
                transforms_v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )
    flipped_dataset = rl_ds.PerceptionImageDataset(
        NPY_FILE,
        IMAGE_DATADIR,
        transform=transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
                transforms_v2.RandomVerticalFlip(p=0.5),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float16, scale=True),
                transforms_v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )
    elastic_dataset = rl_ds.PerceptionImageDataset(
        NPY_FILE,
        IMAGE_DATADIR,
        transform=transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
                transforms_v2.ElasticTransform(alpha=100, sigma=5),
                transforms_v2.ToDtype(torch.float16, scale=True),
                transforms_v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )
    affine_dataset = rl_ds.PerceptionImageDataset(
        NPY_FILE,
        IMAGE_DATADIR,
        transform=transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
                transforms_v2.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms_v2.ToDtype(torch.float16, scale=True),
                transforms_v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )

    unnnormalize_transform = transforms_v2.Compose(
        [rl_ds.UnNormalize(mean=[0.5], std=[0.5]), transforms_v2.ToDtype(torch.uint8, scale=True)]
    )

    for i in range(10):
        original_img = dataset[i].numpy()
        rotated_img = rotated_dataset[i].numpy()
        flipped_img = flipped_dataset[i].numpy()
        elastic_img = elastic_dataset[i].numpy()
        affine_img = affine_dataset[i].numpy()
        fig = plt.figure()
        axs = fig.subplot_mosaic(
            [
                ["original", "rotated"],
                ["flipped", "elastic"],
                ["affine", ""],
            ]
        )
        plot_image(axs["original"], original_img[0, :, :], "Original")
        plot_image(axs["rotated"], rotated_img[0, :, :], "Rotated")
        plot_image(axs["flipped"], flipped_img[0, :, :], "Flipped")
        plot_image(axs["elastic"], elastic_img[0, :, :], "Elastic")
        plot_image(axs["affine"], affine_img[0, :, :], "Affine")
        plt.show()
