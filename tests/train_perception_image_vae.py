#  Use argument parser to set arguments of experiment name
import argparse
import inspect
import math

# import module for random sampling
import random
import time
from collections import deque
from pathlib import Path
from random import shuffle
from typing import Tuple

import colav_simulator.common.paths as cs_dp
import colav_simulator.scenario_generator as cs_sg
import rl_rrt_mpc.common.paths as rl_dp
import tensorflow as tf
import torch as th
import torch.nn as nn
import torchvision
import yaml
from rl_rrt_mpc.common.datasets import PerceptionImageDataset
from rl_rrt_mpc.networks.variational_autoencoder import VAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="default")
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--load_model_path", type=str, default=None)

# BASE_PATH: Path = Path("/home/doctor/Desktop/machine_learning/data")
BASE_PATH: Path = Path("/Users/trtengesdal/Desktop/machine_learning/data")
EXPERIMENT_NAME: str = "vae_training1"
EXPERIMENT_PATH = BASE_PATH / EXPERIMENT_NAME
SAVE_MODEL_FILE: Path = BASE_PATH / "models"  # "_epochxx.pth" appended in training
LOAD_MODEL_FILE: Path = BASE_PATH / "vae_models" / "first.pth"  # "_epochxx.pth" appended in training

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("GPU error")
        print(e)


device = th.device("cuda")
device0 = th.device("cuda:0")


class RunningLoss:
    def __init__(self, batch_size: int) -> None:
        self.batch_size: int = batch_size
        self.aggregated_loss: float = 0.0
        self.n_samples: int = 0
        self.average_loss: float = 0.0
        self.reset()

    def update(self, loss: float) -> None:
        self.aggregated_loss += loss / self.batch_size
        self.n_samples += 1
        self.average_loss = self.aggregated_loss / (self.n_samples)

    def reset(self):
        self.aggregated_loss = 0.0
        self.n_samples = 0
        self.average_loss = 0.0


def make_grid_for_tensorboard(images_list: list, n_grids: int = 2):
    """ """
    joined_images = []
    for images in images_list:
        joined_images.extend(images[:n_grids])
    return torchvision.utils.make_grid(joined_images, nrow=n_grids, padding=5)


def get_noise(means, std_dev, const_multiplier) -> th.Tensor:
    """ """
    return const_multiplier * th.normal(means, std_dev)


def process_for_training(
    input_image: th.Tensor, filled_input_image: th.Tensor, max_intensity: int = 250, min_intensity: float = 0.0
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    Function to process the input image for training
    """
    processed_input_image = input_image.clone()
    processed_filled_input_image = filled_input_image.clone()

    processed_input_image[processed_input_image > max_intensity] = max_intensity
    processed_input_image[processed_input_image < min_intensity] = -1.0
    processed_input_image = processed_input_image / max_intensity
    processed_input_image[processed_input_image < 0] = -1.0

    processed_filled_input_image = th.clamp(processed_filled_input_image, min=0, max=max_intensity)
    processed_filled_input_image[processed_filled_input_image < min_intensity] = max_intensity
    processed_filled_input_image = processed_filled_input_image / max_intensity

    processed_input_image_with_noise = processed_input_image.clone()
    image_to_reconstruct = processed_input_image.clone()
    if ADD_NOISE_TO_INPUT:
        std_dev = th.zeros_like(input_image)
        std_dev[:] = (
            input_image * min_intensity / max_intensity
        )  # interpret this as: std_dev at max depth = 0.15m. std_dev at min depth = 0.0m. linearly increasing in between.
        processed_input_image_with_noise = image_to_reconstruct + get_noise(
            th.zeros_like(image_to_reconstruct), th.ones_like(image_to_reconstruct), std_dev
        )
        processed_input_image_with_noise[processed_input_image_with_noise > 1.0] = 1.0
        processed_input_image_with_noise[input_image < 0] = -1.0

    return processed_input_image_with_noise, image_to_reconstruct, processed_input_image, processed_filled_input_image


def train_vae(
    model: VAE,
    training_dataloader: DataLoader,
    test_dataloader: DataLoader,
    writer: SummaryWriter,
    n_epochs: int,
    batch_size: int,
    optimizer: th.optim.Adam,
    save_interval: int = 10,
) -> None:
    """Trains the variation autoencoder model.

    Args:
        model (VAE): The VAE model to train
        training_dataloader (DataLoader): The training dataloader
        test_dataloader (DataLoader): The test dataloader
        writer (SummaryWriter): The tensorboard writer
        n_epochs (int): The number of epochs to train the model
        batch_size (int): The batch size
        optimizer (th.optim.Adam): The optimizer, typically Adam.
        save_interval (int, optional): The interval at which to save the model. Defaults to 10.
    """
    th.autograd.set_detect_anomaly(True)
    model.train()
    reconstruction_loss = nn.MSELoss()
    kullback_leibler_loss = nn.KLDivLoss()

    loss_meter = RunningLoss(batch_size)
    n_batches = int(len(training_dataloader) / batch_size)
    n_test_batches = int(len(test_dataloader) / batch_size)

    # Create training data + test data
    for epoch in range(n_epochs):
        model.train()
        epoch_start_time = time.time()

        for batch_idx, (n_envs, images_per_env, perception_images, filtered_images) in enumerate(training_dataloader):
            batch_start_time = time.time()
            model.zero_grad()
            optimizer.zero_grad()

            # Process the input image for training
            (
                noisy_image,
                image_to_reconstruct,
                processed_input_image,
                processed_filled_input_image,
            ) = process_for_training(perception_images, filtered_images)

            # Forward pass
            reconstructed_image, means, log_vars, sampled_latens_vars = model(noisy_image)
            mse_loss = reconstruction_loss(reconstructed_image, image_to_reconstruct)
            kld_loss = kullback_leibler_loss(means, log_vars)
            loss = mse_loss + kld_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the tensorboard
            if batch_idx % save_interval == 0 and batch_idx != 0:
                writer.add_scalar("Training/Loss", loss.item() / batch_size, epoch * n_batches + batch_idx)
                writer.add_scalar("Training/KLD Loss", kld_loss.item() / batch_size, epoch * n_batches + batch_idx)
                print(
                    f"[TRAINING] Epoch: {epoch}/{n_epochs} Batch: {batch_idx}/{n_batches} Avg. Train Loss: {loss.item()/batch_size:.4f}, KL Div Loss.: {kld_loss.item()/batch_size:.4f}"
                    f"Time: {time.time() - batch_start_time:.2f}s, Est. time remaining: {(n_batches - batch_idx)*avg_iter_time :.2f}s"
                )

                # add image to the tensorboard
                grid = make_grid_for_tensorboard(
                    [
                        filled_filtered_data,
                        depth_data_to_reconstruct,
                        noisy_image,
                        th.sigmoid(reconstructed_image),
                        semantic_data,
                    ],
                    n_grids=4,
                )
                writer.add_image("training/images", grid, global_step=epoch * n_batches + batch_idx)
                if batch_idx % (5 * save_interval) == 0:
                    torchvision.utils.save_image(
                        grid,
                        EXPERIMENT_PATH
                        + "/training_images"
                        + "/"
                        + EXPERIMENT_NAME
                        + "_epoch_"
                        + str(epoch)
                        + "_batch_"
                        + str(batch_idx)
                        + ".png",
                    )

        # Print the statistics
        print("Epoch: %d, Loss: %.4f, Time: %.4f" % (epoch, loss_meter.average_loss, time.time() - epoch_start_time))
        # Reset the loss meter
        loss_meter.reset()
        print("Saving model...")
        save_path = f"{str(EXPERIMENT_PATH)}/models/{EXPERIMENT_NAME}_LD_{LATENT_DIM}_epoch_{epoch}.pth"
        th.save(
            model.state_dict(),
            save_path,
        )
        print("[DONE] Savng model at ", str(EXPERIMENT_PATH) + "/models")

        # # Evaluate the model
        model.eval()
        model.set_inference_mode(True)
        for batch_idx, (num_envs, images_per_env, depth_data, filtered_data, semantic_data) in enumerate(
            test_dataloader
        ):
            model.zero_grad()
            optimizer.zero_grad()

            depth_data = depth_data.to(device).unsqueeze(1)
            filtered_data = filtered_data.to(device).unsqueeze(1)
            semantic_data = semantic_data.to(device).unsqueeze(1)

            noisy_image, depth_data_to_reconstruct, filtered_data, filled_filtered_data, semantic_data = (
                process_for_training(depth_data, filtered_data, semantic_data)
            )

            # Forward pass
            reconstructed_image, means, log_vars, sampled_latent_vars = model(noisy_image)

            mse_loss = reconstruction_loss(reconstructed_image, image_to_reconstruct)
            kld_loss = kullback_leibler_loss(means, log_vars)
            loss = mse_loss + kld_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            grid = make_grid_for_tensorboard(
                [
                    filled_filtered_data,
                    depth_data_to_reconstruct,
                    noisy_image,
                    th.sigmoid(reconstructed_image),
                    semantic_data,
                ],
                n_grids=4,
            )
            writer.add_image("testing/images", grid, global_step=epoch)
            if batch_idx % (save_interval) == 0 and batch_idx != 0:
                print(
                    f"[TESTING] Epoch: {epoch}/{n_epochs} Batch: {batch_idx}/{n_test_batches} Avg. Train Loss: {loss.item()/batch_size:.4f}, KL Div Loss.: {kld_loss.item()/batch_size:.4f}"
                )
                torchvision.utils.save_image(
                    grid,
                    EXPERIMENT_PATH
                    + "/testing_images"
                    + "/"
                    + EXPERIMENT_NAME
                    + "_epoch_"
                    + str(epoch)
                    + "_batch_"
                    + str(batch_idx)
                    + ".png",
                )
                # Update the tensorboard
                writer.add_scalar("Test/Loss", loss.item() / batch_size, epoch * n_test_batches + batch_idx)
                writer.add_scalar("Test/KL Div Loss", kld_loss.item() / batch_size, epoch * n_test_batches + batch_idx)

        # Print the statistics
        print("Test Loss:", loss_meter.average_loss)
        loss_meter.reset()

    return model


if __name__ == "__main__":
    latent_dim = 64
    vae = VAE(n_input_channels=3, latent_dim=latent_dim).to(device)

    save_model = False
    load_model = False
    save_interval = 10  # Save the model every 10 batches
    batch_size = 64
    num_epochs = 40
    learning_rate = 1e-4

    FILL_UNDEFINED_PIXELS_WITH_NEGATIVE_VALUES = True
    ADD_NOISE_TO_INPUT = False

    # data_dir = Path("/home/doctor/Desktop/machine_learning/data/vae/")
    data_dir = Path("/Users/trtengesdal/Desktop/machine_learning/data/vae/")

    training_dataset = PerceptionImageDataset(npy_file="perception.npy", data_dir=data_dir / "training")
    summary(vae, (3, 400, 400))
    test_dataset = PerceptionImageDataset(npy_file="test.npy", data_dir=data_dir / "test")

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter()
    optimizer = th.optim.Adam(vae.parameters(), lr=learning_rate)
    train_vae(
        model=vae,
        training_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        writer=writer,
        n_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        save_interval=save_interval,
    )
