#  Use argument parser to set arguments of experiment name
import argparse
import inspect
import time
from pathlib import Path
from sys import platform

import rl_rrt_mpc.common.datasets as rl_ds
import rl_rrt_mpc.networks.loss_functions as loss_functions
import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
import yaml
from rl_rrt_mpc.common.datasets import PerceptionImageDataset
from rl_rrt_mpc.networks.variational_autoencoder import VAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

if platform == "linux" or platform == "linux2":
    BASE_PATH: Path = Path("/home/doctor/Desktop/machine_learning/data/vae/")
elif platform == "darwin":
    BASE_PATH: Path = Path("/Users/trtengesdal/Desktop/machine_learning/data/vae/")

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="default")
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--load_model_path", type=str, default=None)

EXPERIMENT_NAME: str = "training1"
EXPERIMENT_PATH: Path = BASE_PATH / EXPERIMENT_NAME
SAVE_MODEL_FILE: Path = BASE_PATH / "models"  # "_epochxx.pth" appended in training
LOAD_MODEL_FILE: Path = BASE_PATH / "models" / "first.pth"  # "_epochxx.pth" appended in training


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


def make_grid_for_tensorboard(batch_images, reconstructed_images, n_rows: int = 2):
    """ """
    joined_images = []
    # joined_images.extend([batch_images[i, 2, :, :].unsqueeze(0) for i in range(len(batch_images.shape[1]))])
    # joined_images.extend([reconstructed_images[i, 2, :, :].unsqueeze(0) for i in range(len(reconstructed_images))])
    # joined_images.extend([batch_images[i, 1, :, :].unsqueeze(0) for i in range(len(batch_images))])
    # joined_images.extend([reconstructed_images[i, 1, :, :].unsqueeze(0) for i in range(len(reconstructed_images))])
    # joined_images.extend([batch_images[i, 0, :, :].unsqueeze(0) for i in range(len(batch_images))])
    # joined_images.extend([reconstructed_images[i, 0, :, :].unsqueeze(0) for i in range(len(reconstructed_images))])
    for j in range(len(batch_images)):
        for i in reversed(range(3)):
            joined_images.append(batch_images[j, i, :, :].unsqueeze(0))
            joined_images.append(reconstructed_images[j, i, :, :].unsqueeze(0))
    # grid = torchvision.utils.make_grid(
    #     [

    #         # batch_images[:n_batch_images_to_show, 0, :, :],  # .unsqueeze(0),
    #         # reconstructed_images[:n_batch_images_to_show, 0, :, :],  # .unsqueeze(0),
    #         # batch_images[:n_batch_images_to_show, 1, :, :],  # .unsqueeze(0),
    #         # reconstructed_images[:n_batch_images_to_show, 1, :, :],  # .unsqueeze(0),
    #         # batch_images[:n_batch_images_to_show, 2, :, :],  # .unsqueeze(0),
    #         # reconstructed_images[:n_batch_images_to_show, 2, :, :],  # .unsqueeze(0),
    #     ],
    #     nrow=6,
    #     padding=2,
    # )

    return torchvision.utils.make_grid(joined_images, nrow=n_rows, padding=2)


def get_noise(means, std_dev, const_multiplier) -> torch.Tensor:
    """ """
    return const_multiplier * torch.normal(means, std_dev)


def train_vae(
    model: VAE,
    training_dataloader: DataLoader,
    test_dataloader: DataLoader,
    writer: SummaryWriter,
    n_epochs: int,
    batch_size: int,
    optimizer: torch.optim.Adam,
    save_interval: int = 10,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Trains the variation autoencoder model.

    Args:
        model (VAE): The VAE model to train
        training_dataloader (DataLoader): The training dataloader
        test_dataloader (DataLoader): The test dataloader
        writer (SummaryWriter): The tensorboard writer
        n_epochs (int): The number of epochs to train the model
        batch_size (int): The batch size
        optimizer (torch.optim.Adam): The optimizer, typically Adam.
        save_interval (int, optional): The interval at which to save the model. Defaults to 10.
        device (torch.device, optional): The device to train the model on. Defaults to "cpu".
    """
    torch.autograd.set_detect_anomaly(True)
    model.train()

    loss_meter = RunningLoss(batch_size)
    n_batches = int(len(training_dataloader))
    n_test_batches = int(len(test_dataloader))

    train_images_path = EXPERIMENT_PATH / "training_images"
    if not train_images_path.exists():
        train_images_path.mkdir()
    test_images_path = EXPERIMENT_PATH / "testing_images"
    if not test_images_path.exists():
        test_images_path.mkdir()
    model_path = EXPERIMENT_PATH / "models"
    if not model_path.exists():
        model_path.mkdir()

    n_batch_images_to_show = batch_size if batch_size < 7 else 6
    beta = 20.0
    n_channels, H, W = model.input_image_dim
    beta_norm = beta * model.latent_dim / (n_channels * H * W)

    # Create training data + test data
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        for batch_idx, (batch_images) in enumerate(training_dataloader):
            batch_start_time = time.time()
            model.zero_grad()
            optimizer.zero_grad()

            batch_images = batch_images.to(device)

            # Forward pass
            log_sigma_opt = 0.0
            reconstructed_images, means, log_vars, sampled_latent_vars = model(batch_images)
            mse_loss, log_sigma_opt = loss_functions.sigma_reconstruction(reconstructed_images, batch_images)
            kld_loss = loss_functions.kullback_leibler_divergence(means, log_vars)
            # kld_loss = beta_norm * kld_loss
            loss = mse_loss + kld_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the tensorboard
            if batch_idx % save_interval == 0:  # and batch_idx != 0:
                writer.add_scalar("Training/Loss", loss.item() / batch_size, epoch * n_batches + batch_idx)
                writer.add_scalar("Training/KLD Loss", kld_loss.item() / batch_size, epoch * n_batches + batch_idx)
                writer.add_scalar("Training/MSE Loss", mse_loss.item() / batch_size, epoch * n_batches + batch_idx)
                writer.add_scalar("Training/Sigma Opt", log_sigma_opt.exp() / batch_size, epoch * n_batches + batch_idx)
                print(
                    f"[TRAINING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_batches} | Avg. Train Loss: {loss.item() / batch_size:.4f} | KL Div. Loss: {kld_loss.item()/batch_size:.4f} | "
                    f"Batch processing time: {time.time() - batch_start_time:.2f}s | Est. time remaining: {(n_batches - batch_idx) * avg_iter_time * (n_epochs - epoch + 1) :.2f}s"
                )

                grid = make_grid_for_tensorboard(
                    batch_images[:n_batch_images_to_show],
                    reconstructed_images[:n_batch_images_to_show],
                    n_rows=6,
                )
                writer.add_image("training/images", grid, global_step=epoch * n_batches + batch_idx)
                if batch_idx % (5 * save_interval) == 0:
                    torchvision.utils.save_image(
                        grid,
                        train_images_path
                        / (EXPERIMENT_NAME + "_epoch_" + str(epoch) + "_batch_" + str(batch_idx) + ".png"),
                    )

        print(f"Epoch: {epoch} | Loss: {loss_meter.average_loss} | Time: {time.time() - epoch_start_time}")
        loss_meter.reset()
        print("Saving model...")
        save_path = f"{str(model_path)}/{EXPERIMENT_NAME}_LD_{model.latent_dim}_epoch_{epoch}.pth"
        torch.save(
            model.state_dict(),
            save_path,
        )
        print("[DONE] Savng model at ", str(model_path))

        model.eval()
        model.set_inference_mode(True)
        if True:
            continue

        for batch_idx, batch_images in enumerate(test_dataloader):
            model.zero_grad()
            optimizer.zero_grad()

            batch_images = batch_images.to(device)

            # Forward pass
            reconstructed_images, means, log_vars, sampled_latent_vars = model(batch_images)
            mse_loss = loss_functions.reconstruction(reconstructed_images, batch_images)
            kld_loss = loss_functions.kullback_leibler_divergence(means, log_vars)
            kld_loss = beta_norm * kld_loss
            loss = mse_loss + kld_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            grid = make_grid_for_tensorboard(
                batch_images[:n_batch_images_to_show],
                reconstructed_images[:n_batch_images_to_show],
                n_rows=6,
            )
            writer.add_image("testing/images", grid, global_step=epoch)
            if batch_idx + 1 % save_interval == 0 and batch_idx != 0:
                print(
                    f"[TESTING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_test_batches} | Avg. Train Loss: {loss.item()/batch_size:.4f} | KL Div Loss.: {kld_loss.item()/batch_size:.4f}"
                )
                torchvision.utils.save_image(
                    grid,
                    test_images_path / (EXPERIMENT_NAME + "_epoch_" + str(epoch) + "_batch_" + str(batch_idx) + ".png"),
                )
                # Update the tensorboard
                writer.add_scalar("Test/Loss", loss.item() / batch_size, epoch * n_test_batches + batch_idx)
                writer.add_scalar("Test/KL Div Loss", -kld_loss.item() / batch_size, epoch * n_test_batches + batch_idx)

        # Print the statistics
        print("Test Loss:", loss_meter.average_loss)
        loss_meter.reset()

    return model


if __name__ == "__main__":
    latent_dim = 64
    input_image_dim = (3, 400, 400)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_image_dim=input_image_dim, latent_dim=latent_dim).to(device)
    # summary(vae, (3, 400, 400))

    load_model = False
    save_interval = 10
    batch_size = 4
    num_epochs = 500
    learning_rate = 0.0001

    log_dir = EXPERIMENT_PATH / "logs"
    data_dir = Path("/home/doctor/Desktop/machine_learning/data/vae/")
    # data_dir = Path("/Users/trtengesdal/Desktop/machine_learning/data/vae/")
    training_npy_filename = "perception_images_rogaland_random_everything_vecenv"
    test_npy_filename = "perception_images_rogaland_random_everything_vecenv_test"

    training_transform = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch.uint8, scale=True),
            transforms_v2.RandomChoice(
                [
                    transforms_v2.ToDtype(torch.uint8, scale=True),
                    transforms_v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.RandomRotation(10),
                    transforms_v2.ElasticTransform(alpha=100, sigma=5),
                ],
                p=[0.5, 0.5, 0.5, 0.5, 0.5],
            ),
            transforms_v2.ToDtype(torch.float32, scale=True),
        ]
    )
    test_transform = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch.float32, scale=True),
        ]
    )

    training_dataset = rl_ds.PerceptionImageDataset(training_npy_filename + ".npy", data_dir, transform=test_transform)
    test_dataset = rl_ds.PerceptionImageDataset(test_npy_filename + ".npy", data_dir, transform=test_transform)

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training dataset length: {len(training_dataset)} | Test dataset length: {len(test_dataset)}")
    print(f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}")
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    training_config = {
        "base_path": BASE_PATH,
        "experiment_path": EXPERIMENT_PATH,
        "experiment_name": EXPERIMENT_NAME,
        "latent_dim": latent_dim,
        "input_image_dim": input_image_dim,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "save_interval": save_interval,
        "load_model": load_model,
    }
    with Path(EXPERIMENT_PATH / "config.yaml").open(mode="w", encoding="utf-8") as fp:
        yaml.dump(training_config, fp)

    train_vae(
        model=vae,
        training_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        writer=writer,
        n_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        save_interval=save_interval,
        device=device,
    )
