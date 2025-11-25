#  Use argument parser to set arguments of experiment name
import argparse
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import rlmpc.common.datasets as rl_ds
import rlmpc.common.helper_functions as rl_hf
import rlmpc.networks.loss_functions as loss_functions
from rlmpc.common.running_loss import RunningLoss
from rlmpc.networks.vqvae2 import VQVAE

BASE_PATH: Path = Path.home() / "machine_learning/perception_vae/"

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="default")
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--load_model_path", type=str, default=None)

EXPERIMENT_NAME: str = "training_vqvae3"
EXPERIMENT_PATH: Path = BASE_PATH / EXPERIMENT_NAME
SAVE_MODEL_FILE: Path = BASE_PATH / "models"  # "_epochxx.pth" appended in training
LOAD_MODEL_FILE: Path = (
    BASE_PATH / "models" / "first.pth"
)  # "_epochxx.pth" appended in training


def train(
    model: VQVAE,
    training_dataloader: DataLoader,
    test_dataloader: DataLoader,
    writer: SummaryWriter,
    n_epochs: int,
    batch_size: int,
    optimizer: torch.optim.Adam,
    save_interval: int = 10,
    device: torch.device = torch.device("cpu"),
    input_image_dim: tuple = (3, 256, 256),
    early_stopping_patience: int = 10,
) -> None:
    """Trains the variation autoencoder model.

    Args:
        model (VQVAE): The VQVAE model to train
        training_dataloader (DataLoader): The training dataloader
        test_dataloader (DataLoader): The test dataloader
        writer (SummaryWriter): The tensorboard writer
        n_epochs (int): The number of epochs to train the model
        batch_size (int): The batch size
        optimizer (torch.optim.Adam): The optimizer, typically Adam.
        save_interval (int, optional): The interval at which to save the model. Defaults to 10.
        device (torch.device, optional): The device to train the model on. Defaults to "cpu".
        input_image_dim (tuple, optional): The input image dimensions. Defaults to (3, 256, 256).
        early_stopping_patience (int, optional): The number of epochs to wait before stopping training. Defaults to 10.
    """
    torch.autograd.set_detect_anomaly(True)

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

    n_batch_images_to_show = batch_size if batch_size < 4 else 4

    best_test_loss = 1e20
    best_epoch = 0

    # Create training data + test data
    num_nondecreasing_loss_iters = 0
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        model.train()
        for batch_idx, (batch_images, semantic_masks) in enumerate(training_dataloader):
            batch_start_time = time.time()
            model.zero_grad()
            optimizer.zero_grad()

            batch_images = batch_images.to(device)
            semantic_masks = semantic_masks.to(device)

            # Forward pass
            reconstructed_images = model(batch_images)
            vq_loss, commit_loss = model.get_vqvae_loss(batch_images)
            recon_loss = loss_functions.semantically_weighted_reconstruction(
                reconstructed_images, batch_images, semantic_masks
            )
            # recon_loss = loss_functions.vanilla_reconstruction(reconstructed_images, batch_images)
            loss = recon_loss + vq_loss + commit_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the tensorboard
            if batch_idx % save_interval == 0:
                writer.add_scalar(
                    "Training/Loss",
                    loss.item() / batch_size,
                    epoch * n_batches + batch_idx,
                )
                writer.add_scalar(
                    "Training/Reconstruction Loss",
                    recon_loss.item() / batch_size,
                    epoch * n_batches + batch_idx,
                )
                writer.add_scalar(
                    "Training/VQ Loss",
                    vq_loss.item() / batch_size,
                    epoch * n_batches + batch_idx,
                )
                writer.add_scalar(
                    "Training/Commitment Loss",
                    commit_loss.item() / batch_size,
                    epoch * n_batches + batch_idx,
                )
                print(
                    f"[TRAINING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_batches} | Losses (total, recon, vq): ({loss.item() / batch_size:.4f}, {recon_loss.item() / batch_size:.4f}, {vq_loss.item() / batch_size:.4f}) |"
                    f"Batch processing time: {time.time() - batch_start_time:.2f}s | Est. time remaining: {(n_batches - batch_idx) * avg_iter_time * (n_epochs - epoch + 1):.2f}s"
                )

                grid = rl_hf.make_grid_for_tensorboard(
                    batch_images[:n_batch_images_to_show],
                    reconstructed_images[:n_batch_images_to_show],
                    semantic_masks[:n_batch_images_to_show],
                    n_rows=6,
                )
                writer.add_image(
                    "training/images", grid, global_step=epoch * n_batches + batch_idx
                )
                if batch_idx % (5 * save_interval) == 0:
                    torchvision.utils.save_image(
                        grid,
                        train_images_path
                        / (
                            EXPERIMENT_NAME
                            + "_epoch_"
                            + str(epoch)
                            + "_batch_"
                            + str(batch_idx)
                            + ".png"
                        ),
                    )

        print(
            f"Epoch: {epoch + 1} | Loss: {loss_meter.average_loss} | Time: {time.time() - epoch_start_time}"
        )
        loss_meter.reset()
        print("Saving model...")
        save_path = f"{str(model_path)}/{EXPERIMENT_NAME}_LD_{model.embedding_dim}_NE_{model.num_embeddings}_NH_{model.hidden_dim}_epoch_{epoch}.pth"
        torch.save(
            model.state_dict(),
            save_path,
        )
        print("[DONE] Saving model at ", str(model_path))

        model.eval()
        for batch_idx, (batch_images, semantic_masks) in enumerate(test_dataloader):
            model.zero_grad()
            optimizer.zero_grad()

            batch_start_time = time.time()

            batch_images = batch_images.to(device)
            semantic_masks = semantic_masks.to(device)

            # Forward pass
            reconstructed_images = model(batch_images)
            vq_loss, commit_loss = model.get_vqvae_loss(batch_images)
            recon_loss = loss_functions.semantically_weighted_reconstruction(
                reconstructed_images, batch_images, semantic_masks
            )
            loss = recon_loss + vq_loss + commit_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            grid = rl_hf.make_grid_for_tensorboard(
                batch_images[:n_batch_images_to_show],
                reconstructed_images[:n_batch_images_to_show],
                semantic_masks[:n_batch_images_to_show],
                n_rows=6,
            )
            writer.add_image("testing/images", grid, global_step=epoch)
            if batch_idx % save_interval == 0:
                print(
                    f"[TESTING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_test_batches} | Losses (total, recon, vq): ({loss.item() / batch_size:.4f}, {recon_loss.item() / batch_size:.4f}, {vq_loss.item() / batch_size:.4f}) |"
                    f"Batch processing time: {time.time() - batch_start_time:.2f}s"
                )
                # Update the tensorboard
                writer.add_scalar(
                    "Testing/Loss",
                    loss.item() / batch_size,
                    epoch * n_test_batches + batch_idx,
                )
                writer.add_scalar(
                    "Testing/Reconstruction Loss",
                    recon_loss.item() / batch_size,
                    epoch * n_test_batches + batch_idx,
                )
                writer.add_scalar(
                    "Testing/VQ Loss",
                    vq_loss.item() / batch_size,
                    epoch * n_test_batches + batch_idx,
                )
                writer.add_scalar(
                    "Testing/Commitment Loss",
                    commit_loss.item() / batch_size,
                    epoch * n_test_batches + batch_idx,
                )
                if batch_idx % (5 * save_interval) == 0:
                    torchvision.utils.save_image(
                        grid,
                        test_images_path
                        / (
                            EXPERIMENT_NAME
                            + "_test_epoch_"
                            + str(epoch)
                            + "_batch_"
                            + str(batch_idx)
                            + ".png"
                        ),
                    )

        # Print the statistics
        print("Test Loss:", loss_meter.average_loss)
        if loss_meter.average_loss < best_test_loss:
            best_test_loss = loss_meter.average_loss
            best_epoch = epoch
            num_nondecreasing_loss_iters = 0
            print(
                f"Current best model at epoch {best_epoch + 1} with test loss {best_test_loss}"
            )
            best_model_path = f"{EXPERIMENT_PATH}/{EXPERIMENT_NAME}_model_LD_{model.embedding_dim}_NE_{model.num_embeddings}_NH_{model.hidden_dim}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            torchvision.utils.save_image(
                grid,
                EXPERIMENT_PATH
                / (
                    EXPERIMENT_NAME
                    + "_test_epoch_"
                    + str(epoch)
                    + "_batch_"
                    + str(batch_idx)
                    + "_best.png"
                ),
            )
        else:
            print(
                f"Test loss has not decreased for {num_nondecreasing_loss_iters} iterations."
            )
            num_nondecreasing_loss_iters += 1
        loss_meter.reset()

        if num_nondecreasing_loss_iters > early_stopping_patience:
            print(
                f"Test loss has not decreased for {early_stopping_patience} epochs. Stopping training."
            )
            break

    return model, best_test_loss, best_epoch


if __name__ == "__main__":
    input_image_dim = (3, 256, 256)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vqvae = VQVAE(
        channels=3,
        num_embeddings=2000,
        hidden_dim=256,
        embedding_dim=1,
        n_pixelcnn_res_blocks=2,
        n_pixelcnn_conv_blocks=2,
    ).to(device)

    # summary(vqvae, (3, 400, 400))

    load_model = False
    save_interval = 10
    batch_size = 32
    num_epochs = 40
    learning_rate = 2e-4

    log_dir = EXPERIMENT_PATH / "logs"
    data_dir = Path.home() / "machine_learning/data/vae/"
    training_data_npy_filename1 = "perception_data_rogaland_random_everything.npy"
    training_masks_npy_filename1 = "segmentation_masks_rogaland_random_everything.npy"

    training_data_npy_filename2 = (
        "perception_data_rogaland_random_everything_many_vessels.npy"
    )
    training_masks_npy_filename2 = (
        "segmentation_masks_rogaland_random_everything_many_vessels.npy"
    )
    test_data_npy_filename = "perception_data_rogaland_random_everything_test.npy"
    test_masks_npy_filename = "segmentation_masks_rogaland_random_everything_test.npy"

    training_transform = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch.uint8, scale=True),
            transforms_v2.RandomChoice(
                [
                    transforms_v2.ToDtype(torch.float32, scale=True),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.RandomRotation(2),
                    transforms_v2.RandomVerticalFlip(),
                    transforms_v2.ElasticTransform(alpha=30, sigma=3),
                ],
                p=[0.5, 0.5, 0.4, 0.2, 0.1],
            ),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize((input_image_dim[1], input_image_dim[2])),
        ]
    )
    test_transform = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize((input_image_dim[1], input_image_dim[2])),
        ]
    )

    training_dataset1 = rl_ds.PerceptionImageDataset(
        training_data_npy_filename1,
        data_dir,
        training_masks_npy_filename1,
        transform=training_transform,
    )
    training_dataset2 = rl_ds.PerceptionImageDataset(
        training_data_npy_filename2,
        data_dir,
        training_masks_npy_filename2,
        transform=training_transform,
    )
    training_dataset = torch.utils.data.ConcatDataset(
        [training_dataset1, training_dataset2]
    )

    test_dataset = rl_ds.PerceptionImageDataset(
        test_data_npy_filename,
        data_dir,
        test_masks_npy_filename,
        transform=test_transform,
    )
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(
        f"Training dataset length: {len(training_dataset)} | Test dataset length: {len(test_dataset)}"
    )
    print(
        f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}"
    )
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)

    training_config = {
        "base_path": BASE_PATH,
        "experiment_path": EXPERIMENT_PATH,
        "experiment_name": EXPERIMENT_NAME,
        "latent_dim": vqvae.embedding_dim,
        "num_embeddings": vqvae.num_embeddings,
        "num_hiddens": vqvae.hidden_dim,
        "num_residual_layers": vqvae.hidden_dim,
        "input_image_dim": input_image_dim,
        "architecture": "VQVAE3",
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "save_interval": save_interval,
        "load_model": load_model,
    }
    with Path(EXPERIMENT_PATH / "config.yaml").open(mode="w", encoding="utf-8") as fp:
        yaml.dump(training_config, fp)

    train(
        model=vqvae,
        training_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        writer=writer,
        n_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        save_interval=save_interval,
        device=device,
        early_stopping_patience=10,
    )
