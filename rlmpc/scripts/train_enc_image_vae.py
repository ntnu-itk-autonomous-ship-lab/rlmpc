import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import optuna
import rlmpc.common.datasets as rl_ds
import rlmpc.common.helper_functions as rl_hf
import rlmpc.networks.loss_functions as loss_functions
import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
import yaml
from rlmpc.common.running_loss import RunningLoss

# from rlmpc.networks.enc_vae_128.vae import VAE
from rlmpc.networks.enc_vae_128_simple.vae import VAE
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_vae(
    model: VAE,
    training_dataloader: DataLoader,
    test_dataloader: DataLoader,
    writer: SummaryWriter,
    n_epochs: int,
    batch_size: int,
    optimizer: torch.optim.Adam,
    lr_schedule: torch.optim.lr_scheduler,
    experiment_path: Path,
    experiment_name: str,
    save_interval: int = 10,
    device: torch.device = torch.device("cpu"),
    early_stopping_patience: int = 10,
    save_intermittent_models: bool = False,
    verbose: bool = False,
) -> Tuple[VAE, float, int, List[List[float]], List[List[float]]]:
    """Trains the variation autoencoder model.

    Args:
        model (VAE): The VAE model to train
        training_dataloader (DataLoader): The training dataloader
        test_dataloader (DataLoader): The test dataloader
        writer (SummaryWriter): The tensorboard writer
        n_epochs (int): The number of epochs to train the model
        batch_size (int): The batch size
        optimizer (torch.optim.Adam): The optimizer, typically Adam.
        lr_schedule: (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts): The learning rate scheduler
        experiment_path (Path): The path to the experiment directory
        experiment_name (str): The name of the experiment
        save_interval (int, optional): The interval at which to save the model. Defaults to 10.
        device (torch.device, optional): The device to train the model on. Defaults to "cpu".
        early_stopping_patience (int, optional): The number of epochs to wait before stopping training if the loss does not decrease. Defaults to 10.
        save_intermittent_models (bool, optional): Whether to save the model at each epoch. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Tuple[VAE, float, int, List[List[float]], List[List[float]]]: The trained model, the best test loss, the epoch at which the best test loss occurred, the training losses, and the testing losses.
    """
    torch.autograd.set_detect_anomaly(True)

    loss_meter = RunningLoss(batch_size)
    n_batches = int(len(training_dataloader))
    n_test_batches = int(len(test_dataloader))

    train_images_path = experiment_path / "training_images"
    if not train_images_path.exists():
        train_images_path.mkdir()
    test_images_path = experiment_path / "testing_images"
    if not test_images_path.exists():
        test_images_path.mkdir()
    model_path = experiment_path / "models"
    if not model_path.exists():
        model_path.mkdir()

    n_batch_images_to_show = 12
    beta = 5.0
    n_channels, H, W = model.input_image_dim
    beta_norm = beta * model.latent_dim / (n_channels * H * W)

    best_test_loss = 1e20
    best_epoch = 0
    random_indices = torch.randint(0, batch_size, (n_batch_images_to_show,))

    training_losses = []
    testing_losses = []

    # Create training data + test data
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        training_batch_losses = []

        model.train()
        model.set_inference_mode(False)
        for batch_idx, (batch_images, semantic_masks) in enumerate(training_dataloader):
            batch_start_time = time.time()
            optimizer.zero_grad()

            batch_images = batch_images.to(device)
            semantic_masks = semantic_masks.to(device)

            # Forward pass
            log_sigma_opt = 0.0
            reconstructed_images, means, log_vars, sampled_latent_vars = model(batch_images)
            # mse_loss, log_sigma_opt = loss_functions.sigma_semantically_weighted_reconstruction(
            #     reconstructed_images, batch_images, semantic_masks
            # )
            mse_loss = loss_functions.semantically_weighted_reconstruction(
                reconstructed_images, batch_images, semantic_masks
            )
            kld_loss = loss_functions.kullback_leibler_divergence(means, log_vars)
            kld_loss = kld_loss * beta_norm
            loss = mse_loss + kld_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)
            training_batch_losses.append(loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the tensorboard
            if batch_idx % save_interval == 0:
                writer.add_scalar("Training/Loss", loss.item() / batch_size, epoch * n_batches + batch_idx)
                writer.add_scalar("Training/KLD Loss", kld_loss.item() / batch_size, epoch * n_batches + batch_idx)
                writer.add_scalar("Training/MSE Loss", mse_loss.item() / batch_size, epoch * n_batches + batch_idx)
                # writer.add_scalar("Training/Sigma Opt", log_sigma_opt.exp() / batch_size, epoch * n_batches + batch_idx)
                if verbose:
                    print(
                        f"[TRAINING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_batches} | Avg. Train Loss: {loss.item() / batch_size:.4f} | KL Div. Loss: {kld_loss.item()/batch_size:.4f} | "
                        f"Batch processing time: {time.time() - batch_start_time:.2f}s | Est. time remaining: {(n_batches - batch_idx) * avg_iter_time * (n_epochs - epoch + 1) :.2f}s"
                    )

                grid = rl_hf.make_grid_for_tensorboard(
                    batch_images[random_indices],
                    reconstructed_images[random_indices],
                    semantic_masks[random_indices],
                    n_rows=6,
                )
                writer.add_image("training/images", grid, global_step=epoch * n_batches + batch_idx)
                if batch_idx % (5 * save_interval) == 0:
                    torchvision.utils.save_image(
                        grid,
                        train_images_path
                        / (experiment_name + "_epoch_" + str(epoch) + "_batch_" + str(batch_idx) + ".png"),
                    )

        lr_schedule.step()

        if verbose:
            print(f"Epoch: {epoch} | Loss: {loss_meter.average_loss} | Time: {time.time() - epoch_start_time}")
        loss_meter.reset()

        if save_intermittent_models:
            save_path = f"{str(model_path)}/{experiment_name}_LD_{model.latent_dim}_epoch_{epoch}.pth"
            torch.save(
                model.state_dict(),
                save_path,
            )

        training_losses.append(training_batch_losses)

        model.eval()
        model.set_inference_mode(True)
        test_batch_losses = []
        for batch_idx, (batch_images, semantic_masks) in enumerate(test_dataloader):
            model.zero_grad()
            optimizer.zero_grad()

            batch_start_time = time.time()

            batch_images = batch_images.to(device)
            semantic_masks = semantic_masks.to(device)

            # Forward pass
            log_sigma_opt = 0.0
            reconstructed_images, means, log_vars, sampled_latent_vars = model(batch_images)
            # mse_loss, log_sigma_opt = loss_functions.sigma_semantically_weighted_reconstruction(
            #     reconstructed_images, batch_images, semantic_masks
            # )
            mse_loss = loss_functions.semantically_weighted_reconstruction(
                reconstructed_images, batch_images, semantic_masks
            )
            kld_loss = loss_functions.kullback_leibler_divergence(means, log_vars)
            kld_loss = kld_loss * beta_norm
            loss = mse_loss + kld_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)
            test_batch_losses.append(loss.item())

            if batch_idx % save_interval == 0:
                if verbose:
                    print(
                        f"[TESTING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_test_batches} | Avg. Test Loss: {loss.item()/batch_size:.4f} | KL Div Loss.: {kld_loss.item()/batch_size:.4f}"
                    )
                writer.add_scalar("Test/Loss", loss.item() / batch_size, epoch * n_test_batches + batch_idx)
                writer.add_scalar("Test/KL Div Loss", -kld_loss.item() / batch_size, epoch * n_test_batches + batch_idx)

                grid = rl_hf.make_grid_for_tensorboard(
                    batch_images[random_indices],
                    reconstructed_images[random_indices],
                    semantic_masks[random_indices],
                    n_rows=6,
                )
                writer.add_image("testing/images", grid, global_step=epoch * n_batches + batch_idx)
                if batch_idx % (5 * save_interval) == 0:
                    torchvision.utils.save_image(
                        grid,
                        test_images_path
                        / (experiment_name + "_epoch_" + str(epoch) + "_batch_" + str(batch_idx) + ".png"),
                    )

        testing_losses.append(test_batch_losses)

        if verbose:
            print("Test Loss:", loss_meter.average_loss)
        if loss_meter.average_loss < best_test_loss:
            best_test_loss = loss_meter.average_loss
            best_epoch = epoch
            num_nondecreasing_loss_iters = 0
            if verbose:
                print(f"Current best model at epoch {best_epoch + 1} with test loss {best_test_loss}")
            best_model_path = f"{experiment_path}/{experiment_name}_model_LD_{model.latent_dim}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            torchvision.utils.save_image(
                grid,
                experiment_path
                / (experiment_name + "_test_epoch_" + str(epoch) + "_batch_" + str(batch_idx) + "_best.png"),
            )
        else:
            if verbose:
                print(f"Test loss has not decreased for {num_nondecreasing_loss_iters} iterations.")
            num_nondecreasing_loss_iters += 1
        loss_meter.reset()

        if num_nondecreasing_loss_iters > early_stopping_patience:
            if verbose:
                print(f"Test loss has not decreased for {early_stopping_patience} epochs. Stopping training.")
            break

    training_losses = np.array(training_losses)
    testing_losses = np.array(testing_losses)
    np.save(experiment_path / "training_losses.npy", training_losses)
    np.save(experiment_path / "testing_losses.npy", testing_losses)

    return model, best_test_loss, best_epoch


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for the VAE hyperparameter optimization.

    Args:
        trial (optuna.Trial): The optuna trial object used for optimization

    Returns:
        float: The loss value to minimize
    """
    BASE_PATH: Path = Path.home() / "Desktop/machine_learning/enc_vae/"
    latent_dim = trial.suggest_int("latent_dim", 12, 40)  # 40
    fc_dim = trial.suggest_int("fc_dim", 32, 512)  # 512
    encoder_conv_block_dims = [64, 128, 256]
    decoder_conv_block_dims = [256, 128, 128]
    input_image_dim = (1, 128, 128)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = VAE(
        input_image_dim=input_image_dim,
        latent_dim=latent_dim,
        fc_dim=fc_dim,
        encoder_conv_block_dims=encoder_conv_block_dims,
        decoder_conv_block_dims=decoder_conv_block_dims,
    ).to(device)

    experiment_name: str = "LD_" + str(latent_dim) + "_FC_" + str(fc_dim) + "_128x128"
    experiment_path: Path = BASE_PATH / experiment_name
    if not experiment_path.exists():
        experiment_path.mkdir(parents=True)

    log_dir = BASE_PATH / "logs"
    data_dir = Path.home() / "Desktop/machine_learning/enc_vae/data"

    training_data_list = []
    training_masks_list = []
    for i in range(8):
        training_data_list.append(f"perception_training_data_rogaland{i}.npy")
        training_masks_list.append(f"segmentation_masks_training_data_rogaland{i}.npy")

    test_data_list = []
    test_masks_list = []
    for i in range(8, 10):
        test_data_list.append(f"perception_training_data_rogaland{i}.npy")
        test_masks_list.append(f"segmentation_masks_training_data_rogaland{i}.npy")

    training_transform = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch.uint8, scale=True),
            transforms_v2.RandomChoice(
                [
                    transforms_v2.ToDtype(torch.float32, scale=True),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.ElasticTransform(alpha=50, sigma=3),
                ],
                p=[0.5, 0.5, 0.5],
            ),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize((input_image_dim[1], input_image_dim[2])),
        ]
    )
    test_transform = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch.uint8, scale=True),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize((input_image_dim[1], input_image_dim[2])),
        ]
    )
    training_dataset = torch.utils.data.ConcatDataset(
        [
            rl_ds.ENCImageDataset(
                data_npy_file=data_file, mask_npy_file=mask_file, data_dir=data_dir, transform=training_transform
            )
            for data_file, mask_file in zip(training_data_list, training_masks_list)
        ]
    )
    test_dataset = torch.utils.data.ConcatDataset(
        [
            rl_ds.ENCImageDataset(
                data_npy_file=data_file, mask_npy_file=mask_file, data_dir=data_dir, transform=test_transform
            )
            for data_file, mask_file in zip(test_data_list, test_masks_list)
        ]
    )

    save_interval = 10
    batch_size = 64
    num_epochs = 60
    learning_rate = 2e-04
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # print(f"Training dataset length: {len(training_dataset)} | Test dataset length: {len(test_dataset)}")
    # print(f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}")

    writer = SummaryWriter(log_dir=log_dir)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)

    training_config = {
        "base_path": BASE_PATH,
        "experiment_path": experiment_path,
        "experiment_name": experiment_name,
        "latent_dim": latent_dim,
        "input_image_dim": input_image_dim,
        "fc_dim": fc_dim,
        "encoder_conv_block_dims": encoder_conv_block_dims,
        "decoder_conv_block_dims": decoder_conv_block_dims,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "save_interval": save_interval,
    }
    with Path(experiment_path / "config.yaml").open(mode="w", encoding="utf-8") as fp:
        yaml.dump(training_config, fp)

    model, opt_loss, opt_epoch = train_vae(
        model=vae,
        training_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        writer=writer,
        n_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        experiment_name=experiment_name,
        experiment_path=experiment_path,
        save_interval=save_interval,
        device=device,
        early_stopping_patience=6,
        save_intermittent_models=False,
        verbose=True,
    )

    return opt_loss


def main():
    study_name = "enc_vae_hpo"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name,
        storage=storage_name,
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, n_trials=1000)

    print(f"Best objective value: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")


if __name__ == "__main__":
    main()
