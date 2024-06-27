#  Use argument parser to set arguments of experiment name
import argparse
import inspect
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytorch_warmup as warmup
import rlmpc.common.datasets as rl_ds
import rlmpc.networks.loss_functions as loss_functions
import torch
import torchvision.transforms.v2 as transforms_v2
import yaml
from rlmpc.networks.tracking_vae.vae import VAE
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

BASE_PATH: Path = Path.home() / "Desktop/machine_learning/tracking_vae/"

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="default")
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--load_model_path", type=str, default=None)


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


def train_vae(
    model: VAE,
    training_dataloader: DataLoader,
    test_dataloader: DataLoader,
    writer: SummaryWriter,
    n_epochs: int,
    batch_size: int,
    optimizer: torch.optim.Adam,
    lr_schedule: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    experiment_path: Path,
    save_interval: int = 10,
    device: torch.device = torch.device("cpu"),
    early_stopping_patience: int = 10,
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
        lr_schedule (torch.optim.lr_scheduler): The learning rate scheduler
        save_interval (int, optional): The interval at which to save the model. Defaults to 10.
        device (torch.device, optional): The device to train the model on. Defaults to "cpu".
        early_stopping_patience (int, optional): The number of epochs to wait before stopping training if the loss does not decrease. Defaults to 10.

    Returns:
        Tuple[VAE, float, int, List[List[float]], List[List[float]]]: The trained model, the best test loss, the epoch at which the best test loss occurred, the training losses, and the testing losses.
    """
    torch.autograd.set_detect_anomaly(True)

    loss_meter = RunningLoss(batch_size)
    n_batches = int(len(training_dataloader))
    n_test_batches = int(len(test_dataloader))

    model_path = experiment_path / "models"
    if not model_path.exists():
        model_path.mkdir()

    best_test_loss = 1e20
    best_train_loss = 1e20
    best_epoch = 0

    beta = 0.001
    training_losses = []
    testing_losses = []

    # warmup_period = n_batches
    # warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

    experiment_name = experiment_path.name

    # Create training data + test data
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        training_batch_losses = []

        model.train()
        model.set_inference_mode(False)
        for batch_idx, batch_obs in enumerate(training_dataloader):
            batch_start_time = time.time()
            model.zero_grad()
            optimizer.zero_grad()

            batch_obs = batch_obs.to(device)

            # extract length of valid obstacle observations
            seq_lengths = (
                torch.sum(batch_obs[:, 0, :] < 0.95, dim=1).to("cpu").type(torch.int64)
            )  # idx 0 is normalized distance, where vals = 1.0 is max dist of 1e4++ and thus not valid
            seq_lengths[seq_lengths == 0] = 1
            max_seq_length = seq_lengths.max().item()
            batch_obs = batch_obs[:, :, -max_seq_length:]
            batch_obs = batch_obs.permute(0, 2, 1)  # permute to (batch, max_seq_len, input_dim)

            # Forward pass
            reconstructed_obs, means, log_vars, _ = model(batch_obs, seq_lengths)
            mse_loss = loss_functions.reconstruction_rnn(reconstructed_obs, batch_obs, seq_lengths)
            kld_loss = loss_functions.kullback_leibler_divergence(means, log_vars)
            beta_norm = beta * model.latent_dim / (max_seq_length * 6)
            loss = mse_loss + beta_norm * kld_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)
            training_batch_losses.append(loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the tensorboard
            if batch_idx % save_interval == 0:
                # if epoch * n_batches + batch_idx < warmup_period:
                #     lr = warmup_scheduler.lrs[-1]
                # else:
                lr = lr_schedule.get_last_lr()[0]
                writer.add_scalar("Training/Learning Rate", lr, epoch * n_batches + batch_idx)
                writer.add_scalar("Training/Loss", loss.item(), epoch * n_batches + batch_idx)
                writer.add_scalar("Training/KLD Loss", kld_loss.item(), epoch * n_batches + batch_idx)
                writer.add_scalar("Training/MSE Loss", mse_loss.item(), epoch * n_batches + batch_idx)
                # writer.add_scalar("Training/Sigma Opt", log_sigma_opt.exp() / batch_size, epoch * n_batches + batch_idx)
                print(
                    f"[TRAINING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_batches} | Train Loss: {loss.item():.4f} | KL Div. Loss: {kld_loss.item():.4f} | "
                    f"Batch processing time: {time.time() - batch_start_time:.2f}s | Est. time remaining: {(n_batches - batch_idx) * avg_iter_time * (n_epochs - epoch + 1) :.2f}s"
                )
                # writer

            # with warmup_scheduler.dampening():
            #     if warmup_scheduler.last_step + 1 >= warmup_period:
        lr_schedule.step()

        print(f"Epoch: {epoch} | Loss: {loss_meter.average_loss} | Time: {time.time() - epoch_start_time}")
        loss_meter.reset()
        print("Saving model...")
        save_path = f"{str(model_path)}/{experiment_name}_epoch_{epoch}.pth"
        torch.save(
            model.state_dict(),
            save_path,
        )
        print("[DONE] Saving model at ", str(model_path))

        training_losses.append(training_batch_losses)

        model.eval()
        model.set_inference_mode(True)
        test_batch_losses = []
        for batch_idx, batch_obs in enumerate(test_dataloader):

            batch_obs = batch_obs.to(device)
            # extract length of valid obstacle observations
            seq_lengths = (
                torch.sum(batch_obs[:, 0, :] < 0.95, dim=1).to("cpu").type(torch.int64)
            )  # idx 0 is normalized distance, where vals = 1.0 is max dist of 1e4++ and thus not valid
            seq_lengths[seq_lengths == 0] = 1
            max_seq_length = seq_lengths.max().item()
            batch_obs = batch_obs[:, :, -max_seq_length:]
            batch_obs = batch_obs.permute(0, 2, 1)  # permute to (batch, max_seq_len, input_dim)

            # Forward pass
            reconstructed_obs, means, log_vars, _ = model(batch_obs, seq_lengths)
            mse_loss = loss_functions.reconstruction_rnn(reconstructed_obs, batch_obs, seq_lengths)
            kld_loss = loss_functions.kullback_leibler_divergence(means, log_vars)
            beta_norm = beta * model.latent_dim / (max_seq_length * 6)
            loss = mse_loss + beta_norm * kld_loss
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)
            test_batch_losses.append(loss.item())

            if batch_idx % save_interval == 0:
                print(
                    f"[TESTING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_test_batches} | Test Loss: {loss.item():.4f} | KL Div Loss.: {kld_loss.item()/batch_size:.4f}"
                )
                # Update the tensorboard
                writer.add_scalar("Test/Loss", loss.item(), epoch * n_test_batches + batch_idx)
                writer.add_scalar("Test/MSE Loss", mse_loss.item(), epoch * n_test_batches + batch_idx)
                writer.add_scalar("Test/KL Div Loss", kld_loss.item(), epoch * n_test_batches + batch_idx)

        testing_losses.append(test_batch_losses)
        weights = torch.ones_like(batch_obs)
        weights[torch.where(batch_obs[:, :, 0] > 0.98)] = 0.0
        diff_ex = (reconstructed_obs[0] - batch_obs[0]) * weights
        print(f"diff_example = {diff_ex[0]}")
        print(f"diff weights = {weights[0]}")
        # Print the statistics
        print("Test Loss:", loss_meter.average_loss)
        if loss_meter.average_loss < best_test_loss:
            best_test_loss = loss_meter.average_loss
            best_epoch = epoch
            num_nondecreasing_loss_iters = 0
            print(f"Current best model at epoch {best_epoch + 1} with test loss {best_test_loss}")
            best_model_path = f"{experiment_path}/{experiment_name}_best.pth"
            torch.save(model.state_dict(), best_model_path)

        else:
            print(f"Test loss has not decreased for {num_nondecreasing_loss_iters} iterations.")
            num_nondecreasing_loss_iters += 1

        if num_nondecreasing_loss_iters > early_stopping_patience:
            print(f"Test loss has not decreased for {early_stopping_patience} epochs. Stopping training.")
            break

    training_losses = np.array(training_losses)
    testing_losses = np.array(testing_losses)
    best_train_loss = np.min(np.mean(training_losses, axis=1))
    np.save(experiment_path / "training_losses.npy", training_losses)
    np.save(experiment_path / "testing_losses.npy", testing_losses)
    return model, best_test_loss, best_train_loss, best_epoch


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    latent_dims = [5, 10, 15, 20, 30]
    rnn_types = [torch.nn.GRU, torch.nn.LSTM]
    num_rnn_layers = 2
    fc_dim = 16
    input_dim = 6

    load_model = False
    save_interval = 20
    batch_size = 128
    num_epochs = 60
    learning_rate = 1e-4

    data_dir = Path("/home/doctor/Desktop/machine_learning/data/tracking_vae/")
    # data_dir = Path("/Users/trtengesdal/Desktop/machine_learning/data/vae/")
    training_data_npy_filename = "tracking_gru_training_data_rogaland.npy"
    test_data_npy_filename = "tracking_gru_test_data_rogaland.npy"
    training_data_npy_filename2 = "tracking_gru_training_data_rogaland2.npy"
    test_data_npy_filename2 = "tracking_gru_test_data_rogaland2.npy"
    training_data_npy_filename3 = "tracking_gru_training_data_rogaland3.npy"
    training_data_npy_filename4 = "tracking_gru_training_data_rogaland4.npy"
    training_data_npy_filename5 = "tracking_gru_training_data_rogaland5.npy"
    training_data_npy_filename6 = "tracking_gru_training_data_rogaland6.npy"
    training_data_npy_filename7 = "tracking_gru_training_data_rogaland7.npy"
    training_data_npy_filename8 = "tracking_gru_training_data_rogaland8.npy"
    training_data_npy_filename9 = "tracking_gru_training_data_rogaland9.npy"

    # training_transform = transforms_v2.Compose(
    #     [
    #         transforms_v2.ToDtype(torch.float32, scale=True),
    #         transforms_v2.RandomChoice(
    #             [
    #                 transforms_v2.ToDtype(torch.float32, scale=True),
    #                 transforms_v2.transforms_v2.ElasticTransform(alpha=50, sigma=3),
    #             ],
    #             p=[0.5, 0.5, 0.5],
    #         ),
    #         transforms_v2.ToDtype(torch.float32, scale=True),
    #     ]
    # )

    training_dataset1 = rl_ds.TrackingObservationDataset(training_data_npy_filename, data_dir)
    test_dataset1 = rl_ds.TrackingObservationDataset(test_data_npy_filename, data_dir)
    training_dataset2 = rl_ds.TrackingObservationDataset(training_data_npy_filename2, data_dir)
    test_dataset2 = rl_ds.TrackingObservationDataset(test_data_npy_filename2, data_dir)

    training_dataset3 = rl_ds.TrackingObservationDataset(training_data_npy_filename3, data_dir)
    training_dataset4 = rl_ds.TrackingObservationDataset(training_data_npy_filename4, data_dir)
    training_dataset5 = rl_ds.TrackingObservationDataset(training_data_npy_filename5, data_dir)
    training_dataset6 = rl_ds.TrackingObservationDataset(training_data_npy_filename6, data_dir)
    training_dataset7 = rl_ds.TrackingObservationDataset(training_data_npy_filename7, data_dir)
    training_dataset8 = rl_ds.TrackingObservationDataset(training_data_npy_filename8, data_dir)
    training_dataset9 = rl_ds.TrackingObservationDataset(training_data_npy_filename9, data_dir)

    training_dataset = torch.utils.data.ConcatDataset(
        [
            training_dataset1,
            training_dataset2,
            training_dataset3,
            training_dataset4,
            training_dataset5,
            training_dataset6,
            training_dataset7,
            training_dataset8,
            training_dataset9,
        ]
    )
    test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])
    # training_dataset = training_dataset1
    # test_dataset = test_dataset1

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training dataset length: {len(training_dataset)} | Test dataset length: {len(test_dataset)}")
    print(f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}")

    # SAVE_MODEL_FILE: Path = BASE_PATH / "models"  # "_epochxx.pth" appended in training
    # LOAD_MODEL_FILE: Path = BASE_PATH / "models" / "first.pth"  # "_epochxx.pth" appended in training
    log_dir = BASE_PATH / "logs"

    best_experiment = ""
    best_loss_sofar = 1e20
    exp_counter = 0
    for rnn_type in rnn_types:
        if rnn_type == torch.nn.GRU:
            rnn_type_str = "GRU"
        else:
            rnn_type_str = "LSTM"

        for latent_dim in latent_dims:
            vae = VAE(
                latent_dim=latent_dim,
                input_dim=input_dim,
                num_layers=num_rnn_layers,
                fc_dim=fc_dim,
                inference_mode=False,
                rnn_type=rnn_type,
                rnn_hidden_dim=20,
                bidirectional=True,
            ).to(device)

            name = f"tracking_vae{exp_counter+1}_NL_{num_rnn_layers}_LD_{latent_dim}_{rnn_type_str}"
            experiment_path = BASE_PATH / name

            writer = SummaryWriter(log_dir=log_dir / name)
            optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
            # T_max = len(train_dataloader) * num_epochs
            lr_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2, eta_min=1e-5)
            # lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)
            # lr_schedule = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

            if not experiment_path.exists():
                experiment_path.mkdir(parents=True)

            training_config = {
                "base_path": BASE_PATH,
                "experiment_path": experiment_path,
                "experiment_name": name,
                "latent_dim": latent_dim,
                "num_rnn_layers": num_rnn_layers,
                "input_dim": input_dim,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "save_interval": save_interval,
                "load_model": load_model,
            }
            with Path(experiment_path / "config.yaml").open(mode="w", encoding="utf-8") as fp:
                yaml.dump(training_config, fp)

            model, opt_loss, opt_train_loss, opt_epoch = train_vae(
                model=vae,
                training_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                writer=writer,
                n_epochs=num_epochs,
                batch_size=batch_size,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                save_interval=save_interval,
                device=device,
                early_stopping_patience=60,  # num_epochs,
                experiment_path=experiment_path,
            )

            print(
                f"[EXPERIMENT: {exp_counter + 1}]: LD={latent_dim}, NL={num_rnn_layers}, rnn_type={rnn_type_str} | Optimal loss: {opt_loss} at epoch {opt_epoch}"
            )

            if opt_loss < best_loss_sofar:
                best_loss_sofar = opt_loss
                best_experiment = name

            batch_obs = next(iter(test_dataloader))
            batch_obs = batch_obs.to(device)
            # extract length of valid obstacle observations
            seq_lengths = (
                torch.sum(batch_obs[:, 0, :] < 0.95, dim=1).to("cpu").type(torch.int64)
            )  # idx 0 is normalized distance, where vals = 1.0 is max dist of 1e4++ and thus not valid
            seq_lengths[seq_lengths == 0] = 1
            max_seq_length = seq_lengths.max().item()
            batch_obs = batch_obs[:, :, -max_seq_length:]
            batch_obs = batch_obs.permute(0, 2, 1)  # permute to (batch, max_seq_len, input_dim)
            recon_obs, _, _, _ = model(batch_obs, seq_lengths)
            weights = torch.ones_like(batch_obs)
            weights[torch.where(batch_obs[:, :, 0] > 0.98)] = 0.0
            diff = weights * (recon_obs - batch_obs)
            print(f"diff = {diff[0]}")

            exp_counter += 1

    print(f"BEST EXPERIMENT: {best_experiment} WITH LOSS: {best_loss_sofar}")
