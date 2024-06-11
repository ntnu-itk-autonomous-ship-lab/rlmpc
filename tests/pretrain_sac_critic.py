#  Use argument parser to set arguments of experiment name
import argparse
import inspect
import time
from pathlib import Path
from sys import platform
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import rlmpc.common.datasets as rl_ds
import rlmpc.common.paths as rl_dp
import rlmpc.networks.feature_extractors as rl_fe
import rlmpc.networks.loss_functions as loss_functions
import rlmpc.policies as rl_policies
import torch
import yaml
from rlmpc.networks.tracking_vae_attention.vae import VAE
from stable_baselines3.sac.policies import ContinuousCritic
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if platform == "linux" or platform == "linux2":
    BASE_PATH: Path = Path("/home/doctor/Desktop/machine_learning/tracking_vae/")
elif platform == "darwin":
    BASE_PATH: Path = Path("/Users/trtengesdal/Desktop/machine_learning/tracking_vae/")

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


def train_critic(
    model: ContinuousCritic,
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
        model (ContinuousCritic): The model to train
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

    beta = 0.5
    training_losses = []
    testing_losses = []

    input_dim = 6
    max_seq_length = 10

    # warmup_period = n_batches
    # warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

    experiment_name = experiment_path.name

    # Create training data + test data
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        training_batch_losses = []

        model.train()
        model.set_inference_mode(False)
        for batch_idx, (batch_features, batch_mpc_values) in enumerate(training_dataloader):
            batch_start_time = time.time()
            model.zero_grad()
            optimizer.zero_grad()

            batch_obs = batch_features.to(device)
            batch_mpc_values = batch_mpc_values.to(device)

            # Forward pass
            action_state_value = model(batch_obs)
            loss = loss_functions.critic_mse_loss(action_state_value, batch_obs)
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
                print(
                    f"[TRAINING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_batches} | Train Loss: {loss.item():.4f} |"
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
        for batch_idx, (batch_features, batch_mpc_values) in enumerate(test_dataloader):
            batch_start_time = time.time()

            batch_obs = batch_features.to(device)
            batch_mpc_values = batch_mpc_values.to(device)

            # Forward pass
            action_state_value = model(batch_obs)
            loss = loss_functions.critic_mse_loss(action_state_value, batch_obs)
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)
            test_batch_losses.append(loss.item())

            if batch_idx % save_interval == 0:
                print(
                    f"[TESTING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_test_batches} | Test Loss: {loss.item():.4f} | KL Div Loss.: {kld_loss.item()/batch_size:.4f}"
                )
                # Update the tensorboard
                writer.add_scalar("Test/Loss", loss.item(), epoch * n_test_batches + batch_idx)

        testing_losses.append(test_batch_losses)

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

    feature_extractor = rl_fe.CombinedExtractor

    num_layers = [1, 2, 3]
    hidden_dims = [64, 128, 256]
    load_model = False
    save_interval = 20
    batch_size = 128
    num_epochs = 40
    learning_rate = 2e-4

    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            "perception_image_observation",
            "relative_tracking_observation",
            "navigation_3dof_state_observation",
            "tracking_observation",
            "ground_truth_tracking_observation",
            "disturbance_observation",
            "time_observation",
        ]
    }

    scenario_folder = rl_dp.scenarios / "training_data" / "rlmpc_scenario_ms_channel"
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": scenario_folder,
        "merge_loaded_scenario_episodes": True,
        "max_number_of_episodes": 1,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        "test_mode": False,
        "render_update_rate": 1.0,
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "shuffle_loaded_scenario_data": True,
        "identifier": "training_env",
        "seed": 15,
    }
    env = gym.make(id=env_id, **env_config)

    data_dir = Path("/home/doctor/Desktop/machine_learning/rlmpc/critic_data/")
    # data_dir = Path("/Users/trtengesdal/Desktop/machine_learning/rlmpc/critic_data/")
    training_data_filename_list = []
    for i in range(1, 2):
        training_data_filename = f"critic_training_data_rogaland{i}.npy"
        training_data_filename_list.append(training_data_filename)

    test_data_filename_list = []
    for i in range(1, 2):
        test_data_filename_list.append(f"critic_test_data_rogaland{i}.npy")
        if i < 3:
            test_data_filename_list.append(f"critic_test_data_rogaland_v2_{i}.npy")

    training_dataset = torch.utils.data.ConcatDataset(
        [
            rl_ds.TrackingObservationDataset(training_data_file, data_dir)
            for training_data_file in training_data_filename_list
        ]
    )

    test_dataset = torch.utils.data.ConcatDataset(
        [rl_ds.TrackingObservationDataset(test_data_file, data_dir) for test_data_file in test_data_filename_list]
    )

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training dataset length: {len(training_dataset)} | Test dataset length: {len(test_dataset)}")
    print(f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}")

    log_dir = BASE_PATH / "logs"

    best_experiment = ""
    best_loss_sofar = 1e20
    exp_counter = 0
    for layers in num_layers:
        for hidden_dim in hidden_dims:
            critic = ContinuousCritic(
                observation_space=env.observation_space,
                action_space=env.action_space,
                hidden_dims=[hidden_dim for _ in range(layers)],
                n_critics=2,
                features_extractor=feature_extractor,
                features_dim=feature_extractor.features_dim,
            )

            name = f"critic{exp_counter+1}_NL_{layers}_HD_{hidden_dim}"
            experiment_path = BASE_PATH / name

            writer = SummaryWriter(log_dir=log_dir / name)
            optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
            # T_max = len(train_dataloader) * num_epochs
            # lr_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2, eta_min=1e-5)
            lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)
            # lr_schedule = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

            if not experiment_path.exists():
                experiment_path.mkdir(parents=True)

            training_config = {
                "base_path": BASE_PATH,
                "experiment_path": experiment_path,
                "experiment_name": name,
                "num_layers": layers,
                "hidden_dim": hidden_dim,
                "input_dim": observation_dim,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "save_interval": save_interval,
                "load_model": load_model,
            }
            with Path(experiment_path / "config.yaml").open(mode="w", encoding="utf-8") as fp:
                yaml.dump(training_config, fp)

            model, opt_loss, opt_train_loss, opt_epoch = train_critic(
                model=critic,
                training_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                writer=writer,
                n_epochs=num_epochs,
                batch_size=batch_size,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                save_interval=save_interval,
                device=device,
                early_stopping_patience=10,  # num_epochs,
                experiment_path=experiment_path,
            )

            print(
                f"[EXPERIMENT: {exp_counter + 1}]: LD={latent_dim}, NL={num_rnn_layers_decoder}, HD={rnn_hidden_dim_decoder}, NH={num_heads}, ED={embedding_dim} | Optimal loss: {opt_loss} at epoch {opt_epoch}"
            )

            if opt_loss < best_loss_sofar:
                best_loss_sofar = opt_loss
                best_experiment = name

            exp_counter += 1

    print(f"BEST EXPERIMENT: {best_experiment} WITH LOSS: {best_loss_sofar}")
