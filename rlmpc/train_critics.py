"""
    train_soft_critics.py

    Summary:
        This script trains the critic and the target critic softly using replayed interaction data.

    Author: Trym Tengesdal
"""

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rlmpc.sac as rlmpc_sac
import torch
from rlmpc.common.running_loss import RunningLoss
from torch.utils.tensorboard import SummaryWriter


def train_critics(
    model: rlmpc_sac.SAC,
    writer: SummaryWriter,
    n_epochs: int,
    batch_size: int,
    experiment_path: Path,
    save_interval: int = 10,
    verbose: bool = False,
    save_intermittent_models: bool = False,
) -> Tuple[rlmpc_sac.SAC, float, int]:
    """Trains the input list of critics for a set of epochs (NOTE: Without cross-validation.)

    Args:
        model (rlmpc_sac.SAC): SAC model that contains the critics.
        writer (SummaryWriter): The tensorboard writer
        n_epochs (int): The number of epochs to train the model
        batch_size (int): The batch size
        save_interval (int, optional): The interval at which to save stats to tensorboard. Defaults to 10.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        save_intermittent_models (bool, optional): Whether to save the model at each epoch. Defaults to False.

    Returns:
        Tuple[List[ContinuousCritic], float, int]: The trained model, the best training loss and corresponding epoch.
    """
    torch.autograd.set_detect_anomaly(True)
    loss_meter = RunningLoss(batch_size)

    if save_intermittent_models:
        model_path = experiment_path / "models"
        if not model_path.exists():
            model_path.mkdir()

    best_epoch = 0
    training_losses = []

    experiment_name = experiment_path.name
    num_iterations_per_epoch = 100
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        training_batch_losses = []

        for batch_idx in range(num_iterations_per_epoch):
            replay_data = model.replay_buffer.sample(batch_size=batch_size)
            loss = model.train_critics(replay_data=replay_data, gradient_step=batch_idx, ent_coef=1.0)
            training_batch_losses.append(loss)

            if batch_idx % save_interval == 0:
                writer.add_scalar("Training/Loss", loss, epoch * num_iterations_per_epoch + batch_idx)
                if verbose:
                    print(
                        f"[TRAINING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{num_iterations_per_epoch} | Train Loss: {loss.item():.4f} | "
                    )

        if verbose:
            print(f"Epoch: {epoch} | Loss: {loss_meter.average_loss} | Time: {time.time() - epoch_start_time}")

        loss_meter.reset()
        training_losses.append(np.array(training_batch_losses).mean())

        if save_intermittent_models:
            save_path = f"{str(model_path)}/{experiment_name}_epoch_{epoch}.pth"
            model.save_critics(path=save_path)

    training_losses = np.array(training_losses)
    best_train_loss = np.min(np.mean(training_losses))
    best_epoch = int(np.argmin(training_losses))
    np.save(experiment_path / "training_losses.npy", training_losses)
    return model, best_train_loss, best_epoch
