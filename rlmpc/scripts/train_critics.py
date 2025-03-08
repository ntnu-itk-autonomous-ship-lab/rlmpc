"""
train_soft_critics.py

Summary:
    This script trains the critic and the target critic softly using replayed interaction data.

Author: Trym Tengesdal
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import optuna
import rlmpc.sac as rlmpc_sac
import rlmpc.standard_sac as rlmpc_ssac
import torch
from rlmpc.common.running_loss import RunningLoss
from torch.utils.tensorboard import SummaryWriter


def train_critics(
    model: rlmpc_sac.SAC | rlmpc_ssac.SAC,
    writer: SummaryWriter,
    n_epochs: int,
    batch_size: int,
    experiment_path: Path,
    save_interval: int = 10,
    verbose: bool = False,
    save_intermittent_models: bool = False,
    early_stopping_patience: int = 6,
    ent_coef: float = 0.001,
    optuna_trial: Optional[optuna.Trial] = None,
) -> Tuple[rlmpc_sac.SAC, float, int]:
    """Trains the input list of critics for a set of epochs (NOTE: Without cross-validation.)

    Args:
        model (rlmpc_sac.SAC): SAC model that contains the critics.
        writer (SummaryWriter): The tensorboard writer
        n_epochs (int): The number of epochs to train the model
        batch_size (int): The batch size
        save_interval (int, optional): The interval at which to save stats to tensorboard.
        verbose (bool, optional): Whether to print verbose output.
        save_intermittent_models (bool, optional): Whether to save the model at each epoch.
        early_stopping_patience (int, optional): The number of epochs to wait before early stopping.
        ent_coef (int, optional): Temperature coefficient
        optuna_trial (Optional[optuna.Trial], optional): The optuna trial object.

    Returns:
        Tuple[List[ContinuousCritic], float, int]: The trained model, the best training loss and corresponding epoch.
    """
    torch.autograd.set_detect_anomaly(True)
    loss_meter = RunningLoss(batch_size)
    n_batches = model.replay_buffer.size() // batch_size

    model_path = experiment_path / "models"
    if not model_path.exists():
        model_path.mkdir()

    best_epoch = 0
    best_train_loss = 1e12
    training_losses = []

    experiment_name = experiment_path.name
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        training_batch_losses = []
        model.policy.set_training_mode(mode=True)
        for batch_idx in range(n_batches):
            batch_start_time = time.time()
            replay_data = model.replay_buffer.sample(batch_size=batch_size)
            loss = model.train_critics(
                replay_data=replay_data, gradient_step=batch_idx, ent_coef=ent_coef
            )
            training_batch_losses.append(loss.item())
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            if batch_idx % save_interval == 0:
                writer.add_scalar("Training/Loss", loss, epoch * n_batches + batch_idx)
                if verbose:
                    print(
                        f"[TRAINING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_batches} | Train Loss: {loss.item():.4f} | "
                        f"Batch processing time: {time.time() - batch_start_time:.2f}s | Est. time remaining: {(n_batches - batch_idx) * avg_iter_time * (n_epochs - epoch + 1):.2f}s"
                    )

        if verbose:
            print(
                f"Epoch: {epoch} | Avg loss: {loss_meter.average_loss:4f} | Time: {time.time() - epoch_start_time:.4f}"
            )

        if loss_meter.average_loss < best_train_loss:
            best_train_loss = loss_meter.average_loss
            best_epoch = epoch
            save_path = f"{str(model_path)}/best_model"
            model.save_critics(path=save_path)
            print(f"New best model saved at epoch {epoch} with loss {best_train_loss}")

        #  else:
        #     print(f"Train loss has not decreased for {num_nondecreasing_loss_iters} iterations.")
        #     num_nondecreasing_loss_iters += 1

        if optuna_trial is not None:
            optuna_trial.report(loss_meter.average_loss, epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        loss_meter.reset()
        training_losses.append(np.array(training_batch_losses).mean())

        if save_intermittent_models and epoch % 4 == 0:
            save_path = f"{str(model_path)}/{experiment_name}_epoch_{epoch}.pth"
            model.save_critics(path=save_path)

    training_losses = np.array(training_losses)
    best_epoch = int(np.argmin(training_losses))
    np.save(experiment_path / "training_losses.npy", training_losses)
    return model, best_train_loss, best_epoch
