"""
train_mpc_param_provider.py

Summary:
    This script trains the MPC parameter provider.

Author: Trym Tengesdal
"""

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rlmpc.networks.loss_functions as loss_functions
import torch
from rlmpc.common.running_loss import RunningLoss
from rlmpc.policies import MPCParameterDNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_mpc_param_dnn(
    model: MPCParameterDNN,
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
    verbose: bool = False,
    save_intermittent_models: bool = False,
) -> Tuple[MPCParameterDNN, float, int, List[List[float]], List[List[float]]]:
    """Trains the variation autoencoder model.

    Args:
        model (MPCParameterDNN): The model to train
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
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        save_intermittent_models (bool, optional): Whether to save the model at each epoch. Defaults to False.

    Returns:
        Tuple[VAE, float, int, List[List[float]], List[List[float]]]: The trained model, the best test loss, the epoch at which the best test loss occurred, the training losses, and the testing losses.
    """
    torch.autograd.set_detect_anomaly(True)

    loss_meter = RunningLoss(batch_size)
    n_batches = int(len(training_dataloader))
    n_test_batches = int(len(test_dataloader))

    if save_intermittent_models:
        model_path = experiment_path / "models"
        if not model_path.exists():
            model_path.mkdir()

    best_test_loss = 1e20
    best_train_loss = 1e20
    best_epoch = 0
    training_losses = []
    testing_losses = []

    experiment_name = experiment_path.name
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        training_batch_losses = []

        model.train()
        for batch_idx, (batch_dnn_inputs, batch_param_incr) in enumerate(
            training_dataloader
        ):
            batch_start_time = time.time()
            model.zero_grad()
            optimizer.zero_grad()

            batch_dnn_inputs = batch_dnn_inputs.to(device)
            batch_param_incr = batch_param_incr.to(device)

            # Forward pass
            predicted_param_incr = model(batch_dnn_inputs)
            loss = loss_functions.mpc_parameter_provider_loss(
                predicted_param_incr, batch_param_incr
            )
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)
            training_batch_losses.append(loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the tensorboard
            if batch_idx % save_interval == 0:
                lr = lr_schedule.get_last_lr()[0]
                writer.add_scalar(
                    "Training/Learning Rate", lr, epoch * n_batches + batch_idx
                )
                writer.add_scalar(
                    "Training/Loss", loss.item(), epoch * n_batches + batch_idx
                )
                if verbose:
                    print(
                        f"[TRAINING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_batches} | Train Loss: {loss.item():.4f} | "
                        f"Batch processing time: {time.time() - batch_start_time:.2f}s | Est. time remaining: {(n_batches - batch_idx) * avg_iter_time * (n_epochs - epoch + 1):.2f}s"
                    )

        lr_schedule.step()

        if verbose:
            print(
                f"Epoch: {epoch} | Loss: {loss_meter.average_loss} | Time: {time.time() - epoch_start_time}"
            )
        loss_meter.reset()

        if save_intermittent_models:
            save_path = f"{str(model_path)}/{experiment_name}_epoch_{epoch}.pth"
            torch.save(
                model.state_dict(),
                save_path,
            )

        training_losses.append(training_batch_losses)

        model.eval()
        test_batch_losses = []
        for batch_idx, (batch_dnn_inputs, batch_param_incr) in enumerate(
            test_dataloader
        ):
            batch_start_time = time.time()

            batch_dnn_inputs = batch_dnn_inputs.to(device)
            batch_param_incr = batch_param_incr.to(device)

            # Forward pass
            predicted_param_incr = model(batch_dnn_inputs)
            loss = loss_functions.mpc_parameter_provider_loss(
                predicted_param_incr, batch_param_incr
            )
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)
            test_batch_losses.append(loss.item())

            # Update the tensorboard
            if batch_idx % save_interval == 0:
                if verbose:
                    print(
                        f"[TESTING] Epoch: {epoch + 1}/{n_epochs} | Batch: {batch_idx + 1}/{n_test_batches} | Test Loss: {loss.item():.4f}"
                    )
                writer.add_scalar(
                    "Test/Loss", loss.item(), epoch * n_test_batches + batch_idx
                )

        testing_losses.append(test_batch_losses)
        if verbose:
            print("Test Loss:", loss_meter.average_loss)
        if loss_meter.average_loss < best_test_loss:
            best_test_loss = loss_meter.average_loss
            best_epoch = epoch
            num_nondecreasing_loss_iters = 0
            if verbose:
                print(
                    f"Current best model at epoch {best_epoch + 1} with test loss {best_test_loss}"
                )
            best_model_path = f"{experiment_path}/best_model.pth"
            torch.save(model.state_dict(), best_model_path)
        else:
            if verbose:
                print(
                    f"Test loss has not decreased for {num_nondecreasing_loss_iters} iterations."
                )
            num_nondecreasing_loss_iters += 1

        if num_nondecreasing_loss_iters > early_stopping_patience:
            if verbose:
                print(
                    f"Test loss has not decreased for {early_stopping_patience} epochs. Stopping training."
                )
            break

    training_losses = np.array(training_losses)
    testing_losses = np.array(testing_losses)
    best_train_loss = np.min(np.mean(training_losses, axis=1))
    np.save(experiment_path / "training_losses.npy", training_losses)
    np.save(experiment_path / "testing_losses.npy", testing_losses)
    return model, best_test_loss, best_train_loss, best_epoch
