"""
loss_functions.py

Summary:
    Contains various loss functions used in the training of VAE models.

Author: Trym Tengesdal
"""

from typing import Tuple

import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def mpc_parameter_provider_loss(
    predicted_param_increment: th.Tensor, target_param_increment: th.Tensor
) -> th.Tensor:
    """Compute the loss for the MPC parameter provider.

    Args:
        predicted_param_increment (th.Tensor): Predicted parameter increment.
        target_param_increment (th.Tensor): Target parameter increment.

    Returns:
        th.Tensor: The loss.
    """
    target_loss = F.mse_loss(predicted_param_increment, target_param_increment)
    chatter_loss = F.mse_loss(
        predicted_param_increment, th.zeros_like(predicted_param_increment)
    )
    loss = target_loss + 0.1 * chatter_loss
    return loss


def vqvae(
    recon_x: th.Tensor,
    z_e_x: th.Tensor,
    z_q_x: th.Tensor,
    x: th.Tensor,
    semantic_mask: th.Tensor,
    beta: float = 1.0,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Loss function for VQ-VAE.

    Args:
        recon_x (th.Tensor): Reconstructed image.
        z_e_x (th.Tensor): Latent space.
        z_q_x (th.Tensor): Quantized latent space.
        x (th.Tensor): Input image.
        semantic_mask (th.Tensor): Semantic mask.
        beta (float): Weight for the commitment loss.

    Returns:
        Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]: The total loss, the reconstruction loss, the VQ loss, and the commitment loss.
    """
    dims = [i for i in range(1, len(x.shape))]
    weight_matrix = compute_weights_from_semantic_mask(semantic_mask)
    loss_recon = 0.5 * nn.MSELoss(reduction="none")(recon_x, x) * weight_matrix
    loss_recon = th.mean(th.sum(loss_recon, dim=dims))

    # Vector quantization objective
    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
    # Commitment objective
    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

    loss = loss_recon + loss_vq + beta * loss_commit
    return loss, loss_recon, loss_vq, loss_commit


def compute_weights_from_semantic_mask_land_ownship_only(
    semantic_mask: th.Tensor,
) -> th.Tensor:
    """Compute weights from semantic mask.

    Args:
        semantic_mask (th.Tensor): Semantic mask tensor.

    Returns:
        th.Tensor: The weight matrix.
    """
    weight_matrix = th.ones_like(semantic_mask)
    # weight_matrix[semantic_mask > 0.05] = 5.0  # land interior
    weight_matrix[semantic_mask > 0.1] = 10.0  # land edges
    weight_matrix[semantic_mask > 0.6] = 20.0  # ownship
    return weight_matrix


def compute_weights_from_semantic_mask(semantic_mask: th.Tensor) -> th.Tensor:
    """Compute weights from semantic mask, land and ownship only.

    Args:
        semantic_mask (th.Tensor): Semantic mask tensor.

    Returns:
        th.Tensor: The weight matrix.
    """
    weight_matrix = th.ones_like(semantic_mask)
    semantic_mask = (semantic_mask * 1000.0).int()
    unique, counts = th.unique(semantic_mask, return_counts=True)
    weights = [100.0, 50.0, 150.0]  # land edges, nominal path, ships
    for idx, (val, _) in enumerate(zip(unique[-3:], counts[-3:])):
        weight_matrix[semantic_mask == val] = weights[idx]
    return weight_matrix


def sigma_semantically_weighted_reconstruction(
    recon_x: th.Tensor, x: th.Tensor, semantic_mask: th.Tensor
) -> th.Tensor:
    """Loss function for semantically weighted reconstruction.

    Args:
        recon_x (th.Tensor): Reconstructed image, normalized to [0, 1].
        x (th.Tensor): Input image, normalized to [0, 1].
        semantic_mask (th.Tensor): Semantic mask.

    Returns:
        th.Tensor: The reconstruction loss.
    """
    dims = [i for i in range(1, len(x.shape))]
    weight_matrix = compute_weights_from_semantic_mask(semantic_mask)
    mse_nonreduced_nonscaled = nn.MSELoss(reduction="none")(recon_x, x) * weight_matrix
    mse = th.mean(th.sum(mse_nonreduced_nonscaled, dim=dims))
    log_sigma_opt = 0.5 * (mse + 1e-7).log()

    if False:
        fig, ax = plt.subplots(1, 4)
        plt.show(block=False)
        ax[0].imshow(x[0, 0].detach().cpu().numpy())
        ax[1].imshow(semantic_mask[0, 0].detach().cpu().numpy())
        ax[2].imshow(weight_matrix[0, 0].detach().cpu().numpy())
        ax[3].imshow(recon_x[0, 0].detach().cpu().numpy())
    recon_loss = (
        0.5 * th.pow((recon_x - x) / log_sigma_opt.exp(), 2) * weight_matrix
        + log_sigma_opt
    )
    recon_loss = th.mean(th.sum(recon_loss, dim=dims))
    return recon_loss, log_sigma_opt + 1e-7


def semantically_weighted_reconstruction(
    recon_x: th.Tensor, x: th.Tensor, semantic_mask: th.Tensor
) -> th.Tensor:
    """Loss function for semantically weighted reconstruction.

    Args:
        recon_x (th.Tensor): Reconstructed image.
        x (th.Tensor): Input image.
        semantic_mask (th.Tensor): Semantic mask.

    Returns:
        th.Tensor: The reconstruction loss.
    """
    dims = [i for i in range(1, len(x.shape))]
    weight_matrix = compute_weights_from_semantic_mask_land_ownship_only(semantic_mask)
    mse_nonreduced_nonscaled = nn.MSELoss(reduction="none")(recon_x, x) * weight_matrix
    recon_loss = th.mean(th.sum(mse_nonreduced_nonscaled, dim=dims))
    return recon_loss


def sigma_reconstruction(recon_x: th.Tensor, x: th.Tensor) -> th.Tensor:
    """
    Compute the reconstruction loss of the VAE.

    Args:
        recon_x (th.Tensor): The reconstructed image of dim [batch_size, n_channels, height, width].
        x (th.Tensor): The input image of dim [batch_size, n_channels, height, width].

    Returns:
        th.Tensor: The reconstruction loss.
    """
    mse = F.mse_loss(recon_x, x, reduction="mean")
    dims = [i for i in range(1, len(x.shape))]
    log_sigma_opt = 0.5 * (mse + 1e-7).log()
    recon_loss = 0.5 * th.pow((recon_x - x) / log_sigma_opt.exp(), 2) + log_sigma_opt
    recon_loss = th.mean(th.sum(recon_loss, dim=dims))
    return recon_loss, log_sigma_opt + 1e-7


def sigma_reconstruction_rnn(
    recon_x: th.Tensor, x: th.Tensor, seq_lengths: th.Tensor
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Compute the reconstruction loss of the sigma-VAE.

    Args:
        recon_x (th.Tensor): The reconstructed image of dim [batch_size, n_channels, height, width].
        x (th.Tensor): The input image of dim [batch_size, n_channels, height, width].

    Returns:
        Tuple[th.Tensor, th.Tensor]: The reconstruction loss and the log sigma.
    """
    dims = [i for i in range(1, len(x.shape))]
    weights = th.ones_like(x)
    weights[th.where(x[:, :, :3] > -1.0)] = (
        5.0  # increase the weight for the first three channels
    )
    weights[th.where(x[:, :, 0] > 0.98)] = 0.0
    mse = F.mse_loss(recon_x, x, reduction="none") * weights
    mse = th.mean(th.sum(mse, dim=dims))
    log_sigma_opt = 0.5 * (mse + 1e-7).log()
    recon_loss = (
        0.5 * th.pow((recon_x - x) / log_sigma_opt.exp(), 2) * weights + log_sigma_opt
    )
    recon_loss = th.mean(th.sum(recon_loss, dim=dims))
    return recon_loss, log_sigma_opt + 1e-7


def reconstruction_rnn(
    recon_x: th.Tensor, x: th.Tensor, seq_lengths: th.Tensor, threshold_dist: float
) -> th.Tensor:
    """
    Compute the reconstruction loss of the VAE.

    Args:
        recon_x (th.Tensor): The reconstructed image of dim [batch_size, n_channels, height, width].
        x (th.Tensor): The input image of dim [batch_size, n_channels, height, width].
        seq_lengths (th.Tensor): The length of the sequences in the batch.
        threshold_dist (float): The threshold distance for the normalized obstacle distance.

    Returns:
        th.Tensor: The reconstruction loss.
    """
    dims = [i for i in range(1, len(x.shape))]
    weights = 0.1 * th.ones_like(x)
    weights[:, :, 0][th.where(x[:, :, 0] >= -1.0)] = (
        100.0  # increase the weight for the first four channels
    )
    weights[:, :, 1][th.where(x[:, :, 1] >= -1.0)] = 50.0
    weights[:, :, 2][th.where(x[:, :, 2] >= -1.0)] = 50.0
    weights[:, :, 3][th.where(x[:, :, 3] >= -1.0)] = 50.0
    weights[th.where(x[:, :, 0] > threshold_dist)] = 0.0
    if weights.sum() == 0.0:
        print("Weights are zero!")
    seq_lengths = seq_lengths.clamp(min=1)
    mse = F.mse_loss(recon_x, x, reduction="none") * weights
    recon_loss = th.sum(mse, dim=dims) / seq_lengths.to(x.device).float()
    recon_loss = th.mean(recon_loss)
    return recon_loss


def vanilla_reconstruction(recon_x: th.Tensor, x: th.Tensor) -> th.Tensor:
    """Compute the reconstruction loss of the VAE.

    Args:
        recon_x (th.Tensor): Reconstructed image.
        x (th.Tensor): Input image.

    Returns:
        th.Tensor: The reconstruction loss.
    """
    mse_nonreduced = nn.MSELoss(reduction="none")(recon_x, x)
    dims = [i for i in range(1, len(x.shape))]
    mse_loss = th.mean(th.sum(mse_nonreduced, dim=dims))
    return mse_loss


def kullback_leibler_divergence(mean: th.Tensor, logvar: th.Tensor) -> th.Tensor:
    """Compute the Kullback-Leibler divergence loss of the VAE.

    Args:
        mean (th.Tensor): Mean of the latent space.
        logvar (th.Tensor): Log variance of the latent space.

    Returns:
        th.Tensor: The Kullback-Leibler divergence loss.
    """
    # logvar = th.log(logvar.exp() + 1e-7)  # To avoid singularity
    # print(f"latent variance (min, max): ({logvar.exp().min()}, {logvar.exp().max()})")
    kld_loss = -0.5 * th.mean(
        th.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1), dim=0
    )
    return kld_loss
