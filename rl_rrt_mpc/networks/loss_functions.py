"""
    loss_functions.py

    Summary:
        Contains feature extractors for neural networks (NNs) used in DRL. Feature extractor inspired by stable-baselines3 (SB3) implementation, the CNN-work of Thomas Larsen, and variational autoencoders (VAEs).

    Author: Trym Tengesdal
"""

from typing import Tuple

import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def vqvae(
    recon_x: th.Tensor, z_e_x: th.Tensor, z_q_x: th.Tensor, x: th.Tensor, semantic_mask: th.Tensor, beta: float = 1.0
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
    weight_matrix = semantic_mask * 100.0
    loss_recon = 0.5 * nn.MSELoss(reduction="none")(recon_x, x) * weight_matrix
    loss_recon = th.mean(th.sum(loss_recon, dim=[1, 2, 3]))

    # Vector quantization objective
    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
    # Commitment objective
    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

    loss = loss_recon + loss_vq + beta * loss_commit
    return loss, loss_recon, loss_vq, loss_commit


def sigma_semantically_weighted_reconstruction(recon_x: th.Tensor, x: th.Tensor, semantic_mask: th.Tensor) -> th.Tensor:
    """Loss function for semantically weighted reconstruction.

    Args:
        recon_x (th.Tensor): Reconstructed image.
        x (th.Tensor): Input image.
        semantic_mask (th.Tensor): Semantic mask.

    Returns:
        th.Tensor: The reconstruction loss.
    """
    # unique, counts = th.unique(semantic_mask, return_counts=True)
    weight_matrix = semantic_mask * 100.0
    mse_nonreduced_nonscaled = nn.MSELoss(reduction="none")(recon_x, x) * weight_matrix
    mse = th.mean(th.sum(mse_nonreduced_nonscaled, dim=[1, 2, 3]))
    log_sigma_opt = 0.5 * (mse + 1e-7).log()

    if False:
        for b in range(semantic_mask.shape[0]):
            for c in range(semantic_mask.shape[1]):
                fig, ax = plt.subplots(1, 4)
                plt.show(block=False)
                ax[0].imshow(x[b, c].detach().cpu().numpy())
                ax[1].imshow(semantic_mask[b, c].detach().cpu().numpy())
                ax[2].imshow(weight_matrix[b, c].detach().cpu().numpy())
                ax[3].imshow(recon_x[b, c].detach().cpu().numpy())
    recon_loss = 0.5 * th.pow((recon_x - x) / log_sigma_opt.exp(), 2) * weight_matrix + log_sigma_opt
    recon_loss = th.mean(th.sum(recon_loss, dim=[1, 2, 3]))
    return recon_loss, log_sigma_opt + 1e-7


def semantic_reconstruction_loss(recon_x: th.Tensor, x: th.Tensor, semantic_mask: th.Tensor) -> th.Tensor:
    """Loss function for semantically weighted reconstruction.

    Args:
        recon_x (th.Tensor): Reconstructed image.
        x (th.Tensor): Input image.
        semantic_mask (th.Tensor): Semantic mask.

    Returns:
        th.Tensor: The reconstruction loss.
    """
    # Shift recon_x to [0, 1] range if it is in [-1, 1] range
    if th.min(recon_x) < 0:
        recon_x = (recon_x + 1) / 2

    weight_matrix = semantic_mask * 100.0
    mse_nonreduced_nonscaled = nn.MSELoss(reduction="none")(recon_x, x) * weight_matrix
    recon_loss = th.mean(th.sum(mse_nonreduced_nonscaled, dim=[1, 2, 3]))
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
    log_sigma_opt = 0.5 * (mse + 1e-7).log()
    recon_loss = 0.5 * th.pow((recon_x - x) / log_sigma_opt.exp(), 2) + log_sigma_opt
    recon_loss = th.mean(th.sum(recon_loss, dim=[1, 2, 3]))
    return recon_loss, log_sigma_opt + 1e-7


def vanilla_reconstruction(recon_x: th.Tensor, x: th.Tensor) -> th.Tensor:
    """Compute the reconstruction loss of the VAE.

    Args:
        recon_x (th.Tensor): Reconstructed image.
        x (th.Tensor): Input image.

    Returns:
        th.Tensor: The reconstruction loss.
    """
    mse_nonreduced = nn.MSELoss(reduction="none")(recon_x, x)
    mse_loss = th.mean(th.sum(mse_nonreduced, dim=[1, 2, 3]))
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
    kld_loss = -0.5 * th.sum(1 + logvar - mean.pow(2) - (logvar.exp()), dim=[1, 0])
    return kld_loss
