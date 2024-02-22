"""
    loss_functions.py

    Summary:
        Contains feature extractors for neural networks (NNs) used in DRL. Feature extractor inspired by stable-baselines3 (SB3) implementation, the CNN-work of Thomas Larsen, and variational autoencoders (VAEs).

    Author: Trym Tengesdal
"""

from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
