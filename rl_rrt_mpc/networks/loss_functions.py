"""
    loss_functions.py

    Summary:
        Contains feature extractors for neural networks (NNs) used in DRL. Feature extractor inspired by stable-baselines3 (SB3) implementation, the CNN-work of Thomas Larsen, and variational autoencoders (VAEs).

    Author: Trym Tengesdal
"""

from typing import Tuple

import torch as th
import torch.nn as nn


def reconstruction(recon_x: th.Tensor, x: th.Tensor) -> th.Tensor:
    """
    Compute the reconstruction loss of the VAE.

    Args:
        recon_x (th.Tensor): The reconstructed image.
        x (th.Tensor): The input image.
        mean (th.Tensor): The mean of the latent space.
        logvar (th.Tensor): The log variance of the latent space.

    Returns:
        th.Tensor: The reconstruction loss.
    """
    recon_loss = th.mean((recon_x - x).pow(2))


def kullback_leibler_divergence(mean: th.Tensor, logvar: th.Tensor) -> th.Tensor:
    """Compute the Kullback-Leibler divergence loss of the VAE.

    Args:
        mean (th.Tensor): Mean of the latent space.
        logvar (th.Tensor): Log variance of the latent space.

    Returns:
        th.Tensor: The Kullback-Leibler divergence loss.
    """
    latent_dim = mean.shape[1]
    k = latent_dim
    # kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    kld_loss = 0.5 * th.mean(th.sum(logvar.exp() + mean.pow(2) - k - logvar, dim=1))
    return kld_loss
