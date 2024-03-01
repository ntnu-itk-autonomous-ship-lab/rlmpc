"""
    vae.py

    Summary:
        Contains the vanilla variational autoencoder (VAE) network for processing and reconstructing images from the environment.

    Author: Trym Tengesdal
"""

from typing import Tuple

import torch as th
import torch.nn as nn
from rlmpc.networks.vanilla_vae.decoder import PerceptionImageDecoder
from rlmpc.networks.vanilla_vae.encoder import PerceptionImageEncoder


class Lambda(nn.Module):
    """Lambda function that accepts tensors as input."""

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x: th.Tensor):
        return self.func(x)


class VAE(nn.Module):
    """Variational Autoencoder for reconstruction of depth images."""

    def __init__(
        self,
        latent_dim: int = 64,
        input_image_dim: Tuple[int, int, int] = (3, 400, 400),
        first_deconv_input_dim: int = 18,
        inference_mode: bool = False,
    ):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            input_image_dim (Tuple[int, int, int]): Dimensions of the input image
            inference_mode (bool): Whether to use inference mode or not
        """

        super(VAE, self).__init__()

        self.input_image_dim = input_image_dim
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.encoder = PerceptionImageEncoder(n_input_channels=input_image_dim[0], latent_dim=latent_dim)
        self.decoder = PerceptionImageDecoder(
            n_input_channels=input_image_dim[0], latent_dim=latent_dim, first_deconv_input_dim=first_deconv_input_dim
        )

        self.mean_params = Lambda(lambda x: x[:, : self.latent_dim])  # mean parameters
        self.logvar_params = Lambda(lambda x: x[:, self.latent_dim :])  # log variance parameters

    def forward(self, img: th.Tensor) -> th.Tensor:
        """Do a forward pass of the VAE. Generates a reconstructed image based on the input image.

        Args:
            img (th.Tensor): The input image.

        Returns:
            th.Tensor: The reconstructed image.
        """
        z = self.encoder(img)

        # Reparametrization trick
        mean = self.mean_params(z)
        logvars = self.logvar_params(z)
        logvars = th.log(logvars.exp() + 1e-7)  # To avoid singularity
        std = th.exp(0.5 * logvars)
        eps = th.randn_like(std)
        if self.inference_mode:
            eps = th.zeros_like(eps)
        z_sampled = mean + eps * std

        img_recon = self.decoder(z_sampled)
        return img_recon, mean, logvars, z_sampled

    def forward_test(self, image: th.Tensor) -> th.Tensor:
        """Do a forward pass of the VAE. Generates a reconstructed image based on input image.

        Args:
            imgage (th.Tensor): The input image.

        Returns:
            th.Tensor: The reconstructed image.
        """
        z = self.encoder(image)

        # Reparametrization trick
        mean = self.mean_params(z)
        logvars = self.logvar_params(z)
        logvars = th.log(logvars.exp() + 1e-7)  # To avoid singularity
        std = th.exp(0.5 * logvars)
        eps = th.randn_like(std)
        if self.inference_mode:
            eps = th.zeros_like(eps)
        z_sampled = mean + eps * std

        img_recon = self.decoder(z_sampled)
        return img_recon, mean, logvars, z_sampled

    def encode(self, image: th.Tensor) -> th.Tensor:
        """Do a forward pass of the VAE. Generates a latent vector based on input image.

        Args:
            image (th.Tensor): The input image.

        Returns:
            th.Tensor: The latent vector.
        """
        z = self.encoder(image)

        means = self.mean_params(z)
        logvars = self.logvar_params(z)
        logvars = th.log(logvars.exp() + 1e-7)  # To avoid singularity
        std = th.exp(0.5 * logvars)
        eps = th.randn_like(logvars)
        if self.inference_mode:
            eps = th.zeros_like(eps)
        z_sampled = means + eps * std

        return z_sampled, means, std

    def decode(self, z: th.Tensor) -> th.Tensor:
        """Do a forward pass of the VAE. Generates a reconstructed image based on z

        Args:
            z (th.Tensor): The latent vector.

        Returns:
            th.Tensor: The reconstructed image.
        """
        reconstructed_image = self.decoder(z)
        return reconstructed_image

    def set_inference_mode(self, mode):
        self.inference_mode = mode


if __name__ == "__main__":
    from torchsummary import summary

    LATENT_DIM = 128
    device = th.device("cpu")
    encoder = PerceptionImageEncoder(n_input_channels=3, latent_dim=LATENT_DIM).to(device)
    summary(encoder, input_size=(3, 400, 400), batch_size=-1, device=device.type)

    decoder = PerceptionImageDecoder(n_input_channels=3, latent_dim=LATENT_DIM).to(device)
    summary(decoder, input_size=(1, LATENT_DIM), batch_size=-1, device=device.type)

    vae = VAE(latent_dim=LATENT_DIM, input_image_dim=(3, 400, 400)).to(device)
    summary(vae, input_size=(3, 400, 400), batch_size=-1, device=device.type)
