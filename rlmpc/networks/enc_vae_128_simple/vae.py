"""
    vae.py

    Summary:
        Contains the vanilla variational autoencoder (VAE) network for processing and reconstructing images from the environment.

    Author: Trym Tengesdal
"""

from typing import List, Tuple

import torch as th
import torch.nn as nn
from rlmpc.networks.enc_vae_128_simple.decoder import ENCDecoder
from rlmpc.networks.enc_vae_128_simple.encoder import ENCEncoder


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
        input_image_dim: Tuple[int, int, int] = (1, 128, 128),
        encoder_conv_block_dims: List[int] = [32, 64, 128],
        decoder_conv_block_dims: List[int] = [128, 128, 64],
        fc_dim: int = 32,
        inference_mode: bool = False,
    ):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            input_image_dim (Tuple[int, int, int]): Dimensions of the input image
            encoder_conv_block_dims (Tuple[int, int, int, int]): Dimensions of the convolutional blocks
            decoder_conv_block_dims (Tuple[int, int, int, int]): Dimensions of the deconvolutional blocks
            inference_mode (bool): Inference mode
        """

        super(VAE, self).__init__()

        self.input_image_dim = input_image_dim
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.encoder = ENCEncoder(
            n_input_channels=input_image_dim[0],
            latent_dim=latent_dim,
            conv_block_dims=tuple(encoder_conv_block_dims),
            fc_dim=fc_dim,
        )
        self.decoder = ENCDecoder(
            n_input_channels=input_image_dim[0],
            latent_dim=latent_dim,
            first_deconv_input_dim=self.encoder.last_conv_block_dim,
            deconv_block_dims=decoder_conv_block_dims,
            fc_dim=fc_dim,
        )

        self.mean_params = Lambda(lambda x: x[:, : self.latent_dim])  # mean parameters
        self.logvar_params = Lambda(lambda x: x[:, self.latent_dim :])  # log variance parameters

    def forward(self, img: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
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

    def encode(self, image: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
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

    LATENT_DIM = 32
    image_dim = (1, 128, 128)
    fc_dim = 128
    device = th.device("cpu")
    vae = VAE(
        latent_dim=LATENT_DIM,
        input_image_dim=image_dim,
        encoder_conv_block_dims=(128, 128, 128),
        decoder_conv_block_dims=(128, 128, 64),
        fc_dim=fc_dim,
    ).to(device)
    summary(vae, input_size=image_dim, batch_size=-1, device=device.type)
