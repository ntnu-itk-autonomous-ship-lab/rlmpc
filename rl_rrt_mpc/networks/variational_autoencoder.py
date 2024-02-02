"""
    variational_autoencoder.py

    Summary:
        Contains the variational autoencoder (VAE) network for processing and reconstructing images from the environment.

    Author: Trym Tengesdal
"""


import torch as th
import torch.nn as nn
from rl_rrt_mpc.networks.perception_image_decoder import PerceptionImageDecoder
from rl_rrt_mpc.networks.perception_image_encoder import PerceptionImageEncoder


class Lambda(nn.Module):
    """Lambda function that accepts tensors as input."""

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x: th.Tensor):
        return self.func(x)


# def reconstruction_loss(recon_x: th.Tensor, x: th.Tensor, mean: th.Tensor, logvar: th.Tensor) -> th.Tensor:
#     """
#     Compute the reconstruction loss of the VAE.

#     Args:
#         recon_x (th.Tensor): The reconstructed image.
#         x (th.Tensor): The input image.
#         mean (th.Tensor): The mean of the latent space.
#         logvar (th.Tensor): The log variance of the latent space.

#     Returns:
#         th.Tensor: The reconstruction loss.
#     """
#     recon_loss = nn.MSELoss(reduction="none")


# def kullback_leibler_divergence_loss(mean: th.Tensor, logvar: th.Tensor) -> th.Tensor:
#     """Compute the Kullback-Leibler divergence loss of the VAE.

#     Args:
#         mean (th.Tensor): Mean of the latent space.
#         logvar (th.Tensor): Log variance of the latent space.

#     Returns:
#         th.Tensor: The Kullback-Leibler divergence loss.
#     """
#     kld_loss = -0.5 * th.sum(1 + logvar - mean.pow(2) - logvar.exp())


class VAE(nn.Module):
    """Variational Autoencoder for reconstruction of depth images."""

    def __init__(self, n_input_channels: int = 1, latent_dim: int = 64, inference_mode: bool = False):
        """
        Args:
            n_input_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            inference_mode (bool): Whether to use inference mode or not
        """

        super(VAE, self).__init__()

        self.n_input_channels = n_input_channels
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.encoder = PerceptionImageEncoder(n_input_channels=n_input_channels, latent_dim=latent_dim)
        self.decoder = PerceptionImageDecoder(n_input_channels=n_input_channels, latent_dim=latent_dim)

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
        logvar = self.logvar_params(z)
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        if self.inference_mode:
            eps = th.zeros_like(eps)
        z_sampled = mean + eps * std

        img_recon = self.decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

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
        logvar = self.logvar_params(z)
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        if self.inference_mode:
            eps = th.zeros_like(eps)
        z_sampled = mean + eps * std

        img_recon = self.decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

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

    LATENT_DIM = 64
    device = th.device("cpu")
    encoder = PerceptionImageEncoder(n_input_channels=1, latent_dim=LATENT_DIM).to(device)
    summary(encoder, input_size=(1, 256, 256), batch_size=-1, device=device.type)

    decoder = PerceptionImageDecoder(n_input_channels=1, latent_dim=LATENT_DIM).to(device)
    summary(decoder, input_size=(1, LATENT_DIM), batch_size=-1, device=device.type)

    vae = VAE(n_input_channels=1, latent_dim=LATENT_DIM).to(device)
    summary(vae, input_size=(1, 256, 256), batch_size=-1, device=device.type)
