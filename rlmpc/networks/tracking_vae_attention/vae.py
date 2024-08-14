"""
    vae.py

    Summary:
        Contains the vanilla variational autoencoder (VAE) network for processing and reconstructing images from the environment.

    Author: Trym Tengesdal
"""

from typing import Tuple

import torch as th
import torch.nn as nn
from rlmpc.networks.tracking_vae_attention.decoder import TrackingDecoder
from rlmpc.networks.tracking_vae_attention.encoder import TrackingEncoder


class Lambda(nn.Module):
    """Lambda function that accepts tensors as input."""

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x: th.Tensor):
        return self.func(x)


class VAE(nn.Module):
    """Variational Autoencoder for reconstruction of tracking observations."""

    def __init__(
        self,
        embedding_dim: int = 6,
        num_heads: int = 10,
        latent_dim: int = 10,
        input_dim: int = 7,
        num_layers: int = 1,
        inference_mode: bool = False,
        rnn_hidden_dim: int = 20,
        rnn_type: nn.Module = nn.LSTM,
        bidirectional: bool = False,
        max_seq_len: int = 10,
    ):
        """
        Args:
            embedding_dim (int): Dimension of the embedding space
            num_heads (int): Number of attention heads
            latent_dim (int): Dimension of the latent space
            input_dim (int): Dimensions of the input tensor
            num_layers (int): Number of GRU layers
            inference_mode (bool): Inference mode
            rnn_hidden_dim (int): Hidden dimension of the RNN
            rnn_type (nn.Module): Type of RNN module to use (GRU or LSTM)
            bidirectional (bool): Whether to use a bidirectional RNN
            max_seq_len (int): Maximum sequence length
        """
        super(VAE, self).__init__()
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.encoder = TrackingEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
        )
        self.decoder = TrackingDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            bidirectional=bidirectional,
        )

        self.mvnormal = th.distributions.MultivariateNormal(th.zeros(self.latent_dim), 0.6 * th.eye(self.latent_dim))

        self.mean_params = Lambda(lambda x: x[:, : self.latent_dim])  # mean parameters
        self.logvar_params = Lambda(lambda x: x[:, self.latent_dim :])  # log variance parameters

        num_params = sum(p.numel() for p in self.parameters())
        # print(f"Initialized tracking RNN Attention-VAE with {num_params} parameters")

    def preprocess_obs(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if observations.ndim < 3:
            observations = observations.unsqueeze(0)
        # extract length of valid obstacle observations
        seq_lengths = (
            th.sum(observations[:, 0, :] < 0.99, dim=1).to("cpu").type(th.int64)
        )  # idx 0 is normalized distance, where vals = 1.0 is max dist of 1e4++ and thus not valid
        observations = observations.permute(0, 2, 1)  # permute to (batch, max_seq_len, input_dim)

        return observations, seq_lengths

    def forward(self, x: th.Tensor, seq_lengths: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Do a forward pass of the VAE. Generates a reconstructed tracking observation based on input observation.

        Args:
            x (th.Tensor): The input observation with shape (batch_size, max_seq_len, input_dim).
            seq_lengths (th.Tensor): The length of the sequences in the batch.

        Returns:
            Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]: The reconstructed observation, mean, log variance, and possibly sampled latent vector.
        """
        z = self.encoder(x, seq_lengths)
        z_sampled, mean, logvars = self.sample(z)

        x_recon = self.decoder(z_sampled, max_seq_len=self.max_seq_len)
        return x_recon, mean, logvars, z_sampled

    def sample(self, z: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample/reparameterization trick using the encoder output

        Args:
            z (th.Tensor): The input latent representation.

        Returns:
            Tuple[th.Tensor, th.Tensor, th.Tensor]: The sampled ltent vector, mean, and logvar vectors.
        """
        # Reparametrization trick
        mean = self.mean_params(z)
        logvars = self.logvar_params(z)
        logvars = th.log(logvars.exp() + 1e-7)  # To avoid singularity
        std = th.exp(0.5 * logvars)
        # eps = th.randn_like(std)
        eps = self.mvnormal.sample(sample_shape=mean.shape[:1]).to(z.device)

        if self.inference_mode:
            eps = th.zeros_like(eps)

        z_sampled = mean + eps * std
        return z_sampled, mean, logvars

    def encode(self, x: th.Tensor, seq_lengths: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Do a forward pass of the VAE. Generates a latent vector based on input tracking observation.

        Args:
            x (th.Tensor): The input observation with shape (batch_size, max_seq_len, input_dim).
            seq_lengths (th.Tensor): The length of the sequences in the batch.

        Returns:
            th.Tensor: The latent vector, mean and std vectors.
        """
        z = self.encoder(x, seq_lengths)

        z_sampled, means, logvars = self.sample(z)

        return z_sampled, means, logvars

    def decode(self, z: th.Tensor, max_seq_length: int) -> th.Tensor:
        """Generates a reconstructed tracking observation based on input latent vector.

        Args:
            z (th.Tensor): The input latent vector.
            max_seq_length (int): The maximum sequence length.

        Returns:
            th.Tensor: The reconstructed observation.
        """
        x_recon = self.decoder(z, max_seq_length)
        return x_recon

    def set_inference_mode(self, mode):
        self.inference_mode = mode


if __name__ == "__main__":

    latent_dimension = 10
    x = th.rand(2, 10, 6).to("cuda")
    seq_lengths = th.tensor([10, 5])

    vae = VAE(
        latent_dim=latent_dimension, input_dim=6, num_layers=1, rnn_type=nn.LSTM, num_heads=10, embedding_dim=100
    ).to("cuda")
    out = vae(x, seq_lengths)
    print(f"Output shape: {out[0].shape}")
