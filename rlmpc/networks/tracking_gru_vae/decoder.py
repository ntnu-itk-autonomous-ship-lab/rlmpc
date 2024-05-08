"""
    decoder.py

    Summary:
        Contains the decoder network for reconstructing images from the latent space.

    Author: Trym Tengesdal
"""

from typing import Tuple

import torch as th
import torch.nn as nn


class TrackingDecoder(nn.Module):
    """Generates a tracking observatio   reconstruction."""

    def __init__(
        self,
        input_dim: int = 10,
        latent_dim: int = 6,
        num_layers: int = 1,
    ):
        """

        Args:
            input_dim (int): Number of input channels to GRUs, equal to the encoder latent dim
            latent_dim (int): Dimension of the latent space,
            num_layers (int): Number of GRU layers.
        """
        super(TrackingDecoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=latent_dim, num_layers=num_layers, batch_first=True)

    def forward(self, z: th.Tensor) -> th.Tensor:
        return self.decode(z)

    def decode(self, z: th.Tensor) -> th.Tensor:

        x = z
        return x


if __name__ == "__main__":
    from torchsummary import summary

    latent_dimension = 128
    img_decoder = TrackingDecoder(latent_dim=latent_dimension, input_dim=6, num_layers=1).to("cuda")
    summary(img_decoder, (1, latent_dimension), device="cuda")
