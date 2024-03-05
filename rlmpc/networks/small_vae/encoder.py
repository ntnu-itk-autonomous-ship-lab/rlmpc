"""
    encoder.py

    Summary:
        Contains the encoder network for processing images from the environment.

    Author: Trym Tengesdal
"""

from typing import Tuple

import torch as th
import torch.nn as nn


def weights_init(m):
    """Initializes the weights of the network"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(m.bias)


class PerceptionImageEncoder(nn.Module):
    """We use a ResNet8 architecture for now."""

    def __init__(
        self,
        n_input_channels: int,
        latent_dim: int,
        conv_block_dims: Tuple[int, int, int, int] | int = (32, 64, 128, 128),
        fc_dim: int = 32,
    ) -> None:
        """

        Args:
            n_input_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            conv_block_dims (Tuple[int, int, int, int]): Dimensions of the convolutional blocks
        """

        # Formula for output image/tensor size after conv2d:
        # ((n - f + 2p) / s) + 1, where
        # n = input number of pixels (assume square image)
        # f = number of kernels (assume square kernel)
        # p = padding
        # s = stride
        # Number of output channels are random/tuning parameter.

        super(PerceptionImageEncoder, self).__init__()
        self.n_input_channels = n_input_channels
        self.latent_dim = latent_dim

        if isinstance(conv_block_dims, int):
            hidden_dim = conv_block_dims

        self.conv_block = nn.Sequential(
            nn.Conv2d(n_input_channels, hidden_dim, 5, 2, 2, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, 2, 2, bias=False),
            # nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.to_latent = nn.Sequential(
            nn.Conv2d(hidden_dim, latent_dim, 1, 1, 0, bias=True),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

        # Fully connected layers
        self.last_conv_block_dim = (self.latent_dim, 10, 10)
        self.last_conv_block_flattened_dim = (
            self.last_conv_block_dim[0] * self.last_conv_block_dim[1] * self.last_conv_block_dim[2]
        )
        self.fc_dim = fc_dim
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=self.last_conv_block_flattened_dim, out_features=self.fc_dim),
            nn.ELU(),
            nn.Linear(in_features=self.fc_dim, out_features=2 * latent_dim),
        )

    def encode(self, image: th.Tensor) -> th.Tensor:
        """Encodes the input image into a latent vector"""
        x = self.conv_block(x)
        x = x + self.res_block(x)
        x = self.bn(x)
        x = self.to_latent(x)
        x4 = self.fc_block(x.view(-1, self.last_conv_block_flattened_dim))
        return x4

    def forward(self, image: th.Tensor) -> th.Tensor:
        """Encodes the input image into a latent vector"""
        return self.encode(image)


if __name__ == "__main__":
    from torchsummary import summary

    #
    latent_dimension = 32
    encoder = PerceptionImageEncoder(
        n_input_channels=3, latent_dim=latent_dimension, conv_block_dims=(128, 128, 128, 128), fc_dim=512
    ).to("cuda")
    summary(encoder, (3, 256, 256), device="cuda")
