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


class ENCEncoder(nn.Module):
    """Encoder network for processing ENC images from the environment"""

    def __init__(
        self,
        n_input_channels: int,
        latent_dim: int,
        conv_block_dims: Tuple[int, int, int, int] = (32, 64, 128),
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

        super(ENCEncoder, self).__init__()
        self.n_input_channels = n_input_channels
        self.latent_dim = latent_dim

        self.block_0_dim = conv_block_dims[0]
        self.conv_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=self.block_0_dim, kernel_size=5, stride=2, padding=3),
            nn.Conv2d(in_channels=self.block_0_dim, out_channels=self.block_0_dim, kernel_size=3, stride=2, padding=3),
            nn.ELU(),
        )

        self.block_1_dim = conv_block_dims[1]
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.block_0_dim, out_channels=self.block_1_dim, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=self.block_1_dim, out_channels=self.block_1_dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

        # Jump connection from last layer of zeroth block to first layer of second block
        self.conv0_jump_to_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.block_0_dim, out_channels=self.block_1_dim, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )

        # Second (Third) block of convolutions
        self.block_2_dim = conv_block_dims[2]
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.block_1_dim, out_channels=self.block_2_dim, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )

        # Jump connection from last layer of first block to FC layer
        self.conv1_jump_to_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.block_1_dim, out_channels=self.block_2_dim, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
        )

        # Fully connected layers
        self.last_conv_block_dim = (self.block_2_dim, 9, 9)
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
        debug = False

        x0 = self.conv_block_0(image)
        if debug:
            print("x0.shape", x0.shape)

        x1 = self.conv_block_1(x0)
        if debug:
            print("x1.shape", x1.shape)

        x0_jump_to_2 = self.conv0_jump_to_2(x0)
        if debug:
            print("x0_jump_to_2.shape", x0_jump_to_2.shape)

        x1 = x1 + x0_jump_to_2

        x2 = self.conv_block_2(x1)
        if debug:
            print("x2.shape", x2.shape)

        x1_jump_to_3 = self.conv1_jump_to_3(x1)
        if debug:
            print("x1_jump_to_3.shape", x1_jump_to_3.shape)
        x2 = x2 + x1_jump_to_3

        x3 = self.fc_block(x2.view(-1, self.last_conv_block_flattened_dim))
        # print("x4.shape", x4.shape)
        return x3

    def forward(self, image: th.Tensor) -> th.Tensor:
        """Encodes the input image into a latent vector"""
        return self.encode(image)


if __name__ == "__main__":
    from torchsummary import summary

    #
    latent_dimension = 32
    encoder = ENCEncoder(
        n_input_channels=1, latent_dim=latent_dimension, conv_block_dims=(64, 128, 128), fc_dim=128
    ).to("cuda")
    summary(encoder, (1, 128, 128), device="cuda")
