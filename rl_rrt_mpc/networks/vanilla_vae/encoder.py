"""
    encoder.py

    Summary:
        Contains the encoder network for processing images from the environment.

    Author: Trym Tengesdal
"""

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

    def __init__(self, n_input_channels: int, latent_dim: int) -> None:
        """

        Args:
            n_input_channels: Number of input channels
            latent_dim: Dimension of the latent space
        """

        # Formula for output image/tensor size after conv2d:
        # ((n - f + 2p) / s) + 1, where
        # n = input number of pixels (assume square image)
        # f = number of kernels (assume square kernel)
        # p = padding
        # s = stride
        # Number of output channels are random/tuning parameter.

        # Zeroth block:
        # => for 3x3 kernel, stride 1, padding 0, input 300x300 image:
        # ((300 - 3 + 2 * 0) / 1 + 1 = 298x298
        # => for 3x3 kernel, stride 2, padding 0, input 298x298 image:
        # ((298 - 4 + 2*0) / 2) + 1 = 148x148

        # First block:
        # => for 3x3 kernel, stride 1, padding 1, input 148x148 image:
        # ((148
        # => for 3x3 kernel, stride 1, padding 1, input 37x37 image:
        # ((37 - 4 + 2*1) / 1) + 1 = 36x36

        # Second block:
        # => for 3x3 kernel, stride 2, padding 1 input 36x36 image:
        # ((35 - 5 + 2*1) / 2) + 1 = 17x17
        # => for 3x3 kernel, stride 1, padding 1:
        # ((17 - 4 + 2*1) / 1) + 1 = 16x16

        # Jump connection block 0 to 2: input image 75x75: desired size: 36x36
        # => for 5x5 kernel, stride 2, padding 0, input 75x75 image:
        # ((75 - 5 + 2*0) / 2) + 1 = 36x36

        # Jump connection block 1 to 3: input image 36x36: desired size: 16x16
        # => for 6x6 kernel, stride 2, padding 0:
        # ((36 - 6 + 2*0) / 2) + 1 = 16x16

        # Third block:
        # => for 4x4 kernel, stride 2, padding 1, input 16x16 image:
        # ((16 - 4 + 2*1) / 2) + 1 = 8x8

        super(PerceptionImageEncoder, self).__init__()
        self.n_input_channels = n_input_channels
        self.latent_dim = latent_dim

        self.elu = nn.ELU()

        # Zeroth block of convolutions
        self.conv00: nn.Conv2d = nn.Conv2d(
            in_channels=n_input_channels, out_channels=16, kernel_size=3, stride=1, padding=0
        )
        self.conv01: nn.Conv2d = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0)
        nn.init.xavier_uniform_(self.conv00.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv00.bias)
        nn.init.xavier_uniform_(self.conv01.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv01.bias)

        # First block of convolutions
        self.conv10: nn.Conv2d = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv10.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv10.bias)

        self.conv11: nn.Conv2d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        nn.init.xavier_uniform_(self.conv11.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv11.bias)

        # # Second block of convolutions
        # # ELU activation function
        self.conv20: nn.Conv2d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv20.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv20.bias)

        # self.conv21: nn.Conv2d = nn.Conv2d(in_channels=128, out_channels=latent_dim, kernel_size=3, stride=1, padding=1)
        # nn.init.xavier_uniform_(self.conv21.weight, gain=nn.init.calculate_gain("linear"))
        # nn.init.zeros_(self.conv21.bias)

        # Jump connection from last layer of zeroth block to first layer of second block
        self.conv0_jump_to_2: nn.Conv2d = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0)
        # Jump connection from last layer of first block to first layer of second block
        self.conv1_jump_to_3: nn.Conv2d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        # Third (Fourth) block of convolutions
        self.conv30: nn.Conv2d = nn.Conv2d(in_channels=64, out_channels=latent_dim, kernel_size=3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv30.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv30.bias)

        # Fully connected layers
        self.fc0 = nn.Linear(in_features=25 * 25 * latent_dim, out_features=2 * latent_dim)
        # ELU activation function
        # self.fc1 = nn.Linear(in_features=2 * 512, out_features=2 * latent_dim)

    def encode(self, image: th.Tensor) -> th.Tensor:
        """Encodes the input image into a latent vector"""
        # Zeroth block of convolutions
        x00 = self.conv00(image)
        # print(f"x00 shape: {x00.shape}")
        x01 = self.elu(self.conv01(x00))
        # print(f"x01 shape: {x01.shape}")

        # First block of convolutions
        x10 = self.conv10(x01)
        # print(f"x10 shape: {x10.shape}")
        x11 = self.conv11(x10)
        # print(f"x11 shape: {x11.shape}")

        x0_jump_to_2 = self.conv0_jump_to_2(x01)
        # print(f"x0_jump_to_2 shape: {x0_jump_to_2.shape}")
        x11 = x11 + x0_jump_to_2

        # Second block of convolutions
        x20 = self.conv20(self.elu(x11))
        # print(f"x20 shape: {x20.shape}")
        x21 = x20  # self.conv21(x20)

        x1_jump_to_3 = self.conv1_jump_to_3(x11)
        # print(f"x1_jump_to_3 shape: {x1_jump_to_3.shape}")
        x21 = x21 + x1_jump_to_3

        # Third (Fourth) block of convolutions
        x30 = self.conv30(self.elu(x21))
        # print(f"x30 shape: {x30.shape}")

        # Fully connected layers
        x40 = self.fc0(x30.view(x30.size(0), -1))
        x40 = self.elu(x40)
        # x41 = self.fc1(x40)
        x = x40
        return x

    def forward(self, image: th.Tensor) -> th.Tensor:
        """Encodes the input image into a latent vector"""
        return self.encode(image)


if __name__ == "__main__":
    from torchsummary import summary

    latent_dimension = 64
    encoder = PerceptionImageEncoder(latent_dim=latent_dimension, n_input_channels=3).to("cuda")
    summary(encoder, (3, 400, 400), device="cuda")
