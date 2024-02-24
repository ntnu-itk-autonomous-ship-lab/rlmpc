"""
    decoder.py

    Summary:
        Contains the decoder network for reconstructing images from the latent space.

    Author: Trym Tengesdal
"""

import torch as th
import torch.nn as nn


class PerceptionImageDecoder(nn.Module):
    """Generates a perception image reconstruction using ConvTranspose2d layers.

    Adapted from https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation/blob/master/racing_models/cmvae.py
    """

    def __init__(self, n_input_channels: int = 3, latent_dim: int = 128, first_deconv_input_dim: int = 8):
        """

        Args:
            n_input_channels: Number of input channels
            latent_dim: Dimension of the latent space
            first_deconv_input_dim: Dimension of the first input convolutional layer
        """

        super(PerceptionImageDecoder, self).__init__()

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.n_input_channels = n_input_channels
        self.first_deconv_input_dim = first_deconv_input_dim
        self.latent_dim = latent_dim

        #  Formula for output image/tensor size after deconv2d:
        # n = (o - 1) * s + f - 2p
        # o = ((n - f + 2p) / s) + 1, where
        # n = input number of pixels (assume square image)
        # o = output number of pixels (assume square image), after convolution
        # f = number of kernels (assume square kernel)
        # p = padding
        # s = stride
        # Number of output channels are random/tuning parameter.

        # Zeroth deconv:
        # Input 256 x 8 x 8
        # => for 3x3 kernel, stride 1, padding 1, input 8x8 image:
        # (8 - 1) * 1 + 3 - 2*1 = 8x8
        # First deconv:
        # => for 4x4 kernel, stride 2, padding (1, 1), input 8x8 image:
        # (8 - 1) * 2 + 4 - 2*1 = 16x16
        # Second deconv:
        # => for 5x5 kernel, stride 2, padding (1, 1), input 16x16 image:
        # (16 - 1) * 2 + 5 - 2*1 = 33x33
        # Third deconv:
        # => for 6x6 kernel, stride 3, padding (1, 1), input 33x33 image:
        # (33 - 1) * 3 + 6 - 2*1 = 100x100
        # Fourth deconv:
        # => for 6x6 kernel, stride 2, padding (2, 2), input 100x100 image:
        # (100 - 1) * 2 + 6 - 2*2 = 200x200
        # Fifth deconv:
        # => for 4x4 kernel, stride 2, padding (1, 1), input 200x200 image:
        # (200 - 1) * 2 + 4 - 2*1 = 400x400

        # Fully connected layers
        # Version 1
        self.fc0 = nn.Linear(latent_dim, 512)
        # Relu activation
        self.fc1 = nn.Linear(512, self.first_deconv_input_dim * self.first_deconv_input_dim * latent_dim)

        # Pytorch docs: output_padding is only used to find output shape, but does not actually add zero-padding to output

        # Deconvolutional layers
        self.deconv0 = nn.ConvTranspose2d(in_channels=latent_dim, out_channels=256, kernel_size=3, stride=1, padding=1)
        # Relu activation
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # Relu activation
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1)
        # Relu activation
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=3, padding=1)
        # Relu activation
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=2)
        # Relu activation
        self.deconv5 = nn.ConvTranspose2d(16, self.n_input_channels, kernel_size=4, stride=2, padding=1)
        # Sigmoid activation

        # Version 2
        # self.deconv0 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # # Relu activation
        # self.deconv1 = nn.ConvTranspose2d(
        #     128, 64, kernel_size=5, stride=2, padding=(2, 2), output_padding=(0, 1), dilation=1
        # )
        # # Relu activation
        # self.deconv2 = nn.ConvTranspose2d(
        #     64, 32, kernel_size=6, stride=4, padding=(2, 2), output_padding=(0, 0), dilation=1
        # )
        # # Relu activation
        # self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=(0, 0), output_padding=(0, 1))
        # # Relu activation
        # self.deconv4 = nn.ConvTranspose2d(
        #     16, self.n_channels, kernel_size=4, stride=2, padding=2
        # )  # tanh activation or sigmoid
        # Sigmoid activation

    def forward(self, z: th.Tensor) -> th.Tensor:
        return self.decode(z)

    def decode(self, z: th.Tensor) -> th.Tensor:
        # Fully connected layers
        # Version 1
        x = self.fc0(z)
        x = self.relu(x)
        x = self.fc1(x)
        x = x.view(x.size(0), self.latent_dim, self.first_deconv_input_dim, self.first_deconv_input_dim)

        # Deconvolutional layers
        # Version 1
        x = self.deconv0(x)
        x = self.relu(x)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.relu(x)
        x = self.deconv5(x)

        x = self.sigmoid(x)
        # print(f"latent vector: {z}")
        # print(f"After sigmoid, mean: {x.mean():.3f} var: {x.var():.3f}")
        return x


if __name__ == "__main__":
    from torchsummary import summary

    latent_dimension = 256
    img_decoder = PerceptionImageDecoder(latent_dim=latent_dimension, n_input_channels=3).to("cuda")
    summary(img_decoder, (1, latent_dimension), device="cuda")
