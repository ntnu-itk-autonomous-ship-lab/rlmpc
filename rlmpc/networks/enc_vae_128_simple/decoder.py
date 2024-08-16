"""
    decoder.py

    Summary:
        Contains the decoder network for reconstructing images from the latent space.

    Author: Trym Tengesdal
"""

from typing import List, Tuple

import torch as th
import torch.nn as nn


class ENCDecoder(nn.Module):
    """Generates an ENC image reconstruction using ConvTranspose2d layers.

    Adapted from https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation/blob/master/racing_models/cmvae.py
    """

    def __init__(
        self,
        n_input_channels: int = 3,
        latent_dim: int = 128,
        first_deconv_input_dim: Tuple[int, int] = (128, 9, 9),
        deconv_block_dims: List[int] = [256, 128, 128, 32],
        fc_dim: int = 32,
    ):
        """

        Args:
            n_input_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            first_deconv_input_dim (Tuple[int, int]): Dimensions of the first deconvolutional block
            deconv_block_dims (List[int]): Dimensions of the deconvolutional blocks
            fc_dim (int): Dimension of the fully connected layer
        """

        super(ENCDecoder, self).__init__()

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

        self.fc_dim = fc_dim
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, first_deconv_input_dim[0] * first_deconv_input_dim[1] * first_deconv_input_dim[2]),
        )

        deconv_input_channels = first_deconv_input_dim[0]
        # Pytorch docs: output_padding is only used to find output shape, but does not actually add zero-padding to output
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=deconv_input_channels, out_channels=deconv_block_dims[0], kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=deconv_block_dims[0], out_channels=deconv_block_dims[1], kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=deconv_block_dims[1], out_channels=deconv_block_dims[2], kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=deconv_block_dims[2],
                out_channels=n_input_channels,
                kernel_size=4,
                stride=4,
                padding=2,
                dilation=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        return self.decode(z)

    def decode(self, z: th.Tensor) -> th.Tensor:
        # Fully connected layers
        x = self.fc_block(z)
        x = x.view(
            x.size(0), self.first_deconv_input_dim[0], self.first_deconv_input_dim[1], self.first_deconv_input_dim[2]
        )

        x = self.deconv_block(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    latent_dimension = 32
    img_decoder = ENCDecoder(
        latent_dim=latent_dimension,
        n_input_channels=1,
        first_deconv_input_dim=(128, 5, 5),
        deconv_block_dims=[128, 64, 32],
        fc_dim=512,
    ).to("cuda")
    summary(img_decoder, (1, latent_dimension), device="cuda")
