"""
    encoder.py

    Summary:
        Contains the encoder network for processing images from the environment.

    Author: Trym Tengesdal
"""

from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


def weights_init(m):
    """Initializes the weights of the network"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(m.bias)


class TrackingEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 5,
        num_layers: int = 1,
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

        super(TrackingEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=latent_dim, num_layers=num_layers, batch_first=True)

    def encode(self, x: th.Tensor) -> Tuple[th.Tensor]:
        batch_size = x.shape[0]
        hidden = th.zeros(self.num_layers, batch_size, self.latent_dim)

        # extract length of valid obstacle observations
        seq_lengths = th.sum(
            x[:, 0, :] < 0.9, dim=1
        )  # idx 0 is normalized distance, where vals = 1.0 is max dist of 1e4++ and thus not valid
        seq_lengths[seq_lengths == 0] = 1
        max_seq_length = seq_lengths.max().item()
        x = x[:, :, :max_seq_length]

        x = x.permute(0, 2, 1)
        packed_seq = rnn_utils.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_seq, hidden)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        hidden_out = hidden.permute(1, 0, 2).reshape(-1, self.num_layers * self.latent_dim)
        return hidden_out

    def init_hidden(self, batch_size: int) -> th.Tensor:
        return th.zeros(self.num_layers, batch_size, self.latent_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Encodes the input into a latent vector"""
        return self.encode(x)


if __name__ == "__main__":
    from torchsummary import summary

    #
    latent_dimension = 10
    encoder = TrackingEncoder(input_dim=6, latent_dim=latent_dimension, num_layers=1).to("cpu")
    summary(encoder, (6, 10), device="cpu")
