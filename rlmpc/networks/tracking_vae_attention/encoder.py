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
        input_dim: int = 6,
        latent_dim: int = 5,
        num_layers: int = 1,
        fc_dim: int = 1024,
        rnn_type: nn.Module = nn.GRU,
    ) -> None:
        """

        Args:
            input_dim (int: Dimensions of the input tensor
            latent_dim (int): Dimension of the latent space
            num_layers (int): Number of GRU layers
            fc_dim (int): Dimension of the fully connected layer.
            rnn_type (nn.Module): Type of RNN module to use (GRU or LSTM)
        """
        super(TrackingEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.fc_dim = fc_dim
        self.rnn1 = rnn_type(input_size=self.input_dim, hidden_size=latent_dim, num_layers=num_layers, batch_first=True)
        self.fc0 = nn.Linear(self.latent_dim, fc_dim)  # 2 * self.latent_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(fc_dim, 2 * self.latent_dim)

    def encode(self, x: th.Tensor, seq_lengths: th.Tensor) -> th.Tensor:
        """Encodes the input into the VAE mean and logvar

        Args:
            x (th.Tensor): Input tensor of shape (batch_size, max_seq_len, input_dim)
            seq_lengths (th.Tensor): Length of the sequences in the batch

        Returns:
            th.Tensor: Encoder output of shape (batch_size, 2 * latent_dim)
        """
        # print(
        #     f"Input shape: {x.shape}, seq_lengths shape: {seq_lengths.shape}, Input (min, max): ({x.min()}, {x.max()})"
        # )
        batch_size = x.shape[0]
        hidden = th.zeros(self.num_layers, batch_size, self.latent_dim).to(x.device)
        packed_seq = rnn_utils.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        if isinstance(self.rnn1, nn.LSTM):
            cell = th.zeros_like(hidden)
            output_seq, (hidden, last_cell) = self.rnn1(packed_seq, (hidden, cell))
        else:  # GRU
            output_seq, hidden = self.rnn1(packed_seq, hidden)
        # unpacked, _ = rnn_utils.pad_packed_sequence(output_seq, batch_first=True)

        # hidden dim = (num_layers, batch_size, latent_dim) and we want to reshape to (batch_size, num_layers * latent_dim)
        last_hidden = hidden[-1].unsqueeze(0)
        last_hidden = last_hidden.permute(1, 0, 2).reshape(-1, self.latent_dim)

        z_enc = self.fc0(last_hidden)
        z_enc = self.elu(z_enc)
        # z_enc = self.dropout(z_enc)
        z_enc = self.fc1(z_enc)
        return z_enc

    def init_hidden(self, batch_size: int) -> th.Tensor:
        return th.zeros(self.num_layers, batch_size, self.latent_dim)

    def forward(self, x: th.Tensor, seq_lengths: th.Tensor) -> th.Tensor:
        z_enc = self.encode(x, seq_lengths)
        return z_enc


if __name__ == "__main__":
    latent_dimension = 10
    encoder = TrackingEncoder(input_dim=6, latent_dim=latent_dimension, num_layers=1, rnn_type=nn.LSTM).to("cpu")

    x = th.rand(2, 6, 6)
    seq_lengths = th.tensor([6, 5])
    out = encoder(x, seq_lengths)
    print(f"In: {x.shape}, Out: {out.shape}")
