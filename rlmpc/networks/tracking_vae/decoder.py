"""
    decoder.py

    Summary:
        Contains the decoder network for reconstructing tracking observations from the latent space.

    Author: Trym Tengesdal
"""

import torch as th
import torch.nn as nn


class TrackingDecoder(nn.Module):
    """Generates a tracking observation reconstruction."""

    def __init__(
        self,
        latent_dim: int = 10,
        output_dim: int = 6,
        num_layers: int = 1,
        fc_dim: int = 1024,
        rnn_type: nn.Module = nn.GRU,
    ):
        """
        Args:
            latent_dim (int): Dimension of the latent space,
            output_dim (int): Number of output from the GRU, should be equal to the input_dim of the encoder
            num_layers (int): Number of GRU layers.
            fc_dim (int): Dimension of the fully connected layer.
            rnn_type (nn.Module): Type of RNN module to use (GRU or LSTM)
        """
        super(TrackingDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.fc_dim = fc_dim
        self.fc0 = nn.Linear(self.latent_dim, fc_dim)
        self.relu = nn.ReLU()
        self.dropout0 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(fc_dim, self.latent_dim)
        self.dropout1 = nn.Dropout(p=0.2)

        self.rnn = rnn_type(input_size=latent_dim, hidden_size=latent_dim, num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Linear(self.latent_dim, output_dim)

    def forward(self, z: th.Tensor, max_seq_len: int) -> th.Tensor:
        return self.decode(z, max_seq_len)

    def decode(self, z: th.Tensor, max_seq_len: int) -> th.Tensor:
        z = z.unsqueeze(1).repeat(1, max_seq_len, 1)
        z = self.fc0(z)
        z = self.relu(z)
        z = self.dropout0(z)
        z = self.fc1(z)
        z = self.dropout1(z)
        output, _ = self.rnn(z)  # hidden state is dont care for decoder.
        output = self.fc2(output)
        return output

    def init_hidden(self, batch_size: int) -> th.Tensor:
        return th.zeros(self.num_layers, batch_size, self.latent_dim)


if __name__ == "__main__":
    latent_dimension = 10
    decoder = TrackingDecoder(latent_dim=latent_dimension, output_dim=6, num_layers=1, rnn_type=nn.LSTM).to("cuda")
    x = th.rand(2, latent_dimension).to("cuda")
    out = decoder(x, max_seq_len=6)
    print(f"In: {x.shape}, Out: {out.shape}")
