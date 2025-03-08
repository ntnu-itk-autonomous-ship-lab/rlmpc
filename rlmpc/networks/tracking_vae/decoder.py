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
        rnn_hidden_dim: int = 256,
        rnn_type: nn.Module = nn.GRU,
        bidirectional: bool = False,
    ):
        """
        Args:
            latent_dim (int): Dimension of the latent space,
            output_dim (int): Number of output from the GRU, should be equal to the input_dim of the encoder
            num_layers (int): Number of GRU layers.
            fc_dim (int): Dimension of the fully connected layer.
            rnn_hidden_dim (int): Hidden dimension of the RNN.
            rnn_type (nn.Module): Type of RNN module to use (GRU or LSTM)
            bidirectional (bool): Whether the RNN is bidirectional.
        """
        super(TrackingDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.fc_dim = fc_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.output_dim = output_dim
        self.elu = nn.ELU()
        self.bidirectional = bidirectional
        self.rnn = rnn_type(
            input_size=latent_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1,
        )
        rnn_output_dim = rnn_hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(rnn_output_dim, output_dim)
        self.init_weights(self.fc1)

    def forward(self, z: th.Tensor, max_seq_len: int) -> th.Tensor:
        return self.decode(z, max_seq_len)

    def decode(self, z: th.Tensor, max_seq_len: int) -> th.Tensor:
        z = z.unsqueeze(1).repeat(1, max_seq_len, 1)
        output, _ = self.rnn(z)  # output hidden state is dont care for decoder.
        output = self.fc1(output)
        output = self.elu(output)
        # print(f"Decoder output shape: {output.shape}, output (min, max): ({output.min()}, {output.max()})")
        return output

    def init_hidden(self, batch_size: int) -> th.Tensor:
        return th.zeros(self.num_layers, batch_size, self.latent_dim)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            print(f"Initialize layer with nn.init.xavier_uniform_: {layer}")
            th.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)


if __name__ == "__main__":
    latent_dimension = 10
    decoder = TrackingDecoder(
        latent_dim=latent_dimension, output_dim=6, num_layers=1, rnn_type=nn.LSTM
    ).to("cuda")
    x = th.rand(2, latent_dimension).to("cuda")
    out = decoder(x, max_seq_len=6)
    print(f"In: {x.shape}, Out: {out.shape}")
