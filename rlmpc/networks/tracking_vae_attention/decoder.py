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
        rnn_hidden_dim: int = 256,
        rnn_type: nn.Module = nn.GRU,
        bidirectional: bool = False,
        embedding_dim: int = 512,
        num_heads: int = 8,
    ):
        """
        Args:
            latent_dim (int): Dimension of the latent space,
            output_dim (int): Number of output from the GRU, should be equal to the input_dim of the encoder
            num_layers (int): Number of GRU layers.
            rnn_hidden_dim (int): Hidden dimension of the RNN.
            rnn_type (nn.Module): Type of RNN module to use (GRU or LSTM)
            bidirectional (bool): Whether the RNN is bidirectional.
            embedding_dim (int): Dimension of the embedding space.
            num_heads (int): Number of attention heads.
        """
        super(TrackingDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.disable_rnn = False
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.fc0 = nn.Linear(self.latent_dim, self.embedding_dim)
        self.relu = nn.ReLU()
        self.fc01 = nn.Linear(self.embedding_dim, self.embedding_dim)
        if self.disable_rnn:
            self.multihead_attention1 = nn.MultiheadAttention(
                embed_dim=self.embedding_dim, num_heads=self.num_heads, batch_first=True
            )
            self.fc1 = nn.Linear(self.embedding_dim, self.output_dim)
        else:
            self.rnn_hidden_init = nn.Parameter(
                th.zeros(self.num_layers, 1, self.rnn_hidden_dim)
            )
            self.rnn = rnn_type(
                input_size=self.embedding_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0,
            )
            rnn_output_dim = rnn_hidden_dim * (2 if bidirectional else 1)
            self.h_0 = nn.Parameter(
                th.zeros(
                    self.num_layers * 2 if bidirectional else self.num_layers,
                    1,
                    self.rnn_hidden_dim,
                )
            )
            self.fc1 = nn.Linear(rnn_output_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(self.embedding_dim)
        self.init_weights(self.fc0)
        self.init_weights(self.fc1)

    def forward(self, z: th.Tensor, max_seq_len: int) -> th.Tensor:
        return self.decode(z, max_seq_len)

    def decode(self, z: th.Tensor, max_seq_len: int) -> th.Tensor:
        z = z.unsqueeze(1).repeat(1, max_seq_len, 1)
        z = self.fc0(z)
        z = self.relu(z)
        z = self.fc01(z)

        if self.disable_rnn:
            output, _ = self.multihead_attention1(z, z, z)
            # output = self.layer_norm1(output + z)
        else:
            batch_size = z.size(0)
            h_0 = self.h_0.expand(-1, batch_size, -1).contiguous()
            output, _ = self.rnn(
                z, h_0
            )  # output hidden state is dont care for decoder.

        output = self.fc1(output)
        return output

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # print(f"Initialize layer with nn.init.xavier_uniform_: {layer}")
            th.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def init_hidden(self, batch_size: int) -> th.Tensor:
        return th.zeros(self.num_layers, batch_size, self.latent_dim)


if __name__ == "__main__":
    latent_dimension = 10
    decoder = TrackingDecoder(
        latent_dim=latent_dimension,
        output_dim=4,
        num_layers=2,
        rnn_type=nn.GRU,
        rnn_hidden_dim=128,
        bidirectional=False,
        embedding_dim=128,
        num_heads=2,
    ).to("cuda")
    x = th.rand(2, latent_dimension).to("cuda")
    out = decoder(x, max_seq_len=10)
    print(f"In: {x.shape}, Out: {out.shape}")
    print(
        f"Decoder number of parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad)}"
    )
