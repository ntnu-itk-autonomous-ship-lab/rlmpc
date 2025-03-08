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
        embedding_dim: int = 100,
        num_heads: int = 10,
        latent_dim: int = 5,
    ) -> None:
        """

        Args:
            input_dim (int: Dimensions of the input tensor
            embedding_dim (int): Dimension of the embedding space
            num_heads (int): Number of attention heads
            latent_dim (int): Dimension of the latent space
        """
        super(TrackingEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.enable_rnn = False

        if self.enable_rnn:
            self.rnn_hidden_dim = embedding_dim // 2
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        self.fc1 = nn.Linear(
            self.input_dim, self.input_dim if self.enable_rnn else self.embedding_dim
        )
        self.multihead_attention1 = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.multihead_attention2 = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.elu = nn.ELU()
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.fc4 = nn.Linear(embedding_dim, 2 * latent_dim)

        self.padding_token = nn.Parameter(th.zeros(embedding_dim))

        self.init_weights(self.fc1)
        self.init_weights(self.fc2)
        self.init_weights(self.fc3)
        self.init_weights(self.fc4)

    def encode(self, x: th.Tensor, seq_lengths: th.Tensor) -> th.Tensor:
        """Encodes the input into the VAE mean and logvar

        Args:
            x (th.Tensor): Input tensor of shape (batch_size, max_seq_len, input_dim)
            seq_lengths (th.Tensor): Length of the sequences in the batch

        Returns:
            th.Tensor: Encoder output of shape (batch_size, 2 * latent_dim)
        """
        x = self.fc1(x)  # (batch_size, seq_len, embedding_dim)

        # Create an attention mask
        seq_len = x.size(1)
        mask = th.arange(seq_len).unsqueeze(0).to(x.device) < (
            seq_len - seq_lengths.to(x.device)
        ).unsqueeze(1)  # (batch_size, seq_len)

        # print("Seq lengths are zero!")
        idx_nan = th.where(seq_lengths == 0)
        idx_nonnan = th.where(seq_lengths != 0)
        max_seq_len = seq_lengths.max().item()

        x_nonnan = x[idx_nonnan]

        attn_input = x_nonnan
        attn_output1 = th.zeros_like(x)
        if self.enable_rnn:
            attn_input = th.zeros_like(x_nonnan)
            packed_seq = rnn_utils.pack_padded_sequence(
                x_nonnan,
                seq_lengths[idx_nonnan],
                batch_first=True,
                enforce_sorted=False,
            )
            attn_input_packed, _ = self.rnn(packed_seq)
            rnn_output, _ = rnn_utils.pad_packed_sequence(
                attn_input_packed, batch_first=True
            )
            attn_input = rnn_output
            mask = mask[:, -max_seq_len:]
            attn_output = th.zeros(x.shape[0], max_seq_len, self.embedding_dim).to(
                x.device
            )

        mask = mask.to(x.device)
        attn_output_sub1, _ = self.multihead_attention1(
            attn_input, attn_input, attn_input, key_padding_mask=mask[idx_nonnan]
        )
        attn_output1[idx_nonnan] = attn_output_sub1 + x_nonnan
        attn_output1[idx_nan] = self.padding_token

        attn_output1 = self.layer_norm1(attn_output1)
        attn_output1 = self.fc2(attn_output1)

        attn_output_sub2, _ = self.multihead_attention2(
            attn_output1[idx_nonnan],
            attn_output1[idx_nonnan],
            attn_output1[idx_nonnan],
            key_padding_mask=mask[idx_nonnan],
        )
        attn_output2 = th.zeros_like(x)
        attn_output2[idx_nonnan] = attn_output_sub2 + attn_output1[idx_nonnan]
        attn_output2[idx_nan] = self.padding_token
        # attn_output2 = self.layer_norm2(attn_output2)

        # Mean pooling over the valid tokens (ignoring padding)
        valid_seq_lengths = (
            seq_lengths.clamp(min=1).unsqueeze(1).to(x.device)
        )  # (batch_size, 1)

        # Max pooling over the valid tokens (ignoring padding)
        # pooled_output = attn_output2.max(dim=1).values  # (batch_size, embedding_dim)

        # Average pooling over the valid tokens (ignoring padding)
        pooled_output = (
            attn_output2.sum(dim=1) / valid_seq_lengths.float()
        )  # (batch_size, embedding_dim)

        attn_output2 = self.fc3(attn_output2)
        attn_output2 = self.elu(attn_output2)
        z_enc = self.fc4(pooled_output)
        return z_enc

    def forward(self, x: th.Tensor, seq_lengths: th.Tensor) -> th.Tensor:
        z_enc = self.encode(x, seq_lengths)
        return z_enc

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # print(f"Initialize layer with nn.init.xavier_uniform_: {layer}")
            th.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def init_hidden(self, batch_size: int) -> th.Tensor:
        return th.zeros(self.num_layers, batch_size, self.latent_dim)


if __name__ == "__main__":
    latent_dimension = 10
    encoder = TrackingEncoder(
        input_dim=4, embedding_dim=12, num_heads=6, latent_dim=latent_dimension
    )

    x = th.rand(2, 6, 4)
    seq_lengths = th.tensor([6, 5])
    out = encoder(x, seq_lengths)
    print(f"In: {x.shape}, Out: {out.shape}")
    print(
        f"Encoder number of parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}"
    )
