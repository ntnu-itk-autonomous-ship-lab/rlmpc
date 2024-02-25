import os

import torch
import torch.distributed as dist_fn
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook**2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            "Trying to call `.grad()` on graph containing "
            "`VectorQuantization`. The function `VectorQuantization` "
            "is not differentiable. Use `VectorQuantizationStraightThrough` "
            "if you want a straight-through estimator of the gradient."
        )


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = grad_output.contiguous().view(-1, embedding_size)
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = VectorQuantization.apply(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = VectorQuantizationStraightThrough.apply(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, in_dim: int = 256, num_res_channels: int = 32, out_dim: int = 256):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, num_res_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_res_channels, out_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim: int = 3, latent_dim: int = 3, num_res_channels=32, conv_dim: int = 256, K: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = K
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=4, stride=2, padding=1),
            ResBlock(in_dim=conv_dim, num_res_channels=num_res_channels, out_dim=conv_dim),
            ResBlock(in_dim=conv_dim, num_res_channels=num_res_channels, out_dim=conv_dim),
        )

        self.codebook = VQEmbedding(K, latent_dim)

        self.decoder = nn.Sequential(
            ResBlock(latent_dim, conv_dim),
            ResBlock(conv_dim, conv_dim),
            nn.ConvTranspose2d(conv_dim, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(conv_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


if __name__ == "__main__":
    import torchsummary

    device = torch.device("cpu")
    model = VectorQuantizedVAE(input_dim=3, latent_dim=5, conv_dim=256, K=512).to(device)
    torchsummary.summary(model, (3, 256, 256))
