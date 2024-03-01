import torch
import torchsummary
from einops import rearrange
from torch import nn
from torch.nn import functional as F

# References:
# https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
# https://keras.io/examples/generative/vq_vae/
# https://github.com/singh-hrituraj/PixelCNN-Pytorch/blob/master/MaskedCNN.py
# https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/05_autoregressive/02_pixelcnn/pixelcnn.ipynb


torch.set_printoptions(linewidth=70)


class Encoder(nn.Module):
    # "All having 256 hidden units."
    def __init__(self, channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        # "The encoder consists of 3 strided convolutional layers with stride 2 and window size 4 × 4,
        # followed by two residual 3 × 3 blocks.
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 5, 2, 2, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, 2, 2, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.to_latent = nn.Sequential(
            nn.Conv2d(hidden_dim, latent_dim, 1, 1, 0, bias=True),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x + self.res_block(x)
        x = self.bn(x)
        x = self.to_latent(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, latents):
        # Compute L2 distances between latents and embedding weights
        dist = torch.linalg.vector_norm(latents.movedim(1, -1).unsqueeze(-2) - self.embedding.weight, dim=-1)
        encoding_inds = torch.argmin(dist, dim=-1)  # Get the number of the nearest codebook vector
        quantized_latents = self.quantize(encoding_inds)  # Quantize the latents

        # Compute the VQ Losses
        codebook_loss = F.mse_loss(latents.detach(), quantized_latents)
        commitment_loss = F.mse_loss(latents, quantized_latents.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Make the gradient with respect to latents be equal to the gradient with respect to quantized latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents  # , vq_loss

    def quantize(self, encoding_indices):
        z = self.embedding(encoding_indices)
        z = z.movedim(-1, 1)  # Move channels back
        return z


class Decoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        # The decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions
        # with stride 2 and window size 4 × 4.
        self.res_block = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, 1, 1, 0, bias=True),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, 2, 2, bias=False),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 6, 1, 1, bias=False),
            nn.ConvTranspose2d(hidden_dim, channels, 5, 1, 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        x = self.conv_block(x)
        return x


class MaskedConv(nn.Conv2d):
    def __init__(self, *args, mask_type, **kwargs):
        super().__init__(*args, **kwargs)

        weight = self.weight.data
        self.register_buffer("mask", torch.zeros_like(weight))
        _, _, h, w = weight.shape
        self.mask[..., : h // 2, :] = 1
        self.mask[..., h // 2, : w // 2] = 1
        if mask_type == "B":
            self.mask[..., h // 2, w // 2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True),
            nn.ReLU(),
            MaskedConv(
                hidden_dim,
                hidden_dim // 2,
                3,
                1,
                1,
                bias=True,
                mask_type="B",
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1, 1, 0, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x)


class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, hidden_dim, n_res_blocks, n_conv_blocks):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(num_embeddings, hidden_dim)
        self.layers = nn.Sequential(
            # "Mask A is applied only to the first convolutional layer."
            MaskedConv(hidden_dim, hidden_dim, 7, 1, 3, bias=True, mask_type="A"),
            nn.ReLU(),
            *[layer for _ in range(n_res_blocks) for layer in [ResBlock(hidden_dim), nn.ReLU()]],
            *[
                layer
                for _ in range(n_conv_blocks)
                for layer in [
                    nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True),
                    nn.ReLU(),
                ]
            ],
            nn.Conv2d(hidden_dim, num_embeddings, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        return x


class VQVAE(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        num_embeddings: int = 512,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        n_pixelcnn_res_blocks: int = 2,
        n_pixelcnn_conv_blocks: int = 2,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.enc = Encoder(channels=channels, hidden_dim=hidden_dim, latent_dim=embedding_dim)
        self.vect_quant = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.dec = Decoder(channels=channels, hidden_dim=hidden_dim, latent_dim=embedding_dim)

        self.pixelcnn = PixelCNN(
            num_embeddings=num_embeddings,
            hidden_dim=hidden_dim,
            n_res_blocks=n_pixelcnn_res_blocks,
            n_conv_blocks=n_pixelcnn_conv_blocks,
        )

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, z):
        x = self.dec(z)
        return x

    def forward(self, ori_image):
        # "The model takes an input $x$, that is passed through an encoder
        # producing output $z_{e}(x)$.
        z_e = self.encode(ori_image)
        z_q = self.vect_quant(z_e)
        x = self.decode(z_q)
        return x

    def get_vqvae_loss(self, ori_image, commit_weight=0.25):
        z_e = self.encode(ori_image)  # "$z_{e}(x)$"
        z_q = self.vect_quant(z_e)  # "$z_{q}(x)$"
        # "The VQ objective uses the $l_{2}$ error to move the embedding vectors $e_{i}$
        # towards the encoder outputs $z_{e}(x)$."
        # "$\Vert \text{sg}[z_{e}(x)] - e \Vert^{2}_{2}$"
        # This term trains the vector quantizer.
        vq_loss = F.mse_loss(z_e.detach(), z_q, reduction="mean")
        # "To make sure the encoder commits to an embedding and its output does not grow,
        # we add a commitment loss."
        # "$\beta \Vert z_{e}(x) - \text{sg}[e] \Vert^{2}_{2}$"
        # This term trains the encoder.
        commit_loss = commit_weight * F.mse_loss(z_e, z_q.detach(), reduction="mean")
        # "We approximate the gradient similar to the straight-through estimator and just
        # copy gradients from decoder input $z_{q}(x)$ to encoder output $z_{e}(x)$."
        z_q = z_e + (z_q - z_e).detach()

        recon_image = self.decode(z_q)
        # This term trains the decoder (and the encoder).
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        return vq_loss, commit_loss

    def get_post_q(self, ori_image):
        z_e = self.encode(ori_image)
        q = self.vect_quant.vector_quantize(z_e)  # "$q(z \vert x)$"
        return q

    def get_pixelcnn_loss(self, q):
        pred_q = self.pixelcnn(q)
        loss = F.cross_entropy(
            rearrange(pred_q, pattern="b c h w -> (b h w) c"),
            q.view(
                -1,
            ),
            reduction="mean",
        )
        return loss

    @staticmethod
    def sample_from_distr(x, temp=1):
        prob = F.softmax(x / temp, dim=1)
        return torch.multinomial(prob, num_samples=1, replacement=True)[:, 0]

    def q_to_image(self, q):
        x = self.vect_quant.embed_space(q)
        return self.decode(x.permute(0, 3, 1, 2))

    @torch.no_grad()
    def sample_post_q(self, batch_size, q_size, device, temp=1):
        sampled_q = torch.zeros(
            size=(batch_size, q_size, q_size),
            dtype=torch.int64,
            device=device,
        )
        for row in range(q_size):
            for col in range(q_size):
                pred_q = self.pixelcnn(sampled_q.detach())
                recon_q = self.sample_from_distr(pred_q[..., row, col], temp=temp)
                sampled_q[:, row, col] = recon_q
        return sampled_q

    def sample(self, batch_size, q_size, device, temp=1):
        post_q = self.sample_post_q(
            batch_size=batch_size,
            q_size=q_size,
            device=device,
            temp=temp,
        )
        return self.q_to_image(post_q)


if __name__ == "__main__":
    prob = F.softmax(torch.randn(4, 128), dim=1)
    device = torch.device("cuda")
    model = VQVAE(
        channels=3,
        num_embeddings=1024,
        embedding_dim=128,
        hidden_dim=256,
        n_pixelcnn_res_blocks=2,
        n_pixelcnn_conv_blocks=2,
    ).to(device)

    torchsummary.summary(model, (3, 256, 256))  # torchsummary err?
