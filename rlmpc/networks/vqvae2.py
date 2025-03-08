import torch
import torchsummary
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.reset_parameters()

    def forward(self, latents):
        # Compute L2 distances between latents and embedding weights
        dist = torch.linalg.vector_norm(
            latents.movedim(1, -1).unsqueeze(-2) - self.embedding.weight, dim=-1
        )
        encoding_inds = torch.argmin(
            dist, dim=-1
        )  # Get the number of the nearest codebook vector
        quantized_latents = self.quantize(encoding_inds)  # Quantize the latents

        # Compute the VQ Losses
        codebook_loss = F.mse_loss(latents.detach(), quantized_latents)
        commitment_loss = F.mse_loss(latents, quantized_latents.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Make the gradient with respect to latents be equal to the gradient with respect to quantized latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss

    def quantize(self, encoding_indices):
        z = self.embedding(encoding_indices)
        z = z.movedim(-1, 1)  # Move channels back
        return z

    def reset_parameters(self):
        nn.init.uniform_(
            self.embedding.weight, -1 / self.num_embeddings, 1 / self.num_embeddings
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self[0].weight)
        bn = self[1]
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)


class Encoder(nn.Module):
    def __init__(self, in_channels, channels_list, latent_channels):
        super().__init__()
        self.stem = ConvBlock(in_channels, channels_list[0], 3)
        self.blocks = Stack(channels_list, DownBlock)
        self.to_latent = ConvBlock(channels_list[-1], latent_channels, 3)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.to_latent(x)
        return x


class Stack(nn.Sequential):
    def __init__(self, channels_list, block):
        layers = []
        for in_channels, out_channels in zip(channels_list[:-1], channels_list[1:]):
            layers.append(block(in_channels, out_channels))
        super().__init__(*layers)


class DownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, 3))


class UpBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(in_channels, out_channels, 3),
        )


class Decoder(nn.Module):
    def __init__(self, latent_channels, channels_list, out_channels):
        super().__init__()
        self.stem = ConvBlock(latent_channels, channels_list[0], 3)
        self.blocks = Stack(channels_list, UpBlock)
        self.to_output = nn.Conv2d(channels_list[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.to_output(x)
        x = torch.sigmoid(x)
        return x


class VQVAE(nn.Module):
    def __init__(
        self,
        num_downsamplings,
        latent_channels,
        num_embeddings,
        channels=32,
        in_channels=3,
    ):
        super().__init__()
        self.embedding_dim = latent_channels
        self.num_embeddings = num_embeddings
        self.num_hiddens = channels
        channels_list = [channels * 2**i for i in range(num_downsamplings + 1)]
        self.encoder = Encoder(in_channels, channels_list, latent_channels)
        self.vq = VectorQuantizer(num_embeddings, latent_channels)
        channels_list.reverse()
        self.decoder = Decoder(latent_channels, channels_list, in_channels)
        self.reduction = 2**num_downsamplings
        self.num_embeddings = num_embeddings

    def forward(self, x):
        latents = self.encoder(x)
        z, vq_loss = self.vq(latents)
        out = self.decoder(z)
        return out, vq_loss

    def sample(self, num_samples, shape, device):
        latent_shape = (
            num_samples,
            shape[0] // self.reduction,
            shape[1] // self.reduction,
        )
        ind = torch.randint(0, self.num_embeddings, latent_shape, device=device)
        with torch.no_grad():
            z = self.vq.quantize(ind)
            out = self.decoder(z)
        return out


if __name__ == "__main__":
    IMAGE_SIZE = 128
    IMAGE_CHANNELS = 3
    ENCODER_CHANNELS = 16
    LATENT_CHANNELS = 16
    NUM_DOWNSAMPLINGS = 5
    NUM_EMBEDDINGS = 256

    EPOCHS = 20
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-2
    device = torch.device("cuda")
    model = VQVAE(
        num_downsamplings=NUM_DOWNSAMPLINGS,
        latent_channels=LATENT_CHANNELS,
        num_embeddings=NUM_EMBEDDINGS,
        channels=ENCODER_CHANNELS,
        in_channels=IMAGE_CHANNELS,
    ).to(device)

    x = torch.rand(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
    out, vq_loss = model(x)

    torchsummary.summary(model, (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
