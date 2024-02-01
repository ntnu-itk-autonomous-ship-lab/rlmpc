"""
    encoder.py

    Summary:
        Contains the encoder network for processing images from the environment.

    Author: Trym Tengesdal
"""
import torch.nn as nn


def weights_init(m):
    """Initializes the weights of the network"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(m.bias)


class PerceptionImageEncoder(nn.Module):
    """We use a ResNet8 architecture for now."""

    def __init__(self, n_input_channels: int, latent_dim: int) -> None:
        """

        Args:
            n_input_channels: Number of input channels
            latent_dim: Dimension of the latent space
        """
        super(PerceptionImageEncoder, self).__init__()
        self.n_input_channels = n_input_channels
        self.latent_dim = latent_dim

        self.elu = nn.ELU()

        # First block of convolutions
        self.conv00: nn.Conv2d = nn.Conv2d(
            in_channels=n_input_channels, out_channels=32, kernel_size=5, stride=2, padding=2
        )
        self.conv01: nn.Conv2d = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=2)
        nn.init.xavier_uniform_(self.conv00.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv00.bias)
        nn.init.xavier_uniform_(self.conv01.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv01.bias)

        # Second block of convolutions
        self.conv10: nn.Conv2d = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv11: nn.Conv2d = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2)
        nn.init.xavier_uniform_(self.conv10.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv10.bias)
        nn.init.xavier_uniform_(self.conv11.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv11.bias)

        # Third block of convolutions
        # ELU activation function
        self.conv20: nn.Conv2d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv21: nn.Conv2d = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2)
        nn.init.xavier_uniform_(self.conv20.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv20.bias)
        nn.init.xavier_uniform_(self.conv21.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv21.bias)

        # Jump connection from last layer of first block to first layer of second block
        self.conv0_jump_to_2: nn.Conv2d = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        # Jump connection from last layer of second block to first layer of third block
        self.conv1_jump_to_3: nn.Conv2d = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, stride=4, padding=(2, 1)
        )

        # Fourth block of convolutions
        # ELU activation function
        self.conv30: nn.Conv2d = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3 * 6 * 128, out_features=512)
        # ELU activation function
        self.fc2 = nn.Linear(
            in_features=512, out_features=2 * latent_dim
        )  # 2 * latent_dim because we need to output both the mean and the log variance

    def encode(self, image):
        """Encodes the input image into a latent vector"""
        # First block of convolutions
        x00 = self.conv00(image)
        x01 = self.elu(self.conv01(x00))

        x10 = self.conv10(x01)
        x11 = self.conv11(x10)

        x0_jump_to_2 = self.conv0_jump_to_2(x01)

        x11 = x11 + x0_jump_to_2
        x20 = self.conv20(self.elu(x11))
        x21 = self.conv21(x20)

        x1_jump_to_3 = self.conv1_jump_to_3(x11)
        x21 = x21 + x1_jump_to_3

        x30 = self.conv30(self.elu(x21))

        x40 = self.fc1(x30.view(x30.size(0), -1))
        x41 = self.fc2(self.elu(x40))

        x = x41
        return x

    def forward(self, image):
        """Encodes the input image into a latent vector"""
        return self.encode(image)


if __name__ == "__main__":
    from torchsummary import summary

    latent_dimension = 128
    encoder = PerceptionImageEncoder(latent_dim=latent_dimension, n_input_channels=1).to("cuda")
    summary(encoder, (1, 270, 480), device="cuda")
