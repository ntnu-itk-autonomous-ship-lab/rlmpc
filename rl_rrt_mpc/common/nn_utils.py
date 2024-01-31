"""
    nn_utils.py

    Summary:
        Contains utility classes (feature extractors, etc.) for neural networks (NNs) used in RL. Feature extractor inspired by stable-baselines3 (SB3) implementation
        and https://github.com/ThomasNLarsen/gym-auv-SB3/blob/main/gym_auv/utils/radarCNN.py.

    Author: Trym Tengesdal
"""


import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PerceptionImageCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension ()
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    # def __init__(self, observation_space: gym.spaces.Box, sensor_dim: int = 180, features_dim: int = 32, kernel_overlap: float = 0.05):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ):
        super(PerceptionImageCNN, self).__init__(observation_space, features_dim=features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.n_input_channels = observation_space.shape[0]

        self.kernel_size = 8
        self.padding = 0
        self.stride = 4

        print("PerceptionImageCNN CONFIG")
        print("\tIN_CHANNELS =", self.n_input_channels)
        print("\tKERNEL_SIZE =", self.kernel_size)
        print("\tPADDING     =", self.padding)
        print("\tSTRIDE      =", self.stride)
        self.perception_image_cnn = nn.Sequential(
            # in_channels: (static obstacles + nominal path + dynamic obstacles) x winow size
            nn.Conv2d(
                in_channels=self.n_input_channels,
                out_channels=32,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self.kernel_size / 2,
                padding=self.padding,
                stride=self.stride / 2,
            ),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        self.n_flatten = 0
        sample = th.as_tensor(observation_space.sample()).float()
        print("Observation space - sample shape:", sample.shape)
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        with th.no_grad():
            print("PerceptionImageCNN initializing, input is", sample.shape, "and", end=" ")
            flatten = self.perception_image_cnn(sample)
            self.n_flatten = flatten.shape[1]
            print("output is", flatten.shape)

        self.linear = nn.Sequential(nn.Linear(self.n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)
        # return self.linear(self.cnn(observations))
        # return self.cnn(observations)

    def get_features(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.cnn:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.cpu().detach().numpy())
        return feat

    def get_activations(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.cnn:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out)

        for layer in self.linear:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out.detach().numpy())

        return feat


class NavigatioNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 6):
        """Feature extractor for the navigation state. This is a simple passthrough layer.

        Args:
            observation_space (gym.spaces.Box): Navigation state observation space.
            features_dim (int, optional): State vector length. Defaults to 6, i.e. [x, y, psi, u, v, r]. Could be other choices as well ([psi, u, v, r, y_e, chi_e, etc..])
        """
        super(NavigatioNN, self).__init__(observation_space, features_dim=features_dim)

        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        shape = observations.shape
        observations = observations[:, 0, :].reshape(shape[0], shape[-1])
        return self.passthrough(observations)


class PerceptionImageNavigationExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(PerceptionImageNavigationExtractor, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "perception_image":
                # Pass sensor readings through CNN
                extractors[key] = PerceptionImageCNN(subspace, features_dim=subspace.shape)
                total_concat_size += features_dim  # extractors[key].n_flatten
            elif key == "navigation":
                # Pass navigation features straight through to the MlpPolicy.
                extractors[key] = NavigatioNN(subspace, features_dim=subspace.shape[-1])  # nn.Identity()
                total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
