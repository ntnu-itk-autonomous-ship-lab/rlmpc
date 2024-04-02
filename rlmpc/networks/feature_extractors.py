"""
    feature_extractors.py

    Summary:
        Contains feature extractors for neural networks (NNs) used in DRL. Feature extractor inspired by stable-baselines3 (SB3) implementation, the CNN-work of Thomas Larsen, and variational autoencoders (VAEs).

    Author: Trym Tengesdal
"""

import pathlib
from sys import platform
from typing import Tuple

import rlmpc.networks.vanilla_vae_arch2.vae as vae_arch2
import torch as th
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

if platform == "linux" or platform == "linux2":
    VAE_DATADIR: pathlib.Path = pathlib.Path("/home/doctor/Desktop/machine_learning/data/vae/")
elif platform == "darwin":
    VAE_DATADIR: pathlib.Path = pathlib.Path("/Users/trtengesdal/Desktop/machine_learning/vae_models/")


class PerceptionImageVAE(BaseFeaturesExtractor):
    """ """

    def __init__(
        self,
        observation_space: spaces.Box,
        encoder_conv_block_dims=(32, 64, 128, 128),
        fc_dim=512,
        latent_dim: int = 128,
        model_file: str | None = None,
    ):
        super(PerceptionImageVAE, self).__init__(observation_space, features_dim=latent_dim)

        self.input_image_dim = (observation_space.shape[0], observation_space.shape[1], observation_space.shape[2])

        if model_file is None:
            model_file = VAE_DATADIR / "training_vae1_model_LD_128_best.pth"
        self.vae: vae_arch2.VAE = vae_arch2.VAE(
            latent_dim=latent_dim,
            input_image_dim=(observation_space.shape[0], observation_space.shape[1], observation_space.shape[2]),
            encoder_conv_block_dims=encoder_conv_block_dims,
            fc_dim=fc_dim,
        )

        self.vae.load_state_dict(
            th.load(
                str(model_file),
                map_location=th.device("cpu"),
            )
        )
        self.vae.eval()
        self.vae.set_inference_mode(True)
        self.latent_dim = self.vae.latent_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        assert self.vae.inference_mode, "VAE must be in inference mode before usage as a feature extractor."
        with th.no_grad():
            z_e, _, _ = self.vae.encode(observations)
            # print(f"z_e shape: {z_e.shape}")
            return z_e


class PathRelativeNavigationNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 5):
        """Feature extractor for the navigation state. This is a simple passthrough layer.

        Args:
            observation_space (gym.spaces.Box): Navigation state observation space.
            features_dim (int, optional): Length of Features [d2path, speed_ref_diff, u, v, r]
        """
        super(PathRelativeNavigationNN, self).__init__(observation_space, features_dim=features_dim)
        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.passthrough(observations)


class DisturbanceNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 4):
        """Feature extractor for the navigation state. This is a simple passthrough layer.

        Args:
            observation_space (gym.spaces.Box): Navigation state observation space.
            features_dim (int, optional): Disturbance vector length. Defaults to 4, i.e. [V_c, beta_c, V_w, beta_w].
        """
        super(DisturbanceNN, self).__init__(observation_space, features_dim=features_dim)

        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.passthrough(observations)


class TrackingGRU(BaseFeaturesExtractor):
    """Feature extractor for the tracking state."""

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 30,
        num_layers: int = 1,
        batch_size: int = 1,
    ) -> None:
        super(TrackingGRU, self).__init__(observation_space, features_dim=features_dim)

        self.input_dim = observation_space.shape[0]
        self.hidden_dim = features_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=features_dim, num_layers=num_layers, batch_first=True)

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor]:
        batch_size = observations.shape[0]
        hidden = th.zeros(self.num_layers, batch_size, self.hidden_dim)

        observations = observations.permute(0, 2, 1)
        packed_seq = rnn_utils.pack_padded_sequence(
            observations, [len(obs) for obs in observations], batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_seq, hidden)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        hidden_out = hidden.permute(1, 0, 2).reshape(-1, self.num_layers * self.hidden_dim)
        return hidden_out

    def init_hidden(self, batch_size: int) -> th.Tensor:
        return th.zeros(self.num_layers, batch_size, self.hidden_dim)


class CombinedExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256, batch_size: int = 1):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CombinedExtractor, self).__init__(observation_space, features_dim)
        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "PerceptionImageObservation":
                extractors[key] = PerceptionImageVAE(subspace)
                total_concat_size += extractors[key].latent_dim
            elif key == "PathRelativeNavigationObservation":
                extractors[key] = PathRelativeNavigationNN(subspace, features_dim=subspace.shape[-1])  # nn.Identity()
                total_concat_size += subspace.shape[-1]
            elif key == "RelativeTrackingObservation":
                extractors[key] = TrackingGRU(subspace, features_dim=30, num_layers=1, batch_size=batch_size)
                total_concat_size += extractors[key].features_dim
            elif key == "DisturbanceObservation":
                extractors[key] = DisturbanceNN(subspace, features_dim=subspace.shape[-1])
                total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)


if __name__ == "__main__":
    import colav_simulator.common.paths as cs_dp
    import gymnasium as gym
    import rlmpc.common.paths as rl_dp

    scenario_choice = 0
    if scenario_choice == 0:
        scenario_name = "rlmpc_scenario_cr_ss"
        config_file = rl_dp.scenarios / (scenario_name + ".yaml")
    elif scenario_choice == 1:
        scenario_name = "rlmpc_scenario_head_on_channel"
        config_file = rl_dp.scenarios / "rlmpc_scenario_easy_headon_no_hazards.yaml"
    elif scenario_choice == 2:
        scenario_name = "rogaland_random_rl"
        config_file = cs_dp.scenarios / "rogaland_random_rl.yaml"
    elif scenario_choice == 3:
        scenario_name = "rogaland_random_rl_2"
        config_file = rl_dp.scenarios / "rogaland_random_rl_2.yaml"
    elif scenario_choice == 4:
        scenario_name = "rl_scenario"
        config_file = rl_dp.scenarios / "rl_scenario.yaml"

    observation_type = {
        "dict_observation": [
            "perception_image_observation",
            "path_relative_observation",
            "relative_tracking_observation",
            "tracking_observation",
            "disturbance_observation",
            "time_observation",
        ]
    }
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": rl_dp.scenarios / "training_data" / scenario_name,
        "max_number_of_episodes": 1,
        "test_mode": False,
        "render_update_rate": 0.5,
        "observation_type": observation_type,
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "seed": 15,
    }
    env = gym.make(id=env_id, **env_config)

    feature_extractor = CombinedExtractor(env.observation_space)
