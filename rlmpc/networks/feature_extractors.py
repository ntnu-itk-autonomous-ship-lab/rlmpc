"""Contains feature extractors for neural networks (NNs) used in DRL.

Feature extractor inspired by stable-baselines3 (SB3) implementation,
he CNN-work of Thomas Larsen, and variational autoencoders (VAEs).

Author: Trym Tengesdal
"""

import pathlib
from typing import Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import rlmpc.common.paths as rl_dp
import rlmpc.networks.enc_vae_128.vae as enc_vae
import rlmpc.networks.tracking_vae_attention.vae as tracking_vae

# ENCVAE_DATADIR: pathlib.Path = pathlib.Path.home() / "machine_learning/enc_vae/"
# TRACKINGVAE_DATADIR: pathlib.Path = pathlib.Path.home() / "machine_learning/tracking_vae/chosen"
ENCVAE_DATADIR: pathlib.Path = rl_dp.package / "networks" / "models"
TRACKINGVAE_DATADIR: pathlib.Path = rl_dp.package / "networks" / "models"


class ENCVAE(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        encoder_conv_block_dims=[64, 128, 256, 256],
        decoder_conv_block_dims=[256, 128, 128, 64, 32],
        fc_dim=1024,
        latent_dim: int = 40,
        model_file: str | None = None,
    ):
        super(ENCVAE, self).__init__(observation_space, features_dim=latent_dim)

        self.input_image_dim = (
            observation_space.shape[0],
            observation_space.shape[1],
            observation_space.shape[2],
        )

        if model_file is None:
            # model_file = ENCVAE_DATADIR / "LD_40_128x128/model_LD_40_best.pth"
            model_file = ENCVAE_DATADIR / "enc_vae.pth"
        self.vae: enc_vae.VAE = enc_vae.VAE(
            latent_dim=latent_dim,
            input_image_dim=(
                observation_space.shape[0],
                observation_space.shape[1],
                observation_space.shape[2],
            ),
            encoder_conv_block_dims=encoder_conv_block_dims,
            decoder_conv_block_dims=decoder_conv_block_dims,
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
        self._features_dim = self.latent_dim
        self.scaling_factor = 55.0
        self.training = False

    def set_inference_mode(self, inference_mode: bool) -> None:
        self.vae.set_inference_mode(inference_mode)

    def display_image(self, image: th.Tensor) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("TkAgg")

        img: np.ndarray = image[0].numpy().copy()
        img = img.transpose(1, 2, 0)
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.show(block=False)

    def reconstruct(self, observations: th.Tensor) -> th.Tensor:
        with th.no_grad():
            z = self.vae.encode(observations)[0]
            recon_obs = self.vae.decode(z)
            # self.display_image(reconstructed_image[0])
            # self.display_image(recon_obs[0])
            return recon_obs

    def forward(self, observations: th.Tensor) -> th.Tensor:
        assert (
            self.vae.inference_mode
        ), "VAE must be in inference mode before usage as a feature extractor."
        # self.display_image(observations[0])
        z_e, _, _ = self.vae.encode(observations)
        # print(f"z_e shape: {z_e.shape}")
        z_e = z_e / self.scaling_factor
        if z_e.max() > 1.0 or z_e.min() < -1.0:
            print("WARNING: z_e max value > 1.0 or min value < -1.0")
        return z_e


class PathRelativeNavigationNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 5):
        """Feature extractor for the navigation state. This is a simple passthrough layer.

        Args:
            observation_space (gym.spaces.Box): Navigation state observation space.
            features_dim (int, optional): Length of Features
        """
        super(PathRelativeNavigationNN, self).__init__(
            observation_space, features_dim=features_dim
        )
        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.passthrough(observations)


class DisturbanceFeedforward(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 4):
        """Feature extractor for the navigation state. This is a simple passthrough layer.

        Args:
            observation_space (gym.spaces.Box): Navigation state observation space.
            features_dim (int, optional): Disturbance vector length. Defaults to 4, i.e. [V_c, beta_c, V_w, beta_w].
        """
        super(DisturbanceFeedforward, self).__init__(
            observation_space, features_dim=features_dim
        )

        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.passthrough(observations)


class MPCParameterFeedforward(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 9):
        """Feature extractor for the current MPC params. This is a simple passthrough layer.

        Args:
            observation_space (gym.spaces.Box): Navigation state observation space.
            features_dim (int, optional): MPC parameter vector length. Defaults to 9, i.e. [Q_p (3), K_app(2), w_colregs (3), r_safe_do].
        """
        super(MPCParameterFeedforward, self).__init__(
            observation_space, features_dim=features_dim
        )
        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.passthrough(observations)


class SimpleTrackingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 12):
        """Feature extractor for the three closest DOs. This is a simple passthrough layer.

        Args:
            observation_space (gym.spaces.Box): Relative tracking observation space.
        """
        super(SimpleTrackingFeatureExtractor, self).__init__(
            observation_space, features_dim=features_dim
        )
        self.num_considered_dos = 3
        self.input_dim = observation_space.shape[0]
        self._features_dim = self.num_considered_dos * self.input_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        closest_dos = observations[:, :, -self.num_considered_dos :].permute(0, 2, 1)
        closest_dos = closest_dos.reshape(-1, self.num_considered_dos * self.input_dim)
        return closest_dos


class ClosestENCHazardFeedForward(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 2):
        """Feature extractor for ENC data, feedforward of a simple computation of the distance to the closest hazard.

        Args:
            observation_space (gym.spaces.Box): ENC image observation space.
        """
        super(ClosestENCHazardFeedForward, self).__init__(
            observation_space, features_dim=features_dim
        )
        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.passthrough(observations)


class TrackingVAE(BaseFeaturesExtractor):
    """Feature extractor for the tracking state."""

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 12,
        num_layers: int = 1,
        model_file: str | None = None,
    ) -> None:
        super(TrackingVAE, self).__init__(observation_space, features_dim=features_dim)

        self.input_dim = observation_space.shape[0]
        self.max_seq_len = observation_space.shape[1]

        if model_file is None:
            # model_name = "tracking_avae_mdd21_NL_2_nonbi_HD_64_LD_10_NH_4_ED_16"
            # model_file = TRACKINGVAE_DATADIR / model_name / f"{model_name}_best.pth"
            model_file = TRACKINGVAE_DATADIR / "tracking_avae.pth"

        self.vae: tracking_vae.VAE = tracking_vae.VAE(
            input_dim=self.input_dim,
            embedding_dim=16,
            num_heads=8,
            rnn_hidden_dim=64,
            latent_dim=12,
            num_layers=1,
            rnn_type=th.nn.GRU,
            bidirectional=False,
            max_seq_len=self.max_seq_len,
            inference_mode=True,
        )

        self.vae.load_state_dict(
            th.load(
                str(model_file),
                map_location=th.device("cpu"),
                weights_only=True,
            )
        )
        self.vae.eval()
        self.vae.set_inference_mode(True)
        self.latent_dim = self.vae.latent_dim
        self._features_dim = self.latent_dim
        self.scaling_factor = 10.0

    def set_inference_mode(self, inference_mode: bool) -> None:
        self.vae.set_inference_mode(inference_mode)

    def preprocess_obs(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.vae.preprocess_obs(observations)

    def reconstruct(self, observations: th.Tensor) -> th.Tensor:
        observations, seq_lengths = self.preprocess_obs(observations)
        recon_obs = self.vae(observations, seq_lengths)
        return recon_obs

    def forward(self, observations: th.Tensor) -> th.Tensor:
        assert (
            self.vae.inference_mode
        ), "VAE must be in inference mode before usage as a feature extractor."
        observations, seq_lengths = self.preprocess_obs(observations)
        z_e, _, _ = self.vae.encode(observations, seq_lengths)
        z_e = z_e / self.scaling_factor
        if z_e.max() > 1.0 or z_e.min() < -1.0:
            print("WARNING: z_e max value > 1.0 or min value < -1.0")
        return z_e


class CombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor that combines multiple feature extractors into one."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        batch_size: int = 1,
    ) -> None:
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
                extractors[key] = ENCVAE(subspace)
                total_concat_size += extractors[key].latent_dim
            elif key == "PathRelativeNavigationObservation":
                extractors[key] = PathRelativeNavigationNN(
                    subspace, features_dim=subspace.shape[-1]
                )  # nn.Identity()
                total_concat_size += subspace.shape[-1]
            elif key == "RelativeTrackingObservation":
                extractors[key] = TrackingVAE(subspace, features_dim=12)
                total_concat_size += extractors[key].latent_dim
            # elif key == "MPCParameterObservation":
            #     extractors[key] = MPCParameterFeedforward(subspace, features_dim=subspace.shape[-1])
            #     total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        with th.no_grad():
            encoded_tensor_list = []
            for key, extractor in self.extractors.items():
                extracted_tensor = extractor(observations[key])
                if extracted_tensor.max() > 1.0:
                    print(f"WARNING: {key} extracted_tensor max value > 1.0")
                elif extracted_tensor.min() < -1.0:
                    print(f"WARNING: {key} extracted_tensor min value < -1.0")
                encoded_tensor_list.append(extracted_tensor)

            return th.cat(encoded_tensor_list, dim=1)


class SimpleCombinedExtractor(BaseFeaturesExtractor):
    """Simple feature extractor that combines multiple feature extractors into one."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 19,
        batch_size: int = 1,
    ) -> None:
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(SimpleCombinedExtractor, self).__init__(observation_space, features_dim)
        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "ClosestENCHazardObservation":
                extractors[key] = ClosestENCHazardFeedForward(subspace)
                total_concat_size += subspace.shape[-1]
            elif key == "PathRelativeNavigationObservation":
                extractors[key] = PathRelativeNavigationNN(
                    subspace, features_dim=subspace.shape[-1]
                )  # nn.Identity()
                total_concat_size += subspace.shape[-1]
            elif key == "RelativeTrackingObservation":
                extractors[key] = TrackingVAE(subspace, features_dim=12)
                total_concat_size += extractors[key].latent_dim
            elif key == "MPCParameterObservation":
                extractors[key] = MPCParameterFeedforward(
                    subspace, features_dim=subspace.shape[-1]
                )
                total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            extracted_tensor = extractor(observations[key])
            if extracted_tensor.max() > 1.0:
                print(f"WARNING: {key} extracted_tensor max value > 1.0")
            elif extracted_tensor.min() < -1.0:
                print(f"WARNING: {key} extracted_tensor min value < -1.0")
            encoded_tensor_list.append(extracted_tensor)

        return th.cat(encoded_tensor_list, dim=1)


if __name__ == "__main__":
    import gymnasium as gym

    import rlmpc.common.paths as rl_dp

    scenario_name = "rlmpc_scenario_cr_ss"
    config_file = rl_dp.scenarios / (scenario_name + ".yaml")

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
