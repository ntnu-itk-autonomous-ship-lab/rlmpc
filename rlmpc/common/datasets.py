"""
datasets.py

Summary:
    Contains classes for pytorch datasets.

Author: Trym Tengesdal
"""

from pathlib import Path
from typing import List, Optional, Tuple

import colav_simulator.common.math_functions as csmf
import colav_simulator.gym.logger as csenv_logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.mpc.parameters as mpc_params
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms_v2


class UnNormalize(transforms_v2.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1.0 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


class ENCImageDataset(Dataset):
    """Class for perception image dataset from the colav-environment."""

    # consider augmenting dataset by flipping and rotating images, cropping, color manipulaton,

    def __init__(
        self,
        data_npy_file: str,
        data_dir: Path,
        mask_npy_file: Optional[str] = None,
        transform=transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
            ]
        ),
    ):
        """Initializes the dataset.
        Args:
            - data_npy_file (str): The name of the npy file containing the data.
            - mask_npy_file (str): The name of the npy file containing the segmentation masks.
            - data_dir (Path): The path to the data directory in which the numpy file is found.
            - transform (transforms_v2...): The transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = np.load(
            data_dir / data_npy_file, mmap_mode="r", allow_pickle=True
        ).astype(np.uint8)
        # self.data = self.data[3:100, 0, :, :, :]  # disregard 3 first.
        self.masks = None
        if mask_npy_file is not None:
            self.masks = np.load(
                data_dir / mask_npy_file, mmap_mode="r", allow_pickle=True
            ).astype(np.uint8)
            # self.masks = self.masks[3:100, 0, :, :, :]  # disregard 3 first.

        if len(self.data.shape) == 4:
            self.data = np.expand_dims(self.data, axis=0)
            self.masks = (
                np.expand_dims(self.masks, axis=0) if self.masks is not None else None
            )

        self.n_samples, self.n_envs, self.n_channels, self.height, self.width = (
            self.data.shape
        )

        self.unnormalize_transform = transforms_v2.Compose(
            [
                UnNormalize(mean=[0.5], std=[0.5]),
                transforms_v2.ToDtype(torch.uint8, scale=True),
            ]
        )

    def get_datainfo(self) -> Tuple[int, int, int, int, int]:
        """Returns the data information."""
        return self.n_samples, self.n_envs, self.n_channels, self.height, self.width

    def show_random_image(self):
        """Shows a random image from the dataset."""
        idx = np.random.randint(0, self.n_samples)
        env_idx = np.random.randint(0, self.n_envs)
        plt.imshow(self.data[idx, env_idx, 0, :, :])
        plt.show()

    def __len__(self):
        return self.n_samples * self.n_envs

    def __getitem__(self, idx: int) -> torch.Tensor:
        assert idx < self.n_samples * self.n_envs, "Index out of range"
        env_idx = idx % self.n_envs
        sample_idx = idx // self.n_envs
        sample = torch.from_numpy(
            self.data[sample_idx, env_idx, :, :, :].copy().astype(np.uint8)
        )
        if self.masks is not None:
            mask = torch.from_numpy(self.masks[sample_idx, env_idx, :, :, :].copy())
        else:
            mask = 255 * torch.ones_like(sample, dtype=torch.uint8)

        if self.transform:
            both_images = torch.cat((sample.unsqueeze(0), mask.unsqueeze(0)), 0)
            transformed = self.transform(both_images)
            sample = transformed[0]
            mask = transformed[1]
        assert not torch.isinf(sample).any(), "Sample contains inf"
        # print(f"Sample min: {sample.min()}, max: {sample.max()}")
        return sample, mask


class TrackingObservationDataset(Dataset):
    """Class for tracking observation dataset from the colav-environment.

    The dataset is a numpy array of shape (n_samples, n_envs, max_num_do, do_info_dim),

    where n_samples is the number of samples, n_envs is the number of environments,
    max_num_do is the maximum number of dynamic obstacles in the environment,
    and do_info_dim is the dimension of the dynamic obstacle information.
    """

    def __init__(
        self,
        data_npy_file: str,
        data_dir: Path,
        transform=None,
    ):
        """Initializes the dataset.
        Args:
            - data_npy_file (str): The name of the npy file containing the observation data.
            - data_dir (Path): The path to the data directory in which the numpy file is found.
            - transform (transforms_v2...): The transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = np.load(
            data_dir / data_npy_file, mmap_mode="r", allow_pickle=True
        ).astype(np.float32)
        # self.data = self.data[:1, :1]
        self.n_samples, self.n_envs, self.max_num_do, self.do_info_dim = self.data.shape
        self.do_info_dim = min(4, self.do_info_dim)

    def get_datainfo(self) -> Tuple[int, int, int, int, int]:
        """Returns the data information."""
        return self.n_samples, self.n_envs, self.max_num_do, self.do_info_dim

    def get_data(self):
        return self.data

    def __len__(self):
        return self.n_samples * self.n_envs

    def __getitem__(self, idx: int) -> torch.Tensor:
        assert idx < self.n_samples * self.n_envs, "Index out of range"
        env_idx = idx % self.n_envs
        sample_idx = idx // self.n_envs
        sample = torch.from_numpy(
            self.data[sample_idx, env_idx, :, :].copy().astype(np.float32)
        )
        sample = sample[: self.do_info_dim, :]
        # sort sample after entry 0 (distance)
        if self.transform:
            sample = self.transform(sample)
        return sample


class ParameterProviderDataset(Dataset):
    """Class for a dataset containing preferred parameter sets to provide to
    an MPC scheme, given the current situation (environment state) processed through
    the CombinedFeatureExtractor in this repo.
    """

    def __init__(
        self,
        env_data_pkl_file: str,
        data_dir: Path,
        param_list: List[str],
        transform=None,
    ):
        """Initializes the dataset.
        Args:
            - env_data_pkl_file (str): The name of the pkl file containing the environment data.
            - data_dir (Path): The path to the data directory in which the pkl file is found.
            - num_adjustable_mpc_params (int): Number of paramters adjusted by the DNN param provider.
            - transform (transforms_v2...): The transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.env_data_logger = csenv_logger.Logger(
            experiment_name="parameter_provider_dataset", log_dir=data_dir
        )
        self.env_data_logger.load_from_pickle(name=env_data_pkl_file)
        self.timestep = 2.0
        self.param_list = param_list
        (
            self.dnn_out_parameter_ranges,
            self.dnn_out_parameter_incr_ranges,
            self.dnn_out_parameter_lengths,
        ) = mpc_params.MidlevelMPCParams.get_adjustable_parameter_info()
        offset = 0
        self.out_parameter_indices = {}
        for param in self.param_list:
            self.out_parameter_indices[param] = offset
            offset += self.dnn_out_parameter_lengths[param]
        self.num_adjustable_mpc_params = offset
        self._setup_data()

    def get_datainfo(self) -> Tuple[int, int, int, int]:
        """Returns the data information."""
        return self.n_episodes, self.n_features, self.n_mpc_params

    def _setup_data(self):
        """Sets up the param provider dataset: Extracts input data from the env data file and computes ad hoc parameter preferences for the parameter provider dataset, based on the env data provided."""

        env_data = self.env_data_logger.env_data
        assert "ms_channel" in env_data[0].name, "must be the rlmpc_scenario_ms_channel"
        self.data = []
        for epdata in env_data:
            dnn_input_features = np.array(
                [ainfo["dnn_input_features"] for ainfo in epdata.actor_infos]
            )
            dnn_input_current_norm_mpc_params = np.zeros(
                (dnn_input_features.shape[0], self.num_adjustable_mpc_params)
            )
            dnn_input_current_norm_mpc_params[0, :] = np.array(
                [
                    -0.6552,
                    -0.0345,
                    -0.0345,
                    -0.2081,
                    -0.2081,
                    -0.3333,
                    -0.3333,
                    -0.3333,
                    0.0,
                ]
            )  # dnn_input_features[0, -self.num_adjustable_mpc_params :]
            n_timesteps = dnn_input_features.shape[0]
            norm_param_incr_preferences, new_norm_mpc_params = (
                self._compute_ad_hoc_episode_parameter_preferences(
                    n_timesteps=n_timesteps,
                    dnn_input_current_norm_mpc_params=dnn_input_current_norm_mpc_params,
                    distances_to_collision=epdata.distances_to_collision,
                    add_noise_to_preferences=False,  # True if epdata.episode > 0 else False,
                )
            )
            dnn_input_features[:, -self.num_adjustable_mpc_params :] = (
                new_norm_mpc_params
            )
            self.data.append(
                (
                    epdata.episode,
                    n_timesteps,
                    (dnn_input_features, norm_param_incr_preferences),
                )
            )

        self.n_episodes = len(self.data)
        self.n_mpc_params = new_norm_mpc_params.shape[1]
        self.n_features = dnn_input_features.shape[1]
        self.n_samples = self._compute_number_of_samples()

    def _compute_number_of_samples(self) -> int:
        """Computes the number of samples in the dataset."""
        n_samples = 0
        for processed_epdata in self.data:
            n_samples += processed_epdata[1]
        return n_samples

    def _compute_ad_hoc_episode_parameter_preferences(
        self,
        n_timesteps: int,
        dnn_input_current_norm_mpc_params: np.ndarray,
        distances_to_collision: np.ndarray,
        d2collision_threshold: float = 500.0,
        add_noise_to_preferences: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes ad hoc parameter preferences for the parameter provider dataset, based on the env data provided.

        Args:
            n_timesteps (int): Number of timesteps in the env data.
            dnn_input_current_norm_mpc_params (np.ndarray): The current normalized mpc parameters for the dataset episode.
            distances_to_collision (np.ndarray): The distances to collision for the dataset episode.
            d2collision_threshold (float): The distance to collision threshold for which to apply a factor to the Q_p parameter.
            add_noise_to_preferences (bool): Whether to add noise to the parameter preferences.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The parameter preferences and updated current norm mpc params for the dataset.
        """
        unnorm_param_increments = np.zeros_like(dnn_input_current_norm_mpc_params)
        norm_param_increments = np.zeros_like(dnn_input_current_norm_mpc_params)
        new_unnorm_mpc_params = np.zeros_like(dnn_input_current_norm_mpc_params)
        new_unnorm_mpc_params[0, :] = self._unnormalize_parameters(
            dnn_input_current_norm_mpc_params[0, :]
        )
        new_norm_mpc_params = np.zeros_like(new_unnorm_mpc_params)
        unnorm_params_0 = np.zeros_like(new_unnorm_mpc_params[0, :])
        unnorm_params_1 = np.zeros_like(new_unnorm_mpc_params[0, :])
        unnorm_params_2 = np.zeros_like(new_unnorm_mpc_params[0, :])
        unnorm_params_3 = np.zeros_like(new_unnorm_mpc_params[0, :])
        unnorm_params_4 = np.zeros_like(new_unnorm_mpc_params[0, :])
        unnorm_params_5 = np.zeros_like(new_unnorm_mpc_params[0, :])
        for k in range(n_timesteps):
            t = k * self.timestep
            # ramp1: ned til 5.0 m for r_safe_do i stredet ved ish t = 100.0s. => mink med (5 - r_safe_do_init) / (100.0 - 0.0) per
            # ramp2: Opp til 15.0m frå ish t = 124.0s til 160.0s.
            # ramp3: Ned til 5.0 m igjen frå ish t = 190.0s til 200.0s.
            # ramp4: Opp til 30.0m frå ish t = 234.0s til 254.0s.
            # ved d2collision > 500.0m, ingen endring i r_safe_do pga irrelevant.
            # # Mink då K_app_course og K_app_speed med 25% av max increment ved d2collision > 300.0m
            # søk unary op on numpy array, apply ramp func

            # ved gitt terskel til kollisjon, auke Q_p med max increment
            if t >= 0.0 and t < 30.0:
                if t == 0.0:
                    unnorm_params_0 = new_unnorm_mpc_params[0, :].copy()
                ramp_Q_p_1 = (0.6 - unnorm_params_0[0]) / ((30.0 - 0.0) / self.timestep)
                ramp_Q_p_2 = (40.0 - unnorm_params_0[1]) / (
                    (30.0 - 0.0) / self.timestep
                )
                ramp_Q_p_3 = (40.0 - unnorm_params_0[2]) / (
                    (30.0 - 0.0) / self.timestep
                )
                ramp_K_app = (90.0 - unnorm_params_0[3]) / (
                    (30.0 - 0.0) / self.timestep
                )
                ramp_w_colregs = (120.0 - unnorm_params_0[5]) / (
                    (30.0 - 0.0) / self.timestep
                )
                ramp_r_safe_do = 0.0
            elif t >= 30.0 and t < 100.0:
                if t == 30.0:
                    unnorm_params_1 = new_unnorm_mpc_params[k - 1, :].copy()
                ramp_r_safe_do = (7.0 - unnorm_params_1[-1]) / (
                    (100.0 - 30.0) / self.timestep
                )
                ramp_Q_p_1 = 0.0
                ramp_Q_p_2 = 0.0
                ramp_Q_p_3 = 0.0
                ramp_K_app = 0.0
                ramp_w_colregs = (80.0 - unnorm_params_1[5]) / (
                    (100.0 - 30.0) / self.timestep
                )
            elif t >= 124.0 and t < 160.0:
                if t == 124.0:
                    unnorm_params_2 = new_unnorm_mpc_params[k - 1, :].copy()
                ramp_r_safe_do = (15.0 - unnorm_params_2[-1]) / (
                    (160.0 - 124.0) / self.timestep
                )
                ramp_Q_p_1 = (0.3 - unnorm_params_2[0]) / (
                    (160.0 - 124.0) / self.timestep
                )
                ramp_Q_p_2 = (20.0 - unnorm_params_2[1]) / (
                    (160.0 - 124.0) / self.timestep
                )
                ramp_Q_p_3 = (20.0 - unnorm_params_2[2]) / (
                    (160.0 - 124.0) / self.timestep
                )
                ramp_K_app = 0.0
                ramp_w_colregs = 0.0
            elif t >= 190.0 and t < 200.0:
                if t == 190.0:
                    unnorm_params_3 = new_unnorm_mpc_params[k - 1, :].copy()
                ramp_r_safe_do = (5.0 - unnorm_params_3[-1]) / (
                    (200.0 - 190.0) / self.timestep
                )
                ramp_Q_p_1 = (0.1 - unnorm_params_3[0]) / (
                    (200.0 - 190.0) / self.timestep
                )
                ramp_Q_p_2 = (15.0 - unnorm_params_3[1]) / (
                    (200.0 - 190.0) / self.timestep
                )
                ramp_Q_p_3 = (15.0 - unnorm_params_3[2]) / (
                    (200.0 - 190.0) / self.timestep
                )
                ramp_K_app = 0.0
                ramp_w_colregs = 0.0
            elif t >= 234.0 and t < 254.0:
                if t == 234.0:
                    unnorm_params_4 = new_unnorm_mpc_params[k - 1, :].copy()
                ramp_r_safe_do = (15.0 - unnorm_params_4[-1]) / (
                    (254.0 - 234.0) / self.timestep
                )
                ramp_Q_p_1 = (1.0 - unnorm_params_4[0]) / (
                    (254.0 - 234.0) / self.timestep
                )
                ramp_Q_p_2 = (30.0 - unnorm_params_4[1]) / (
                    (254.0 - 234.0) / self.timestep
                )
                ramp_Q_p_3 = (30.0 - unnorm_params_4[2]) / (
                    (254.0 - 234.0) / self.timestep
                )
                ramp_w_colregs = (60.0 - unnorm_params_4[5]) / (
                    (254.0 - 234.0) / self.timestep
                )
                ramp_K_app = (15.0 - unnorm_params_4[3]) / (
                    (254.0 - 234.0) / self.timestep
                )
            elif t >= 254.0:
                if t == 254.0:
                    unnorm_params_5 = new_unnorm_mpc_params[k - 1, :].copy()
                ramp_r_safe_do = (25.0 - unnorm_params_5[-1]) / (
                    (330.0 - 254.0) / self.timestep
                )
                ramp_Q_p_1 = (2.0 - unnorm_params_5[0]) / (
                    (330.0 - 254.0) / self.timestep
                )
                ramp_Q_p_2 = (40.0 - unnorm_params_5[1]) / (
                    (330.0 - 254.0) / self.timestep
                )
                ramp_Q_p_3 = (40.0 - unnorm_params_5[2]) / (
                    (330.0 - 254.0) / self.timestep
                )
                ramp_w_colregs = (120.0 - unnorm_params_5[5]) / (
                    (330.0 - 254.0) / self.timestep
                )
                ramp_K_app = (10.0 - unnorm_params_5[3]) / (
                    (330.0 - 254.0) / self.timestep
                )
            else:
                ramp_r_safe_do = 0.0
                ramp_Q_p_1 = 0.0
                ramp_Q_p_2 = 0.0
                ramp_Q_p_3 = 0.0
                ramp_w_colregs = 0.0

            unnorm_param_increments[k, 0] = np.clip(
                ramp_Q_p_1,
                self.dnn_out_parameter_incr_ranges["Q_p"][0][0],
                self.dnn_out_parameter_incr_ranges["Q_p"][0][1],
            )
            unnorm_param_increments[k, 1] = np.clip(
                ramp_Q_p_2,
                self.dnn_out_parameter_incr_ranges["Q_p"][1][0],
                self.dnn_out_parameter_incr_ranges["Q_p"][1][1],
            )
            unnorm_param_increments[k, 2] = np.clip(
                ramp_Q_p_3,
                self.dnn_out_parameter_incr_ranges["Q_p"][2][0],
                self.dnn_out_parameter_incr_ranges["Q_p"][2][1],
            )
            unnorm_param_increments[k, 3] = np.clip(
                ramp_K_app,
                self.dnn_out_parameter_incr_ranges["K_app_course"][0],
                self.dnn_out_parameter_incr_ranges["K_app_course"][1],
            )
            unnorm_param_increments[k, 4] = np.clip(
                ramp_K_app,
                self.dnn_out_parameter_incr_ranges["K_app_speed"][0],
                self.dnn_out_parameter_incr_ranges["K_app_speed"][1],
            )
            unnorm_param_increments[k, 5] = np.clip(
                ramp_w_colregs,
                self.dnn_out_parameter_incr_ranges["w_colregs"][0],
                self.dnn_out_parameter_incr_ranges["w_colregs"][1],
            )
            unnorm_param_increments[k, 6] = np.clip(
                ramp_w_colregs,
                self.dnn_out_parameter_incr_ranges["w_colregs"][0],
                self.dnn_out_parameter_incr_ranges["w_colregs"][1],
            )
            unnorm_param_increments[k, 7] = np.clip(
                ramp_w_colregs,
                self.dnn_out_parameter_incr_ranges["w_colregs"][0],
                self.dnn_out_parameter_incr_ranges["w_colregs"][1],
            )
            unnorm_param_increments[k, -1] = np.clip(
                ramp_r_safe_do,
                self.dnn_out_parameter_incr_ranges["r_safe_do"][0],
                self.dnn_out_parameter_incr_ranges["r_safe_do"][1],
            )

            # if next_10s_d2collision > d2collision_threshold
            # idx_10s_ahead = min(k + int(10.0 / self.timestep), n_timesteps - 1)
            # if np.all(distances_to_collision[k:idx_10s_ahead] > 0.6 * d2collision_threshold):
            #     if not (k > 0 and new_unnorm_mpc_params[k - 1, 0] >= 2.0):
            #         unnorm_param_increments[k, 0] = 0.5 * self.dnn_out_parameter_incr_ranges["Q_p"][0][1]
            #         unnorm_param_increments[k, 1] = 0.5 * self.dnn_out_parameter_incr_ranges["Q_p"][1][1]
            #         unnorm_param_increments[k, 2] = 0.5 * self.dnn_out_parameter_incr_ranges["Q_p"][2][1]
            #     unnorm_param_increments[k, 3] = 0.5 * self.dnn_out_parameter_incr_ranges["K_app_course"][0]
            #     unnorm_param_increments[k, 4] = 0.5 * self.dnn_out_parameter_incr_ranges["K_app_speed"][0]
            #     unnorm_param_increments[k, 5] = 0.5 * self.dnn_out_parameter_incr_ranges["w_colregs"][0]
            #     unnorm_param_increments[k, 6] = 0.5 * self.dnn_out_parameter_incr_ranges["w_colregs"][0]
            #     unnorm_param_increments[k, 7] = 0.5 * self.dnn_out_parameter_incr_ranges["w_colregs"][0]

            if add_noise_to_preferences:
                unnorm_param_increments = self.add_noise_to_preferences(
                    unnorm_param_increments, k
                )

            norm_param_increments[k, :] = self._normalize_parameter_increments(
                unnorm_param_increments[k, :]
            )
            if k > 0:
                new_unnorm_mpc_params[k, :] = (
                    new_unnorm_mpc_params[k - 1, :] + unnorm_param_increments[k, :]
                )

            new_norm_mpc_params[k, :] = self._normalize_parameters(
                new_unnorm_mpc_params[k, :].copy()
            )  # clips to -1.0, 1.0
            new_unnorm_mpc_params[k, :] = self._unnormalize_parameters(
                new_norm_mpc_params[k, :].copy()
            )

        # self.plot_ad_hoc_preferences(new_unnorm_mpc_params)
        return norm_param_increments, new_norm_mpc_params

    def plot_ad_hoc_preferences(
        self,
        unnorm_mpc_params_prefs: np.ndarray,
        pred_unnorm_mpc_param: Optional[np.ndarray] = None,
    ) -> None:
        """Plots the ad hoc parameter preferences for the parameter provider dataset.

        Args:
            unnorm_mpc_params_prefs (np.ndarray): Preferred hand-crafted unnormalized mpc params.
            pred_unnorm_mpc_param (Optional[np.ndarray]): Predicted unnormalized mpc params from the DNN (added on top of the original mpc params per step)
        """
        assert unnorm_mpc_params_prefs.shape[1] == 9, (
            "The parameter increments must have shape (n_timesteps, 9)"
        )
        n_timesteps = unnorm_mpc_params_prefs.shape[0]
        times = np.arange(0, n_timesteps * self.timestep, self.timestep)
        matplotlib.use("TkAgg")
        _, ax = plt.subplots(4, 1, figsize=(8, 8))
        if pred_unnorm_mpc_param is not None:
            ax[0].plot(times, unnorm_mpc_params_prefs[:, 0], label="Q_p[0]")
            ax[0].plot(times, unnorm_mpc_params_prefs[:, 1], label="Q_p[1]")
            ax[0].plot(times, unnorm_mpc_params_prefs[:, 2], label="Q_p[2]")
        else:
            ax[0].semilogy(times, unnorm_mpc_params_prefs[:, 0], label="Q_p[0]")
            ax[0].semilogy(times, unnorm_mpc_params_prefs[:, 1], label="Q_p[1]")
            ax[0].semilogy(times, unnorm_mpc_params_prefs[:, 2], label="Q_p[2]")

        ax[1].plot(times, unnorm_mpc_params_prefs[:, 3], label="K_app_course")
        ax[1].plot(times, unnorm_mpc_params_prefs[:, 4], label="K_app_speed")

        ax[2].plot(times, unnorm_mpc_params_prefs[:, 5], label="w_colregs[0]")
        ax[2].plot(times, unnorm_mpc_params_prefs[:, 6], label="w_colregs[1]")
        ax[2].plot(times, unnorm_mpc_params_prefs[:, 7], label="w_colregs[2]")

        ax[3].plot(times, unnorm_mpc_params_prefs[:, 8], label="r_safe_do")
        ax[3].set_xlabel("Time [s]")
        if pred_unnorm_mpc_param is not None:
            ax[0].plot(times, pred_unnorm_mpc_param[:, 0], label="pred Q_p[0]")
            ax[0].plot(times, pred_unnorm_mpc_param[:, 1], label="pred Q_p[1]")
            ax[0].plot(times, pred_unnorm_mpc_param[:, 2], label="pred Q_p[2]")

            ax[1].plot(times, pred_unnorm_mpc_param[:, 3], label="pred K_app_course")
            ax[1].plot(times, pred_unnorm_mpc_param[:, 4], label="pred K_app_speed")

            ax[2].plot(times, pred_unnorm_mpc_param[:, 5], label="pred w_colregs[0]")
            ax[2].plot(times, pred_unnorm_mpc_param[:, 6], label="pred w_colregs[1]")
            ax[2].plot(times, pred_unnorm_mpc_param[:, 7], label="pred w_colregs[2]")

            ax[3].plot(times, pred_unnorm_mpc_param[:, 8], label="pred r_safe_do")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()

        plt.show(block=False)

    def test_model_on_episode_data(self, model) -> None:
        ep_idx = int(np.random.randint(0, self.n_episodes))
        ep_data = self.data[ep_idx]
        dnn_input_features = ep_data[2][0]
        pref_norm_mpc_param_incr = ep_data[2][1]
        pred_norm_mpc_param_incr = (
            model(torch.from_numpy(dnn_input_features.copy())).detach().numpy()
        )

        original_norm_mpc_params = dnn_input_features[
            :, -self.num_adjustable_mpc_params :
        ].copy()
        new_norm_mpc_params = original_norm_mpc_params.copy()
        original_unnorm_mpc_params = np.zeros_like(new_norm_mpc_params)
        new_unnorm_mpc_params = np.zeros_like(new_norm_mpc_params)
        for k in range(dnn_input_features.shape[0]):
            original_unnorm_mpc_params[k, :] = self._unnormalize_parameters(
                original_norm_mpc_params[k, :].copy()
            )
            if k > 0:
                new_norm_mpc_params[k, :] = (
                    new_norm_mpc_params[k - 1, :] + pred_norm_mpc_param_incr[k, :]
                )
            new_norm_mpc_params[k, :] = np.clip(new_norm_mpc_params[k, :], -1.0, 1.0)
            new_unnorm_mpc_params[k, :] = self._unnormalize_parameters(
                new_norm_mpc_params[k, :].copy()
            )
        # self.plot_ad_hoc_preferences(original_unnorm_mpc_params, pred_unnorm_mpc_param=new_unnorm_mpc_params)
        self.plot_ad_hoc_preferences(pref_norm_mpc_param_incr, pred_norm_mpc_param_incr)

    def _unnormalize_parameters(self, x: np.ndarray) -> np.ndarray:
        """Unnormalize the input parameter.

        Args:
            x  (np.ndarray): The normalized parameter

        Returns:
            np.ndarray: The unnormalized parameter
        """
        x_unnorm = np.zeros_like(x)
        for param_name in self.param_list:
            param_range = self.dnn_out_parameter_ranges[param_name]
            param_length = self.dnn_out_parameter_lengths[param_name]
            pindx = self.out_parameter_indices[param_name]
            x_param = x[pindx : pindx + param_length].copy()

            for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
                if param_name == "Q_p":
                    x_param[j] = csmf.linear_map(
                        x_param[j], (-1.0, 1.0), tuple(param_range[j])
                    )
                else:
                    x_param[j] = csmf.linear_map(
                        x_param[j], (-1.0, 1.0), tuple(param_range)
                    )
            x_unnorm[pindx : pindx + param_length] = x_param
        return x_unnorm

    def _normalize_parameters(self, x: np.ndarray) -> np.ndarray:
        """Normalize the input parameter.

        Args:
            x (np.ndarray): The unnormalized parameter

        Returns:
            np.ndarray: The normalized parameter
        """
        x_norm = np.zeros_like(x)
        for param_name in self.param_list:
            param_range = self.dnn_out_parameter_ranges[param_name]
            param_length = self.dnn_out_parameter_lengths[param_name]
            pindx = self.out_parameter_indices[param_name]
            x_param = x[pindx : pindx + param_length].copy()

            for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
                if param_name == "Q_p":
                    x_param[j] = csmf.linear_map(
                        x_param[j], tuple(param_range[j]), (-1.0, 1.0)
                    )
                else:
                    x_param[j] = csmf.linear_map(
                        x_param[j], tuple(param_range), (-1.0, 1.0)
                    )
            x_norm[pindx : pindx + param_length] = x_param
        return np.clip(x_norm, -1.0, 1.0)

    def _normalize_parameter_increments(self, x: np.ndarray) -> np.ndarray:
        """Normalize the input parameter increment.

        Args:
            x (np.ndarray): The unnormalized parameter increment

        Returns:
            np.ndarray: The normalized parameter increment
        """
        x_norm = np.zeros_like(x)
        for param_name in self.param_list:
            param_range = self.dnn_out_parameter_incr_ranges[param_name]
            param_length = self.dnn_out_parameter_lengths[param_name]
            pindx = self.out_parameter_indices[param_name]
            x_param_norm = x[pindx : pindx + param_length].copy()

            for j in range(len(x_param_norm)):  # pylint: disable=consider-using-enumerate
                if param_name == "Q_p":
                    x_param_norm[j] = csmf.linear_map(
                        x_param_norm[j], tuple(param_range[j]), (-1.0, 1.0)
                    )
                else:
                    x_param_norm[j] = csmf.linear_map(
                        x_param_norm[j], tuple(param_range), (-1.0, 1.0)
                    )
            x_norm[pindx : pindx + param_length] = x_param_norm
        return x_norm

    def _unnormalize_parameter_increments(self, x: np.ndarray) -> np.ndarray:
        """Unnormalize the input parameter increment.

        Args:
            x  (np.ndarray): The normalized parameter increment

        Returns:
            np.ndarray: The unnormalized parameter increment
        """
        x_unnorm = np.zeros_like(x)
        for param_name in self.param_list:
            param_range = self.dnn_out_parameter_incr_ranges[param_name]
            param_length = self.dnn_out_parameter_lengths[param_name]
            pindx = self.out_parameter_indices[param_name]
            x_param = x[pindx : pindx + param_length]

            for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
                x_param[j] = csmf.linear_map(
                    x_param[j], (-1.0, 1.0), tuple(param_range)
                )
            x_unnorm[pindx : pindx + param_length] = x_param
        return x_unnorm

    def get_data(self):
        return self.data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert idx < self.n_samples, "Index out of range"
        # map idx to episode bin:
        cumulative_sum = 0
        ep_idx = 0
        considered_samples = self.data[0][2]
        for episode, n_timesteps, samples in self.data:
            cumulative_sum += n_timesteps
            if idx < cumulative_sum:
                considered_samples = samples
                ep_idx = episode
                break

        sample_idx = idx - (cumulative_sum - self.data[ep_idx][1])

        dnn_input_feature_sample = torch.from_numpy(
            considered_samples[0][sample_idx, :].copy().astype(np.float32)
        )
        norm_mpc_param_incr_sample = torch.from_numpy(
            considered_samples[1][sample_idx, :].copy().astype(np.float32)
        )

        if self.transform:
            dnn_input_feature_sample = self.transform(dnn_input_feature_sample)
            norm_mpc_param_incr_sample = self.transform(norm_mpc_param_incr_sample)
        return dnn_input_feature_sample, norm_mpc_param_incr_sample

    def add_noise_to_preferences(
        self, unnorm_param_increments: np.ndarray, k: int
    ) -> np.ndarray:
        """Adds noise to the parameter increments.

        Args:
            unnorm_param_increments (np.ndarray): The unnormalized parameter increments.
            k (int): The timestep index.

        Returns:
            np.ndarray: The updated parameter increments.
        """
        assert unnorm_param_increments.shape[1] == 9, (
            "The parameter increments must have shape (n_timesteps, 9)"
        )
        eps = 1e-6
        if abs(unnorm_param_increments[k, 0]) > eps:
            unnorm_param_increments[k, 0] = np.clip(
                unnorm_param_increments[k, 0] + np.random.normal(0.0, 0.005),
                self.dnn_out_parameter_incr_ranges["Q_p"][0][0],
                self.dnn_out_parameter_incr_ranges["Q_p"][0][1],
            )
        if abs(unnorm_param_increments[k, 1]) > eps:
            unnorm_param_increments[k, 1] = np.clip(
                unnorm_param_increments[k, 1] + np.random.normal(0.0, 0.5),
                self.dnn_out_parameter_incr_ranges["Q_p"][1][0],
                self.dnn_out_parameter_incr_ranges["Q_p"][1][1],
            )
        if abs(unnorm_param_increments[k, 2]) > eps:
            unnorm_param_increments[k, 2] = np.clip(
                unnorm_param_increments[k, 2] + np.random.normal(0.0, 0.5),
                self.dnn_out_parameter_incr_ranges["Q_p"][2][0],
                self.dnn_out_parameter_incr_ranges["Q_p"][2][1],
            )
        if abs(unnorm_param_increments[k, 3]) > eps:
            unnorm_param_increments[k, 3] = np.clip(
                unnorm_param_increments[k, 3] + np.random.normal(0.0, 1.0),
                self.dnn_out_parameter_incr_ranges["K_app_course"][0],
                self.dnn_out_parameter_incr_ranges["K_app_course"][1],
            )
        if abs(unnorm_param_increments[k, 4]) > eps:
            unnorm_param_increments[k, 4] = np.clip(
                unnorm_param_increments[k, 4] + np.random.normal(0.0, 1.0),
                self.dnn_out_parameter_incr_ranges["K_app_speed"][0],
                self.dnn_out_parameter_incr_ranges["K_app_speed"][1],
            )
        if abs(unnorm_param_increments[k, 5]) > eps:
            unnorm_param_increments[k, 5] = np.clip(
                unnorm_param_increments[k, 5] + np.random.normal(0.0, 1.0),
                self.dnn_out_parameter_incr_ranges["w_colregs"][0],
                self.dnn_out_parameter_incr_ranges["w_colregs"][1],
            )
        if abs(unnorm_param_increments[k, 6]) > eps:
            unnorm_param_increments[k, 6] = np.clip(
                unnorm_param_increments[k, 6] + np.random.normal(0.0, 1.0),
                self.dnn_out_parameter_incr_ranges["w_colregs"][0],
                self.dnn_out_parameter_incr_ranges["w_colregs"][1],
            )
        if abs(unnorm_param_increments[k, 7]) > eps:
            unnorm_param_increments[k, 7] = np.clip(
                unnorm_param_increments[k, 7] + np.random.normal(0.0, 1.0),
                self.dnn_out_parameter_incr_ranges["w_colregs"][0],
                self.dnn_out_parameter_incr_ranges["w_colregs"][1],
            )
        if abs(unnorm_param_increments[k, 8]) > eps:
            unnorm_param_increments[k, 8] = np.clip(
                unnorm_param_increments[k, 8] + np.random.normal(0.0, 0.05),
                self.dnn_out_parameter_incr_ranges["r_safe_do"][0],
                self.dnn_out_parameter_incr_ranges["r_safe_do"][1],
            )
        return unnorm_param_increments
