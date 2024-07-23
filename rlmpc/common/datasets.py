"""
    datasets.py

    Summary:
        Contains classes for pytorch datasets.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.common.math_functions as csmf
import colav_simulator.gym.logger as csenv_logger
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms_v2


class UnNormalize(transforms_v2.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1.0 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


class PerceptionImageDataset(Dataset):
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
        self.data = np.load(data_dir / data_npy_file, mmap_mode="r", allow_pickle=True).astype(np.uint8)
        # self.data = self.data[3:100, 0, :, :, :]  # disregard 3 first.
        self.masks = None
        if mask_npy_file is not None:
            self.masks = np.load(data_dir / mask_npy_file, mmap_mode="r", allow_pickle=True).astype(np.uint8)
            # self.masks = self.masks[3:100, 0, :, :, :]  # disregard 3 first.

        if len(self.data.shape) == 4:
            self.data = np.expand_dims(self.data, axis=0)
            self.masks = np.expand_dims(self.masks, axis=0) if self.masks is not None else None

        self.n_samples, self.n_envs, self.n_channels, self.height, self.width = self.data.shape

        self.unnormalize_transform = transforms_v2.Compose(
            [UnNormalize(mean=[0.5], std=[0.5]), transforms_v2.ToDtype(torch.uint8, scale=True)]
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
        sample = torch.from_numpy(self.data[sample_idx, env_idx, :, :, :].copy().astype(np.uint8))
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
        self.data = np.load(data_dir / data_npy_file, mmap_mode="r", allow_pickle=True).astype(np.float32)
        # self.data = self.data[:1, :1]
        self.n_samples, self.n_envs, self.max_num_do, self.do_info_dim = self.data.shape

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
        sample = torch.from_numpy(self.data[sample_idx, env_idx, :, :].copy().astype(np.float32))
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
        num_adjustable_mpc_params: int = 4,
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
        self.num_adjustable_mpc_params = num_adjustable_mpc_params
        self.env_data_logger = csenv_logger.Logger(experiment_name="parameter_provider_dataset", log_dir=data_dir)
        self.env_data_logger.load_from_pickle(name=env_data_pkl_file)
        self.timestep = 2.0
        self.param_list = ["Q_p", "r_safe_do"]
        self.dnn_out_parameter_ranges = {
            "Q_p": [[0.001, 2.5], [0.1, 100.0], [0.1, 100.0]],
            "K_app_course": [0.1, 200.0],
            "K_app_speed": [0.1, 200.0],
            "d_attenuation": [10.0, 1000.0],
            "w_colregs": [0.1, 500.0],
            "r_safe_do": [5.0, 120.0],
        }
        self.dnn_out_parameter_incr_ranges = {
            "Q_p": [[-0.25, 0.25], [-2.0, 2.0], [-2.0, 2.0]],
            "K_app_course": [-5.0, 5.0],
            "K_app_speed": [-5.0, 5.0],
            "d_attenuation": [-50.0, 50.0],
            "w_colregs": [-10.0, 10.0],
            "r_safe_do": [-5.0, 5.0],
        }
        self.dnn_out_parameter_lengths = {
            "Q_p": 3,
            "K_app_course": 1,
            "K_app_speed": 1,
            "d_attenuation": 1,
            "w_colregs": 3,
            "r_safe_do": 1,
        }
        offset = 0
        self.out_parameter_indices = {}
        for param in self.param_list:
            self.out_parameter_indices[param] = offset
            offset += self.dnn_out_parameter_lengths[param]

        self._setup_data()

    def _setup_data(self):
        """Sets up the param provider dataset: Extracts input data from the env data file and computes ad hoc parameter preferences for the parameter provider dataset, based on the env data provided."""

        env_data = self.env_data_logger.env_data
        assert "ms_channel" in env_data[0].name, "must be the rlmpc_scenario_ms_channel"
        self.data = []
        for epdata in env_data:
            dnn_input_features = [ainfo["dnn_input_features"] for ainfo in epdata.actor_infos]
            dnn_input_current_norm_mpc_params = [ainfo["norm_old_mpc_params"] for ainfo in epdata.actor_infos]
            n_timesteps = len(dnn_input_features)
            param_incr_preferences = self._compute_ad_hoc_parameter_preferences(
                n_timesteps=n_timesteps,
                dnn_input_features=dnn_input_features,
                dnn_input_current_norm_mpc_params=dnn_input_current_norm_mpc_params,
                distances_to_collision=epdata.distances_to_collision,
            )
            # add param_incr_preferences to the dnn_input_current_norm_mpc_params for a well-posed dataset
            processed_epdata = list(zip(dnn_input_features, dnn_input_current_norm_mpc_params, param_incr_preferences))
            self.data.extend(processed_epdata)

    def _compute_ad_hoc_parameter_preferences(
        self,
        n_timesteps: int,
        dnn_input_features: np.ndarray,
        dnn_input_current_norm_mpc_params: np.ndarray,
        distances_to_collision: np.ndarray,
        d2collision_threshold: float = 500.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes ad hoc parameter preferences for the parameter provider dataset, based on the env data provided.

        Args:
            n_timesteps (int): Number of timesteps in the env data.
            d2collision_threshold (float): The distance to collision threshold for which to apply a factor to the Q_p parameter.


        Returns:
            Tuple[np.ndarray, np.ndarray]: The parameter preferences and updated current norm mpc params for the dataset.
        """
        unnorm_param_increments = np.zeros((self.num_adjustable_mpc_params, n_timesteps))
        param_increments = np.zeros((self.num_adjustable_mpc_params, n_timesteps))
        unnorm_mpc_params_0 = self._unnormalize_parameters(dnn_input_current_norm_mpc_params[0])
        unnorm_mpc_params_124 = self._unnormalize_parameters(
            dnn_input_current_norm_mpc_params[int(124 // self.timestep)]
        )
        unnorm_mpc_params_190 = self._unnormalize_parameters(
            dnn_input_current_norm_mpc_params[int(190 // self.timestep)]
        )
        unnorm_mpc_params_234 = self._unnormalize_parameters(
            dnn_input_current_norm_mpc_params[int(234 // self.timestep)]
        )
        new_unnorm_mpc_params = np.zeros_like(dnn_input_current_norm_mpc_params)
        for k in range(n_timesteps):
            t = k * self.timestep
            unnorm_mpc_params_k = self._unnormalize_parameters(dnn_input_current_norm_mpc_params[k])
            if t >= 30.0 and t < 100.0:
                ramp = (5.0 - unnorm_mpc_params_0[-1]) / ((100.0 - 30.0) / self.timestep)
            elif t >= 124.0 and t < 160.0:
                ramp = (15.0 - unnorm_mpc_params_124[-1]) / (160.0 - 124.0)
            elif t >= 190.0 and t < 200.0:
                ramp = (5.0 - unnorm_mpc_params_190[-1]) / (200.0 - 190.0)
            elif t >= 234.0 and t < 254.0:
                ramp = (30.0 - unnorm_mpc_params_234[-1]) / (254.0 - 234.0)
            else:
                ramp = 0.0

            unnorm_param_increments[-1, k] = ramp
            # if next_20s_d2collision > d2collision_threshold
            idx_20s_ahead = min(k + int(20.0 / self.timestep), n_timesteps - 1)
            if np.all(distances_to_collision[k:idx_20s_ahead] > d2collision_threshold):
                unnorm_param_increments[0, k] = self.dnn_out_parameter_incr_ranges["Q_p"][0][1]
                unnorm_param_increments[1, k] = self.dnn_out_parameter_incr_ranges["Q_p"][1][1]
                unnorm_param_increments[2, k] = self.dnn_out_parameter_incr_ranges["Q_p"][2][1]

            next_idx = min(k + 1, n_timesteps - 1)
            if (
                distances_to_collision[k] < 0.5 * d2collision_threshold
                and distances_to_collision[next_idx] < distances_to_collision[k]
            ):
                unnorm_param_increments[0, k] = self.dnn_out_parameter_incr_ranges["Q_p"][0][0]
                unnorm_param_increments[1, k] = self.dnn_out_parameter_incr_ranges["Q_p"][1][0]
                unnorm_param_increments[2, k] = self.dnn_out_parameter_incr_ranges["Q_p"][2][0]

            param_increments[:, k] = self._normalize_parameter_increments(unnorm_param_increments[:, k])
            new_unnorm_mpc_params[k] = unnorm_mpc_params_k + unnorm_param_increments[:, k]

            # ramp1: ned til 5.0 m for r_safe_do i stredet ved ish t = 100.0s. => mink med (5 - r_safe_do_init) / (100.0 - 0.0) per
            # ramp2: Opp til 15.0m frå ish t = 124.0s til 160.0s.
            # ramp3: Ned til 5.0 m igjen frå ish t = 190.0s til 200.0s.
            # ramp4: Opp til 30.0m frå ish t = 234.0s til 254.0s.
            # ved d2collision > 500.0m, ingen endring i r_safe_do pga irrelevant.
            # søk unary op on numpy array, apply ramp func

            # ved terskel på 500.0m + til kollisjon, auke Q_p med faktor på 10.0

        # ramp function from initial norm mpc param
        # decrease to
        return param_increments, new_norm_mpc_params

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
            x_param = x[pindx : pindx + param_length]

            for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
                if param_name == "Q_p":
                    x_param[j] = csmf.linear_map(x_param[j], (-1.0, 1.0), tuple(param_range[j]))
                else:
                    x_param[j] = csmf.linear_map(x_param[j], (-1.0, 1.0), tuple(param_range))
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
            x_param = x[pindx : pindx + param_length]

            for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
                if param_name == "Q_p":
                    x_param[j] = csmf.linear_map(x_param[j], tuple(param_range[j]), (-1.0, 1.0))
                else:
                    x_param[j] = csmf.linear_map(x_param[j], tuple(param_range), (-1.0, 1.0))
            x_norm[pindx : pindx + param_length] = x_param
        return x_norm

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
                    x_param_norm[j] = csmf.linear_map(x_param_norm[j], tuple(param_range[j]), (-1.0, 1.0))
                else:
                    x_param_norm[j] = csmf.linear_map(x_param_norm[j], tuple(param_range), (-1.0, 1.0))
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

            for j in range(len(x_param)):
                x_param[j] = csmf.linear_map(x_param[j], (-1.0, 1.0), tuple(param_range))
            x_unnorm[pindx : pindx + param_length] = x_param
        return x_unnorm

    def get_data(self):
        return self.data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = torch.from_numpy(self.data[sample_idx].copy().astype(np.float32))
        # sort sample after entry 0 (distance)
        if self.transform:
            sample = self.transform(sample)
        return sample
