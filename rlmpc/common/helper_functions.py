"""
helper_functions.py

Summary:
    Contains miscellaneous helper functions for the RL-RRT-MPC COLAV system.

Author: Trym Tengesdal
"""

import linecache
import pathlib
import resource
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import casadi as csd
import colav_simulator.common.math_functions as csmf
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.integrators as sim_integrators
import colav_simulator.core.models as sim_models
import colav_simulator.gym.logger as colav_logger
import gymnasium as gym
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.file_utils as fu
import rlmpc.common.logger as rlmpc_logger
import rlmpc.common.math_functions as mf
import rlmpc.common.paths as dp
import seacharts.enc as senc
import shapely.affinity as affinity
import shapely.geometry as geometry
import shapely.ops as ops
import torch
import torch as th
import torchvision
import yaml
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.stats import chi2
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage{bm}",
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}",
            ]
        ),
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{bm}",
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}",
            ]
        ),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


def normalize_mpc_param_tensor(
    x: th.Tensor,
    param_list: List[str],
    parameter_ranges: Dict[str, Any],
    parameter_lengths: Dict[str, Any],
    parameter_indices: Dict[str, Any],
) -> th.Tensor:
    """Normalize the input parameter tensor.

    Args:
        x (th.Tensor): The unnormalized parameter tensor
        param_list (List[str]): The list of parameters to map.
        parameter_ranges (Dict[str, Any]): The parameter ranges.
        parameter_lengths (Dict[str, Any]): The parameter lengths.
        parameter_indices (Dict[str, Any]): The parameter indices.

    Returns:
        th.Tensor: The normalized parameter tensor
    """
    x_norm = th.zeros_like(x, dtype=th.float32)
    for param_name in param_list:
        param_range = parameter_ranges[param_name]
        param_length = parameter_lengths[param_name]
        pindx = parameter_indices[param_name]
        x_param = x[pindx : pindx + param_length]

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
    return x_norm


def unnormalize_mpc_param_tensor(
    x: th.Tensor | np.ndarray,
    param_list: List[str],
    parameter_ranges: Dict[str, Any],
    parameter_lengths: Dict[str, Any],
    parameter_indices: Dict[str, Any],
) -> np.ndarray:
    """Unnormalize the input parameter tensor.

    Args:
        x (th.Tensor): The normalized parameter tensor
        param_list (List[str]): The list of parameters to map.
        parameter_ranges (Dict[str, Any]): The parameter ranges.
        parameter_lengths (Dict[str, Any]): The parameter lengths.
        parameter_indices (Dict[str, Any]): The parameter indices.

    Returns:
        np.ndarray: The unnormalized output as a numpy array
    """
    if isinstance(x, th.Tensor):
        x = x.detach().numpy()
    x_unnorm = np.zeros_like(x, dtype=np.float32)
    for param_name in param_list:
        param_range = parameter_ranges[param_name]
        param_length = parameter_lengths[param_name]
        pindx = parameter_indices[param_name]
        x_param = x[pindx : pindx + param_length]

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


def normalize_mpc_param_increment_tensor(
    x: th.Tensor,
    param_list: List[str],
    parameter_incr_ranges: Dict[str, Any],
    parameter_lengths: Dict[str, Any],
    parameter_indices: Dict[str, Any],
) -> th.Tensor:
    """Normalize the input parameter increment tensor.

    Args:
        x (th.Tensor): The unnormalized parameter increment tensor
        param_list (List[str]): The list of parameters to map.
        parameter_incr_ranges (Dict[str, Any]): The parameter increment ranges.
        parameter_lengths (Dict[str, Any]): The parameter lengths.
        parameter_indices (Dict[str, Any]): The parameter indices.

    Returns:
        th.Tensor: The normalized parameter increment tensor
    """
    x_norm = th.zeros_like(x, dtype=th.float32)
    for param_name in param_list:
        param_incr_range = parameter_incr_ranges[param_name]
        param_length = parameter_lengths[param_name]
        pindx = parameter_indices[param_name]
        x_param = x[pindx : pindx + param_length]

        for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
            if param_name == "Q_p":
                x_param[j] = csmf.linear_map(
                    x_param[j], tuple(param_incr_range[j]), (-1.0, 1.0)
                )
            else:
                x_param[j] = csmf.linear_map(
                    x_param[j], tuple(param_incr_range), (-1.0, 1.0)
                )
        x_norm[pindx : pindx + param_length] = x_param
    return x_norm


def unnormalize_mpc_param_increment_tensor(
    x: th.Tensor,
    param_list: List[str],
    parameter_incr_ranges: Dict[str, Any],
    parameter_lengths: Dict[str, Any],
    parameter_indices: Dict[str, Any],
) -> np.ndarray:
    """Unnormalize the input parameter increment tensor.

    Args:
        x (th.Tensor): The normalized parameter increment tensor
        param_list (List[str]): The list of parameters to map.
        parameter_incr_ranges (Dict[str, Any]): The parameter increment ranges.
        parameter_lengths (Dict[str, Any]): The parameter lengths.
        parameter_indices (Dict[str, Any]): The parameter indices.

    Returns:
        np.ndarray: The unnormalized output as a numpy array
    """
    if x.ndim == 1:
        x = x.unsqueeze(0).detach().numpy()
    x_unnorm = np.zeros_like(x, dtype=np.float32)

    for i in range(x.shape[0]):
        for param_name in param_list:
            param_incr_range = parameter_incr_ranges[param_name]
            param_length = parameter_lengths[param_name]
            pindx = parameter_indices[param_name]
            x_param = x[i, pindx : pindx + param_length].copy()

            for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
                if param_name == "Q_p":
                    x_param[j] = csmf.linear_map(
                        x_param[j], (-1.0, 1.0), tuple(param_incr_range[j])
                    )
                else:
                    x_param[j] = csmf.linear_map(
                        x_param[j], (-1.0, 1.0), tuple(param_incr_range)
                    )
            x_unnorm[i, pindx : pindx + param_length] = x_param
    if x_unnorm.shape[0] == 1:
        x_unnorm = x_unnorm.squeeze(0)
    return x_unnorm


def map_mpc_param_incr_array_to_parameter_dict(
    x: np.ndarray,
    current_params: np.ndarray,
    param_list: List[str],
    parameter_ranges: Dict[str, Any],
    parameter_incr_ranges: Dict[str, Any],
    parameter_lengths: Dict[str, Any],
    parameter_indices: Dict[str, Any],
) -> Dict[str, Union[float, np.ndarray]]:
    """Maps the MPC parameter DNN output parameter increment tensor to a dictionary of unnormalized parameters, given the current parameters.

    Args:
        x (np.ndarray): The DNN output, consisting of normalized parameter increments.
        current_params (np.ndarray): The current parameters, unnormalized.
        param_list (List[str]): The list of parameters to map.
        parameter_ranges (Dict[str, Any]): The parameter ranges.
        parameter_incr_ranges (Dict[str, Any]): The parameter increment ranges.
        parameter_lengths (Dict[str, Any]): The parameter lengths.
        parameter_indices (Dict[str, Any]): The parameter indices.

    Returns:
        Dict[str, Union[float, np.ndarray]]: The dictionary of unnormalized parameters
    """
    params = {}
    x_np = x.copy()
    current_params_np = current_params.copy()
    for param_name in param_list:
        param_range = parameter_ranges[param_name]
        param_incr_range = parameter_incr_ranges[param_name]
        param_length = parameter_lengths[param_name]
        pindx = parameter_indices[param_name]

        x_param_incr = x_np[pindx : pindx + param_length].copy()
        for j in range(len(x_param_incr)):  # pylint: disable=consider-using-enumerate
            if param_name == "Q_p":
                x_param_incr[j] = csmf.linear_map(
                    x_param_incr[j], (-1.0, 1.0), tuple(param_incr_range[j])
                )
            else:
                x_param_incr[j] = csmf.linear_map(
                    x_param_incr[j], (-1.0, 1.0), tuple(param_incr_range)
                )

        x_param_current = current_params_np[pindx : pindx + param_length]
        x_param_new = x_param_current + x_param_incr
        if param_name == "Q_p":
            for j in range(3):
                x_param_new[j] = np.clip(
                    x_param_new[j], param_range[j][0], param_range[j][1]
                )
        else:
            x_param_new = np.clip(x_param_new, param_range[0], param_range[1])
        params[param_name] = x_param_new.astype(np.float32)
    return params


def make_env(env_id: str, env_config: dict, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env. Key thing is to update the env identifier
    such that live plotting and mpc compilation works.

    Args:
        env_id: (str) the environment ID
        env_config: (dict) the environment config
        rank: (int) index of the subprocess
        seed: (int) the inital seed for RNG

    Returns:
        (Callable): a function that creates the environment
    """

    def _init():
        env_config.update(
            {"identifier": env_config["identifier"] + str(rank), "seed": seed + rank}
        )
        env = Monitor(gym.make(env_id, **env_config))
        return env

    set_random_seed(seed)
    return _init


def create_data_dirs(
    base_dir: Path, experiment_name: str, remove_log_files: bool = True
) -> Tuple[Path, Path, Path]:
    base_dir = base_dir / experiment_name
    log_dir = base_dir / "logs"
    model_dir = base_dir / "models"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    else:
        if remove_log_files:
            for file in log_dir.iterdir():
                if file.is_dir():
                    for f in file.iterdir():
                        f.unlink()
                    file.rmdir()
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    return base_dir, log_dir, model_dir


def set_memory_limit(n_bytes: int = 20_000_000_000) -> None:
    """Force Python to raise an exception when it uses more than
    n_bytes bytes of memory.

    Args:
        n_bytes: (int) the maximum number of bytes of memory that Python
        can use before raising an exception
    """
    if n_bytes <= 0:
        return

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)

    resource.setrlimit(resource.RLIMIT_AS, (n_bytes, hard))

    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)

    if n_bytes < soft * 1024:
        resource.setrlimit(resource.RLIMIT_DATA, (n_bytes, hard))


def display_top(snapshot, key_type="lineno", limit=10):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def extract_do_list_from_tracking_observation(
    obs: np.ndarray,
) -> List[Tuple[int, np.ndarray, np.ndarray, float, float]]:
    """Extracts the dynamic obstacle list from the TrackingObservation.

    Args:
        obs (np.ndarray): The TrackingObservation

    Returns:
        List[Tuple[int, np.ndarray, np.ndarray, float, float]]: List of dynamic obstacles on the form (ID, state, cov, length, width).
    """
    max_num_do = obs.shape[1]
    do_list = []
    for i in range(max_num_do):
        if np.sum(obs[1:, i]) > 1.0:  # A proper DO entry has non-zeros in its vector
            cov = obs[7:, i].reshape(4, 4)
            do_list.append((int(obs[0, i]), obs[1:5, i], cov, obs[5, i], obs[6, i]))
    return do_list


def compute_distances_to_dynamic_obstacles(
    ownship_state: np.ndarray, do_list: List
) -> List[Tuple[int, float]]:
    os_pos = ownship_state[0:2]
    os_speed = np.linalg.norm(ownship_state[3:5])
    os_course = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])
    distances2do = []
    for idx, (_, do_state, _, _, _) in enumerate(do_list):
        d2do = np.linalg.norm(do_state[0:2] - os_pos)
        bearing_do = (
            np.arctan2(do_state[1] - os_pos[1], do_state[0] - os_pos[0]) - os_course
        )
        if bearing_do > np.pi:
            distances2do.append((idx, 1e10))
        else:
            distances2do.append((idx, d2do))
    distances2do = sorted(distances2do, key=lambda x: x[1])
    return distances2do


def process_rl_training_data(
    data: rlmpc_logger.RLData, ma_window_size: int = 5
) -> rlmpc_logger.RLData:
    """Smooths out training data from the RL process, given the chosen window size.

    Args:
        data (rlmpc_logger.RLData): RLData structure
        ma_window_size (int, optional): Smoothing (moving average) window size in number of episodes.

    Returns:
        rlmpc_logger.RLData: Smoothed RLData structure.
    """
    smoothed_critic_loss, std_critic_loss = compute_smooted_mean_and_std(
        data.critic_loss, window_size=ma_window_size
    )
    smoothed_actor_loss, std_actor_loss = compute_smooted_mean_and_std(
        data.actor_loss, window_size=ma_window_size
    )
    smoothed_mean_actor_grad_norm, std_mean_actor_grad_norm = (
        compute_smooted_mean_and_std(
            data.mean_actor_grad_norm, window_size=ma_window_size
        )
    )
    smoothed_ent_coeff_loss, std_ent_coeff_loss = compute_smooted_mean_and_std(
        data.ent_coeff_loss, window_size=ma_window_size
    )
    smoothed_ent_coeff, std_ent_coeff = compute_smooted_mean_and_std(
        data.ent_coeff, window_size=ma_window_size
    )

    smoothed_non_optimal_solution_rate, std_non_optimal_solution_rate = (
        compute_smooted_mean_and_std(
            data.non_optimal_solution_rate, window_size=ma_window_size
        )
    )
    smoothed_mean_episode_reward, std_mean_episode_reward = (
        compute_smooted_mean_and_std(
            data.mean_episode_reward, window_size=ma_window_size
        )
    )
    smoothed_mean_episode_length, std_mean_episode_length = (
        compute_smooted_mean_and_std(
            data.mean_episode_length, window_size=ma_window_size
        )
    )
    success_rate, std_success_rate = compute_smooted_mean_and_std(
        data.success_rate, window_size=ma_window_size
    )
    actor_expl_std, std_actor_expl_std = compute_smooted_mean_and_std(
        data.actor_expl_std, window_size=ma_window_size
    )

    smoothed_data = rlmpc_logger.SmoothedRLData(
        n_updates=data.n_updates,
        time_elapsed=data.time_elapsed,
        episodes=data.episodes,
        critic_loss=smoothed_critic_loss,
        std_critic_loss=std_critic_loss,
        actor_loss=smoothed_actor_loss,
        std_actor_loss=std_actor_loss,
        mean_actor_grad_norm=smoothed_mean_actor_grad_norm,
        std_mean_actor_grad_norm=std_mean_actor_grad_norm,
        ent_coeff_loss=smoothed_ent_coeff_loss,
        std_ent_coeff_loss=std_ent_coeff_loss,
        ent_coeff=smoothed_ent_coeff,
        std_ent_coeff=std_ent_coeff,
        non_optimal_solution_rate=smoothed_non_optimal_solution_rate,
        std_non_optimal_solution_rate=std_non_optimal_solution_rate,
        mean_episode_reward=smoothed_mean_episode_reward,
        std_mean_episode_reward=std_mean_episode_reward,
        mean_episode_length=smoothed_mean_episode_length,
        std_mean_episode_length=std_mean_episode_length,
        success_rate=success_rate,
        std_success_rate=std_success_rate,
        actor_expl_std=actor_expl_std,
        std_actor_expl_std=std_actor_expl_std,
    )
    return smoothed_data


def exponential_moving_average(data: List[float], alpha: float = 0.9) -> List[float]:
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * ema[-1] + (1 - alpha) * data[i])
    return ema


def compute_smooted_mean_and_std(
    data: List[float], window_size: int = 10
) -> Tuple[List[float], List[float]]:
    mean = np.convolve(data, np.ones((window_size,)) / window_size, mode="valid")
    std = [np.std(mean[max(0, i - window_size + 1) : i + 1]) for i in range(len(mean))]

    return mean, std


def extract_reward_data(
    data: List[colav_logger.EpisodeData],
    ma_window_size: int = 10,
    scale_reward_components: bool = False,
) -> Dict[str, Any]:
    """Extracts reward metrics from the environment data.

    Args:
        data (List[EpisodeData]): List of EpisodeData objects from training.
        ma_window_size (int, optional): Window size for moving average.
        scale_reward_components (bool, optional): Whether to scale reward components.

    Returns:
        Dict[str, Any]: Dictionary containing reward metrics.
    """
    rewards = []
    r_antigrounding = []
    r_colav = []
    r_colreg = []
    r_trajectory_tracking = []
    r_ra_maneuvering = []
    r_action_chatter = []
    r_dnn_pp = []
    ep_lengths = []
    r_scale = 1.0
    for env_idx, env_data in enumerate(data):
        if not (env_idx % 5 == 0):
            continue
        return_colreg_ep = (
            np.sum([r["r_colreg"] for r in env_data.reward_components]) / r_scale
        )
        return_colav_ep = (
            np.sum([r["r_collision_avoidance"] for r in env_data.reward_components])
            / r_scale
        )
        return_antigrounding_ep = (
            np.sum([r["r_antigrounding"] for r in env_data.reward_components]) / r_scale
        )
        return_trajectory_tracking_ep = (
            np.sum([r["r_trajectory_tracking"] for r in env_data.reward_components])
            / r_scale
        )
        return_readily_apparent_maneuvering_ep = (
            np.sum(
                [
                    r["r_readily_apparent_maneuvering"]
                    for r in env_data.reward_components
                ]
            )
            / r_scale
        )
        return_action_chatter = (
            np.sum([r["r_action_chatter"] for r in env_data.reward_components])
            / r_scale
        )
        return_dnn_pp = (
            np.sum([r["r_dnn_parameters"] for r in env_data.reward_components])
            / r_scale
        )

        r_colreg.append(return_colreg_ep)
        r_colav.append(return_colav_ep)
        r_antigrounding.append(return_antigrounding_ep)
        r_trajectory_tracking.append(return_trajectory_tracking_ep)
        r_ra_maneuvering.append(return_readily_apparent_maneuvering_ep)
        r_action_chatter.append(return_action_chatter)
        r_dnn_pp.append(return_dnn_pp)
        ep_lengths.append(env_data.timesteps)
        rewards.append(env_data.cumulative_reward)

    rewards_smoothed, std_rewards_smoothed = compute_smooted_mean_and_std(
        rewards, ma_window_size
    )
    r_colreg, std_r_colreg = compute_smooted_mean_and_std(r_colreg, ma_window_size)
    r_colav, std_r_colav = compute_smooted_mean_and_std(r_colav, ma_window_size)
    r_antigrounding, std_r_antigrounding = compute_smooted_mean_and_std(
        r_antigrounding, ma_window_size
    )
    r_trajectory_tracking, std_r_trajectory_tracking = compute_smooted_mean_and_std(
        r_trajectory_tracking, ma_window_size
    )
    r_ra_maneuvering, std_r_ra_maneuvering = compute_smooted_mean_and_std(
        r_ra_maneuvering, ma_window_size
    )
    r_action_chatter, std_r_action_chatter = compute_smooted_mean_and_std(
        r_action_chatter, ma_window_size
    )
    r_dnn_pp, std_r_dnn_pp = compute_smooted_mean_and_std(r_dnn_pp, ma_window_size)

    ep_lengths_smoothed, std_ep_lengths_smoothed = compute_smooted_mean_and_std(
        ep_lengths, ma_window_size
    )

    out = {
        "rewards": rewards,
        "rewards_smoothed": rewards_smoothed,
        "std_rewards_smoothed": std_rewards_smoothed,
        "r_colreg": r_colreg,
        "std_r_colreg": std_r_colreg,
        "r_colav": r_colav,
        "std_r_colav": std_r_colav,
        "r_antigrounding": r_antigrounding,
        "std_r_antigrounding": std_r_antigrounding,
        "r_trajectory_tracking": r_trajectory_tracking,
        "std_r_trajectory_tracking": std_r_trajectory_tracking,
        "r_ra_maneuvering": r_ra_maneuvering,
        "std_r_ra_maneuvering": std_r_ra_maneuvering,
        "r_action_chatter": r_action_chatter,
        "std_r_action_chatter": std_r_action_chatter,
        "r_dnn_pp": r_dnn_pp,
        "std_r_dnn_pp": std_r_dnn_pp,
        "ep_lengths": ep_lengths,
        "ep_lengths_smoothed": ep_lengths_smoothed,
        "std_ep_lengths_smoothed": std_ep_lengths_smoothed,
    }
    return out


def make_grid_for_tensorboard(
    batch_images, reconstructed_images, semantic_masks, n_rows: int = 2
) -> torch.Tensor:
    """Create grid of images to show in tensorboard.

    Args:
        batch_images (torch.Tensor): The batch of input images
        reconstructed_images (torch.Tensor): The batch of reconstructed images
        semantic_masks (torch.Tensor): The batch of semantic masks
        n_rows (int, optional): The number of rows in the grid. Defaults to 2.

    Returns:
        torch.Tensor: The grid of images
    """
    joined_images = []
    batch_size, n_channels, _, _ = batch_images.shape
    for j in range(batch_size):
        for i in reversed(range(n_channels)):
            if batch_images[j, i, :, :].dim() > 2:
                print("wrong")
            joined_images.append(batch_images[j, i, :, :].unsqueeze(0))
            joined_images.append(semantic_masks[j, i, :, :].unsqueeze(0))
            joined_images.append(reconstructed_images[j, i, :, :].unsqueeze(0))
    return torchvision.utils.make_grid(joined_images, nrow=n_rows, padding=2)


def create_los_based_trajectory(
    xs: np.ndarray,
    waypoints: np.ndarray,
    speed_plan: np.ndarray,
    los: guidances.LOSGuidance,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a trajectory based on the provided LOS guidance, controller and model.

    Args:
        - xs (np.ndarray): State vector
        - waypoints (np.ndarray): Waypoints
        - speed_plan (np.ndarray): Speed plan
        - los (guidances.LOSGuidance): LOS guidance object
        - dt (float): Time step

    Returns:
        np.ndarray: Trajectory
    """
    model = sim_models.Telemetron()
    controller = controllers.FLSC(model.params)
    trajectory = []
    inputs = []
    xs_k = xs
    t = 0.0
    reached_goal = False
    t_braking = 30.0
    t_brake_start = 0.0
    while t < 2000.0:
        trajectory.append(xs_k)
        references = los.compute_references(waypoints, speed_plan, None, xs_k, dt)
        if reached_goal:
            references[3:] = np.tile(0.0, (references[3:].size, 1))
        u = controller.compute_inputs(references, xs_k, dt)
        inputs.append(u)
        w = None
        xs_k = sim_integrators.erk4_integration_step(
            model.dynamics, model.bounds, xs_k, u, w, dt
        )

        dist2goal = np.linalg.norm(xs_k[0:2] - waypoints[:, -1])
        t += dt
        if dist2goal < 70.0 and not reached_goal:
            reached_goal = True
            t_brake_start = t

        if reached_goal and t - t_brake_start > t_braking:
            break

    return np.array(trajectory).T, np.array(inputs)[:, :2].T


def interpolate_solution(
    trajectory: np.ndarray,
    inputs: np.ndarray,
    slacks: Optional[np.ndarray],
    dt_sim: float,
    T_mpc: float,
    dt_mpc: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolates the solution from the MPC to the time step in the simulation.

    Args:
        - trajectory (np.ndarray): The solution state trajectory.
        - inputs (np.ndarray): The solution input trajectory.
        - slacks (Optional[np.ndarray]): The solution slack variable trajectory.
        - dt_sim (float): The simulation time step.
        - T_mpc (float): The MPC horizon.
        - dt_mpc (float): The MPC time step.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The interpolated solution state trajectory, input trajectory and slack variable trajectory.
    """
    intp_trajectory = trajectory
    intp_inputs = inputs
    intp_slacks = slacks
    if dt_mpc > dt_sim:
        nx = trajectory.shape[0]
        nu = inputs.shape[0]
        sim_times = np.arange(0.0, T_mpc + dt_mpc, dt_sim)
        mpc_times = np.arange(0.0, T_mpc + dt_mpc, dt_mpc)
        n_samples = len(sim_times)
        intp_trajectory = np.zeros((nx, n_samples))
        intp_inputs = np.zeros((nu, n_samples - 1))
        for dim in range(nx):
            intp_trajectory[dim, :] = interp1d(
                mpc_times, trajectory[dim, :], kind="linear", fill_value="extrapolate"
            )(sim_times)
        for dim in range(nu):
            intp_inputs[dim, :] = interp1d(
                mpc_times[:-1], inputs[dim, :], kind="linear", fill_value="extrapolate"
            )(sim_times[:-1])
    return intp_trajectory, intp_inputs, intp_slacks


def shift_nominal_plan(
    nominal_trajectory: np.ndarray,
    nominal_inputs: np.ndarray,
    ownship_state: np.ndarray,
    N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Updates the nominal trajectory and inputs to the MPC based on the current ownship state. This is done by
    find closest point on nominal trajectory to the current state and then shifting the nominal trajectory to this point

    Args:
        - nominal_trajectory (np.ndarray): The nominal trajectory.
        - nominal_inputs (np.ndarray): The nominal inputs.
        - ownship_state (np.ndarray): The ownship state.
        - N (int): MPC horizon length in samples

    Returns:
        Tuple[np.ndarray, np.ndarray]: The shifted nominal trajectory and inputs.
    """
    nx = ownship_state.size
    nu = nominal_inputs.shape[0]
    closest_idx = int(
        np.argmin(
            np.linalg.norm(
                nominal_trajectory[:2, :]
                - np.tile(ownship_state[:2], (len(nominal_trajectory[0, :]), 1)).T,
                axis=0,
            )
        )
    )
    shifted_nominal_trajectory = nominal_trajectory[:, closest_idx:]
    shifted_nominal_inputs = nominal_inputs[:, closest_idx:]
    n_samples = shifted_nominal_trajectory.shape[1]
    if n_samples == 0:  # Done with following nominal trajectory, stop
        shifted_nominal_trajectory = np.tile(
            np.array(
                [ownship_state[0], ownship_state[1], ownship_state[2], 0.0, 0.0, 0.0]
            ),
            (N + 1, 1),
        ).T
        shifted_nominal_inputs = np.zeros((nu, N))
    elif n_samples < N + 1:
        shifted_nominal_trajectory = np.zeros((nx, N + 1))
        shifted_nominal_trajectory[:, :n_samples] = nominal_trajectory[
            :, closest_idx : closest_idx + n_samples
        ]
        shifted_nominal_trajectory[:, n_samples:] = np.tile(
            nominal_trajectory[:, closest_idx + n_samples - 1], (N + 1 - n_samples, 1)
        ).T
        shifted_nominal_inputs = np.zeros((nu, N))
        shifted_nominal_inputs[:, : n_samples - 1] = nominal_inputs[
            :, closest_idx : closest_idx + n_samples - 1
        ]
        shifted_nominal_inputs[:, n_samples - 1 :] = np.tile(
            nominal_inputs[:, closest_idx + n_samples - 2], (N - n_samples + 1, 1)
        ).T
    else:
        shifted_nominal_trajectory = shifted_nominal_trajectory[:, : N + 1]
        shifted_nominal_inputs = shifted_nominal_inputs[:, :N]
    return shifted_nominal_trajectory, shifted_nominal_inputs


def decision_trajectories_from_solution(
    soln: np.ndarray, N: int, nu: int, nx: int, ns: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the input sequence U, state sequence X and the slack variable sequence S from the solution vector soln = w = [U.flattened, X.flattened, Sigma.flattened] from the optimization problem.

    Args:
        soln (np.ndarray): A solution vector from the optimization problem.
        N (int): The prediction horizon.
        nu (int): The input dimension
        nx (int): The state dimension
        ns (int): The slack variable dimension

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The input sequence U, state sequence X and the slack variable sequence S.
    """
    U = np.zeros((nu, N))
    X = np.zeros((nx, N + 1))
    Sigma = np.zeros((ns, N + 1))
    for k in range(N + 1):
        if k < N:
            U[:, k] = soln[k * nu : (k + 1) * nu].ravel()
        X[:, k] = soln[N * nu + k * nx : N * nu + (k + 1) * nx].ravel()
        Sigma[:, k] = soln[
            N * nu + (N + 1) * nx + k * ns : N * nu + (N + 1) * nx + (k + 1) * ns
        ].ravel()
    return U, X, Sigma


def linestring_to_ndarray(line: geometry.LineString) -> np.ndarray:
    """Converts a shapely LineString to a numpy array

    Args:
        - line (LineString): Any LineString object

    Returns:
        np.ndarray: Numpy array containing the coordinates of the LineString
    """
    return np.array(line.coords).transpose()


def ndarray_to_linestring(array: np.ndarray) -> geometry.LineString:
    """Converts a 2D numpy array to a shapely LineString

    Args:
        - array (np.ndarray): Numpy array of 2 x n_samples, containing the coordinates of the LineString

    Returns:
        LineString: Any LineString object
    """
    assert array.shape[0] == 2 and array.shape[1] > 1, (
        "Array must be 2 x n_samples with n_samples > 1"
    )
    return geometry.LineString(list(zip(array[0, :], array[1, :])))


def casadi_potential_field_base_function(x: csd.MX) -> csd.MX:
    """Potential field base function f(x) = x / sqrt(x^2 + 1)

    Args:
        x (csd.MX): Input

    Returns:
        csd.MX: Output f(x)
    """
    return x / csd.sqrt(x**2 + 1)


def casadi_matrix_from_nested_list(M: list) -> csd.MX:
    """Convenience function for making a casadi matrix from lists of lists
    (don't know why this doesn't exist already), the alternative is

    Args:
        M (list): List of lists

    Returns:
        csd.MX: Casadi matrix
    """
    return csd.vertcat(*(csd.horzcat(*row) for row in M))


def casadi_diagonal_matrix_from_vector(v: csd.MX) -> csd.MX:
    """Creates a diagonal matrix from a vector.

    Args:
        v (csd.MX): Vector symbolic representing diagonal entries
    """
    n = v.shape[0]
    llist = []
    for i in range(n):
        nested_list = []
        for j in range(n):
            if i == j:
                nested_list.append(v[i])
            else:
                nested_list.append(0.0)
        llist.append(nested_list)
    return casadi_matrix_from_nested_list(llist)


def casadi_matrix_from_vector(v: csd.MX, n_rows: int, n_cols: int) -> csd.MX:
    """Creates a matrix from a vector.

    Args:
        v (csd.MX): Vector symbolic representing matrix entries
        n_rows (int): Rows in matrix
        n_cols (int): Columns in matrix

    Returns:
        csd.MX: Casadi matrix
    """
    llist = []
    for i in range(n_rows):
        nested_list = []
        for j in range(n_cols):
            nested_list.append(v[i * n_rows + j])
        llist.append(nested_list)
    return casadi_matrix_from_nested_list(llist)


def load_rrt_solution(save_file: pathlib.Path = dp.rrt_solution) -> dict:
    return fu.read_yaml_into_dict(save_file)


def save_rrt_solution(
    rrt_solution: dict, save_file: pathlib.Path = dp.rrt_solution
) -> None:
    save_file.touch(exist_ok=True)
    with save_file.open(mode="w") as file:
        yaml.dump(rrt_solution, file)


def translate_dynamic_obstacle_coordinates(
    dynamic_obstacles: list, x_shift: float, y_shift: float
) -> list:
    """Translates the coordinates of a list of dynamic obstacles by (-y_shift, -x_shift)

    Args:
        dynamic_obstacles (list): List of dynamic obstacle objects on the form (ID, state, cov, length, width)
        x_shift (float): Easting shift
        y_shift (float): Northing shift

    Returns:
        list: List of dynamic obstacles with shifted coordinates
    """
    translated_dynamic_obstacles = []
    for ID, state, cov, length, width in dynamic_obstacles:
        translated_state = state - np.array([y_shift, x_shift, 0.0, 0.0])
        translated_dynamic_obstacles.append((ID, translated_state, cov, length, width))
    return translated_dynamic_obstacles


def translate_polygons(polygons: list, x_shift: float, y_shift: float) -> list:
    """Shifts the coordinates of a list of polygons by (-x_shift, -y_shift)

    Args:
        polygons (list): List of shapely polygons
        x_shift (float): Shift easting
        y_shift (float): Shift northing

    Returns:
        list: List of shifted polygons
    """
    translated_polygons = []
    for polygon in polygons:
        translated_polygon = affinity.translate(polygon, xoff=-x_shift, yoff=-y_shift)
        translated_polygons.append(translated_polygon)
    return translated_polygons


def create_ellipse(
    center: np.ndarray,
    A: Optional[np.ndarray] = None,
    a: float | None = 1.0,
    b: float | None = 1.0,
    phi: float | None = 0.0,
) -> Tuple[list, list]:
    """Create standard ellipse at center, with input semi-major axis, semi-minor axis and angle.

    Either specified by c, A or c, a, b, phi:

    (p - c)^T A (p - c) = 1

    or

    (p - c)^T R^T D R (p - c) = 1

    with R = R(phi) and D = diag(1 / a^2, 1 / b^2)


    Args:
        - center (np.ndarray): Center of ellipse
        - A (Optional[np.ndarray], optional): Hessian matrix. Defaults to None.
        - a (float | None, optional): Semi-major axis. Defaults to 1.0.
        - b (float | None, optional): Semi-minor axis. Defaults to 1.0.
        - phi (float | None, optional): Angle. Defaults to 0.0.


    Returns:
        Tuple[list, list]: List of x and y coordinates
    """

    if A is not None:
        # eigenvalues and eigenvectors of the covariance matrix
        eigenval, eigenvec = np.linalg.eig(A[0:2, 0:2])

        largest_eigenval = max(eigenval)
        largest_eigenvec_idx = np.argwhere(eigenval == max(eigenval))[0][0]
        largest_eigenvec = eigenvec[:, largest_eigenvec_idx]

        smallest_eigenval = min(eigenval)
        # if largest_eigenvec_idx == 0:
        #     smallest_eigenvec = eigenvec[:, 1]
        # else:
        #     smallest_eigenvec = eigenvec[:, 0]

        angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
        angle = mf.wrap_angle_to_02pi(angle)

        a = np.sqrt(largest_eigenval)
        b = np.sqrt(smallest_eigenval)
    else:
        angle = phi

    # the ellipse in "body" x and y coordinates
    t = np.linspace(0, 2.01 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)

    R = mf.Rpsi2D(angle)

    # Rotate to NED by angle phi, N_ell_points x 2
    ellipse_xy = np.array([x, y])
    for i in range(ellipse_xy.shape[1]):
        ellipse_xy[:, i] = R @ ellipse_xy[:, i] + center

    return ellipse_xy[0, :].tolist(), ellipse_xy[1, :].tolist()


def create_probability_ellipse(
    P: np.ndarray, probability: float = 0.99
) -> Tuple[list, list]:
    """Creates a probability ellipse for a covariance matrix P and a given
    confidence level (default 0.99).

    Args:
        P (np.ndarray): Covariance matrix
        probability (float, optional): Confidence level. Defaults to 0.99.

    Returns:
        np.ndarray: Ellipse data in x and y coordinates
    """

    # eigenvalues and eigenvectors of the covariance matrix
    eigenval, eigenvec = np.linalg.eig(P[0:2, 0:2])

    largest_eigenval = max(eigenval)
    largest_eigenvec_idx = np.argwhere(eigenval == max(eigenval))[0][0]
    largest_eigenvec = eigenvec[:, largest_eigenvec_idx]

    smallest_eigenval = min(eigenval)
    # if largest_eigenvec_idx == 0:
    #     smallest_eigenvec = eigenvec[:, 1]
    # else:
    #     smallest_eigenvec = eigenvec[:, 0]

    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
    angle = mf.wrap_angle_to_02pi(angle)

    # Get the ellipse scaling factor based on the confidence level
    chisquare_val = chi2.ppf(q=probability, df=2)

    a = chisquare_val * np.sqrt(largest_eigenval)
    b = chisquare_val * np.sqrt(smallest_eigenval)

    # the ellipse in "body" x and y coordinates
    t = np.linspace(0, 2.01 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)

    R = mf.Rpsi2D(angle)

    # Rotate to NED by angle phi, N_ell_points x 2
    ellipse_xy = np.array([x, y])
    for i in range(len(ellipse_xy)):
        ellipse_xy[:, i] = R @ ellipse_xy[:, i]

    return ellipse_xy[0, :].tolist(), ellipse_xy[1, :].tolist()


def plot_surface_approximation_stuff(
    radial_basis_function: csd.Function,
    radial_basis_function_gradient: csd.Function,
    surface_data_points: Tuple[list, list],
    surface_data_point_mask: list,
    surface_data_points_before_buffering: Tuple[list, list],
    original_polygon: geometry.Polygon,
    polygon: geometry.Polygon,
    polygon_safety: geometry.Polygon,
    polygon_index: int,
    relevant_coastline_safety: geometry.Polygon,
    d_safe: float,
    map_origin: np.ndarray,
    enc: senc.ENC,
) -> Tuple[list, list]:
    """Plots the surface approximation stuff. This is used for debugging purposes, not optimized for readability.

    Args:
        radial_basis_function (csd.Function): Radial basis function
        radial_basis_function_gradient (csd.Function): Gradient of radial basis function
        surface_data_points (Tuple[list, list]): Data points used for interpolation
        surface_data_point_mask (list): Mask of data points used for interpolation
        surface_data_points_before_buffering (Tuple[list, list]): Data points used for interpolation before buffering
        original_polygon (geometry.Polygon): Original polygon
        polygon (geometry.Polygon): Polygon clipped inside trajectory envelope
        polygon_safety (geometry.Polygon): Polygon clipped inside trajectory envelope and safety buffered
        polygon_index (int): Index of polygon
        relevant_coastline_safety (geometry.Polygon): Relevant coastline of the clipped safety polygon
        d_safe (float): Safety distance
        map_origin (np.ndarray): Map origin in NE
        enc (senc.ENC): ENC object

    Returns:
        Tuple[list, list]: List of figures and axes
    """
    x_surface, y_surface = surface_data_points
    cap_style = 2
    join_style = 2
    ax1 = plt.figure().add_subplot(111, projection="3d")
    # ax2 = plt.figure().add_subplot(111, projection="3d")
    # ax3 = plt.figure().add_subplot(111, projection="3d")
    # ax3 = plt.figure().add_subplot(111)
    # ax5 = plt.figure().add_subplot(111, projection="3d")
    poly_min_east, poly_min_north, poly_max_east, poly_max_north = polygon.buffer(
        d_safe + 10.0, cap_style=cap_style, join_style=join_style
    ).bounds

    (
        coastline_min_east,
        coastline_min_north,
        coastline_max_east,
        coastline_max_north,
    ) = relevant_coastline_safety.bounds
    # if polygon_index == 1:
    #     translated_polygon = translate_polygons([polygon], -map_origin[1], -map_origin[0])[0]
    #     enc.draw_polygon(
    #         translated_polygon.buffer(d_safe, cap_style=cap_style, join_style=join_style), color="black", fill=False
    #     )
    #    save_path = dp.figures
    #     enc.save_image(name="enc_island_polygon", path=save_path, extension="pdf")
    #     enc.save_image(name="enc_island_polygon", path=save_path, scale=2.0)

    if polygon_index == 8:
        polygon_diff = ops.split(
            relevant_coastline_safety.buffer(
                10.0, cap_style=cap_style, join_style=join_style
            ),
            geometry.LineString(original_polygon.exterior.coords),
        )
        geom = polygon_diff.geoms[1]
        translated_geom = translate_polygons([geom], -map_origin[1], -map_origin[0])[0]
        enc.draw_polygon(translated_geom, color="black", fill=False)

    y_poly_safety, x_poly_safety = polygon_safety.exterior.coords.xy

    # Compute error approximation
    compute_err_approx = False
    if compute_err_approx:
        n_points = 200
        grid_resolution_y = 0.5
        grid_resolution_x = 0.5
        buffer = 5.0
        npy = int((poly_max_east + 2 * buffer - poly_min_east) / grid_resolution_y)
        npx = int((poly_max_north + 2 * buffer - poly_min_north) / grid_resolution_x)
        north_coords = np.linspace(
            start=poly_min_north - buffer, stop=poly_max_north + buffer, num=npx
        )
        east_coords = np.linspace(
            start=poly_min_east - buffer, stop=poly_max_east + buffer, num=npy
        )

        Y, X = np.meshgrid(east_coords, north_coords, indexing="ij")
        map_coords = np.hstack((Y.reshape(-1, 1), X.reshape(-1, 1)))

        poly_path = mpath.Path(np.array([y_poly_safety, x_poly_safety]).T)
        mask = poly_path.contains_points(points=map_coords, radius=0.00001)
        mask = mask.astype(float).reshape((npy, npx))
        mask[mask > 0.0] = 1.0

        epsilon = 1e-3
        dist_surface_points = np.zeros((npy, npx))
        diff_surface_points = np.zeros((npy, npx))

        for i, east_coord in enumerate(east_coords):
            if polygon_index == 0 and east_coord < coastline_min_east:
                continue
            for ii, north_coord in enumerate(north_coords):
                if polygon_index == 0 and north_coord < coastline_min_north:
                    continue

                if (
                    polygon_index == 0
                    and north_coord < coastline_min_north + 200.0
                    and east_coord < coastline_min_east + 200.0
                ):
                    continue

                if (
                    polygon_index == 0
                    and north_coord < coastline_min_north + 20.0
                    and east_coord > coastline_max_east - 60.0
                ):
                    continue

                if polygon_index == 8 and not geometry.Point(
                    east_coord, north_coord
                ).within(geom):
                    continue

                if polygon_index == 8 and north_coord < 215.0 and east_coord < 324.8:
                    continue

                if polygon_index == 8 and north_coord < -12.0 and east_coord > 1257.0:
                    continue

                if (
                    mask[i, ii] > 0.0
                    and radial_basis_function(
                        np.array([north_coord, east_coord]).reshape((1, 2))
                    )
                    <= 0.0 + epsilon
                ) or (
                    mask[i, ii] <= 0.0
                    and radial_basis_function(
                        np.array([north_coord, east_coord]).reshape((1, 2))
                    )
                    > 0.0 + epsilon
                ):
                    # if mask[i, ii] - radial_basis_function(np.array([north_coord, east_coord]).reshape((1, 2))) > 0.0:
                    #    print("Error: ", mask[i, ii] - radial_basis_function(np.array([north_coord, east_coord]).reshape((1, 2))))
                    d2poly = polygon_safety.distance(
                        geometry.Point(east_coord, north_coord)
                    )
                    dist_surface_points[i, ii] = d2poly
                    diff_surface_points[i, ii] = radial_basis_function(
                        np.array([north_coord, east_coord]).reshape((1, 2))
                    )
        print(
            "polygon_index = {polygon_index} |Max distance of error: ",
            np.max(dist_surface_points),
        )

        n_points = len(x_surface)
        actual_dataset_error = np.zeros(n_points)
        for i, (north_coord, east_coord) in enumerate(zip(x_surface, y_surface)):
            point = np.array([north_coord + 0.000001, east_coord + 0.000001]).reshape(
                1, 2
            )
            actual_dataset_error[i] = (
                surface_data_point_mask[i] - radial_basis_function(point).full()
            )
        mean_error = np.mean(dist_surface_points)
        max_error = np.max(dist_surface_points)
        idx_max_error = np.argmax(actual_dataset_error)
        std_error = np.std(dist_surface_points)
        print(
            f"polygon_index = {polygon_index} | Num interpolation data points: {len(x_surface)} | Num original poly points: {len(x_poly_safety)}"
        )
        print(
            f"Dataset: Mean 0point crossing error: {mean_error}, Max, idx max error: ({max_error}, {idx_max_error}), Std error: {std_error}"
        )

        Y, X = np.meshgrid(
            east_coords + map_origin[1], north_coords + map_origin[0], indexing="ij"
        )
        # Y, X = np.meshgrid(east_coords, north_coords, indexing="ij")
        # ax5.plot_surface(Y, X, dist_surface_points, rcount=100, ccount=100, cmap=cm.coolwarm)
        # # ax5.contourf(Y, X, mask.T, zdir="z", offset=50.0, cmap=cm.coolwarm)
        # ax5.set_xlabel("East [m]")
        # ax5.set_ylabel("North [m]")
        # ax5.set_zlabel("Distance [m]")

        y_surface_orig, x_surface_orig = surface_data_points_before_buffering
        fig6, ax6 = plt.subplots()
        pc6 = ax6.pcolormesh(
            Y, X, dist_surface_points, shading="gouraud", rasterized=True
        )
        ax6.plot(y_surface_orig + map_origin[1], x_surface_orig + map_origin[0], "k")
        # ax6.plot(y_surface_orig, x_surface_orig, "k")
        cbar6 = fig6.colorbar(pc6)
        cbar6.set_label("Distance [m]")
        ax6.set_xlabel("East [m]")
        ax6.set_ylabel("North [m]")

    buffer = 200.0
    n_points = 100
    extra_north_coords = np.linspace(
        start=poly_min_north - buffer, stop=poly_max_north + buffer, num=n_points
    )
    extra_east_coords = np.linspace(
        start=poly_min_east - buffer, stop=poly_max_east + buffer, num=n_points
    )

    surface_points = np.zeros((n_points, n_points))
    surface_grad_points = np.zeros((n_points, n_points, 2))
    for i, east_coord in enumerate(extra_east_coords):
        for ii, north_coord in enumerate(extra_north_coords):
            point = np.array([north_coord, east_coord]).reshape(1, 2)
            surface_points[i, ii] = radial_basis_function(point).full()
            surface_grad_points[i, ii, :] = (
                radial_basis_function_gradient(point).full().flatten()
            )
    yY, xX = np.meshgrid(
        extra_east_coords + map_origin[1],
        extra_north_coords + map_origin[0],
        indexing="ij",
    )

    print(f"Number of gradient NaNs: {np.count_nonzero(np.isnan(surface_grad_points))}")

    fig1 = ax1.figure
    ax1.plot_surface(yY, xX, surface_points, cmap=cm.coolwarm)
    ax1.set_ylabel("North [m]")
    ax1.set_xlabel("East [m]")
    # ax1.set_zlabel(r"$h_j(\bm{\zeta})$")
    # fig1.savefig("surface_approx.pdf", bbox_inches="tight", dpi=50)

    fig2, ax2 = plt.subplots()
    ax2.pcolormesh(yY, xX, surface_points, shading="gouraud")
    p = ax2.scatter(
        y_surface + map_origin[1],
        x_surface + map_origin[0],
        c=np.array(surface_data_point_mask),
        ec="k",
    )
    cbar4 = fig2.colorbar(p)
    # cbar4.set_label(r"$h_j(\bm{\zeta})$")
    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")
    # fig2.savefig("colormesh_island_polygon.pdf", bbox_inches="tight", dpi=50)

    # ax2.plot_surface(yY, xX, surface_grad_points[:, :, 0], rcount=200, ccount=200, cmap=cm.coolwarm)
    # ax2.set_xlabel("East")
    # ax2.set_ylabel("North")
    # ax2.set_zlabel("Mask")
    # ax2.set_title("Spline surface gradient x")

    # ax3.plot_surface(yY, xX, surface_grad_points[:, :, 1], rcount=200, ccount=200, cmap=cm.coolwarm)
    # ax3.set_xlabel("East")
    # ax3.set_ylabel("North")
    # ax3.set_zlabel("Mask")
    # ax3.set_title("Spline surface gradient y")
    plt.show(block=False)
    # ax1.clear()
    # ax2.clear()
    # ax3.clear()
    # ax5.clear()

    return [fig1, fig2], [ax1, ax2]


def create_arc_length_spline(x: list, y: list) -> Tuple[interp1d, interp1d, list]:
    """Creates a spline for the arc length of the input x and y coordinates.

    Args:
        - x (list): List of x coordinates.
        - y (list): List of y coordinates.

    Returns:
        Tuple[interp1d, interp1d, list]: Tuple of arc length splines for x and y coordinates.
    """
    # Interpolate the data to get more points => higher accuracy in the arc length spline
    n_points = len(x)
    y_interp = interp1d(np.arange(n_points), y, kind="linear")
    x_interp = interp1d(np.arange(n_points), x, kind="linear")

    n_expanded_points = 500
    y_expanded = list(y_interp(np.linspace(0, n_points - 1, n_expanded_points)))
    x_expanded = list(x_interp(np.linspace(0, n_points - 1, n_expanded_points)))
    arc_length = [0.0]
    for i in range(1, n_expanded_points):
        pi = np.array([x_expanded[i - 1], y_expanded[i - 1]])
        pj = np.array([x_expanded[i], y_expanded[i]])
        arc_length.append(np.linalg.norm(pi - pj))
    arc_length = np.cumsum(arc_length)
    y_interp_arc_length = interp1d(arc_length, y_expanded, kind="linear")
    x_interp_arc_length = interp1d(arc_length, x_expanded, kind="linear")
    return x_interp_arc_length, y_interp_arc_length, arc_length
