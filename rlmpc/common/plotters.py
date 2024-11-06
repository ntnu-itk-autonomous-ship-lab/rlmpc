"""
    plotters.py

    Summary:
        Contains functions for plotting data from the RL training process.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import colav_simulator.common.image_helper_methods as ihm
import colav_simulator.gym.logger as csenv_logger
import colav_simulator.gym.logger as csgym_logger
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.helper_functions as hf
import rlmpc.common.logger as rlmpc_logger
import rlmpc.common.logger as rl_logger
import seaborn as sns
from matplotlib import gridspec

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
matplotlib.use("TkAgg")
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
        "font.weight": "bold",
        "font.size": SMALL_SIZE,
        "axes.titlesize": SMALL_SIZE,
        "axes.labelsize": MEDIUM_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
    }
)


def plot_single_model_enc_snapshots(
    data: List[csgym_logger.EpisodeData],
    nrows: int = 5,
    ncols: int = 3,
    name: str = None,
    save_path: Path = None,
    save_fig: bool = False,
) -> None:
    if name is None:
        name = "model"

    fig = plt.figure(figsize=(10, 10), num=f"enc_snapshots_{name}")
    fig.tight_layout()
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        fig,
        wspace=0,
        hspace=0.1,
        top=0.95,
        bottom=0.05,
    )
    n_episodes = len(data)
    indices = np.linspace(0, n_episodes - 1, nrows * ncols).astype(int)
    for i, idx in enumerate(indices):
        ax = plt.subplot(gs[i])
        episode = data[idx]
        if episode.frames.size == 0:
            continue
        ep_len = len(episode.frames)
        frame = episode.frames[int(3 * ep_len / 4)]
        # upscale the image
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
        frame = ihm.remove_whitespace(frame)
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(f"Episode {idx}")
    if save_fig:
        save_path = save_path if save_path is not None else Path("./figures")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        plt.savefig(save_path / f"enc_snapshots_{name}.pdf", bbox_inches="tight", dpi=100)
    plt.show(block=False)


def plot_multiple_model_reward_curves(
    model_data: List[Dict[str, Any]], model_names: List[str], save_path: Path, save_fig: bool = False
) -> None:
    fig, axs = plt.subplots(4, 2, figsize=(15, 15), num="training_reward_curves")
    fig.subplots_adjust(hspace=0.4, wspace=0.25)
    colors = sns.color_palette("tab10", n_colors=len(model_data))
    for name, data in zip(model_names, model_data):
        color = colors.pop()
        plot_single_model_reward_curves(axs, data, name, color=color)
    if save_fig:
        save_path = save_path if save_path is not None else Path("./")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        plt.savefig(save_path / "training_reward_curves.pdf", bbox_inches="tight", dpi=100)


def plot_single_model_reward_curves(
    axs: List[plt.Axes], data: Dict[str, Any], model_name: str, color: str = None
) -> None:
    # Plot reward components
    sns.lineplot(
        x=range(len(data["rewards_smoothed"])), y=data["rewards_smoothed"], ax=axs[0, 0], label=model_name, color=color
    )
    axs[0, 0].fill_between(
        range(len(data["rewards_smoothed"])),
        np.array(data["rewards_smoothed"]) - np.array(data["std_rewards_smoothed"]),
        np.array(data["rewards_smoothed"]) + np.array(data["std_rewards_smoothed"]),
        alpha=0.2,
        color=color,
    )
    axs[0, 0].set_title("Total return")
    axs[0, 0].set_ylabel("Avg. return", rotation=90)
    # axs[0, 0].set_xlabel("Episode")

    axs[0, 1].plot(
        range(len(data["ep_lengths_smoothed"])),
        200.0 * np.ones_like(data["ep_lengths_smoothed"]),
        label="Truncation limit",
        color="r",
        linestyle="--",
    )
    sns.lineplot(
        x=range(len(data["ep_lengths_smoothed"])),
        y=data["ep_lengths_smoothed"],
        ax=axs[0, 1],
        label=model_name,
        color=color,
    )
    axs[0, 1].fill_between(
        range(len(data["ep_lengths_smoothed"])),
        np.array(data["ep_lengths_smoothed"]) - np.array(data["std_ep_lengths_smoothed"]),
        np.array(data["ep_lengths_smoothed"]) + np.array(data["std_ep_lengths_smoothed"]),
        alpha=0.2,
        color=color,
    )
    axs[0, 1].set_title("Episode length")
    axs[0, 1].set_ylabel("Avg. length", rotation=90)
    # axs[0, 1].set_xlabel("Episode")

    sns.lineplot(x=range(len(data["r_colreg"])), y=data["r_colreg"], ax=axs[1, 0], label=model_name, color=color)
    axs[1, 0].fill_between(
        range(len(data["r_colreg"])),
        np.array(data["r_colreg"]) - np.array(data["std_r_colreg"]),
        np.array(data["r_colreg"]) + np.array(data["std_r_colreg"]),
        alpha=0.2,
        color=color,
    )
    axs[1, 0].set_title("COLREG")
    # axs[0, 1].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Avg. return", rotation=90)

    sns.lineplot(x=range(len(data["r_colav"])), y=data["r_colav"], ax=axs[1, 1], label=model_name, color=color)
    axs[1, 1].fill_between(
        range(len(data["r_colav"])),
        np.array(data["r_colav"]) - np.array(data["std_r_colav"]),
        np.array(data["r_colav"]) + np.array(data["std_r_colav"]),
        alpha=0.2,
        color=color,
    )
    # axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_title("Collision avoidance")

    sns.lineplot(
        x=range(len(data["r_antigrounding"])),
        y=data["r_antigrounding"],
        ax=axs[2, 0],
        label=model_name,
        color=color,
    )
    axs[2, 0].fill_between(
        range(len(data["r_antigrounding"])),
        np.array(data["r_antigrounding"]) - np.array(data["std_r_antigrounding"]),
        np.array(data["r_antigrounding"]) + np.array(data["std_r_antigrounding"]),
        alpha=0.2,
        color=color,
    )
    axs[2, 0].set_title("Anti-grounding")
    axs[2, 0].set_ylabel("Avg. return", rotation=90)
    # axs[2, 0].set_xlabel("Episode")

    sns.lineplot(
        x=range(len(data["r_trajectory_tracking"])),
        y=data["r_trajectory_tracking"],
        ax=axs[2, 1],
        label=model_name,
        color=color,
    )
    axs[2, 1].fill_between(
        range(len(data["r_trajectory_tracking"])),
        np.array(data["r_trajectory_tracking"]) - np.array(data["std_r_trajectory_tracking"]),
        np.array(data["r_trajectory_tracking"]) + np.array(data["std_r_trajectory_tracking"]),
        alpha=0.2,
        color=color,
    )
    axs[2, 1].set_title("Trajectory tracking")
    # axs[2, 1].set_xlabel("Episode")

    sns.lineplot(
        x=range(len(data["r_ra_maneuvering"])),
        y=data["r_ra_maneuvering"],
        ax=axs[3, 0],
        label=model_name,
        color=color,
    )
    axs[3, 0].fill_between(
        range(len(data["r_ra_maneuvering"])),
        np.array(data["r_ra_maneuvering"]) - np.array(data["std_r_ra_maneuvering"]),
        np.array(data["r_ra_maneuvering"]) + np.array(data["std_r_ra_maneuvering"]),
        alpha=0.2,
        color=color,
    )
    axs[3, 0].set_title("RA maneuvering")
    axs[3, 0].set_xlabel("Episode")
    axs[3, 0].set_ylabel("Avg. return", rotation=90)

    sns.lineplot(
        x=range(len(data["r_dnn_pp"])),
        y=data["r_dnn_pp"],
        ax=axs[3, 1],
        label=model_name,
        color=color,
    )
    axs[3, 1].fill_between(
        range(len(data["r_dnn_pp"])),
        np.array(data["r_dnn_pp"]) - np.array(data["std_r_dnn_pp"]),
        np.array(data["r_dnn_pp"]) + np.array(data["std_r_dnn_pp"]),
        alpha=0.2,
        color=color,
    )
    axs[3, 1].set_title("Parameters")
    axs[3, 1].set_xlabel("Episode")
    plt.show(block=False)


def plot_multiple_model_training_stats(
    model_data: List[rl_logger.RLData], model_names: List[str], save_fig: bool = False, save_path: Path = None
) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), num="training_stats")
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    colors = sns.color_palette("tab10", n_colors=len(model_data))
    for name, data in zip(model_names, model_data):
        color = colors.pop()
        plot_single_model_training_stats(axs, data, name, color=color)
    if save_fig:
        save_path = save_path if save_path is not None else Path("./")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        plt.savefig(save_path / "training_stats.pdf", bbox_inches="tight", dpi=100)


def plot_single_model_training_stats(
    axs: List[plt.Axes],
    data: rl_logger.SmoothedRLData,
    model_name: str,
    color: str = None,
) -> None:
    plt.show(block=False)
    sns.lineplot(x=range(len(data.critic_loss)), y=data.critic_loss, ax=axs[0, 0], label=model_name, color=color)
    axs[0, 0].fill_between(
        range(len(data.critic_loss)),
        data.critic_loss - data.std_critic_loss,
        data.critic_loss + data.std_critic_loss,
        alpha=0.2,
        color=color,
    )
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_xlabel("Training step")
    axs[0, 0].set_title("Critic loss")

    sns.lineplot(x=range(len(data.actor_loss)), y=data.actor_loss, ax=axs[0, 1], label=model_name, color=color)
    axs[0, 1].fill_between(
        range(len(data.actor_loss)),
        data.actor_loss - data.std_actor_loss,
        data.actor_loss + data.std_actor_loss,
        alpha=0.2,
        color=color,
    )
    axs[0, 1].set_title("Actor loss")
    axs[0, 1].set_xlabel("Training step")

    sns.lineplot(
        x=range(len(data.ent_coeff_loss)),
        y=data.ent_coeff_loss,
        ax=axs[0, 2],
        label=model_name,
        color=color,
    )
    axs[0, 2].fill_between(
        range(len(data.ent_coeff_loss)),
        data.ent_coeff_loss - data.std_ent_coeff_loss,
        data.ent_coeff_loss + data.std_ent_coeff_loss,
        alpha=0.2,
        color=color,
    )
    axs[0, 2].set_title("Entropy coeff. loss")
    axs[0, 2].set_xlabel("Training step")

    sns.lineplot(
        x=range(len(data.mean_actor_grad_norm)),
        y=data.mean_actor_grad_norm,
        ax=axs[1, 0],
        label=model_name,
        color=color,
    )
    axs[1, 0].fill_between(
        range(len(data.mean_actor_grad_norm)),
        data.mean_actor_grad_norm - data.std_mean_actor_grad_norm,
        data.mean_actor_grad_norm + data.std_mean_actor_grad_norm,
        alpha=0.2,
        color=color,
    )
    axs[1, 0].set_title("Mean actor grad. norm")
    axs[1, 0].set_xlabel("Training step")

    sns.lineplot(x=range(len(data.ent_coeff)), y=data.ent_coeff, ax=axs[1, 1], label=model_name, color=color)
    axs[1, 1].fill_between(
        range(len(data.ent_coeff)),
        data.ent_coeff - data.std_ent_coeff,
        data.ent_coeff + data.std_ent_coeff,
        alpha=0.2,
        color=color,
    )
    axs[1, 1].set_title("Entropy coeff.")
    axs[1, 1].set_xlabel("Training step")

    sns.lineplot(
        x=range(len(data.non_optimal_solution_rate)),
        y=data.non_optimal_solution_rate,
        ax=axs[1, 2],
        label=model_name,
        color=color,
    )
    axs[1, 2].fill_between(
        range(len(data.non_optimal_solution_rate)),
        data.non_optimal_solution_rate - data.std_non_optimal_solution_rate,
        data.non_optimal_solution_rate + data.std_non_optimal_solution_rate,
        alpha=0.2,
        color=color,
    )
    axs[1, 2].set_title("Non-optimal solution percentage")
    axs[1, 2].set_xlabel("Training step")


def plot_multiple_model_eval_results(
    eval_data_list: List[Dict[str, Any]],
    model_names: List[str],
    save_fig: bool = False,
    save_path: Path = None,
) -> None:
    """Plots the evaluation results for multiple models, with different colors for each model and different markers for different outcomes.

    Args:
        eval_data_list (List[Dict[str, Any]]): List of tuples containing evaluation return data dictionaries (timesteps, episode lengths, and results) and indices for different outcomes (goal reached, colliding, grounding, truncating, actor failed).
        model_names (List[str]): List of model names.
        save_fig (bool, optional): Whether to save the figure.
        save_path (Path, optional): Path to save the figure.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), num="mm_eval_results")
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    colors = sns.color_palette("tab10", n_colors=len(eval_data_list))
    for name, data in zip(model_names, eval_data_list):
        return_data = data[0]
        # indices_with_ep_lengths_above_100 = data[1]
        goal_reached_rate = data[2]
        collision_rate = data[3]
        grounding_rate = data[4]
        truncation_rate = data[5]
        actor_failed_rate = data[6]

        color = colors.pop()
        axs[0].plot(
            data[0]["timesteps"], data[0]["mean_ep_length"], label=name, color=color, marker=".", linestyle="--"
        )
        axs[0].fill_between(
            data[0]["timesteps"],
            np.array(data[0]["mean_ep_length"]) - np.array(data[0]["std_ep_length"]),
            np.array(data[0]["mean_ep_length"]) + np.array(data[0]["std_ep_length"]),
            alpha=0.2,
            color=color,
        )
        axs[0].set_ylabel("Episode length")
        axs[1].plot(
            return_data["timesteps"], return_data["mean_ep_rew"], label=name, color=color, marker=".", linestyle="--"
        )
        axs[1].fill_between(
            return_data["timesteps"],
            np.array(return_data["mean_ep_rew"]) - np.array(return_data["std_ep_rew"]),
            np.array(return_data["mean_ep_rew"]) + np.array(return_data["std_ep_rew"]),
            alpha=0.2,
            color=color,
        )
        axs[1].set_ylabel("Return")
        axs[2].plot(
            return_data["timesteps"],
            goal_reached_rate,
            label="Goals reached rate",
            color=color,
            marker=".",
            linestyle="--",
        )
        # axs[2].plot(return_data["timesteps"], collision_rate, label="Collision rate", color="r")
        # axs[2].plot(return_data["timesteps"], grounding_rate, label="Grounding rate", color="b")
        # axs[2].plot(return_data["timesteps"], truncation_rate, label="Truncation rate", color="k")
        # axs[2].plot(return_data["timesteps"], actor_failed_rate, label="Actor failure rate", color="y")
        axs[2].set_xlabel("Timesteps")
        axs[2].set_ylabel("Rate")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show(block=False)
    if save_fig:
        save_path = save_path if save_path is not None else Path("./")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        fig.savefig(save_path / "mm_eval_results.pdf", bbox_inches="tight", dpi=100)


def plot_multiple_model_worst_and_best_episode_data(
    wb_env_data_list: List[Tuple[csgym_logger.EpisodeData, csgym_logger.EpisodeData]],
    model_names: List[str],
    save_fig: bool = False,
    save_path: Path = None,
) -> None:
    """Plots the worst and best episode data for each model in the list.

    Args:
        wb_env_data_list (List[Tuple[csgym_logger.EpisodeData, csgym_logger.EpisodeData]]): Worst and best episode data for each model.
        model_names (List[str]): List of model names.
        save_fig (bool, optional): Whether to save the figure.
        save_path (Path, optional): Path to save the figure.
    """
    colors = sns.color_palette("tab10", n_colors=len(wb_env_data_list))
    for name, data in zip(model_names, wb_env_data_list):
        color = colors.pop()
        plot_episode_data_series(data[0], name=name + "_worst_ep", save_figs=save_fig, save_path=save_path)
        plot_episode_data_series(data[1], name=name + "_best_ep", save_figs=save_fig, save_path=save_path)


def plot_episode_data_series(
    data: csgym_logger.EpisodeData,
    name: str,
    save_figs: bool = False,
    save_path: Path = None,
) -> Tuple[plt.axes, plt.axes]:
    """Plots the episode data series.

    Args:
        axs (plt.Axes): Axes to plot on.
        data (csgym_logger.EpisodeData): Episode data to plot.
        name (str): Name of the model + episode used in saving the figure.
        save_figs (bool, optional): Whether to save the figures.
        save_path (Path, optional): Path to save the figure.
    """
    mpc_params = np.array([data.actor_infos[i]["new_mpc_params"] for i in range(len(data.actor_infos))])
    r_safe_so = 5.0

    fig1, axs1 = plt.subplots(4, 1, figsize=(10, 15), num=name + "_d2fail_actions")
    fig1.subplots_adjust(hspace=0.3, wspace=0.25)
    times = np.linspace(0, data.duration, len(data.distances_to_collision))
    axs1[0].semilogy(times, data.distances_to_grounding, label="Dist. to grounding", color="b")
    axs1[0].semilogy(times, r_safe_so * np.ones_like(times), label=r"$r_{safe, so}$", color="r", linestyle="--")
    axs1[0].set_ylabel("Distance [m]")

    axs1[1].semilogy(times, data.distances_to_collision, label="Dist. to collision", color="b")
    axs1[1].semilogy(times, mpc_params[:, 8], label=r"$r_{safe, do}$", color="r", linestyle="--")
    axs1[1].set_ylabel("Distance [m]")

    course_refs = np.array([data.actor_infos[i]["applied_refs"][0] for i in range(len(data.actor_infos))])
    courses = data.ownship_states[:, 2] + np.arctan2(data.ownship_states[:, 4], data.ownship_states[:, 3])
    courses = np.unwrap(courses)
    speed_refs = np.array([data.actor_infos[i]["applied_refs"][1] for i in range(len(data.actor_infos))])
    speeds = np.sqrt(data.ownship_states[:, 3] ** 2 + data.ownship_states[:, 4] ** 2)
    axs1[2].plot(times, 180.0 * course_refs / np.pi, label=r"$\chi_{d}$", color="r", linestyle="--")
    axs1[2].plot(times, 180.0 * courses / np.pi, label=r"$\chi$", color="b")
    axs1[2].set_ylabel("Course [deg]")

    axs1[3].plot(times, speed_refs, label=r"$U_{d}$", color="r", linestyle="--")
    axs1[3].plot(times, speeds, label=r"$U$", color="b")
    axs1[3].set_ylabel("Speed [m/s]")
    axs1[3].set_xlabel("Time [s]")

    axs1[0].legend()
    axs1[1].legend()
    axs1[2].legend()
    axs1[3].legend()

    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 15), num=name + "_mpc_params")
    fig2.subplots_adjust(hspace=0.3, wspace=0.25)
    axs2[0].semilogy(times, mpc_params[:, 0], label=r"$K_p$")
    axs2[0].semilogy(times, mpc_params[:, 1], label=r"$K_U$")
    axs2[0].semilogy(times, mpc_params[:, 2], label=r"$K_{\omega}$")
    axs2[1].plot(times, mpc_params[:, 3], label=r"$K_{\dot{\chi}}$")
    axs2[1].plot(times, mpc_params[:, 4], label=r"$K_{\dot{U}}$")
    axs2[2].plot(times, mpc_params[:, 5], label=r"$w_{HO}$")
    axs2[2].plot(times, mpc_params[:, 6], label=r"$w_{CR}$")
    axs2[2].plot(times, mpc_params[:, 7], label=r"$w_{OT}$")
    axs2[2].set_xlabel("Time [s]")
    axs2[0].legend()
    axs2[1].legend()
    axs2[2].legend()

    fig3, axs3 = plt.subplots(4, 2, figsize=(10, 15), num=name + "_rewards")
    r_total = data.rewards
    r_colreg = np.array([data.reward_components[i]["r_colreg"] for i in range(len(data.reward_components))])
    r_app_man = np.array(
        [data.reward_components[i]["r_readily_apparent_maneuvering"] for i in range(len(data.reward_components))]
    )
    r_colav = np.array([data.reward_components[i]["r_collision_avoidance"] for i in range(len(data.reward_components))])
    r_antigrounding = np.array(
        [data.reward_components[i]["r_antigrounding"] for i in range(len(data.reward_components))]
    )
    r_trajectory_tracking = np.array(
        [data.reward_components[i]["r_trajectory_tracking"] for i in range(len(data.reward_components))]
    )
    r_dnn_params = np.array([data.reward_components[i]["r_dnn_parameters"] for i in range(len(data.reward_components))])
    axs3[0, 0].plot(times, r_total, label="Total reward")
    axs3[0, 1].plot(times, r_colreg, label="COLREG reward")
    axs3[1, 0].plot(times, r_colav, label="COLAV reward")
    axs3[1, 1].plot(times, r_antigrounding, label="Anti-grounding reward")
    axs3[2, 0].plot(times, r_trajectory_tracking, label="Trajectory tracking reward")
    axs3[2, 1].plot(times, r_app_man, label="Readily apparent maneuvering reward")
    axs3[3, 0].plot(times, r_dnn_params, label="DNN parameters reward")
    axs3[3, 0].set_xlabel("Time [s]")
    axs3[3, 1].set_xlabel("Time [s]")
    axs3[0, 0].legend()
    axs3[0, 1].legend()
    axs3[1, 0].legend()
    axs3[1, 1].legend()
    axs3[2, 0].legend()
    axs3[2, 1].legend()
    axs3[3, 0].legend()
    plt.show(block=False)

    if save_figs:
        save_path = save_path if save_path is not None else Path("./")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        fig1.savefig(save_path / (name + "_d2fail_actions.pdf"), bbox_inches="tight", dpi=100)
        fig2.savefig(save_path / (name + "_mpc_params.pdf"), bbox_inches="tight", dpi=100)
        fig3.savefig(save_path / (name + "_rewards.pdf"), bbox_inches="tight", dpi=100)


def plot_training_results(
    base_dir: Path, experiment_names: List[str], abbreviations: Optional[List[str]] = None
) -> None:
    """Plots results from training.

    Args:
        base_dir (Path): Base path to the experiment directories
        experiment_names (List[str]): List of experiment names (experiment folder names).
        abbreviations (Optional[List[str]], optional): List of abbreviations for the model names.
    """
    env_data_list = []
    training_stats_list = []
    reward_data_list = []
    training_stats_list = []

    plot_env_snapshots = False
    plot_reward_curves = True
    for experiment_name in experiment_names:
        log_dir = base_dir / experiment_name

        rl_data_logger = rlmpc_logger.Logger(experiment_name=experiment_name, log_dir=log_dir)
        rl_data_logger.load_from_pickle(f"{experiment_name}_training_stats")
        smoothed_training_stats = hf.process_rl_training_data(rl_data_logger.rl_data, ma_window_size=5)
        training_stats_list.append(smoothed_training_stats)

        if plot_env_snapshots or plot_reward_curves:
            env_logger = csenv_logger.Logger(
                experiment_name=experiment_name, log_dir=log_dir, max_num_logged_episodes=1000000
            )
            env_logger.load_from_pickle(f"{experiment_name}_env_training_data")
            env_data_list.append(env_logger.env_data)

            reward_data = hf.extract_reward_data(env_logger.env_data, scale_reward_components=True)
            reward_data_list.append(reward_data)

        if plot_env_snapshots:
            plot_single_model_enc_snapshots(
                env_logger.env_data,
                nrows=5,
                ncols=3,
                save_fig=True,
                save_path=base_dir / experiment_name / "figures",
            )

    model_names = experiment_names
    if abbreviations is not None:
        model_names = abbreviations
    plot_multiple_model_reward_curves(
        model_data=reward_data_list,
        model_names=model_names,
        save_fig=True,
        save_path=base_dir / "figures",
    )
    plot_multiple_model_training_stats(
        model_data=training_stats_list,
        model_names=model_names,
        save_fig=True,
        save_path=base_dir / "figures",
    )


def plot_evaluation_results(
    base_dir: Path, experiment_names: List[str], abbreviations: Optional[List[str]] = None
) -> None:
    """Plots results from training, more specifically the environment data logged by the COLAVENvironment gym logger, and
    the .npz files from each evaluation (stored using stable-baselines3 evaluation callback).

    Args:
        base_dir (Path): Base path to the experiment directories
        experiment_names (List[str]): List of experiment names (experiment folder names).
        abbreviations (Optional[List[str]], optional): List of abbreviations for the model names.
    """
    wb_env_data_list = []
    eval_data_list = []

    plot_env_snapshots = False
    plot_evaluation_curves = True
    plot_worst_and_best_episode_data = True
    for experiment_name in experiment_names:
        log_dir = base_dir / experiment_name
        eval_data_dir = log_dir / "eval_data"
        env_logger = csenv_logger.Logger(experiment_name=experiment_name, log_dir=log_dir)
        env_data_pkl_file_list = [file for file in eval_data_dir.iterdir()]
        env_data_pkl_file_list = [eval_data_dir / file.stem for file in env_data_pkl_file_list if file.suffix == ".pkl"]
        env_data_pkl_file_list.sort(key=lambda x: int(x.stem.split("_")[-3]))

        eval_return_data = {}
        eval_return_data["timesteps"] = []
        eval_return_data["mean_ep_length"] = []
        eval_return_data["std_ep_length"] = []
        eval_return_data["mean_ep_rew"] = []
        eval_return_data["std_ep_rew"] = []
        npz_file_list = [file for file in eval_data_dir.iterdir()]
        npz_file_list = [file for file in npz_file_list if file.suffix == ".npz"]
        npz_file_list.sort(key=lambda x: int(x.stem.split("_")[-1]))
        if not npz_file_list:
            continue

        indices_with_above_100_ep_lengths = []
        goal_reached_rate = []
        collision_rate = []
        grounding_rate = []
        truncating_rate = []
        actor_failed_rate = []
        for idx, npzf in enumerate(npz_file_list):
            if idx > len(env_data_pkl_file_list) - 1:
                continue
            with np.load(npzf) as data:
                timesteps = data["timesteps"][-1]
                mean_ep_length = data["ep_lengths"][-1]
                mean_rew = data["results"][-1]
                if mean_ep_length.size > 1:
                    mean_ep_length = mean_ep_length.mean()
                    mean_rew = mean_rew.mean()
                    std_ep_length = data["ep_lengths"][-1].std()
                    std_rew = data["results"][-1].std()
                eval_return_data["timesteps"].append(timesteps)
                eval_return_data["mean_ep_length"].append(mean_ep_length)
                eval_return_data["std_ep_length"].append(std_ep_length)
                eval_return_data["mean_ep_rew"].append(mean_rew)
                eval_return_data["std_ep_rew"].append(std_rew)
                if mean_ep_length > 100:
                    indices_with_above_100_ep_lengths.append(idx)

                n_eval_eps = data["results"][-1].size
                goals_reached = 0
                collisions = 0
                groundings = 0
                truncations = 0
                actor_failures = 0
                for eidx, ep_len in enumerate(data["ep_lengths"][0]):
                    env_logger.load_from_pickle(str(env_data_pkl_file_list[idx]))
                    if env_logger.env_data[eidx].goal_reached:
                        goals_reached += 1
                    elif env_logger.env_data[eidx].collision:
                        collisions += 1
                    elif env_logger.env_data[eidx].grounding:
                        groundings += 1
                    elif env_logger.env_data[eidx].truncated:
                        truncations += 1
                    elif env_logger.env_data[eidx].actor_failure:
                        actor_failures += 1

                goal_reached_rate.append(goals_reached / n_eval_eps)
                collision_rate.append(collisions / n_eval_eps)
                grounding_rate.append(groundings / n_eval_eps)
                truncating_rate.append(truncations / n_eval_eps)
                actor_failed_rate.append(actor_failures / n_eval_eps)

        eval_data_list.append(
            (
                eval_return_data,
                indices_with_above_100_ep_lengths,
                goal_reached_rate,
                collision_rate,
                grounding_rate,
                truncating_rate,
                actor_failed_rate,
            )
        )
        argmin_reward = int(np.argmin(eval_return_data["mean_ep_rew"]))
        argmax_reward = int(np.argmax([eval_return_data["mean_ep_rew"][i] for i in indices_with_above_100_ep_lengths]))
        env_logger.load_from_pickle(str(env_data_pkl_file_list[argmin_reward]))
        worst_env_data = env_logger.env_data[0]
        env_logger.load_from_pickle(str(env_data_pkl_file_list[argmax_reward]))
        best_env_data = env_logger.env_data[0]
        wb_env_data_list.append((worst_env_data, best_env_data))

        if plot_env_snapshots:
            plot_single_model_enc_snapshots(
                env_logger.env_data,
                nrows=5,
                ncols=3,
                save_fig=True,
                save_path=base_dir / experiment_name / "figures",
            )

    model_names = experiment_names
    if abbreviations is not None:
        model_names = abbreviations

    if plot_evaluation_curves:
        plot_multiple_model_eval_results(
            eval_data_list=eval_data_list,
            model_names=model_names,
            save_fig=True,
            save_path=base_dir / "figures",
        )
    if plot_worst_and_best_episode_data:
        plot_multiple_model_worst_and_best_episode_data(
            wb_env_data_list=wb_env_data_list,
            model_names=model_names,
            save_fig=True,
            save_path=base_dir / "figures",
        )


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    base_dir: Path = Path.home() / "Desktop/machine_learning/rlmpc"
    experiment_names = ["standard_snmpc_1te_4ee_seed1_jid20787312"]
    model_names = ["SAC-NMPC1"]
    plot_training_results(base_dir=base_dir, experiment_names=experiment_names, abbreviations=model_names)
    plot_evaluation_results(base_dir=base_dir, experiment_names=experiment_names, abbreviations=model_names)
    print("Done plotting")
