"""
    plotters.py

    Summary:
        Contains functions for plotting data from the RL training process.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Any, Dict, List

import colav_simulator.common.image_helper_methods as ihm
import colav_simulator.gym.logger as csgym_logger
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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


def plot_single_model_training_enc_snapshots(
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
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        fig,
        wspace=0,
        hspace=0.2,
        top=0.95,
        bottom=0.05,
    )
    n_episodes = len(data)
    indices = np.linspace(0, n_episodes - 1, nrows * ncols).astype(int)
    for i, idx in enumerate(indices):
        ax = plt.subplot(gs[i])
        episode = data[idx]
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
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), num="reward_curves")
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    colors = sns.color_palette("tab10", n_colors=len(model_data))
    for name, data in zip(model_names, model_data):
        color = colors.pop()
        plot_single_model_reward_curves(axs, data, name, color=color)
    if save_fig:
        save_path = save_path if save_path is not None else Path("./")
        plt.savefig(save_path / "reward_curves.pdf", bbox_inches="tight", dpi=100)


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
    axs[0, 0].set_ylabel("Average return", rotation=90)
    axs[0, 0].set_xlabel("Episode")

    sns.lineplot(x=range(len(data["r_colreg"])), y=data["r_colreg"], ax=axs[0, 1], label=model_name, color=color)
    axs[0, 1].fill_between(
        range(len(data["r_colreg"])),
        np.array(data["r_colreg"]) - np.array(data["std_r_colreg"]),
        np.array(data["r_colreg"]) + np.array(data["std_r_colreg"]),
        alpha=0.2,
        color=color,
    )
    axs[0, 1].set_title("COLREG")
    axs[0, 1].set_xlabel("Episode")

    sns.lineplot(x=range(len(data["r_colav"])), y=data["r_colav"], ax=axs[0, 2], label=model_name, color=color)
    axs[0, 2].fill_between(
        range(len(data["r_colav"])),
        np.array(data["r_colav"]) - np.array(data["std_r_colav"]),
        np.array(data["r_colav"]) + np.array(data["std_r_colav"]),
        alpha=0.2,
        color=color,
    )
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_title("Collision avoidance")

    sns.lineplot(
        x=range(len(data["r_antigrounding"])),
        y=data["r_antigrounding"],
        ax=axs[1, 0],
        label=model_name,
        color=color,
    )
    axs[1, 0].fill_between(
        range(len(data["r_antigrounding"])),
        np.array(data["r_antigrounding"]) - np.array(data["std_r_antigrounding"]),
        np.array(data["r_antigrounding"]) + np.array(data["std_r_antigrounding"]),
        alpha=0.2,
        color=color,
    )
    axs[1, 0].set_title("Anti-grounding")
    axs[1, 0].set_ylabel("Average return", rotation=90)
    axs[1, 0].set_xlabel("Episode")

    sns.lineplot(
        x=range(len(data["r_trajectory_tracking"])),
        y=data["r_trajectory_tracking"],
        ax=axs[1, 1],
        label=model_name,
        color=color,
    )
    axs[1, 1].fill_between(
        range(len(data["r_trajectory_tracking"])),
        np.array(data["r_trajectory_tracking"]) - np.array(data["std_r_trajectory_tracking"]),
        np.array(data["r_trajectory_tracking"]) + np.array(data["std_r_trajectory_tracking"]),
        alpha=0.2,
        color=color,
    )
    axs[1, 1].set_title("Trajectory tracking")
    axs[1, 1].set_xlabel("Episode")

    sns.lineplot(
        x=range(len(data["r_ra_maneuvering"])),
        y=data["r_ra_maneuvering"],
        ax=axs[1, 2],
        label=model_name,
        color=color,
    )
    axs[1, 2].fill_between(
        range(len(data["r_ra_maneuvering"])),
        np.array(data["r_ra_maneuvering"]) - np.array(data["std_r_ra_maneuvering"]),
        np.array(data["r_ra_maneuvering"]) + np.array(data["std_r_ra_maneuvering"]),
        alpha=0.2,
        color=color,
    )
    axs[1, 2].set_title("RA maneuvering")
    axs[1, 2].set_xlabel("Episode")

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
        plt.savefig(save_path / "training_stats.pdf", bbox_inches="tight", dpi=100)


def plot_single_model_training_stats(
    axs: List[plt.Axes],
    data: rl_logger.SmoothedRLData,
    model_name: str,
    color: str = None,
) -> None:

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

    plt.show(block=False)
