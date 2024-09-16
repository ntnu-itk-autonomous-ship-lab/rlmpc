"""
    plotters.py

    Summary:
        Contains functions for plotting data from the RL training process.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    fig, axs = plt.subplots(4, 2, figsize=(15, 10), num="reward_curves")
    fig.subplots_adjust(hspace=0.4, wspace=0.25)
    colors = sns.color_palette("tab10", n_colors=len(model_data))
    for name, data in zip(model_names, model_data):
        color = colors.pop()
        plot_single_model_reward_curves(axs, data, name, color=color)
    if save_fig:
        save_path = save_path if save_path is not None else Path("./")
        if not save_path.exists():
            save_path.mkdir(parents=True)
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
    # axs[0, 0].set_xlabel("Episode")

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
    axs[1, 0].set_ylabel("Average return", rotation=90)

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
    axs[2, 0].set_ylabel("Average return", rotation=90)
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
    # axs[3, 0].set_xlabel("Episode")
    axs[3, 0].set_ylabel("Average return", rotation=90)

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


def plot_multiple_model_eval_results(
    eval_data_list: List[Dict[str, Any]], model_names: List[str], save_fig: bool = False, save_path: Path = None
) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), num="mm_eval_results")
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    colors = sns.color_palette("tab10", n_colors=len(eval_data_list))
    for name, data in zip(model_names, eval_data_list):
        color = colors.pop()
        axs.plot(data["timesteps"], data["ep_lengths"], label=name, color=color)
        axs.plot(data["timesteps"], data["results"], label=name, color=color)

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
    fig_worst, axs_worst = plt.subplots(2, 2, figsize=(15, 10), num="worst_episode_data")
    fig_worst.subplots_adjust(hspace=0.3, wspace=0.25)
    fig_best, axs_best = plt.subplots(2, 2, figsize=(15, 10), num="best_episode_data")
    fig_best.subplots_adjust(hspace=0.3, wspace=0.25)
    colors = sns.color_palette("tab10", n_colors=len(wb_env_data_list))
    for name, data in zip(model_names, wb_env_data_list):
        color = colors.pop()
        plot_episode_data_series(axs_worst, data[0], name + "_worst_ep", color=color)
        plot_episode_data_series(axs_best, data[1], name + "_best_ep", color=color)
    if save_fig:
        save_path = save_path if save_path is not None else Path("./")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        fig_worst.savefig(save_path / "worst_episode_data.pdf", bbox_inches="tight", dpi=100)
        fig_best.savefig(save_path / "best_episode_data.pdf", bbox_inches="tight", dpi=100)


def plot_episode_data_series(axs: plt.Axes, data: csgym_logger.EpisodeData, name: str, color: str = None) -> None:
    """Plots the episode data series.

    Args:
        axs (plt.Axes): Axes to plot on.
        data (csgym_logger.EpisodeData): Episode data to plot.
        name (str): Name of the model + episode used in the label
        color (str, optional): Optional color for the plot.
    """


def plot_training_results(base_dir: Path, experiment_names: List[str]) -> None:
    """Plots results from training.

    Args:
        base_dir (Path): Base path to the experiment directories
        experiment_names (List[str]): List of experiment names (experiment folder names).
    """
    env_data_list = []
    training_stats_list = []
    reward_data_list = []
    training_stats_list = []

    plot_env_snapshots = True
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

    plot_multiple_model_reward_curves(
        model_data=reward_data_list,
        model_names=experiment_names,
        save_fig=True,
        save_path=base_dir / "figures",
    )
    plot_multiple_model_training_stats(
        model_data=training_stats_list,
        model_names=experiment_names,
        save_fig=True,
        save_path=base_dir / "figures",
    )


def plot_evaluation_results(base_dir: Path, experiment_names: List[str]) -> None:
    """Plots results from training.

    Args:
        base_dir (Path): Base path to the experiment directories
        experiment_names (List[str]): List of experiment names (experiment folder names).
    """
    env_data_list = []
    wb_env_data_list = []
    eval_data_list = []

    plot_env_snapshots = False
    plot_evaluation_curves = True
    plot_worst_and_best_episode_data = True
    for experiment_name in experiment_names:
        log_dir = base_dir / experiment_name
        eval_data_dir = log_dir / "eval_data"

        eval_data = {}
        eval_data["timesteps"] = []
        eval_data["ep_lengths"] = []
        eval_data["results"] = []
        npz_file_list = [file for file in eval_data_dir.iterdir()]
        npz_file_list = [file for file in npz_file_list if file.suffix == ".npz"]
        npz_file_list.sort(key=lambda x: int(x.stem.split("_")[-1]))
        indices_with_above_100_ep_lengths = []
        for npzf in npz_file_list:
            with np.load(npzf) as data:
                eval_data["timesteps"].append(int(data["timesteps"].item()))
                eval_data["ep_lengths"].append(int(data["ep_lengths"].item()))
                eval_data["results"].append(float(data["results"].item()))
                if int(data["ep_lengths"].item()) > 100:
                    indices_with_above_100_ep_lengths.append(int(data["timesteps"].item()))
        eval_data_list.append(eval_data)

        env_data_pkl_file_list = [file for file in eval_data_dir.iterdir()]
        env_data_pkl_file_list = [eval_data_dir / file.stem for file in env_data_pkl_file_list if file.suffix == ".pkl"]
        env_data_pkl_file_list.sort(key=lambda x: int(x.stem.split("_")[-3]))

        env_logger = csenv_logger.Logger(experiment_name=experiment_name, log_dir=log_dir)
        # Plot mpc params and actions for best and worst episodes

        if plot_worst_and_best_episode_data:
            argmin_reward = int(np.argmin(eval_data["results"]))
            argmax_reward = int(np.argmax([eval_data["results"][i] for i in indices_with_above_100_ep_lengths]))
            env_logger.load_from_pickle(str(env_data_pkl_file_list[argmin_reward]))
            worst_env_data = env_logger.env_data[0]
            best_env_data = env_logger.load_from_pickle(str(env_data_pkl_file_list[argmax_reward]))
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

    if plot_evaluation_curves:
        plot_multiple_model_eval_results(
            eval_data_list=eval_data_list,
            model_names=experiment_names,
            save_fig=True,
            save_path=base_dir / "figures",
        )
    if plot_worst_and_best_episode_data:
        plot_multiple_model_worst_and_best_episode_data(
            wb_env_data_list=wb_env_data_list,
            model_names=experiment_names,
            save_fig=True,
            save_path=base_dir / "figures",
        )


if __name__ == "__main__":
    base_dir: Path = Path.home() / "Desktop/machine_learning/rlmpc"
    experiment_names = ["snmpc_db_1te_1ee_16cpus"]
    # plot_training_results(base_dir=base_dir, experiment_names=experiment_names)
    plot_evaluation_results(base_dir=base_dir, experiment_names=experiment_names)
    print("Done plotting")
