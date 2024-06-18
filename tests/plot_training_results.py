import pickle
import platform
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from colav_simulator.gym.logger import EpisodeData


def plot_training_results() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))


def plot_reward_curves(env_data: List[EpisodeData]) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rewards = [episode_data.cumulative_reward for episode_data in env_data]
    ax.plot(np.arange(len(rewards)), rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward vs. Episode")
    plt.show()


if __name__ == "__main__":
    if platform.system() == "Linux":
        base_dir = Path("/home/doctor/Desktop/machine_learning/rlmpc/")
    elif platform.system() == "Darwin":
        base_dir = Path("/Users/trtengesdal/Desktop/machine_learning/rlmpc/")

    experiment_name = "sac_rlmpc1"
    data_dir = base_dir / "sac_rlmpc"
    eval_infos = None
    env_data_info_path = data_dir / (experiment_name + "env_training_data.pkl")
    training_stats_data_path = data_dir / (experiment_name + "training_data.pkl")

    with open(env_data_info_path, "rb") as f:
        env_data = pickle.load(f)

    with open(training_stats_data_path, "rb") as f:
        training_stats = pickle.load(f)

    plot_training_results(env_data, training_stats)
