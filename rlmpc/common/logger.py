"""
    logger.py

    Summary:
        Contains class for reporting data from the RL training process.

    Author: Trym Tengesdal
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np


class SmoothedRLData(NamedTuple):
    """Smoothed data with std info from the RLMPC training process."""

    timesteps: int
    time_elapsed: float
    episodes: int
    critic_loss: List[float]
    std_critic_loss: List[float]
    actor_loss: List[float]
    std_actor_loss: List[float]
    mean_actor_grad_norm: List[float]
    std_mean_actor_grad_norm: List[float]
    ent_coeff_loss: List[float]
    std_ent_coeff_loss: List[float]
    ent_coeff: List[float]
    std_ent_coeff: List[float]
    infeasible_solutions: List[int]
    std_infeasible_solutions: List[int]
    mean_episode_reward: List[float]
    std_mean_episode_reward: List[float]
    mean_episode_length: List[float]
    std_mean_episode_length: List[float]
    batch_processing_time: List[float] = []
    std_batch_processing_time: List[float] = []
    success_rate: List[float] = []
    std_success_rate: List[float] = []


class RLData(NamedTuple):
    """Data from the RLMPC training process."""

    timesteps: int
    time_elapsed: float
    episodes: int
    critic_loss: List[float]
    actor_loss: List[float]
    mean_actor_grad_norm: List[float]
    ent_coeff_loss: List[float]
    ent_coeff: List[float]
    infeasible_solutions: List[int]
    mean_episode_reward: List[float]
    mean_episode_length: List[float]
    batch_processing_time: List[float] = []
    success_rate: List[float] = []


class Logger:
    """Logs data from the COLAV environment. Supports saving and loading to/from pickle files."""

    def __init__(self, experiment_name: str, log_dir: Path) -> None:
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        self.rl_data: RLData = RLData(
            timesteps=0,
            time_elapsed=0.0,
            episodes=0,
            critic_loss=[],
            actor_loss=[],
            mean_actor_grad_norm=[],
            ent_coeff_loss=[],
            ent_coeff=[],
            infeasible_solutions=[],
            mean_episode_reward=[],
            mean_episode_length=[],
            batch_processing_time=[],
            success_rate=[],
        )
        self.experiment_name: str = experiment_name
        self.log_dir: Path = log_dir

        self.timesteps: int = 0
        self.time_elapsed: float = 0.0
        self.episodes: int = 0
        self.critic_losses: List[float] = []
        self.actor_losses: List[float] = []
        self.mean_actor_grad_norms: List[float] = []
        self.ent_coeff_losses: List[float] = []
        self.ent_coeffs: List[float] = []
        self.infeasible_solutions: List[int] = []
        self.mean_episode_rewards: List[float] = []
        self.mean_episode_lengths: List[float] = []
        self.batch_processing_times: List[float] = []
        self.success_rates: List[float] = []

    def save_as_pickle(self, name: Optional[str] = None) -> None:
        """Saves the environment data to a pickle file.

        Args:
            name (Optional[str]): The name of the pickle file, without the .pkl extension.
        """
        if name is None:
            name = "rl_training_data"

        with open(self.log_dir / (name + ".pkl"), "wb") as f:
            pickle.dump(self.rl_data, f)

    def load_from_pickle(self, name: Optional[str]) -> None:
        """Loads the environment data from a pickle file.
        Args:
            name (Optional[str]): Name of the pickle file, without the .pkl extension.
        """
        if name is None:
            name = "env_data"
        with open(self.log_dir / (name + ".pkl"), "rb") as f:
            self.rl_data = pickle.load(f)

    def update_rollout_metrics(self, rollout_info: dict) -> None:
        """Updates the logger with the latest rollout data.

        Args:
            rollout_info (dict): Dictionary containing training data.
        """
        if not rollout_info:
            return

        # Updated in off_policy_algorithm.py
        self.episodes = rollout_info["episodes"]
        self.mean_episode_rewards.append(rollout_info["mean_episode_reward"])
        self.mean_episode_lengths.append(rollout_info["mean_episode_length"])
        self.success_rates.append(rollout_info["success_rate"])
        self.infeasible_solutions.append(rollout_info["infeasible_solutions"])

    def update_training_metrics(self, training_info: dict) -> None:
        """Updates the logger with the latest training data (compatible with SAC only per now)

        Args:
            training_info (dict): Dictionary containing training data.
        """
        if not training_info:
            return

        self.timesteps = training_info["training_timesteps"]
        self.time_elapsed = training_info["time_elapsed"]
        self.batch_processing_times.append(training_info["batch_processing_time"])
        self.critic_losses.append(training_info["critic_loss"])
        self.actor_losses.append(training_info["actor_loss"])
        self.mean_actor_grad_norms.append(training_info["actor_grad_norm"])
        self.ent_coeff_losses.append(training_info["ent_coeff_loss"])
        self.ent_coeffs.append(training_info["ent_coeff"])

    def push(self) -> None:
        """Updates the logger with the latest data."""
        self.rl_data = RLData(
            timesteps=self.timesteps,
            time_elapsed=self.time_elapsed,
            episodes=self.episodes,
            critic_loss=self.critic_losses,
            actor_loss=self.actor_losses,
            mean_actor_grad_norm=self.mean_actor_grad_norms,
            ent_coeff_loss=self.ent_coeff_losses,
            ent_coeff=self.ent_coeffs,
            infeasible_solutions=self.infeasible_solutions,
            mean_episode_reward=self.mean_episode_rewards,
            mean_episode_length=self.mean_episode_lengths,
            batch_processing_time=self.batch_processing_times,
            success_rate=self.success_rates,
        )

    def _reset(self) -> None:
        """Resets the data structures in preparation of a new training."""
        self.timesteps = 0
        self.time_elapsed = 0.0
        self.episodes = 0
        self.critic_losses = []
        self.actor_losses = []
        self.mean_actor_grad_norms = []
        self.ent_coeff_losses = []
        self.ent_coeffs = []
        self.infeasible_solutions = []
        self.mean_episode_rewards = []
        self.mean_episode_lengths = []
        self.batch_processing_times = []
        self.success_rates = []


if __name__ == "__main__":
    import pathlib

    ptofile = pathlib.Path("test.pkl")
