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

    n_updates: int
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

    n_updates: int
    time_elapsed: float
    episodes: int
    critic_loss: np.ndarray
    actor_loss: np.ndarray
    mean_actor_grad_norm: np.ndarray
    ent_coeff_loss: np.ndarray
    ent_coeff: np.ndarray
    infeasible_solutions: np.ndarray
    mean_episode_reward: np.ndarray
    mean_episode_length: np.ndarray
    batch_processing_time: np.ndarray
    success_rate: np.ndarray


class Logger:
    """Logs data from the COLAV environment. Supports saving and loading to/from pickle files."""

    def __init__(self, experiment_name: str, log_dir: Path, max_num_episodes: int = 10_000) -> None:
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        self.rl_data: RLData = RLData(
            n_updates=0,
            time_elapsed=0.0,
            episodes=0,
            critic_loss=np.array([]),
            actor_loss=np.array([]),
            mean_actor_grad_norm=np.array([]),
            ent_coeff_loss=np.array([]),
            ent_coeff=np.array([]),
            infeasible_solutions=np.array([]),
            mean_episode_reward=np.array([]),
            mean_episode_length=np.array([]),
            batch_processing_time=np.array([]),
            success_rate=np.array([]),
        )
        self.experiment_name: str = experiment_name
        self.log_dir: Path = log_dir

        self.max_num_episodes = max_num_episodes
        self.n_updates: int = 0
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

        with open(self.log_dir / (name + ".pkl"), "ba") as f:
            pickle.dump(self.rl_data, f)

    def load_from_pickle(self, name: Optional[str]) -> None:
        """Loads the rl data from a pickle file.

        Args:
            name (Optional[str]): Name of the pickle file, without the .pkl extension.
        """
        if name is None:
            name = "rl_data"
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

        self.n_updates = training_info["n_updates"]
        self.time_elapsed = training_info["time_elapsed"]
        self.batch_processing_times.append(training_info["batch_processing_time"])
        self.critic_losses.append(training_info["critic_loss"])
        self.actor_losses.append(training_info["actor_loss"])
        self.mean_actor_grad_norms.append(training_info["actor_grad_norm"])
        self.ent_coeff_losses.append(training_info["ent_coef_loss"])
        self.ent_coeffs.append(training_info["ent_coef"])

    def push(self) -> None:
        """Updates the logger with the latest data."""
        self.rl_data = RLData(
            n_updates=self.n_updates,
            time_elapsed=self.time_elapsed,
            episodes=self.episodes,
            critic_loss=np.array(self.critic_losses, dtype=np.float32),
            actor_loss=np.array(self.actor_losses, dtype=np.float32),
            mean_actor_grad_norm=np.array(self.mean_actor_grad_norms, dtype=np.float32),
            ent_coeff_loss=np.array(self.ent_coeff_losses, dtype=np.float32),
            ent_coeff=np.array(self.ent_coeffs, dtype=np.float32),
            infeasible_solutions=np.array(self.infeasible_solutions, dtype=np.int32),
            mean_episode_reward=np.array(self.mean_episode_rewards, dtype=np.float32),
            mean_episode_length=np.array(self.mean_episode_lengths, dtype=np.float32),
            batch_processing_time=np.array(self.batch_processing_times, dtype=np.float32),
            success_rate=np.array(self.success_rates, dtype=np.float32),
        )
        if self.episodes >= self.max_num_episodes:
            self.save_as_pickle()
            self._reset()

    def _reset(self) -> None:
        """Resets the data structures in preparation of a new training."""
        self.n_updates = 0
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
