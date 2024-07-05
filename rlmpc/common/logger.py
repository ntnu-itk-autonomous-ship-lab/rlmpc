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
    non_optimal_solution_rate: List[int]
    std_non_optimal_solution_rate: List[int]
    mean_episode_reward: List[float]
    std_mean_episode_reward: List[float]
    mean_episode_length: List[float]
    std_mean_episode_length: List[float]
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
    non_optimal_solution_rate: np.ndarray
    mean_episode_reward: np.ndarray
    mean_episode_length: np.ndarray
    success_rate: np.ndarray


class Logger:
    """Logs reinforcement learning training statistics data, tailored to a Soft Actor Critic. Supports saving and loading to/from pickle files.

    Args:
        experiment_name (str): Name of the experiment.
        log_dir (Path): Directory to save the log files.
        max_num_entries (int): Maximum number of entries to store in the log file.

    """

    def __init__(self, experiment_name: str, log_dir: Path, max_num_entries: int = 30_000) -> None:
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        self.max_num_entries = max_num_entries
        self.train_pos: int = 0
        self.prev_train_pos: int = 0
        self.rollout_pos: int = 0
        self.prev_rollout_pos: int = 0
        self.experiment_name: str = experiment_name
        self.log_dir: Path = log_dir
        self.rl_data_list: List[RLData] = []
        self.rl_data: RLData = None
        self.n_updates: int = 0
        self.n_updates_prev: int = 0
        self.time_elapsed: float = 0.0
        self.episodes: int = 0
        self.critic_loss: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)
        self.actor_loss: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)
        self.mean_actor_grad_norm: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)
        self.ent_coef_loss: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)
        self.ent_coef: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)
        self.non_optimal_solution_rate: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)
        self.mean_episode_rewards: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)
        self.mean_episode_lengths: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)
        self.success_rates: np.ndarray = np.zeros(max_num_entries, dtype=np.float32)

    def _save_as_pickle(self, name: Optional[str] = None) -> None:
        """Saves the data to a pickle file.

        Args:
            name (Optional[str]): The name of the pickle file, without the .pkl extension.
        """
        if name is None:
            name = self.experiment_name + "_training_stats"

        # Don't save if there is no data
        if not self.rl_data:
            return

        with open(self.log_dir / (name + ".pkl"), "ba") as f:
            pickle.dump(self.rl_data, f)
        self.rl_data = None

    def load_from_pickle(self, name: Optional[str], merge_objects: bool = True) -> None:
        """Loads the rl data from a pickle file.

        Args:
            name (Optional[str]): Name of the pickle file, without the .pkl extension.
            merge_objects (bool): If True, merge all objects into one list.
        """
        if name is None:
            name = "rl_data"
        rl_data_list = []
        # Because multiple objects might be stored in the same file
        # we need to load them one by one
        with open(self.log_dir / (name + ".pkl"), "rb") as f:
            while 1:
                try:
                    rl_data = pickle.load(f)
                    rl_data_list.append(rl_data)
                except EOFError:
                    break
        if len(rl_data_list) == 1:
            self.rl_data = rl_data_list[0]
        else:
            self.rl_data_list = rl_data_list
            print(f"Loaded {len(self.rl_data_list)} RL data objects from {name}.pkl")

        if merge_objects:
            self.rl_data = RLData(
                n_updates=self.rl_data_list[-1].n_updates,
                time_elapsed=self.rl_data_list[-1].time_elapsed,
                episodes=self.rl_data_list[-1].episodes,
                critic_loss=np.concatenate([rl_data.critic_loss for rl_data in self.rl_data_list]),
                actor_loss=np.concatenate([rl_data.actor_loss for rl_data in self.rl_data_list]),
                mean_actor_grad_norm=np.concatenate([rl_data.mean_actor_grad_norm for rl_data in self.rl_data_list]),
                ent_coeff_loss=np.concatenate([rl_data.ent_coeff_loss for rl_data in self.rl_data_list]),
                ent_coeff=np.concatenate([rl_data.ent_coeff for rl_data in self.rl_data_list]),
                non_optimal_solution_rate=np.concatenate(
                    [rl_data.non_optimal_solution_rate for rl_data in self.rl_data_list]
                ),
                mean_episode_reward=np.concatenate([rl_data.mean_episode_reward for rl_data in self.rl_data_list]),
                mean_episode_length=np.concatenate([rl_data.mean_episode_length for rl_data in self.rl_data_list]),
                success_rate=np.concatenate([rl_data.success_rate for rl_data in self.rl_data_list]),
            )
            print(
                f"Merged {len(self.rl_data_list)} RL data objects into one object with {self.rl_data.n_updates} updates."
            )
            self.rl_data_list = None

    def update_rollout_metrics(self, rollout_info: dict) -> None:
        """Updates the logger with the latest rollout data.

        Args:
            rollout_info (dict): Dictionary containing training data.
        """
        if not rollout_info:
            return

        # Updated in off_policy_algorithm.py
        self.episodes = rollout_info["episodes"]
        self.mean_episode_rewards[self.rollout_pos] = rollout_info["mean_episode_reward"]
        self.mean_episode_lengths[self.rollout_pos] = rollout_info["mean_episode_length"]
        self.success_rates[self.rollout_pos] = rollout_info["success_rate"]
        self.non_optimal_solution_rate[self.rollout_pos] = rollout_info["non_optimal_solution_rate"]
        self.rollout_pos += 1

    def update_training_metrics(self, training_info: dict) -> None:
        """Updates the logger with the latest training data (compatible with SAC only per now)

        Args:
            training_info (dict): Dictionary containing training data.
        """
        if not training_info:
            return

        self.n_updates = training_info["n_updates"]
        self.time_elapsed = training_info["time_elapsed"]
        self.critic_loss[self.train_pos] = training_info["critic_loss"]
        self.actor_loss[self.train_pos] = training_info["actor_loss"]
        self.mean_actor_grad_norm[self.train_pos] = training_info["actor_grad_norm"]
        self.ent_coef_loss[self.train_pos] = training_info["ent_coef_loss"]
        self.ent_coef[self.train_pos] = training_info["ent_coef"]
        self.train_pos += 1

    def save(self, name: Optional[str] = None) -> None:
        """Saves the rl data to a pickle file.

        Args:
            name (Optional[str]): The name of the pickle file, without the .pkl extension.
        """
        # Don't save if there is no new data
        if self.n_updates_prev == self.n_updates:
            return

        self.rl_data = RLData(
            n_updates=self.n_updates,
            time_elapsed=self.time_elapsed,
            episodes=self.episodes,
            critic_loss=self.critic_loss[self.prev_train_pos : self.train_pos],
            actor_loss=self.actor_loss[self.prev_train_pos : self.train_pos],
            mean_actor_grad_norm=self.mean_actor_grad_norm[self.prev_train_pos : self.train_pos],
            ent_coeff_loss=self.ent_coef_loss[self.prev_train_pos : self.train_pos],
            ent_coeff=self.ent_coef[self.prev_train_pos : self.train_pos],
            non_optimal_solution_rate=self.non_optimal_solution_rate[self.prev_rollout_pos : self.rollout_pos],
            mean_episode_reward=self.mean_episode_rewards[self.prev_rollout_pos : self.rollout_pos],
            mean_episode_length=self.mean_episode_lengths[self.prev_rollout_pos : self.rollout_pos],
            success_rate=self.success_rates[self.prev_rollout_pos : self.rollout_pos],
        )
        self.prev_train_pos = self.train_pos
        self.prev_rollout_pos = self.rollout_pos

        self._save_as_pickle(name)

        if self.train_pos >= self.max_num_entries or self.rollout_pos >= self.max_num_entries:
            self._reset()

        self.n_updates_prev = self.n_updates

    def _reset(self) -> None:
        """Resets the buffer position pointers."""
        self.episodes = 0
        self.train_pos = 0
        self.rollout_pos = 0
        self.prev_rollout_pos = 0
        self.prev_train_pos = 0


if __name__ == "__main__":
    log_dir = Path.home() / "Desktop" / "machine_learning" / "rlmpc" / "sac_drl1"
    experiment_name = "sac_drl1"
    logger = Logger(experiment_name=experiment_name, log_dir=log_dir, max_num_entries=30_000)
    logger.load_from_pickle(f"{experiment_name}_training_stats")
