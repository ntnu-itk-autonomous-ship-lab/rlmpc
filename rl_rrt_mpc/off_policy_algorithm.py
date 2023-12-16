"""
    off_policy_algorithm.py

    Summary:
        Contains functionality for off-policy RL algorithms. Heavily inspired/boiled from stable-baselines3.


    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import colav_simulator.gym.environment as cs_env
import numpy as np
import rl_rrt_mpc.replay_buffer as rb
import stable_baselines3.common.save_util as sb3_sutils
import stable_baselines3.common.type_aliases as sb3_types
import stable_baselines3.common.utils as sb3_utils
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.noise import (
    ActionNoise,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    VectorizedActionNoise,
)
from stable_baselines3.common.vec_env import VecEnv


class OffPolicyAlgorithm(ABC):
    """Base class for an Off-Policy RL algorithm"""

    def __init__(
        self,
        policy: Any,
        env: cs_env.COLAVEnvironment,
        learning_rate: float,
        buffer_size: int = 1e6,
        learning_starts: int = 0,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[np.ndarray] = None,
    ) -> None:
        self.policy: Any = policy
        self.env: cs_env.COLAVEnvironment = env
        self.learning_rate: float = learning_rate
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.learning_starts: int = learning_starts
        self.tau: float = tau
        self.gamma: float = gamma
        self.gradient_steps: int = gradient_steps
        self.action_noise: np.ndarray = action_noise
        self.verbose: bool = False
        self.replay_buffer: Optional[rb.ReplayBuffer] = None
        self.train_freq: sb3_types.TrainFreq = train_freq
        self._convert_train_freq()

    @abstractmethod
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)

        Args:
            - gradient_steps (int): number of gradient steps
            - batch_size (int): size of the batch to sample from the replay buffer
        """

    @abstractmethod
    def sample_action(
        self,
        action_noise: Optional[np.ndarray] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample action from the policy

        Args:
            - action_noise (Optional[np.ndarray]): Action noise that will be used for exploration
            - n_envs (int): Number of parallel environments
        """

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, sb3_types.TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], sb3_types.TrainFrequencyUnit(train_freq[1]))
            except ValueError as e:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
                ) from e

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = sb3_types.TrainFreq(*train_freq)

    def save_replay_buffer(self, path: Path) -> None:
        """
        Save the replay buffer as a pickle file.

        Args:
            - path (Path): path where the replay buffer should be saved
        """
        sb3_sutils.save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Path,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        Args:
            - path (Path): path where the replay buffer should be loaded from
        """
        self.replay_buffer = sb3_sutils.load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, rb.ReplayBuffer), "The replay buffer must be a ReplayBuffer class"

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: str = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int]:
        """
        Sets the self.start_time attribute and updates self._total_timesteps

        Args:
            - total_timesteps (int): The total number of samples (env steps) to train on
            - callback (str): name of the callback to use (str)
            - reset_num_timesteps (bool): Whether to reset or not the num timesteps attribute
            - tb_log_name (str): the name of the run for tensorboard log
            - progress_bar (bool): Whether to show or not a progress bar
        """
        replay_buffer = self.replay_buffer

    def learn(
        self,
        total_timesteps: int,
        callback: str = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> OffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: sb3_types.TrainFreq,
        replay_buffer: rb.ReplayBuffer,
        action_noise: Optional[np.ndarray] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> sb3_types.RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        Args:
            - env (VecEnv): The training environment
            - callback (BaseCallback): Callback that will be called at each step
                (and at the beginning and end of the rollout)
            - train_freq (TrainFreq): How much experience to collect
                by doing rollouts of current policy.
                Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
                or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            - replay_buffer (ReplayBuffer): Buffer to store the experiences
            - action_noise (Optional[np.ndarray]): Action noise that will be used for exploration
            - learning_starts (int): Number of steps before learning for the warm-up phase.
            - log_interval (Optional[int]): Log data every ``log_interval`` episodes
        """
        # Switch to eval mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert self.train_freq > 0, "Should at least collect one step or episode."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while self._should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return sb3_types.RolloutReturn(
                    num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return sb3_types.RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _should_collect_more_steps(
        self,
        train_freq: int,
        num_collected_steps: int,
        num_collected_episodes: int,
    ) -> bool:
        """
        Helper used in off-policy algorithms
        to determine the termination condition.

        Args:
            - train_freq (int): How much experience should be collected before updating the policy.
            - num_collected_steps (int): The number of already collected steps.
            - num_collected_episodes (int): The number of already collected episodes.

        Returns:
            - bool: Whether to continue or not collecting experience
            by doing rollouts of the current policy.
        """
        if train_freq.unit == sb3_types.TrainFrequencyUnit.STEP:
            return num_collected_steps < train_freq.frequency

        elif train_freq.unit == sb3_types.TrainFrequencyUnit.EPISODE:
            return num_collected_episodes < train_freq.frequency

        else:
            raise ValueError(
                "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
                f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
            )
