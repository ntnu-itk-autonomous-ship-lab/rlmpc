"""
    off_policy_algorithm.py

    Summary:
        Contains functionality for off-policy RL algorithms. Heavily inspired/boiled from stable-baselines3.


    Author: Trym Tengesdal
"""

import sys
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import rlmpc.buffers as rlmpc_buffers
import stable_baselines3.common.callbacks as sb3_callbacks
import stable_baselines3.common.logger as sb3_logger
import stable_baselines3.common.monitor as sb3_monitor
import stable_baselines3.common.noise as sb3_noise
import stable_baselines3.common.policies as sb3_policies
import stable_baselines3.common.save_util as sb3_sutils
import stable_baselines3.common.type_aliases as sb3_types
import stable_baselines3.common.utils as sb3_utils
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")


class OffPolicyAlgorithm(BaseAlgorithm):
    """Base class for an Off-Policy RL algorithm

    Args:
        - policy (Any): The policy model to use (RLMPC, MlpPolicy, CnnPolicy, ...)
        - env (VecEnv): The environment to learn from
            (if registered in Gym, can be str. Can be None for loading trained models)
        - learning_rate (float): learning rate for the optimizer,
            it can be a function of the current progress remaining (from 1 to 0)
        - buffer_size (int): size of the replay buffer
        - learning_starts (int): how many steps of the model to collect transitions for before learning starts
        - batch_size (int): Minibatch size for each gradient update
        - tau (float): the soft update coefficient ("Polyak update", between 0 and 1)
        - gamma (float): the discount factor
        - tensorboard_log (Optional[str]): the log location for tensorboard (if None, no logging)
        - train_freq (Union[int, Tuple[int, str]]): Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
            like ``(5, "step")`` or ``(2, "episode")``.
        - gradient_steps (int): How many gradient steps to do after each rollout (see ``train_freq``)
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout.
        - action_noise (Optional[ActionNoise]): the action noise type (None by default), this can help
            for hard exploration problem. Cf common.noise for the different action noise type.
        - verbose (int): Verbosity level: 0 no output, 1 info, 2 debug
        - device (Union[th.device, str]): Device on which the code should run.
        - support_multi_env (bool): Whether the algorithm supports training
            with multiple environments (as in A2C)
        - monitor_wrapper (bool): When creating an environment, whether to wrap it
            or not in a Monitor wrapper.
        - seed (Optional[int]): Seed for the pseudo random generators
    """

    def __init__(
        self,
        policy: Any,
        env: VecEnv,
        learning_rate: float,
        buffer_size: int = 1e6,
        learning_starts: int = 0,
        batch_size: int = 64,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (50, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[sb3_noise.ActionNoise] = None,
        replay_buffer_class: Optional[Type[rlmpc_buffers.ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "cuda" if th.cuda.is_available() else "cpu",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.policy_kwargs.update(
            {"observation_type": env.unwrapped.observation_type, "action_type": env.unwrapped.action_type}
        )
        self._last_actor_info: List[Dict[str, Any]] = [{} for _ in range(self.n_envs)]
        self.learning_rate = learning_rate
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.learning_starts: int = learning_starts
        self.tau: float = tau
        self.gamma: float = gamma
        self.gradient_steps: int = gradient_steps
        self.action_noise = action_noise
        self.verbose: int = verbose
        self.replay_buffer: Optional[rlmpc_buffers.ReplayBuffer] = None
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs or {}
        self.train_freq: sb3_types.TrainFreq = train_freq
        self._convert_train_freq()
        self._num_timesteps: int = 0
        self._episode_num: int = 0

        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup

    @abstractmethod
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)

        Args:
            - gradient_steps (int): number of gradient steps
            - batch_size (int): size of the batch to sample from the replay buffer
        """

    def predict_with_mpc(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...] | Dict] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        Args:
            - observation (Union[np.ndarray, Dict[str, np.ndarray]]): the input observation
            - state (Optional[Tuple[np.ndarray, ...] | Dict]): The last hidden states or MPC internal info (prev solution, etc.)
            - episode_start (Optional[np.ndarray]): The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            - deterministic (bool): Whether or not to return deterministic actions.

        Returns:
            - Tuple[np.ndarray, Dict]: the MPC's action and the corresponding MPC internal state (solution info etc.)
        """
        return self.policy.predict_with_mpc(observation, state, episode_start, deterministic)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[sb3_noise.ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        Args:
            - learning_starts (int): Number of steps before learning for the warm-up phase.
            - action_noise (Optional[ActionNoise]): Action noise that will be used for exploration
                Required for deterministic policy (e.g. TD3). This can also be used
                in addition to the stochastic policy for SAC.
            - n_envs (int): Number of parallel environments.

        Returns:
            - Tuple[np.ndarray, np.ndarray, List[Dict]]: scaled action(s) to take in the environment, and unscaled action that will be stored in the replay buffer. Also, information on the MPC internal states (solution info etc.) is returned
        """
        # Note: when using continuous actions,
        # we assume that the policy uses tanh to scale the action
        # We use non-deterministic action in the case of SAC, for TD3, it does not matter
        unnormalized_actions, normalized_actions, actor_infos = self.predict_with_mpc(
            observation=self._last_obs, state=self._last_actor_info, deterministic=False
        )

        # Add noise to the action (improve exploration)
        if action_noise is not None:
            normalized_actions = np.clip(normalized_actions + action_noise(), -1.0, 1.0)

        # We store the normalized action in the buffer
        actions = normalized_actions
        batch_size = normalized_actions.shape[0]
        for b in range(batch_size):
            unnormalized_actions[b] = self.policy.action_type.unnormalize(normalized_actions[b])
        buffer_actions = unnormalized_actions
        return actions, buffer_actions, actor_infos

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

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = rlmpc_buffers.DictReplayBuffer
            else:
                self.replay_buffer_class = rlmpc_buffers.ReplayBuffer

        if self.replay_buffer is None:
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=False,
                **replay_buffer_kwargs,  # pytype:disable=wrong-keyword-args
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )

        # self.policy = self.policy.to(self.device) Not applicable for RLMPC
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

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
        assert isinstance(
            self.replay_buffer, rlmpc_buffers.ReplayBuffer
        ), "The replay buffer must be a ReplayBuffer class"

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: str = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, sb3_callbacks.BaseCallback]:
        """
        Sets the self.start_time attribute and updates self._total_timesteps

        Args:
            - total_timesteps (int): The total number of samples (env steps) to train on
            - callback (str): name of the callback to use (str)
            - reset_num_timesteps (bool): Whether to reset or not the num timesteps attribute
            - tb_log_name (str): the name of the run for tensorboard log
            - progress_bar (bool): Whether to show or not a progress bar

        Returns:
            - Tuple[int, BaseCallback]: The updated total number of timesteps and the callback
        """
        replay_buffer = self.replay_buffer
        truncate_last_traj = (
            reset_num_timesteps and replay_buffer is not None and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: sb3_callbacks.BaseCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self._num_timesteps < total_timesteps:
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

            if self._num_timesteps > 0 and self._num_timesteps > self.learning_starts:
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
        callback: sb3_callbacks.BaseCallback,
        train_freq: sb3_types.TrainFreq,
        replay_buffer: rlmpc_buffers.ReplayBuffer,
        action_noise: Optional[np.ndarray] = None,
        learning_starts: int = 10,
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

        assert isinstance(env, VecEnv), "You must pass a VecEnv. "
        assert env.num_envs == 1, "OffPolicyAlgorithm for RLMPC only support single environment for now."
        assert self.train_freq.frequency > 0, "Should at least collect one step or episode."

        # Vectorize action noise if needed
        if (
            action_noise is not None
            and env.num_envs > 1
            and not isinstance(action_noise, sb3_noise.VectorizedActionNoise)
        ):
            action_noise = sb3_noise.VectorizedActionNoise(action_noise, env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while self._should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if env.envs[0].unwrapped.time == 0.0:
                self.policy.initialize_mpc_actor(env.envs[0])

            actions, buffer_actions, actor_infos = self._sample_action(learning_starts, action_noise, env.num_envs)

            new_obs, rewards, dones, infos = env.step(actions)
            for idx, info in enumerate(infos):
                info.update({"actor_info": actor_infos[idx]})

            self._num_timesteps += env.num_envs
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
            self._store_transition(replay_buffer, actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self._num_timesteps, self._total_timesteps)

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

    def _store_transition(
        self,
        replay_buffer: rlmpc_buffers.ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        Args:
            - replay_buffer (ReplayBuffer): Replay buffer object where to store the transition.
            - buffer_action (np.ndarray): normalized action
            - new_obs (Union[np.ndarray, Dict[str, np.ndarray]]): next observation in the current episode
                or first observation of the episode (when dones is True)
            - reward (np.ndarray): reward for the current transition
            - dones (np.ndarray): Termination signals
            - infos (List[Dict[str, Any]]): List of additional information about the transition.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                infos[i][
                    "actor_info"
                ] = {}  # Clear the actor (state) info for the terminal observation, as the actor state should
                # not be used in the next episode
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        self._last_actor_info = [info.get("actor_info") for info in infos]
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    @property
    def logger(self) -> sb3_logger.Logger:
        """Getter for the logger object."""
        return self._logger

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        Args:
            - num_timesteps (int): current number of timesteps
            - total_timesteps (int): total number of timesteps
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _should_collect_more_steps(
        self,
        train_freq: int,
        num_collected_steps: int,
        num_collected_episodes: int,
    ) -> bool:
        """
        Helper used in off-policy algorithms to determine the termination condition.

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

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean", sb3_utils.safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            )
            self.logger.record(
                "rollout/ep_len_mean", sb3_utils.safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
            )
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", sb3_utils.safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)
