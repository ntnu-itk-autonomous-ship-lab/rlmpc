"""
    callbacks.py

    Summary:
        Contains custom callback classes for the RL training process.

    Author: Trym Tengesdal
"""

import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from colav_simulator.gym.environment import COLAVEnvironment
from colav_simulator.gym.logger import Logger as COLAVEnvironmentLogger
from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.logger import Image as sb3_Image
from stable_baselines3.common.logger import Logger as sb3_Logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization


class CollectStatisticsCallback(BaseCallback):
    def __init__(
        self,
        env: gym.Env,
        log_dir: Path,
        experiment_name: str,
        save_stats_freq: int = 1000,
        save_agent_model_freq: int = 100,
        log_stats_freq: int = 100,
        verbose: int = 1,
    ):
        """Initializes the CollectStatisticsCallback class.

        Args:
            env (COLAVEnvironment): The environment to collect statistics from.
            save_stats_freq (int): Frequency to save statistics.
            record_agent_freq (int): Frequency to record the agent.
            log_dir (Path): The directory to save all experiment data to.
            model_dir (Path): The directory to save the models to.
            experiment_name (str): Name of the experiment.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(CollectStatisticsCallback, self).__init__(verbose)
        self.experiment_name = experiment_name
        self.save_stats_freq = save_stats_freq
        self.save_agent_freq = save_agent_model_freq
        self.log_stats_freq = log_stats_freq
        self.log_dir = log_dir
        self.model_save_path = log_dir / "models"
        self.envdata_save_path = log_dir / "envdata"
        self.n_episodes = 0
        self.vec_env = env

        self.envdata_logger: COLAVEnvironmentLogger = COLAVEnvironmentLogger(log_dir, experiment_name)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.model_save_path is not None:
            self.model_save_path.mkdir(parents=True, exist_ok=True)

        if self.envdata_save_path is not None:
            self.envdata_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Checking for both 'done' and 'dones' keywords because:
        # Some models use keyword 'done' (e.g.,: SAC, TD3, DQN, DDPG)
        # While some models use keyword 'dones' (e.g.,: A2C, PPO)
        done_array = np.array(
            self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones")
        )

        if self.num_timesteps % self.log_stats_freq == 0:
            self.envdata_logger(self.vec_env)

            self.logger.record("mpc/infeasible_solutions", self.model.actor.infeasible_solutions)
            frame = self.vec_env.render()
            if frame is not None:
                self.logger.record("env/frame", sb3_Image(frame, "HW"))

        if np.sum(done_array).item() > 0:
            self.n_episodes += np.sum(done_array).item()
            self.logger.record("time/episodes", self.n_episodes)

        if self.num_timesteps % self.save_agent_freq == 0:
            print("Saving agent after", self.num_timesteps, "timesteps")
            assert hasattr(
                self.model, "custom_save"
            ), "Model must have a custom_save method, i.e. be a custom SAC model with an MPC actor."
            self.model.custom_save(self.model_save_path / f"model_{self.experiment_name}_{self.num_timesteps}")

        return True


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    Args:
        - eval_env (Union[COLAVEnvironment, VecEnv]): The environment used for initialization
        - callback_on_new_best (Optional[BaseCallback], optional): Callback to trigger
            when there is a new best model according to the ``mean_reward``
        - callback_after_eval (Optional[BaseCallback], optional): Callback to trigger after every evaluation
        - n_eval_episodes (int, optional): The number of episodes to test the agent. Defaults to 5.
        - eval_freq (int, optional): Evaluate the agent every ``eval_freq`` call of the callback. Defaults to 10000.
        - log_path (Optional[str], optional): Path to a folder where the evaluations (``evaluations.npz``)
            will be saved. It will be updated at each evaluation. Defaults to None.
        - best_model_save_path (Optional[str], optional): Path to a folder where the best model
            according to performance on the eval env will be saved. Defaults to None.
        - deterministic (bool, optional): Whether the evaluation should use a stochastic or deterministic actions. Defaults to True.
        - render (bool, optional): Whether to render or not the environment during evaluation. Defaults to False.
        - verbose (int, optional): Verbosity level: 0 for no output, 1 for indicating information about evaluation results. Defaults to 1.
        - warn (bool, optional): Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
            wrapped with a Monitor wrapper). Defaults to True.
    """

    def __init__(
        self,
        eval_env: Union[COLAVEnvironment, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        Args:
            - locals_ (Dict[str, Any]): Local variables during rollout collection.
            - globals_ (Dict[str, Any]): Global variables during rollout collection.
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_mpc_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        Args:
            - locals_ (Dict[str, Any]): the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def evaluate_mpc_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Custom version of the evaluate_policy function from stable_baselines3.common.evaluation.py.

    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    Args:
        - model (type_aliases.PolicyPredictor): The RL agent you want to evaluate. This can be any object
            that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
            or policy (``BasePolicy``).
        - env (Union[gym.Env, VecEnv]): The gym environment or ``VecEnv`` environment.
        - n_eval_episodes (int): Number of episode to evaluate the agent
        - deterministic (bool): Whether to use deterministic or stochastic actions
        - render (bool): Whether to render the environment or not
        - callback (Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]): callback function to do additional checks,
            called after each step. Gets locals() and globals() passed as parameters.
        - reward_threshold (Optional[float]): Minimum expected reward per episode,
            this will raise an error if the performance is not met
        - return_episode_rewards (bool): If True, a list of rewards and episode lengths
            per episode will be returned instead of the mean.
        - warn (bool): If True (default), warns user about lack of a Monitor wrapper in the
            evaluation environment.

    Returns:
        - Union[Tuple[float, float], Tuple[List[float], List[int]]]: Mean reward per episode, std of reward per episode.
            Returns ([float], [int]) when ``return_episode_rewards`` is True, first
            list containing per-episode rewards and second containing per-episode lengths
            (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    assert n_envs == 1, "Only one environment is supported for now."
    print("Evaluating policy...")
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        if env.envs[0].unwrapped.time < 0.0001:
            model.policy.initialize_mpc_actor(env.envs[0])

        unnormalized_actions, normalized_actions, actor_infos = model.predict_with_mpc(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        states = actor_infos

        new_observations, rewards, dones, infos = env.step(normalized_actions)
        for actor_info, info in zip(actor_infos, infos):
            info.update({"actor_info": actor_info})
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    print(f"Done evaluating policy. | mean_reward: {mean_reward:.2f} | std_reward: {std_reward:.2f}")
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
