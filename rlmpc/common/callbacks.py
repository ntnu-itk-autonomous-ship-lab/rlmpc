"""
    callbacks.py

    Summary:
        Contains custom callback classes for the RL training process.

    Author: Trym Tengesdal
"""

import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import colav_simulator.common.image_helper_methods as ihm
import colav_simulator.gym.environment as colav_env
import colav_simulator.gym.logger as colav_env_logger
import gymnasium as gym
import numpy as np
import rlmpc.common.logger as rlmpc_logger
import torch as th
import torchvision.transforms.v2 as transforms_v2
from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.logger import Image as sb3_Image
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
    VecVideoRecorder,
    is_vecenv_wrapped,
    sync_envs_normalization,
)

# I. How is the agent doing?


# Solve rate
# Wall time -> tells you how fast you can progress or try new ideas.
# Steps per second
# State/Action value function
# Policy entropy
# KL divergence
# Network weights/gradients/activations histograms -> Beware Dying ReLUs / Vanishing or Exploding gradients or activations.
# Policy/Value/Quality/... heads losses
# Average and standard deviation
# Minimum/Maximum value -> inspecting extremes can help spot a bug.
# Median
# What other categories or metrics you observe that help you fix or improve your policy?
class CollectStatisticsCallback(BaseCallback):
    def __init__(
        self,
        env: SubprocVecEnv,
        log_dir: Path,
        model_dir: Path,
        experiment_name: str,
        save_stats_freq: int = 1,
        save_agent_model_freq: int = 100,
        log_freq: int = 1,
        max_num_env_episodes: int = 5_000,
        max_num_training_stats_entries: int = 30_000,
        verbose: int = 1,
    ):
        """Initializes the CollectStatisticsCallback class.

        Args:
            env (SubprocVecEnv): The environment to collect statistics from.
            log_dir (Path): The directory to save all experiment data to.
            model_dir (Path): The directory to save the model to.
            experiment_name (str): Name of the experiment.
            save_stats_freq (int): Frequency to save statistics in number of steps.
            save_agent_model_freq (int): Frequency to save the agent model in number of steps.
            log_freq (int): Frequency to log statistics in number of steps.
            verbose (int, optional): Verbosity level. Defaults to 1.
            max_num_env_episodes (int, optional): Maximum number of episodes to log before save and reset. Defaults to 5_000.
            max_num_training_stats_entries (int, optional): Maximum number of training statistics entries to store. Defaults to 30_000.
        """
        super(CollectStatisticsCallback, self).__init__(verbose)
        self.experiment_name = experiment_name
        self.save_stats_freq = save_stats_freq
        self.save_agent_freq = save_agent_model_freq
        self.log_freq = log_freq
        self.vec_env = env
        self.n_updates_prev: int = 0
        self.n_episodes: int = 0
        self.last_ep_rew_mean: float = 0.0
        self.model_save_path = model_dir
        self.log_dir = log_dir
        self.num_envs = env.num_envs if isinstance(env, SubprocVecEnv) else 1
        self.env_data_logger: colav_env_logger.Logger = colav_env_logger.Logger(
            experiment_name,
            self.log_dir,
            n_envs=self.num_envs,
            max_num_logged_episodes=max_num_env_episodes,
        )
        self.training_stats_logger: rlmpc_logger.Logger = rlmpc_logger.Logger(
            experiment_name, self.log_dir, max_num_entries=max_num_training_stats_entries
        )

        self.img_transform = transforms_v2.Compose(
            [
                transforms_v2.ToDtype(th.float32, scale=True),
                transforms_v2.Resize((128, 128)),
            ]
        )
        self.display_transform = transforms_v2.Compose(
            [
                transforms_v2.ToDtype(th.uint8, scale=True),
            ]
        )
        self.prev_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.model_save_path is not None:
            self.model_save_path.mkdir(parents=True, exist_ok=True)

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.n_updates_prev = 0
        self.last_ep_rew_mean = 0.0

    def extract_training_info(self, model: "type_aliases.PolicyPredictor") -> Tuple[Dict[str, Any], bool]:
        """Extracts training information from the model.

        Args:
            model (type_aliases.PolicyPredictor): The model to extract training information from.

        Returns:
            Tuple[Dict[str, Any], bool]: A dictionary containing training information and a boolean indicating if the model was just trained.
        """
        if hasattr(model, "last_training_info"):
            return model.last_training_info, model.just_trained

        info = {
            "n_updates": model.logger.name_to_value["train/n_updates"],
            "time_elapsed": max((time.time_ns() - model.start_time) / 1e9, sys.float_info.epsilon),
            "batch_processing_time": 0.0,
            "ep_rew_mean": model.logger.name_to_value["train/ep_rew_mean"],
            "actor_loss": model.logger.name_to_value["train/actor_loss"],
            "critic_loss": model.logger.name_to_value["train/critic_loss"],
            "ent_coef_loss": model.logger.name_to_value["train/ent_coef_loss"],
            "ent_coef": model.logger.name_to_value["train/ent_coef"],
            "learning_rate": model.logger.name_to_value["train/learning_rate"],
            "actor_grad_norm": 0.0,
        }
        just_trained = False
        if info["n_updates"] > self.n_updates_prev:
            just_trained = True
        self.n_updates_prev = info["n_updates"]
        return info, just_trained

    def extract_rollout_info(self, model: "type_aliases.PolicyPredictor") -> Tuple[Dict[str, Any], bool]:
        if hasattr(model, "last_rollout_info"):
            return model.last_rollout_info, model.just_dumped_rollout_logs

        info = {
            "episodes": self.n_episodes,
            "mean_episode_reward": model.logger.name_to_value["rollout/ep_rew_mean"],
            "success_rate": model.logger.name_to_value["rollout/success_rate"],
            "mean_episode_length": model.logger.name_to_value["rollout/ep_len_mean"],
            "non_optimal_solution_rate": model.logger.name_to_value["rollout/non_optimal_solution_rate"],
        }
        just_dumped_rollout_logs = False
        if info["mean_episode_reward"] != self.last_ep_rew_mean:
            just_dumped_rollout_logs = True
        self.last_ep_rew_mean = info["mean_episode_reward"]

        return info, just_dumped_rollout_logs

    def _on_step(self) -> bool:
        # Checking for both 'done' and 'dones' keywords because:
        # Some models use keyword 'done' (e.g.,: SAC, TD3, DQN, DDPG)
        # While some models use keyword 'dones' (e.g.,: A2C, PPO)
        done_array = np.array(self.locals.get("dones"))
        if self.locals.get("done") is not None:
            done_array = np.array([self.locals.get("done")])

        infos = list(self.locals.get("infos"))
        if np.any(done_array):
            self.n_episodes += np.sum(done_array).item()

        if self.num_timesteps % self.log_freq == 0 or np.sum(done_array).item() > 0:
            for env_idx in range(self.num_envs):
                if np.any(done_array):  # only one element in done_array for SAC
                    infos[env_idx] = (
                        self.prev_infos[env_idx]
                        if isinstance(self.locals["env"], SubprocVecEnv)
                        else self.locals["env"].envs[env_idx].unwrapped.terminal_info
                    )
            self.env_data_logger(infos)

            last_rollout_info, just_dumped_rollout_logs = self.extract_rollout_info(self.model)
            if just_dumped_rollout_logs:
                self.training_stats_logger.update_rollout_metrics(last_rollout_info)
                if hasattr(self.model, "just_dumped_rollout_logs"):
                    self.model.just_dumped_rollout_logs = False

                    self.logger.record(
                        "mpc/infeasible_solution_percentage",
                        100.0 * self.model.actor.infeasible_solutions / (self.num_timesteps + 1),
                    )
                    mpc_params = self.model.actor.mpc.mpc_params
                    self.logger.record("mpc/r_safe_do", mpc_params.r_safe_do)
                    # self.logger.record("mpc/Q_p_path", mpc_params.Q_p[0, 0])
                    # self.logger.record("mpc/Q_p_speed", mpc_params.Q_p[1, 1])
                    # self.logger.record("mpc/Q_p_s", mpc_params.Q_p[2, 2])
                    # self.logger.record("mpc/K_app_course", mpc_params.K_app_course)
                    # self.logger.record("mpc/K_app_speed", mpc_params.K_app_speed)
                    # self.logger.record("mpc/w_colregs", mpc_params.w_colregs)
                    # self.logger.record("mpc/d_attenuation", mpc_params.d_attenuation)

            last_training_info, just_trained = self.extract_training_info(self.model)
            if just_trained:
                self.training_stats_logger.update_training_metrics(last_training_info)
                if hasattr(self.model, "just_trained"):
                    self.model.just_trained = False

            current_obs = self.model._current_obs if hasattr(self.model, "_current_obs") else self.model._last_obs
            if "PerceptionImageObservation" in current_obs:
                pimg = th.from_numpy(current_obs["PerceptionImageObservation"]).type(th.float32)
                pvae = self.model.critic.features_extractor.extractors["PerceptionImageObservation"]
                recon_frame = pvae.reconstruct(self.img_transform(pimg))

                self.logger.record("env/frame", sb3_Image(pimg[0, 0], "HW"), exclude=("log", "stdout"))
                self.logger.record("env/recon_frame", sb3_Image(recon_frame[0, 0], "HW"), exclude=("log", "stdout"))

        if self.num_timesteps % self.save_agent_freq == 0:
            print("Saving agent after", self.num_timesteps, "timesteps")
            # NMPC SAC model must have a custom_save method
            if hasattr(self.model, "custom_save"):
                self.model.custom_save(self.model_save_path / f"{self.experiment_name}_{self.num_timesteps}_steps")
            else:
                self.model.save(self.model_save_path / f"{self.experiment_name}_{self.num_timesteps}_steps")
            if hasattr(self.model, "save_replay_buffer"):
                self.model.save_replay_buffer(self.model_save_path / f"{self.experiment_name}_replay_buffer")

        if self.num_timesteps % self.save_stats_freq == 0:
            print("Saving training data after", self.num_timesteps, "timesteps")
            self.env_data_logger.save_as_pickle(f"{self.experiment_name}_env_training_data")
            self.training_stats_logger.save(f"{self.experiment_name}_training_stats")

        self.prev_infos = infos
        return True


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    Args:
        - eval_env (Union[colav_env.COLAVEnvironment, VecEnv]): The environment used for initialization
        - eval_model (Optional[BaseAlgorithm], optional): The model to evaluate. If not specified, uses the model associated with the callback.
        - callback_on_new_best (Optional[BaseCallback], optional): Callback to trigger
            when there is a new best model according to the ``mean_reward``
        - callback_after_eval (Optional[BaseCallback], optional): Callback to trigger after every evaluation
        - n_eval_episodes (int, optional): The number of episodes to test the agent. Defaults to 5.
        - eval_freq (int, optional): Evaluate the agent every ``eval_freq`` call of the callback. Defaults to 10000.
        - log_path (Optional[str], optional): Path to a folder where the evaluations (``evaluations.npz``)
            will be saved. It will be updated at each evaluation. Defaults to None.
        - best_model_save_path (Optional[str], optional): Path to a folder where the best model
            according to performance on the eval env will be saved. Defaults to None.
        - video_save_path (Optional[str], optional): Path to a folder where videos of the agent will be saved. Defaults to None.
        - deterministic (bool, optional): Whether the evaluation should use a stochastic or deterministic actions. Defaults to True.
        - render (bool, optional): Whether to render or not the environment during evaluation. Defaults to False.
        - verbose (int, optional): Verbosity level: 0 for no output, 1 for indicating information about evaluation results. Defaults to 1.
        - warn (bool, optional): Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
            wrapped with a Monitor wrapper). Defaults to True.
    """

    def __init__(
        self,
        eval_env: Union[colav_env.COLAVEnvironment, VecEnv],
        log_path: Path,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        experiment_name: str = "eval",
        record: bool = False,
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
        self.render = render
        self.record = record
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.experiment_name = experiment_name
        self.eval_env = eval_env
        self.best_model_save_path = log_path
        self.log_path = log_path
        self.video_save_path = log_path / "eval_videos"
        self.num_envs = eval_env.num_envs if isinstance(eval_env, SubprocVecEnv) else 1

        self.env_data_logger: colav_env_logger.Logger = colav_env_logger.Logger(
            experiment_name=experiment_name + "_env_data",
            log_dir=log_path,
            n_envs=self.num_envs,
            max_num_logged_episodes=100,
        )

        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        # if not isinstance(self.training_env, type(self.eval_env)):
        #     warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        if not self.log_path.exists():
            self.log_path.mkdir(parents=True, exist_ok=True)

        if not self.video_save_path.exists():
            self.video_save_path.mkdir(parents=True, exist_ok=True)
        else:
            # Clean the video folder
            for file in self.video_save_path.iterdir():
                file.unlink()
        if not self.best_model_save_path.exists():
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)

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

            if hasattr(self.model.actor, "mpc"):
                self.model.actor.mpc.close_enc_display()

            print(f"Evaluating policy after {self.num_timesteps} timesteps...")
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                return_episode_rewards=True,
                warn=self.warn,
                record=self.record,
                record_path=self.video_save_path,
                record_name=f"eval_{self.experiment_name}_{self.num_timesteps}",
                callback=self._log_success_callback,
                env_data_logger=self.env_data_logger,
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
                    self.log_path / f"eval_{self.experiment_name}_{self.num_timesteps}.npz",
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

                self.env_data_logger.save_as_pickle(f"{self.experiment_name}_env_eval_data")

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
                    if hasattr(self.model, "custom_save"):
                        self.model.custom_save(Path(self.best_model_save_path / "best_model_eval"))
                    else:
                        self.model.save(Path(self.best_model_save_path / "best_model_eval"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            print(f"Done evaluating policy. | mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        Args:
            - locals_ (Dict[str, Any]): the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 5,
    render: bool = True,
    reward_threshold: Optional[float] = None,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    return_episode_rewards: bool = True,
    warn: bool = True,
    record: bool = False,
    record_path: Optional[Path] = None,
    record_name: str = "eval",
    record_type: str = "gif",
    env_data_logger: Optional[colav_env_logger.Logger] = None,
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
        - render (bool): Whether to render the environment or not
        - reward_threshold (Optional[float]): Minimum expected reward per episode,
            this will raise an error if the performance is not met
        - return_episode_rewards (bool): If True, a list of rewards and episode lengths
            per episode will be returned instead of the mean.
        - callback (Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]): Callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
        - warn (bool): If True (default), warns user about lack of a Monitor wrapper in the
            evaluation environment.
        - record (bool): If True, records the evaluation episodes.
        - record_path (Optional[Path]): Path to the folder where the videos will be recorded.
        - record_name (str): Name of the video.
        - record_type (str): Type of the video. Can be 'mp4' or 'gif'.
        - env_data_logger (Optional[colav_env_logger.Logger]): Logger for environment data.

    Returns:
        - Union[Tuple[float, float], Tuple[List[float]]: Mean reward per episode, std of reward per episode.
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
    episode_rewards = []
    episode_lengths = []
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    if record:
        assert record_path is not None, "record_path must be provided if record is True."
        if not record_path.exists():
            record_path.mkdir(parents=True, exist_ok=True)

        if record_type == "mp4":
            env = VecVideoRecorder(
                env, str(record_path), name_prefix=record_name, record_video_trigger=lambda x: x == 0
            )

    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    if env_data_logger is not None:
        env_data_logger.reset_data_structures(env_idx=0)
    is_mpc_policy = hasattr(model.policy.actor, "mpc")
    frames = []
    while (episode_counts < episode_count_targets).any():
        if env.envs[0].unwrapped.time < 0.0001 and is_mpc_policy:
            states = None
            model.policy.initialize_mpc_actor(env.envs[0], evaluate=True)
            last_actor_info = [{} for _ in range(n_envs)]

        if is_mpc_policy:
            _, normalized_actions, actor_infos = model.predict_with_mpc(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=True,
            )
            states = actor_infos
            actions = normalized_actions
            # For plotting the predicted trajectory
            for env_idx in range(env.num_envs):
                env.envs[env_idx].unwrapped.ownship.set_remote_actor_predicted_trajectory(
                    actor_infos[env_idx]["trajectory"]
                )
                env.envs[env_idx].unwrapped.ownship.set_colav_data(actor_infos[env_idx])
        else:
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=True,
            )
            actor_infos = [{} for _ in range(n_envs)]

        new_observations, rewards, dones, infos = env.step(actions)
        for actor_info, info in zip(actor_infos, infos):
            info.update({"actor_info": actor_info})

        if env_data_logger is not None:
            env_data_logger(infos)

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
                    if env_data_logger is not None:
                        env_data_logger.save_as_pickle(f"{record_name}_env_data")

                    if is_mpc_policy:
                        actor_infos[i] = {}
                        model.actor.mpc.close_enc_display()
                        env.envs[i].unwrapped.terminal_info.update({"actor_info": last_actor_info[i]})
                        last_actor_info[i] = {}
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
            frame = env.render()
            frames.append(frame)

    env.close()

    if record_type == "gif":
        ihm.save_frames_as_gif(frames, record_path / f"{record_name}.gif", verbose=True)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
