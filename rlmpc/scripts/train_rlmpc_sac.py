"""
    train_sac.py

    Summary:
        This script trains the RL agent using the SAC algorithm.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import rlmpc.common.helper_functions as hf
import rlmpc.sac as rlmpc_sac
from colav_simulator.gym.environment import COLAVEnvironment
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback, evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def train_rlmpc_sac(
    model_kwargs: Dict[str, Any],
    n_timesteps: int,
    env_id: str,
    training_env_config: Dict[str, Any],
    n_training_envs: int,
    eval_env_config: Dict[str, Any],
    n_eval_envs: int,
    n_eval_episodes: int,
    eval_freq: int,
    base_dir: Path,
    model_dir: Path,
    experiment_name: str,
    load_critics: bool = False,
    load_critics_path: str = "sac_drl1_critic",
    load_model: bool = True,
    load_model_path: str = "sac_drl1_0_steps",
    load_rb_path: str = "sac_drl1_replay_buffer",
    seed: int = 0,
    iteration: int = 0,
) -> Tuple[rlmpc_sac.SAC, bool]:
    """Train the RL agent using the SAC algorithm.

    Args:
        model_kwargs (Dict[str, Any]): The RL agent model keyword arguments.
        n_timesteps (int): The number of timesteps to train the RL agent.
        env_id (str): The environment ID.
        training_env_config (Dict[str, Any]): The training environment configuration.
        n_training_envs (int): The number of training environments.
        eval_env_config (Dict[str, Any]): The evaluation environment configuration.
        n_eval_envs (int): The number of evaluation environments.
        n_eval_episodes (int): The number of evaluation episodes.
        eval_freq (int): Evaluation callback frequency in number of steps.
        base_dir (Path): The base directory.
        model_dir (Path): The model directory.
        experiment_name (str): The experiment name.
        load_critics (bool, optional): Whether to load the critics.
        load_critics_path (str, optional): The critics path for loading.
        load_model (bool, optional): Whether to load the model with SAC classmethod
        load_model_path (str, optional): The model path for loading.
        load_rb_path (str, optional): The replay buffer path
        seed (int, optional): The seed.
        iteration (int, optional): The iteration used for TB logging naming.

    Returns:
        Tuple[rlmpc_sac.SAC, bool]: The trained RL agent and whether the vector environment failed during a training step.
    """
    if n_training_envs == 1:
        training_env = Monitor(gym.make(id=env_id, **training_env_config))
    else:
        training_env = SubprocVecEnv([hf.make_env(env_id, training_env_config, i + 1) for i in range(n_training_envs)])

    stats_callback = CollectStatisticsCallback(
        env=training_env,
        log_dir=base_dir,
        model_dir=model_dir,
        experiment_name=experiment_name,
        save_stats_freq=20,
        save_agent_model_freq=1000,
        log_freq=5,
        max_num_env_episodes=1000,
        max_num_training_stats_entries=40000,
        verbose=1,
    )
    if n_eval_envs == 1:
        eval_env = Monitor(gym.make(id=env_id, **eval_env_config))
    else:
        eval_env = SubprocVecEnv([hf.make_env(env_id, eval_env_config, i + 1) for i in range(n_eval_envs)])
    eval_callback = EvalCallback(
        eval_env,
        log_path=base_dir / "eval_data",
        eval_freq=max(eval_freq // n_training_envs, 1),
        n_eval_episodes=n_eval_episodes,
        # callback_after_eval=stop_train_callback,
        experiment_name=experiment_name,
        record=True,
        render=True,
        verbose=1,
    )

    if load_model:
        model = rlmpc_sac.SAC.load(
            load_model_path,
            env=training_env,
            learning_rate=model_kwargs["learning_rate"],
            device=model_kwargs["device"],
            tau=model_kwargs["tau"],
            buffer_size=model_kwargs["buffer_size"],
            batch_size=model_kwargs["batch_size"],
            gradient_steps=model_kwargs["gradient_steps"],
            train_freq=model_kwargs["train_freq"],
            replay_buffer_kwargs=model_kwargs["replay_buffer_kwargs"],
            tensorboard_log=base_dir / "logs",
            verbose=1,
        )
        print(f"Loading model at {load_model_path}")
        model.load_replay_buffer(path=load_rb_path)
        print(f"Loading replay buffer at {load_rb_path}")
        model.set_env(training_env)
    else:
        model = rlmpc_sac.SAC(env=training_env, **model_kwargs)

    if load_critics:
        print(f"Loading critic at {load_critics_path}")
        model.load_critics(path=load_critics_path, critic_arch=model_kwargs["policy_kwargs"]["critic_arch"])

    model.set_random_seed(seed)
    model.learn(
        total_timesteps=n_timesteps,
        log_interval=1,
        tb_log_name=experiment_name + f"_{iteration}",
        reset_num_timesteps=True,
        callback=CallbackList(callbacks=[eval_callback, stats_callback]),
        progress_bar=True,
    )

    if not model.vecenv_failed:
        training_env.close()
        eval_env.close()
    del training_env
    del eval_env
    return model, model.vecenv_failed
