"""
    train_sac.py

    Summary:
        This script trains the RL agent using the SAC algorithm.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
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
    n_eval_episodes: int,
    eval_freq: int,
    base_dir: Path,
    model_dir: Path,
    experiment_name: str,
    load_model: bool = True,
    load_model_path: str = "sac_drl1_0_steps",
    load_rb_path: str = "sac_drl1_replay_buffer",
    seed: int = 0,
    iteration: int = 0,
) -> rlmpc_sac.SAC:
    """Train the RL agent using the SAC algorithm.

    Args:
        model_kwargs (Dict[str, Any]): The RL agent model keyword arguments.
        n_timesteps (int): The number of timesteps to train the RL agent.
        env_id (str): The environment ID.
        training_env_config (Dict[str, Any]): The training environment configuration.
        n_training_envs (int): The number of training environments.
        eval_env_config (Dict[str, Any]): The evaluation environment configuration.
        n_eval_episodes (int): The number of evaluation episodes.
        eval_freq (int): Evaluation callback frequency in number of steps.
        base_dir (Path): The base directory.
        model_dir (Path): The model directory.
        experiment_name (str): The experiment name.
        load_model (bool, optional): Whether to load the model. Defaults to True.
        load_model_path (str, optional): The model path for loading.
        load_rb_path (str, optional): The replay buffer path
        seed (int, optional): The seed. Defaults to 0.
        iteration (int, optional): The iteration used for TB logging naming. Defaults to 0.

    Returns:
        rlmpc_sac.SAC: The trained RL agent.
    """
    if n_training_envs == 1:
        training_env = Monitor(gym.make(id=env_id, **training_env_config))
    else:
        training_env = make_vec_env(
            env_id=env_id,
            env_kwargs=training_env_config,
            n_envs=n_training_envs,
            monitor_dir=str(base_dir),
            vec_env_cls=SubprocVecEnv,
            seed=seed,
        )
    stats_callback = CollectStatisticsCallback(
        env=training_env,
        log_dir=base_dir,
        model_dir=model_dir,
        experiment_name=experiment_name,
        save_stats_freq=20,
        save_agent_model_freq=500,
        log_freq=5,
        max_num_env_episodes=1000,
        max_num_training_stats_entries=40000,
        verbose=1,
    )
    eval_env = Monitor(gym.make(id=env_id, **eval_env_config))
    eval_callback = EvalCallback(
        eval_env,
        log_path=base_dir / "eval_data",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        # callback_after_eval=stop_train_callback,
        experiment_name=experiment_name,
        record=True,
        render=True,
        verbose=1,
    )
    model = rlmpc_sac.SAC(env=training_env, **model_kwargs)
    if load_model:
        model.inplace_load(path=load_model_path)
        model.load_replay_buffer(path=load_rb_path)
        model.set_env(training_env)

    model.set_random_seed(seed)
    model.learn(
        total_timesteps=n_timesteps,
        log_interval=2,
        tb_log_name=experiment_name + f"_{iteration}",
        reset_num_timesteps=True,
        callback=CallbackList(callbacks=[eval_callback, stats_callback]),
        progress_bar=True,
    )
    return model
