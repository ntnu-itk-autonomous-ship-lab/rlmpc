"""
    train_sac.py

    Summary:
        This script trains the RL agent using the SAC algorithm.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import matplotlib.pyplot as plt
from colav_simulator.gym.environment import COLAVEnvironment
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback, evaluate_policy
from stable_baselines3 import SAC as sb3_SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def train_sac(
    model_kwargs: Dict[str, Any],
    n_timesteps: int,
    env_id: str,
    training_env_config: Dict[str, Any],
    n_training_envs: int,
    eval_env_config: Dict[str, Any],
    base_dir: Path,
    log_dir: Path,
    model_dir: Path,
    experiment_name: str,
    load_model: bool = True,
    load_model_name: str = "sac_drl1_0_steps",
    load_rb_name: str = "sac_drl1_replay_buffer",
    seed: int = 0,
    iteration: int = 0,
) -> sb3_SAC:
    """Train the RL agent using the SAC algorithm.

    Args:
        model_kwargs (Dict[str, Any]): The RL agent model keyword arguments.
        n_timesteps (int): The number of timesteps to train the RL agent.
        env_id (str): The environment ID.
        training_env_config (Dict[str, Any]): The training environment configuration.
        n_training_envs (int): The number of training environments.
        eval_env_config (Dict[str, Any]): The evaluation environment configuration.
        base_dir (Path): The base directory.
        log_dir (Path): The log directory.
        model_dir (Path): The model directory.
        experiment_name (str): The experiment name.
        load_model (bool, optional): Whether to load the model. Defaults to True.
        load_model_name (str, optional): The model name to load. Defaults to "sac_drl1_0_steps".
        load_rb_name (str, optional): The replay buffer name to load. Defaults to "sac_drl1_replay_buffer".
        seed (int, optional): The seed. Defaults to 0.
        iteration (int, optional): The iteration used for TB logging naming. Defaults to 0.

    Returns:
        sb3_SAC: The trained RL agent model.
    """
    # training_vec_env = SubprocVecEnv([make_env(env_id, training_env_config, i + 1) for i in range(num_cpu)])
    training_vec_env = make_vec_env(
        env_id=env_id,
        env_kwargs=training_env_config,
        n_envs=n_training_envs,
        monitor_dir=str(log_dir),
        vec_env_cls=SubprocVecEnv,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=model_dir,
        name_prefix=experiment_name,
        verbose=1,
        save_replay_buffer=True,
    )
    # stats_callback = CollectStatisticsCallback(
    #     env=training_vec_env,
    #     log_dir=base_dir,
    #     model_dir=model_dir,
    #     experiment_name=args.experiment_name,
    #     save_stats_freq=1000,
    #     save_agent_model_freq=50000,
    #     log_freq=10,
    #     max_num_env_episodes=1000,
    #     max_num_training_stats_entries=30000,
    #     verbose=1,
    # )
    eval_env = Monitor(gym.make(id=env_id, **eval_env_config))
    eval_callback = EvalCallback(
        eval_env,
        log_path=base_dir / "eval_data",
        eval_freq=30000,
        n_eval_episodes=1,
        # callback_after_eval=stop_train_callback,
        experiment_name=experiment_name,
        record=True,
        render=True,
        verbose=1,
    )

    if load_model:
        model = sb3_SAC.load(path=model_dir / load_model_name, env=training_vec_env, **model_kwargs)
        model.load_replay_buffer(path=model_dir / load_rb_name)
    else:
        model = sb3_SAC(env=training_vec_env, **model_kwargs)

    model.set_random_seed(seed)
    model.learn(
        total_timesteps=n_timesteps,
        log_interval=2,
        tb_log_name=experiment_name + f"_{iteration}",
        reset_num_timesteps=True,
        callback=CallbackList(callbacks=[eval_callback, checkpoint_callback]),
        progress_bar=True,
    )
    return model
