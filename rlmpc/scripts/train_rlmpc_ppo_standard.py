"""
    train_sac.py

    Summary:
        This script trains the RL agent using the standard SAC algorithm.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import stable_baselines3.ppo as ppo
from colav_simulator.gym.environment import COLAVEnvironment
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import rlmpc.common.helper_functions as hf
from rlmpc.common.callbacks import (
    CollectStatisticsCallback,
    EvalCallback,
    evaluate_policy,
)


def train_rlmpc_ppo_standard(
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
    load_model_path: str = "ppo_drl1_0_steps",
    seed: int = 0,
    iteration: int = 0,
    reset_num_timesteps: int = False,
) -> Tuple[ppo.PPO, bool]:
    """Train the RL agent using the standard PPO algorithm.

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
        load_model_path (str, optional): The model path for loading.
        seed (int, optional): The seed used for the enviroment, action spaces etc.
        iteration (int, optional): The iteration used for TB logging naming.
        reset_num_timesteps (bool, optional): Whether to reset model num timesteps before learning.

    Returns:
        Tuple[rlmpc_sac.SAC, bool]: The trained RL agent and whether the vector environment failed during a training step.
    """
    if n_training_envs == 1:
        training_env = Monitor(gym.make(id=env_id, **training_env_config))
    else:
        training_env = SubprocVecEnv(
            [
                hf.make_env(env_id=env_id, env_config=training_env_config, rank=i + 1, seed=seed)
                for i in range(n_training_envs)
            ]
        )

    stats_callback = CollectStatisticsCallback(
        env=training_env,
        log_dir=base_dir,
        model_dir=model_dir,
        experiment_name=experiment_name,
        save_stats_freq=2000,
        save_agent_model_freq=40000,
        log_freq=n_training_envs,
        max_num_env_episodes=1000,
        max_num_training_stats_entries=40000,
        minimal_logging=True,
        verbose=1,
    )
    if n_eval_envs == 1:
        eval_env = Monitor(gym.make(id=env_id, **eval_env_config))
    else:
        eval_env = SubprocVecEnv(
            [hf.make_env(env_id=env_id, env_config=eval_env_config, rank=i + 1, seed=seed) for i in range(n_eval_envs)]
        )
    eval_callback = EvalCallback(
        eval_env,
        log_path=base_dir / "eval_data",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        minimal_env_logging=False,
        experiment_name=experiment_name,
        record=True,
        render=True,
        verbose=1,
    )

    if load_model_path:
        model = ppo.PPO.load(
            load_model_path,
            env=training_env,
            learning_rate=model_kwargs["learning_rate"],
            device=model_kwargs["device"],
            tau=model_kwargs["tau"],
            batch_size=model_kwargs["batch_size"],
            n_epochs=model_kwargs["n_epochs"],
            gamma=model_kwargs["gamma"],
            gae_lambda=model_kwargs["gae_lambda"],
            n_steps=model_kwargs["n_steps"],
            tensorboard_log=base_dir / "logs",
            verbose=1,
        )
        print(f"Before learn: | Number of timesteps completed: {model.num_timesteps}")
        print(f"Loading model at {load_model_path}")
        model.set_env(training_env)
    else:
        model = ppo.PPO(env=training_env, **model_kwargs)

    model.set_random_seed(seed)
    model.learn(
        total_timesteps=n_timesteps,
        log_interval=3,
        tb_log_name=experiment_name + f"_{iteration}",
        reset_num_timesteps=reset_num_timesteps,
        callback=CallbackList(callbacks=[eval_callback, stats_callback]),
        progress_bar=False,
    )

    return model, False
