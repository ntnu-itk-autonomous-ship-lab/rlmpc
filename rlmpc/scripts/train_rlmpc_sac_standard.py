"""
    train_sac.py

    Summary:
        This script trains the RL agent using the standard SAC algorithm.

    Author: Trym Tengesdal
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import rlmpc.common.helper_functions as hf
import rlmpc.standard_sac as rlmpc_ssac
from colav_simulator.gym.environment import COLAVEnvironment
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback, evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def train_rlmpc_sac_standard(
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
    reset_num_timesteps: int = False,
) -> Tuple[rlmpc_ssac.SAC, bool]:
    """Train the RL agent using the standard SAC algorithm.

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
        save_stats_freq=1000,
        save_agent_model_freq=10000,
        log_freq=n_training_envs,
        max_num_env_episodes=2000,
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

    if load_model:
        model = rlmpc_ssac.SAC.load(
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
        print(
            f"Before learn: | Number of timesteps completed: {model.num_timesteps} | Number of episodes completed: {model.num_episodes}"
        )
        print(f"Loading model at {load_model_path}")
        model.load_replay_buffer(path=load_rb_path)
        print(f"Loading replay buffer at {load_rb_path}")
        model.set_env(training_env)
    else:
        model = rlmpc_ssac.SAC(env=training_env, **model_kwargs)

    if load_critics:
        print(f"Loading critic at {load_critics_path}")
        model.load_critics(path=load_critics_path, critic_arch=model_kwargs["policy_kwargs"]["critic_arch"])

    model.set_random_seed(seed)
    model.learn(
        total_timesteps=n_timesteps,
        log_interval=1,
        tb_log_name=experiment_name + f"_{iteration}",
        reset_num_timesteps=reset_num_timesteps,
        callback=CallbackList(callbacks=[eval_callback, stats_callback]),
        progress_bar=False,
    )

    if not model.vecenv_failed:
        training_env.close()
        eval_env.close()
    del training_env
    del eval_env
    return model, model.vecenv_failed
