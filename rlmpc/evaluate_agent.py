"""
    train_sac.py

    Summary:
        This script trains the RL agent using the SAC algorithm.

    Author: Trym Tengesdal
"""

import argparse
import sys
import colav_simulator.gym.logger as csenv_logger
from pathlib import Path
from typing import Any, Dict, List, Tuple

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import numpy as np
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.policies as rlmpc_policies
import rlmpc.sac as rlmpc_sac
import stable_baselines3.sac as sb3_sac
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from rlmpc.common.callbacks import evaluate_policy
from rlmpc.networks.feature_extractors import CombinedExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def evaluate(
    model: Any,
    env_id: str,
    env_config: Dict[str, Any],
    log_dir: Path,
    experiment_name: str,
    n_eval_episodes: int = 5,
    record: bool = True,
) -> Tuple[float, float, List[float]]:
    """Train the RL agent using the SAC algorithm.

    Args:
        model (Any): The RL agent model to evaluate.
        env_id (str): The environment ID.
        env_config (Dict[str, Any]): The evaluation environment configuration.
        log_dir (Path): The log directory.
        experiment_name (str): The experiment name.
        n_eval_episodes (int, optional): The number of evaluation episodes.
        record (bool, optional): Whether to record the evaluation.

    Returns:
        Tuple[float, float, List[float]]: The mean reward, standard deviation of rewards, and rewards.
    """

    env = Monitor(gym.make(id=env_id, **env_config))
    env_data_logger = csenv_logger.Logger(log_dir=log_dir, experiment_name=experiment_name, save_freq=4, n_envs=1, max_num_logged_episodes=500)
    ep_rewards, ep_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        record=record,
        record_path=log_dir / "eval_data" / "eval_videos",
        record_name=experiment_name + "_eval",
        return_episode_rewards=True,
        env_data_logger=env_data_logger,
    )
    print(
        f"{experiment_name} evaluation results | mean_reward: {np.mean(ep_rewards)}, std_reward: {np.std(ep_rewards)}"
    )
    return np.mean(ep_rewards), np.std(ep_rewards), ep_rewards


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=rlmpc_sac.SAC, default=rlmpc_sac.SAC)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--record", type=bool, default=True)
    parser.add_argument("--model_kwargs", type=dict, default={})
    parser.add_argument("--base_dir", type=str, default=str(Path.home() / "Desktop/machine_learning/rlmpc"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_rlmpc_parameter_provider", type=bool, default=True)
    parser.add_argument("--experiment_name", type=str, default="sac_rlmpc1")
    args = parser.parse_args(args)
    args.base_dir = Path(args.base_dir)
    print("Provided args to SAC RLMPC eval:")
    print("".join(f"{k}={v}\n" for k, v in vars(args).items()))

    base_dir, log_dir, model_dir = hf.create_data_dirs(base_dir=args.base_dir, experiment_name=args.experiment_name)

    # map_size: [4000.0, 4000.0]
    # map_origin_enu: [-33524.0, 6572500.0]
    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            "perception_image_observation",
            "relative_tracking_observation",
            "navigation_3dof_state_observation",
            "tracking_observation",
            "time_observation",
        ]
    }

    scenario_names = [
        "rlmpc_scenario_ms_channel"
    ]  # ["rlmpc_scenario_ho", "rlmpc_scenario_cr_ss", "rlmpc_scenario_random_many_vessels"]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    eval_sim_config = cs_sim.Config.from_file(rl_dp.config / "eval_simulator.yaml")
    scen_gen_config = cs_sg.Config.from_file(rl_dp.config / "scenario_generator.yaml")
    env_id = "COLAVEnvironment-v0"
    eval_env_config = {
        "scenario_file_folder": test_scenario_folders,
        "scenario_generator_config": scen_gen_config,
        "max_number_of_episodes": 20,
        "simulator_config": eval_sim_config,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        "render_update_rate": 0.5,
        "render_mode": "rgb_array",
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "merge_loaded_scenario_episodes": True,
        "shuffle_loaded_scenario_data": True,
        "identifier": "eval_env",
        "seed": args.seed,
    }

    if isinstance(args.model_class, rlmpc_sac.SAC):
        mpc_config_file = rl_dp.config / "rlmpc.yaml"
        # actor_noise_std_dev = np.array([0.004, 0.004, 0.025])  # normalized std dev for the action space [x, y, speed]
        actor_noise_std_dev = np.array([0.004, 0.004])  # normalized std dev for the action space [course, speed]
        mpc_param_provider_kwargs = {
            "param_list": ["Q_p", "r_safe_do"],
            "hidden_sizes": [512, 512],
            "activation_fn": th.nn.ELU,
        }
        policy_kwargs = {
            "features_extractor_class": CombinedExtractor,
            "critic_arch": [512, 512],
            "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
            "mpc_config": mpc_config_file,
            "activation_fn": th.nn.ReLU,
            "std_init": actor_noise_std_dev,
            "disable_parameter_provider": True,
            "debug": False,
        }
        model_kwargs = {
            "policy": rlmpc_policies.SACPolicyWithMPC,
            "policy_kwargs": policy_kwargs,
            "device": "cpu",
            "ent_coef": "auto",
            "verbose": 1,
            "tensorboard_log": str(log_dir),
        }
        model = args.model_class(**model_kwargs)
        model.inplace_load(path=args.model_load_path)
        model.load_replay_buffer(path=args.model_load_path)
    else:
        model_kwargs = {
            "policy": "MultiInputPolicy",
            "device": "cpu",
            "ent_coef": "auto",
            "verbose": 1,
            "tensorboard_log": str(log_dir),
            "policy_kwargs": {
                "features_extractor_class": CombinedExtractor,
                "net_arch": [512, 512],
                "log_std_init": -5.0,
                "use_sde": True,
            },
            "replay_buffer_kwargs": {"handle_timeout_termination": True},
        }
        model = args.model_class.load(path=model_dir / args.experiment_name, **model_kwargs)

    mean_reward, std_reward, rewards = evaluate(
        model=model,
        env_id=env_id,
        env_config=eval_env_config,
        log_dir=base_dir,
        experiment_name=args.experi
        n_eval_episodes=args.n_eval_episodes,
        record=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
