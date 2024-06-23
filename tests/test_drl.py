"""Test a standard SAC DRL agent on the COLAV environment.
"""

from pathlib import Path
from typing import Tuple

import colav_simulator.common.paths as dp
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
from colav_simulator.gym.environment import COLAVEnvironment
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback
from rlmpc.networks.feature_extractors import CombinedExtractor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
import platform


def create_data_dirs(experiment_name: str) -> Tuple[Path, Path, Path, Path]:
    if platform.system() == "Linux":
        base_dir = Path("/home/doctor/Desktop/machine_learning/rlmpc/")
    elif platform.system() == "Darwin":
        base_dir = Path("/Users/trtengesdal/Desktop/machine_learning/rlmpc/")
    base_dir = base_dir / experiment_name
    log_dir = base_dir / "logs"
    model_dir = base_dir / "models"
    best_model_dir = model_dir / "best_model"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    else:
        # remove folders in log_dir
        for file in log_dir.iterdir():
            if file.is_dir():
                for f in file.iterdir():
                    f.unlink()
                file.rmdir()
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    if not best_model_dir.exists():
        best_model_dir.mkdir(parents=True)
    return base_dir, log_dir, model_dir, best_model_dir


if __name__ == "__main__":
    config_file = dp.scenarios / "rl_scenario.yaml"
    experiment_name = "sac_drl1"
    base_dir, log_dir, model_dir, best_model_dir = create_data_dirs(experiment_name=experiment_name)

    scenario_names = [
        "rlmpc_scenario_ms_channel"
    ]  # ["rlmpc_scenario_ho", "rlmpc_scenario_cr_ss", "rlmpc_scenario_random_many_vessels"]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            "perception_image_observation",
            "relative_tracking_observation",
            "navigation_3dof_state_observation",
            "ground_truth_tracking_observation",
        ]
    }

    rewarder_config = rewards.Config.from_file(rl_dp.config / "rewarder.yaml")
    training_sim_config = cs_sim.Config.from_file(rl_dp.config / "training_simulator.yaml")
    eval_sim_config = cs_sim.Config.from_file(rl_dp.config / "eval_simulator.yaml")
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": [training_scenario_folders[0]],
        "merge_loaded_scenario_episodes": True,
        "max_number_of_episodes": 300,
        "simulator_config": training_sim_config,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        "rewarder_class": rewards.MPCRewarder,
        "rewarder_kwargs": {"config": rewarder_config},
        "render_update_rate": 1.0,
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": True,
        "show_loaded_scenario_data": False,
        "shuffle_loaded_scenario_data": True,
        "identifier": "training_env1",
        "seed": 0,
    }
    env = Monitor(gym.make(id=env_id, **env_config))

    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "arch": [258, 128],
    }
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=100000,
        batch_size=32,
        gradient_steps=1,
        train_freq=(8, "step"),
        device="cpu",
        tensorboard_log=str(log_dir),
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=100, log_interval=4, tb_log_name=experiment_name)
    print("done")
