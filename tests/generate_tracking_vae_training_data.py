import os
from pathlib import Path
from sys import platform
from typing import Callable, Tuple

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from rlmpc.networks.tracking_vae.vae import VAE
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

# For macOS users, you might need to set the environment variable
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


def make_env(env_id: str, env_config: dict, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    Args:
        env_id: (str) the environment ID
        env_config: (dict) the environment config
        rank: (int) index of the subprocess
        seed: (int) the inital seed for RNG

    Returns:
        (Callable): a function that creates the environment
    """

    def _init():
        env = gym.make(env_id, **env_config)
        env.unwrapped.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if platform == "linux" or platform == "linux2":
    TRACKINGOBS_DATADIR: Path = Path("/home/doctor/Desktop/machine_learning/data/tracking_vae/")
elif platform == "darwin":
    TRACKINGOBS_DATADIR: Path = Path("/Users/trtengesdal/Desktop/machine_learning/data/tracking_vae/")


if __name__ == "__main__":
    scenario_names = [
        "rlmpc_scenario_ms_channel",
        "rlmpc_scenario_random_many_vessels",
    ]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    generate = False
    if generate:
        scenario_generator = cs_sg.ScenarioGenerator(config_file=rl_dp.config / "scenario_generator.yaml")
        for idx, name in enumerate(scenario_names):
            if idx == 1:
                continue

            scenario_generator.seed(idx + 1)
            _ = scenario_generator.generate(
                config_file=rl_dp.scenarios / (name + ".yaml"),
                new_load_of_map_data=False if idx == 0 else False,
                save_scenario=True,
                save_scenario_folder=rl_dp.scenarios / "training_data" / name,
                show_plots=True,
                episode_idx_save_offset=0,
                n_episodes=90,
                delete_existing_files=True,
            )

            scenario_generator.seed(idx + 1003)
            _ = scenario_generator.generate(
                config_file=rl_dp.scenarios / (name + ".yaml"),
                new_load_of_map_data=False,
                save_scenario=True,
                save_scenario_folder=rl_dp.scenarios / "test_data" / name,
                show_plots=True,
                episode_idx_save_offset=0,
                n_episodes=20,
                delete_existing_files=True,
            )

    # map_size: [4000.0, 4000.0]
    # map_origin_enu: [-33524.0, 6572500.0]
    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            "perception_image_observation",
            "relative_tracking_observation",
            "navigation_3dof_state_observation",
            "tracking_observation",
            "ground_truth_tracking_observation",
            "disturbance_observation",
            "time_observation",
        ]
    }

    rewarder_config = rewards.Config.from_file(rl_dp.config / "rewarder.yaml")
    training_sim_config = cs_sim.Config.from_file(rl_dp.config / "training_simulator.yaml")
    eval_sim_config = cs_sim.Config.from_file(rl_dp.config / "eval_simulator.yaml")
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": training_scenario_folders,  # [training_scenario_folders[0]],
        "merge_loaded_scenario_episodes": True,
        "max_number_of_episodes": 1000,
        "simulator_config": training_sim_config,
        "action_sample_time": 1.0 / 0.4,  # from rlmpc.yaml config file
        "rewarder_class": rewards.MPCRewarder,
        "rewarder_kwargs": {"config": rewarder_config},
        "test_mode": False,
        "render_update_rate": 1.0,
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "shuffle_loaded_scenario_data": True,
        "identifier": "training_env1",
        "seed": 52,
    }

    TRACKING_GRU_TRAINING_DATA_SAVE_FILE = "tracking_vae_training_data_rogaland9.npy"
    TRACKING_GRU_TEST_DATA_SAVE_FILE = "tracking_vae_training_data_rogaland10.npy"

    use_vec_env = True
    if use_vec_env:
        num_cpu = 18
        training_vec_env = SubprocVecEnv([make_env(env_id, env_config, i + 1) for i in range(num_cpu)])
        obs = training_vec_env.reset()
        observations = [obs]
        frames = []
        tracking_obs_dim = list(obs["RelativeTrackingObservation"].shape)
        n_steps = 1100
        tracking_observations = np.zeros((n_steps, *tracking_obs_dim), dtype=np.float32)
        for i in range(n_steps):
            actions = np.array([training_vec_env.action_space.sample() for _ in range(num_cpu)])
            obs, reward, dones, info = training_vec_env.step(actions)
            # training_vec_env.render()

            tracking_observations[i] = obs["RelativeTrackingObservation"]
            print(f"Progress: {i}/{n_steps}")

        np.save(TRACKINGOBS_DATADIR / TRACKING_GRU_TRAINING_DATA_SAVE_FILE, tracking_observations)
        training_vec_env.close()

        # tracking_data = np.load(
        #     TRACKINGOBS_DATADIR / TRACKING_TRAINING_DATA_SAVE_FILE, mmap_mode="r", allow_pickle=True
        # )
        # m = np.load(IMAGE_DATADIR / SEGMASKS_SAVE_FILE, mmap_mode="r", allow_pickle=True).astype(np.uint8)

        env_config.update(
            {
                "max_number_of_episodes": 9000000,
                "scenario_file_folder": test_scenario_folders,
                "merge_loaded_scenario_episodes": True,
                "seed": 56,
                "test_mode": True,
                "simulator_config": eval_sim_config,
                "reload_map": False,
                "identifier": "eval_env1",
            }
        )

        test_vec_env = SubprocVecEnv([make_env(env_id, env_config, i + 1) for i in range(num_cpu)])
        obs = test_vec_env.reset()
        observations = [obs]
        frames = []
        tracking_obs_dim = list(obs["RelativeTrackingObservation"].shape)
        n_steps = 1100
        tracking_observations = np.zeros((n_steps, *tracking_obs_dim), dtype=np.float32)
        for i in range(n_steps):
            actions = np.array([test_vec_env.action_space.sample() for _ in range(num_cpu)])
            obs, reward, dones, info = test_vec_env.step(actions)
            test_vec_env.render()

            tracking_observations[i] = obs["RelativeTrackingObservation"]
            print(f"Progress: {i}/{n_steps}")

        np.save(TRACKINGOBS_DATADIR / TRACKING_GRU_TEST_DATA_SAVE_FILE, tracking_observations)

        # tracking_data = np.load(TRACKINGOBS_DATADIR / TRACKING_TRAINING_DATA_SAVE_FILE, mmap_mode="r", allow_pickle=True)
        # m = np.load(IMAGE_DATADIR / SEGMASKS_SAVE_FILE, mmap_mode="r", allow_pickle=True).astype(np.uint8)

        test_vec_env.close()
    else:
        env = gym.make(id=env_id, **env_config)
        obs, info = env.reset(seed=1)
        tracking_obs_dim = list(obs["RelativeTrackingObservation"].shape)
        observations = []
        frames = []

        # vae = VAE(latent_dim=10, input_dim=6, num_layers=1, inference_mode=True, rnn_type=th.nn.GRU).to(
        #     th.device("cpu")
        # )

        # vae.load_state_dict(
        #     th.load(
        #         "/home/doctor/Desktop/machine_learning/data/tracking_vae/tracking_vae2_BS_32_LD_10_GRU/tracking_vae2_BS_32_LD_10_GRU_best.pth",
        #         map_location=th.device("cpu"),
        #     )
        # )
        # vae.eval()
        # vae.set_inference_mode(True)

        n_steps = 500
        tracking_observations = np.zeros((n_steps, *tracking_obs_dim), dtype=np.float32)
        for i in range(n_steps):
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)
            observations.append(obs)

            tracking_observations[i] = obs["RelativeTrackingObservation"]

            pobs, seq_lengths = vae.preprocess_obs(th.from_numpy(tracking_observations[i]).unsqueeze(0))
            recon_obs, _, _, _ = vae(pobs, seq_lengths)

            if terminated or truncated:
                env.reset()

        env.close()
