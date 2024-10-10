import os
from pathlib import Path
from typing import Callable, Tuple

import colav_simulator.behavior_generator as cs_bg
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
        env_config.update({"identifier": env_config["identifier"] + str(rank), "seed": seed + rank})
        env = gym.make(env_id, **env_config)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    TRACKINGOBS_DATADIR: Path = Path.home() / "Desktop/machine_learning/tracking_vae/data"

    scenario_names = [
        "rlmpc_scenario_ms_channel",
        "rlmpc_scenario_random_many_vessels",
    ]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    # map_size: [4000.0, 4000.0]
    # map_origin_enu: [-33524.0, 6572500.0]
    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            # "perception_image_observation",
            "relative_tracking_observation",
            "tracking_observation",
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
        "max_number_of_episodes": 100000,
        "simulator_config": training_sim_config,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        # "rewarder_class": rewards.MPCRewarder,
        # "rewarder_kwargs": {"config": rewarder_config},
        "test_mode": False,
        "render_update_rate": 1.0,
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "shuffle_loaded_scenario_data": True,
        "identifier": "training_env",
        "seed": 0,
    }

    TRACKING_VAE_TRAINING_DATA_SAVE_FILE = "tracking_vae_training_data_rogaland_new"
    TRACKING_VAE_TEST_DATA_SAVE_FILE = "tracking_vae_test_data_rogaland_new"

    n_files = 200
    for f in range(100, n_files):
        training_filename = TRACKING_VAE_TRAINING_DATA_SAVE_FILE + str(f) + ".npy"
        test_filename = TRACKING_VAE_TEST_DATA_SAVE_FILE + str(f) + ".npy"
        use_vec_env = True
        if use_vec_env:
            num_cpu = 12
            env_config.update(
                {
                    "scenario_file_folder": training_scenario_folders,
                    "seed": f,
                    "simulator_config": training_sim_config,
                    "identifier": "training_env",
                }
            )
            training_vec_env = SubprocVecEnv([make_env(env_id, env_config, i + 1) for i in range(num_cpu)])
            env_config.update(
                {
                    "scenario_file_folder": test_scenario_folders,
                    "seed": f + 500,
                    "simulator_config": eval_sim_config,
                    "identifier": "eval_env",
                }
            )
            test_vec_env = SubprocVecEnv([make_env(env_id, env_config, i + 1) for i in range(num_cpu)])
        else:
            env = gym.make(id=env_id, **env_config)

        if use_vec_env:
            training_vec_env.seed(f * 200)
            obs = training_vec_env.reset()
            observations = [obs]
            frames = []
            tracking_obs_dim = list(obs["RelativeTrackingObservation"].shape)
            n_steps = 1200
            tracking_observations = np.zeros((n_steps, *tracking_obs_dim), dtype=np.float32)
            for i in range(n_steps):
                actions = np.array([training_vec_env.action_space.sample() for _ in range(num_cpu)])
                obs, reward, dones, info = training_vec_env.step(actions)
                # training_vec_env.render()

                tracking_observations[i] = obs["RelativeTrackingObservation"]
                print(f"Progress: {i}/{n_steps}")

            np.save(TRACKINGOBS_DATADIR / training_filename, tracking_observations)

            # tracking_data = np.load(
            #     TRACKINGOBS_DATADIR / TRACKING_TRAINING_DATA_SAVE_FILE, mmap_mode="r", allow_pickle=True
            # )
            # m = np.load(IMAGE_DATADIR / SEGMASKS_SAVE_FILE, mmap_mode="r", allow_pickle=True).astype(np.uint8)

            test_vec_env.seed(f * 1510)
            obs = test_vec_env.reset()
            observations = [obs]
            frames = []
            tracking_obs_dim = list(obs["RelativeTrackingObservation"].shape)
            n_steps = 1200
            tracking_observations = np.zeros((n_steps, *tracking_obs_dim), dtype=np.float32)
            for i in range(n_steps):
                actions = np.array([test_vec_env.action_space.sample() for _ in range(num_cpu)])

                obs, reward, dones, info = test_vec_env.step(actions)
                # test_vec_env.render()

                tracking_observations[i] = obs["RelativeTrackingObservation"]
                print(f"Progress: {i}/{n_steps}")

            np.save(TRACKINGOBS_DATADIR / test_filename, tracking_observations)

            # tracking_data = np.load(TRACKINGOBS_DATADIR / TRACKING_TRAINING_DATA_SAVE_FILE, mmap_mode="r", allow_pickle=True)
            # m = np.load(IMAGE_DATADIR / SEGMASKS_SAVE_FILE, mmap_mode="r", allow_pickle=True).astype(np.uint8)

        else:
            obs, info = env.reset(seed=1)
            tracking_obs_dim = list(obs["RelativeTrackingObservation"].shape)
            observations = []
            frames = []

            # vae = VAE(latent_dim=10, input_dim=6, num_layers=1, inference_mode=True, rnn_type=th.nn.GRU).to(
            #     th.device("cpu")
            # )

            # vae.load_state_dict(
            #     th.load(
            #         str(Path.home() / "Desktop/machine_learning/data/tracking_vae/tracking_vae2_BS_32_LD_10_GRU/tracking_vae2_BS_32_LD_10_GRU_best.pth",
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

    if use_vec_env:
        training_vec_env.close()
        test_vec_env.close()
    else:
        env.close()
