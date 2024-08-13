"""Test module for gym.py

    Shows how to use the gym environment, and how to save a video + gif of the simulation.
"""

import colav_simulator.common.image_helper_methods as ihm
import colav_simulator.common.paths as dp
import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import numpy as np
import rlmpc.action as mpc_action
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
from colav_simulator.gym.environment import COLAVEnvironment
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
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
    env_id = "COLAVEnvironment-v0"
    rewarder_config = rewards.Config.from_file(rl_dp.config / "rewarder.yaml")
    training_sim_config = cs_sim.Config.from_file(rl_dp.config / "training_simulator.yaml")
    training_sim_config.visualizer.matplotlib_backend = "TkAgg"  # to show the live viz
    scen_gen_config = cs_sg.Config.from_file(rl_dp.config / "scenario_generator.yaml")
    mpc_config_path = rl_dp.config / "rlmpc.yaml"
    actor_noise_std_dev = np.array([0.004, 0.004])  # normalized std dev for the action space [course, speed]
    action_kwargs = {
        "mpc_config_path": mpc_config_path,
        "mpc_param_list": ["Q_p", "w_colregs", "r_safe_do"],
        "debug": True,
        "std_init": actor_noise_std_dev,
    }

    scenario_names = ["rlmpc_scenario_ms_channel"]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    env_config = {
        "scenario_file_folder": [training_scenario_folders[0]],
        "scenario_generator_config": scen_gen_config,
        "max_number_of_episodes": 1,
        "simulator_config": training_sim_config,
        "action_type_class": mpc_action.MPCParameterSettingAction,
        "action_sample_time": 1.0 / 0.5,
        "action_kwargs": action_kwargs,
        "rewarder_class": rewards.MPCRewarder,
        "rewarder_kwargs": {"config": rewarder_config},
        "render_update_rate": 0.5,
        "render_mode": "rgb_array",
        "observation_type": observation_type,
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "identifier": "trainining_env",
        "seed": 0,
    }

    use_vec_env = False
    if use_vec_env:
        num_cpu = 2
        env = SubprocVecEnv([hf.make_env(env_id, env_config, i + 1) for i in range(num_cpu)])
    else:
        env = gym.make(id=env_id, **env_config)

    obs = env.reset()
    frames = []
    for i in range(100):
        actions = np.zeros(env.action_space.shape[0]) if use_vec_env else np.zeros(env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(actions)

        frames.append(env.render())

        if terminated or truncated:
            env.reset()

    env.close()

    save_gif = False
    if save_gif:
        ihm.save_frames_as_gif(frames, dp.animation_output / "demo.gif")

    print("done")
