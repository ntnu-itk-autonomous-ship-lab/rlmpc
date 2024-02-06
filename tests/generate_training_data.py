from pathlib import Path
from typing import Callable

import colav_simulator.behavior_generator as cs_bg
import colav_simulator.gym.observation as cs_obs
import colav_simulator.scenario_generator as cs_sg
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.paths as rl_dp
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


def save_frames_as_gif(frame_list: list, filename: Path) -> None:
    # Mess with this to change frame size
    fig = plt.figure(figsize=(frame_list[0].shape[1] / 72.0, frame_list[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frame_list[0], aspect="auto")
    plt.axis("off")

    def init():
        patch.set_data(frame_list[0])
        return (patch,)

    def animate(i):
        patch.set_data(frame_list[i])
        return (patch,)

    anim = animation.FuncAnimation(
        fig=fig, func=animate, init_func=init, blit=True, frames=len(frame_list), interval=50, repeat=True
    )
    anim.save(
        filename=filename.as_posix(),
        writer=animation.PillowWriter(fps=20),
        progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"),
    )


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
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    scenario_choice = 5
    if scenario_choice == 0:
        scenario_name = "rlmpc_scenario_cr_ss"
        config_file = rl_dp.scenarios / (scenario_name + ".yaml")
    elif scenario_choice == 1:
        scenario_name = "rlmpc_scenario_head_on_channel"
        config_file = rl_dp.scenarios / "rlmpc_scenario_easy_headon_no_hazards.yaml"
    elif scenario_choice == 3:
        scenario_name = "rogaland_random_rl_2"
        config_file = rl_dp.scenarios / "rogaland_random_rl_2.yaml"
    elif scenario_choice == 4:
        scenario_name = "rl_scenario"
        config_file = rl_dp.scenarios / "rl_scenario.yaml"
    elif scenario_choice == 5:
        scenario_name = "rlmpc_scenario_random_everything"
        config_file = rl_dp.scenarios / "rlmpc_scenario_random_everything.yaml"

    scenario_generator = cs_sg.ScenarioGenerator(seed=0)

    # scen = scenario_generator.load_scenario_from_folder(
    #     rl_dp.scenarios / "training_data" / scenario_name, scenario_name, show=True
    # )

    scenario_data = scenario_generator.generate(
        config_file=config_file,
        new_load_of_map_data=False,
        save_scenario=True,
        save_scenario_folder=rl_dp.scenarios / "training_data" / scenario_name,
        show_plots=True,
        reset_episode_counter=False,
    )
    print("done")

    # Collect perception image data by executing random actions in N environments over the scenarios.
    observation_type = {
        "dict_observation": [
            "navigation_3dof_state_observation",
            "time_observation",
            # "tracking_observation",
            "perception_image_observation",
        ]
    }
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": rl_dp.scenarios / "training_data" / scenario_name,
        "max_number_of_episodes": 1,
        "test_mode": False,
        "observation_type": observation_type,
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "seed": 0,
    }
    env = gym.make(id=env_id, **env_config)
    num_cpu = 12  # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, env_config, i) for i in range(num_cpu)])

    # load randomized episode data from selected map area (vary nr og actions, random ships etc..)
    # save perception images and actions from each episode to a dataset folder for training the VAE

    record = False
    if record:
        video_path = rl_dp.animations / "demo.mp4"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

    obs = env.reset()
    frames = []
    perception_images = []
    nonscaled_observations = []
    for i in range(200):
        obs, reward, done, info = env.step(np.array([-0.25, 0.0]))

        nonscaled_obs = info["unnormalized_obs"]
        img = env.render()
        frames.append(img)
        perception_images.append(obs["PerceptionImageObservation"])
        nonscaled_observations.append(nonscaled_obs)
        if done:
            env.reset()

    env.close()

    save_gif = False
    if save_gif:
        save_frames_as_gif(frames, rl_dp.animations / "demo2.gif")
