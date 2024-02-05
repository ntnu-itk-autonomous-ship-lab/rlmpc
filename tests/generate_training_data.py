from pathlib import Path

import colav_simulator.behavior_generator as cs_bg
import colav_simulator.scenario_generator as cs_sg
import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.paths as rl_dp
import rl_rrt_mpc.sac as sac_rlmpc
import skimage as skimg
import stable_baselines3.common.vec_env as sb3_vec_env
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from rl_rrt_mpc.networks.feature_extractors import PerceptionImageNavigationExtractor
from stable_baselines3.common.evaluation import evaluate_policy

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


if __name__ == "__main__":
    scenario_choice = 0
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

    scenario_generator = cs_sg.ScenarioGenerator(seed=0)

    # scen = scenario_generator.load_scenario_from_folder(
    #     rl_dp.scenarios / "training_data" / scenario_name, scenario_name, show=True
    # )

    # scenario_data = scenario_generator.generate(
    #     config_file=config_file,
    #     new_load_of_map_data=False,
    #     save_scenario=True,
    #     save_scenario_folder=rl_dp.scenarios / "training_data" / scenario_name,
    #     show_plots=True,
    #     reset_episode_counter=False,
    # )
    # print("done")

    # Collect perception image data by executing random actions in N environments over the scenarios.
    observation_type = {"dict_observation": ["navigation_3dof_state_observation", "time_observation"]}
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

    record = False
    if record:
        video_path = rl_dp.animations / "demo.mp4"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

    env.reset(seed=1)
    frames = []
    for i in range(200):
        obs, reward, terminated, truncated, info = env.step(np.array([-0.2, 0.0]))

        img = env.render()
        frames.append(img)
        if terminated or truncated:
            env.reset()

    env.close()

    img = frames[0]
    gray_img = skimg.color.rgb2gray(img)
    im_handle = plt.imshow(img, aspect="equal")

    plt.axis("off")
    plt.tight_layout()
    # find way to rotate the image such that the vessel is pointing upwards
    # then, extract a subimage around the vessel, and use that as the input to the VAE
    # then, downsample the image to 256x256

    # plt.savefig(rl_dp.animations / "demo2.png", bbox_inches="tight", pad_inches=0)
    # convert img to grayscale

    save_gif = False
    if save_gif:
        save_frames_as_gif(frames, rl_dp.animations / "demo2.gif")
