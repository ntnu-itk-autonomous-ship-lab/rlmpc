import os
from pathlib import Path
from sys import platform
from typing import Callable

import colav_simulator.common.image_helper_methods as cs_ihm
import colav_simulator.common.paths as cs_dp
import colav_simulator.scenario_generator as cs_sg
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.paths as rl_dp
import torch as th
import torchvision.transforms.v2 as transforms_v2
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from rlmpc.networks.perception_vae.vae import VAE
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# For macOS users, you might need to set the environment variable
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

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
        env.unwrapped.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if platform == "linux" or platform == "linux2":
    IMAGE_DATADIR: Path = Path("/home/doctor/Desktop/machine_learning/data/vae/")
elif platform == "darwin":
    IMAGE_DATADIR: Path = Path("/Users/trtengesdal/Desktop/machine_learning/data/vae/")


if __name__ == "__main__":
    scenario_choice = 0
    if scenario_choice == -1:
        scenario_name = "crossing_give_way"
        config_file = cs_dp.scenarios / (scenario_name + ".yaml")
    elif scenario_choice == 0:
        scenario_name = "boknafjorden_generation_test"
        config_file = rl_dp.scenarios / (scenario_name + ".yaml")
    elif scenario_choice == 1:
        scenario_name = "rlmpc_scenario_ms_channel"
        config_file = rl_dp.scenarios / "rlmpc_scenario_easy_headon_no_hazards.yaml"
    elif scenario_choice == 3:
        scenario_name = "rogaland_random_rl"
        config_file = rl_dp.scenarios / "rogaland_random_rl.yaml"
    elif scenario_choice == 4:
        scenario_name = "rl_scenario"
        config_file = rl_dp.scenarios / "rl_scenario.yaml"
    elif scenario_choice == 5:
        scenario_name = "rlmpc_scenario_random_everything"
        config_file = rl_dp.scenarios / "rlmpc_scenario_random_everything.yaml"
    elif scenario_choice == 6:
        scenario_name = "rlmpc_scenario_random_everything_test"
        config_file = rl_dp.scenarios / "rlmpc_scenario_random_everything_test.yaml"
    elif scenario_choice == 7:
        scenario_name = "rlmpc_scenario_random_many_vessels"
        config_file = rl_dp.scenarios / "rlmpc_scenario_random_many_vessels.yaml"

    scenario_generator = cs_sg.ScenarioGenerator(seed=5)

    # scen = scenario_generator.load_scenario_from_folder(
    #     rl_dp.scenarios / "training_data" / scenario_name, scenario_name, show=True
    # )
    scenario_data = scenario_generator.generate(
        config_file=config_file,
        new_load_of_map_data=True,
        save_scenario=True,
        save_scenario_folder=rl_dp.scenarios / "training_data" / scenario_name,
        show_plots=True,
        episode_idx_save_offset=0,
    )

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
        "max_number_of_episodes": 100000000,
        "test_mode": False,
        "render_update_rate": 0.5,
        "observation_type": observation_type,
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "seed": 11,
    }
    IMG_SAVE_FILE = "perception_data_rogaland_random_everything_land_only2.npy"
    SEGMASKS_SAVE_FILE = "segmentation_masks_rogaland_random_everything_land_only2.npy"

    use_vec_env = False
    if use_vec_env:
        num_cpu = 18
        vec_env = SubprocVecEnv([make_env(env_id, env_config, i + 1) for i in range(num_cpu)])
        obs = vec_env.reset()
        observations = [obs]
        frames = []
        img_dim = list(obs["PerceptionImageObservation"].shape)
        n_steps = 2000
        perception_images = np.zeros((n_steps, *img_dim), dtype=np.uint8)
        masks = np.zeros((n_steps, *img_dim), dtype=np.uint8)
        for i in range(n_steps):
            actions = np.array([vec_env.action_space.sample() for _ in range(num_cpu)])
            obs, reward, dones, info = vec_env.step(actions)
            vec_env.render()

            perception_images[i] = obs["PerceptionImageObservation"]
            masks[i] = cs_ihm.create_simulation_image_segmentation_mask(perception_images[i])
            if False:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(perception_images[i, 0, 0], cmap="gray")
                ax[1].imshow(masks[i, 0, 0], cmap="gray")
                plt.show(block=False)

            print(f"Progress: {i}/{n_steps}")

        np.save(IMAGE_DATADIR / IMG_SAVE_FILE, perception_images)
        np.save(IMAGE_DATADIR / SEGMASKS_SAVE_FILE, masks)

        # imgs = np.load(IMAGE_DATADIR / IMG_SAVE_FILE, mmap_mode="r", allow_pickle=True).astype(np.uint8)
        # m = np.load(IMAGE_DATADIR / SEGMASKS_SAVE_FILE, mmap_mode="r", allow_pickle=True).astype(np.uint8)

        vec_env.close()
    else:
        env = gym.make(id=env_id, **env_config)

        record = False
        if record:
            video_path = rl_dp.animations / "demo.mp4"
            env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

        obs, info = env.reset(seed=1)
        img_dim = obs["PerceptionImageObservation"].shape
        observations = []
        frames = []

        vae = VAE(
            latent_dim=160, input_image_dim=(1, 256, 256), encoder_conv_block_dims=(32, 128, 256, 256), fc_dim=512
        ).to(th.device("cpu"))

        vae.load_state_dict(
            th.load(
                "/Users/trtengesdal/Desktop/machine_learning/vae_models/training_vae2_model_LD_160_best.pth",
                map_location=th.device("cpu"),
            )
        )
        vae.eval()
        vae.set_inference_mode(True)
        img_transform = transforms_v2.Compose(
            [
                transforms_v2.ToDtype(th.float32, scale=True),
                transforms_v2.Resize((256, 256)),
            ]
        )
        display_transform = transforms_v2.Compose(
            [
                transforms_v2.ToDtype(th.uint8, scale=True),
            ]
        )
        n_steps = 500
        perception_images = np.zeros((n_steps, *img_dim), dtype=np.uint8)
        masks = np.zeros((n_steps, *img_dim), dtype=np.uint8)
        for i in range(n_steps):
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)
            img = env.render()

            frames.append(img)
            observations.append(obs)

            perception_images[i] = obs["PerceptionImageObservation"]
            masks = cs_ihm.create_simulation_image_segmentation_mask(obs["PerceptionImageObservation"])
            pi_tensor = img_transform(th.tensor(perception_images[i]))
            reconstructed_image, mean, log_var, sampled_latent_var = vae(pi_tensor.unsqueeze(0))
            if True:
                fig, ax = plt.subplots(1, 1)
                ax.imshow(display_transform(pi_tensor)[0, :, :].detach().numpy(), cmap="hot")
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                fig, ax = plt.subplots(1, 1)
                ax.imshow(display_transform(reconstructed_image)[0, 0, :, :].detach().numpy(), cmap="hot")
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                # ax[1].imshow(masks[i, 0, 0], cmap="gray")
                plt.show(block=False)
            if terminated or truncated:
                env.reset()

        env.close()

    save_gif = False
    if save_gif:
        save_frames_as_gif(frames, rl_dp.animations / "demo2.gif")
