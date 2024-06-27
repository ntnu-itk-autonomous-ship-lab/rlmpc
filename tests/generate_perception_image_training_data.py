import os
from pathlib import Path
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


if __name__ == "__main__":
    IMAGE_DATADIR: Path = Path.home() / "Desktop/machine_learning/vae/data"
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
            "perception_image_observation",
            "relative_tracking_observation",
            "navigation_3dof_state_observation",
            "tracking_observation",
            "ground_truth_tracking_observation",
            "disturbance_observation",
            "time_observation",
        ]
    }

    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": training_scenario_folders,
        "max_number_of_episodes": 100000,
        "render_update_rate": 0.5,
        "observation_type": observation_type,
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "seed": 0,
    }

    PERCEPTION_TRAINING_DATA_SAVE_FILE = "perception_training_data_rogaland"
    SEGMASKS_TRAINING_SAVE_FILE = "segmentation_masks_training_data_rogaland"
    PERCEPTION_TEST_DATA_SAVE_FILE = "perception_test_data_rogaland"
    SEGMASKS_TEST_SAVE_FILE = "segmentation_masks_test_data_rogaland"

    n_savefiles = 4

    for sf in range(n_savefiles):
        training_save_filename = PERCEPTION_TRAINING_DATA_SAVE_FILE + str(sf) + ".npy"
        segmasks_training_save_filename = SEGMASKS_TRAINING_SAVE_FILE + str(sf) + ".npy"
        test_save_filename = PERCEPTION_TEST_DATA_SAVE_FILE + str(sf) + ".npy"
        segmasks_test_save_filename = SEGMASKS_TEST_SAVE_FILE + str(sf) + ".npy"

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

            np.save(IMAGE_DATADIR / training_save_filename, perception_images)
            np.save(IMAGE_DATADIR / segmasks_training_save_filename, masks)

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

            use_vae = False
            if use_vae:
                vae = VAE(
                    latent_dim=160,
                    input_image_dim=(1, 256, 256),
                    encoder_conv_block_dims=(32, 128, 256, 256),
                    fc_dim=512,
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

                if use_vae:
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
