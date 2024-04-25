import argparse
import copy
import platform
from pathlib import Path
from typing import Tuple

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
import rlmpc.sac as sac_rlmpc
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback
from rlmpc.networks.feature_extractors import CombinedExtractor
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

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


def main():
    experiment_name = "test_sac_rlmpc"
    base_dir, log_dir, model_dir, best_model_dir = create_data_dirs(experiment_name=experiment_name)

    scenario_names = ["rlmpc_scenario_ho", "rlmpc_scenario_cr_ss", "rlmpc_scenario_random_many_vessels"]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    generate = False
    if generate:
        scenario_generator = cs_sg.ScenarioGenerator(config_file=rl_dp.config / "scenario_generator.yaml")
        for idx, name in enumerate(scenario_names[:3]):
            if idx == 0:
                continue

            scenario_generator.seed(idx + 2)
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

            scenario_generator.seed(idx + 103)
            _ = scenario_generator.generate(
                config_file=rl_dp.scenarios / (name + ".yaml"),
                new_load_of_map_data=False,
                save_scenario=True,
                save_scenario_folder=rl_dp.scenarios / "test_data" / name,
                show_plots=True,
                episode_idx_save_offset=0,
                n_episodes=40,
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
        "scenario_file_folder": training_scenario_folders,
        "merge_loaded_scenario_episodes": True,
        "max_number_of_episodes": 10,
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
        "seed": 15,
    }
    total_training_timesteps = 100
    env = Monitor(gym.make(id=env_id, **env_config))
    env_config.update(
        {
            "max_number_of_episodes": 2,
            "scenario_file_folder": test_scenario_folders,
            "merge_loaded_scenario_episodes": True,
            "seed": 100,
            "test_mode": True,
            "simulator_config": eval_sim_config,
            "reload_map": False,
            "identifier": "eval_env1",
        }
    )
    eval_env = Monitor(gym.make(id=env_id, **env_config))

    mpc_config_file = rl_dp.config / "rlmpc.yaml"
    policy = sac_rlmpc.SACPolicyWithMPC
    # actor_noise_std_dev = np.array([0.004, 0.004, 0.025])  # normalized std dev for the action space [x, y, speed]
    actor_noise_std_dev = np.array(
        [0.001, 0.001, 0.001, 0.001]
    )  # normalized std dev for the action space [course, speed, course, speed]

    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "critic_arch": [256, 128],
        "mpc_config": mpc_config_file,
        "activation_fn": th.nn.ReLU,
        "use_sde": False,
        "std_init": actor_noise_std_dev,
        "use_expln": False,
        "clip_mean": 2.0,
    }
    model = sac_rlmpc.SAC(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=500,
        learning_starts=0,
        batch_size=16,
        gradient_steps=1,
        train_freq=(5, "step"),
        device="cpu",
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        log_path=base_dir,
        eval_freq=2,
        n_eval_episodes=1,
        callback_after_eval=stop_train_callback,
        experiment_name="sac_rlmpc1",
        deterministic=True,
        record=True,
        render=True,
        verbose=1,
    )
    stats_callback = CollectStatisticsCallback(
        env,
        log_dir=base_dir,
        experiment_name="sac_rlmpc1",
        save_stats_freq=10,
        save_agent_model_freq=100,
        log_stats_freq=2,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_training_timesteps,
        progress_bar=True,
        log_interval=2,
        callback=CallbackList([stats_callback, eval_callback]),
    )
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    record = True
    if record:
        video_path = rl_dp.animations / "test_sac_rlmpc.mp4"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

    n_episodes = 1
    for i in range(n_episodes):
        obs, _ = env.reset(seed=1)
        done = False
        frames = []
        while not done:
            action = model.predict_with_mpc(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            frames.append(env.render())

            done = terminated or truncated

    env.close()

    save_gif = True
    if save_gif:
        save_frames_as_gif(frames, rl_dp.animations / "test_sac_rlmpc.gif")
    print("done")


if __name__ == "__main__":
    main()
