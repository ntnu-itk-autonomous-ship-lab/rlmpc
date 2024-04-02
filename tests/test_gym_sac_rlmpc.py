import platform
from pathlib import Path

import colav_simulator.scenario_generator as cs_sg
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
import rlmpc.sac as sac_rlmpc
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from rlmpc.networks.feature_extractors import CombinedExtractor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement
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
    if platform.system() == "Unix":
        base_dir = Path("/home/doctor/machine_learning/rlmpc/")
    elif platform.system() == "Darwin":
        base_dir = Path("/Users/trtengesdal/machine_learning/rlmpc/")
    log_dir = base_dir / "logs"
    model_dir = base_dir / "models"
    best_model_dir = model_dir / "best_model"

    scenario_choice = 0
    if scenario_choice == 0:
        scenario_name = "rlmpc_scenario_cr_ss"
        config_file = rl_dp.scenarios / (scenario_name + ".yaml")
    elif scenario_choice == 1:
        scenario_name = "rlmpc_scenario_head_on_channel"
        config_file = rl_dp.scenarios / "rlmpc_scenario_easy_headon_no_hazards.yaml"

    # scenario_generator = cs_sg.ScenarioGenerator(seed=2)
    # scenario_data = scenario_generator.generate(
    #     config_file=config_file,
    #     new_load_of_map_data=False,
    #     save_scenario=True,
    #     save_scenario_folder=rl_dp.scenarios / "test_data" / scenario_name,
    #     show_plots=True,
    #     episode_idx_save_offset=0,
    # )

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

    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": rl_dp.scenarios / "training_data" / scenario_name,
        "max_number_of_episodes": 1,
        "action_sampling_time": 1.0 / 0.2,  # from rlmpc.yaml config file
        "rewarder_class": rewards.MPCRewarder,
        "rewarder_kwargs": {"config": rewarder_config},
        "test_mode": False,
        "render_update_rate": 0.5,
        "observation_type": observation_type,
        "action_type": "continuous_relative_los_reference_action",
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "seed": 15,
    }
    env = gym.make(id=env_id, **env_config)
    env_config.update(
        {"scenario_file_folder": rl_dp.scenarios / "test_data" / scenario_name, "seed": 100, "test_mode": True}
    )
    eval_env = gym.make(id=env_id, **env_config)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=5,
        callback_after_eval=stop_train_callback,
        best_model_save_path=best_model_dir,
        deterministic=True,
        render=True,
        verbose=1,
    )

    mpc_config_file = rl_dp.config / "rlmpc.yaml"
    policy = sac_rlmpc.SACPolicyWithMPC
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "critic_arch": [256, 128, 32],
        "mpc_config": mpc_config_file,
        "activation_fn": th.nn.ReLU,
        "use_sde": False,
        "log_std_init": -3.0,
        "use_expln": False,
        "clip_mean": 2.0,
    }
    model = sac_rlmpc.SAC(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=100,
        learning_starts=0,
        batch_size=2,
        gradient_steps=2,
        train_freq=(2, "step"),
        device="cpu",
        tensorboard_log=log_dir,
        verbose=1,
    )

    model.learn(total_timesteps=100, progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    record = True
    if record:
        video_path = rl_dp.animations / "demo.mp4"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

    env.reset(seed=1)
    frames = []
    for i in range(250):
        obs, reward, terminated, truncated, info = env.step(np.array([-0.2, 0.0]))

        frames.append(env.render())

        if terminated or truncated:
            env.reset()

    env.close()

    save_gif = True
    if save_gif:
        save_frames_as_gif(frames, rl_dp.animations / "demo2.gif")

    print("done")
