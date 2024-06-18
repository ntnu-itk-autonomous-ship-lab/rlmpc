import argparse
import platform
from pathlib import Path
from typing import Tuple

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.paths as rl_dp
import rlmpc.policies as rlmpc_policies
import rlmpc.rewards as rewards
import rlmpc.sac as sac_rlmpc
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback, evaluate_mpc_policy
from rlmpc.networks.feature_extractors import CombinedExtractor
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnNoModelImprovement
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


# tuning:
# horizon
# tau/barrier param
# edge case shit
# constraint satisfaction highly dependent on tau/barrier
# if ship gets too much off path/course it will just continue off course


# optimize runtime?
# fix enc display whiteness in training
# upd scen gen to spawn obstacles along waypoints instead of only near init pos
# add more scenarios
def main():
    experiment_name = "sac_rlmpc"
    base_dir, log_dir, model_dir, best_model_dir = create_data_dirs(experiment_name=experiment_name)

    scenario_names = [
        "rlmpc_scenario_ms_channel"
    ]  # ["rlmpc_scenario_ho", "rlmpc_scenario_cr_ss", "rlmpc_scenario_random_many_vessels"]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    generate = False
    if generate:
        scenario_generator = cs_sg.ScenarioGenerator(config_file=rl_dp.config / "scenario_generator.yaml")
        for idx, name in enumerate(scenario_names):

            scenario_generator.seed(idx + 2)
            _ = scenario_generator.generate(
                config_file=rl_dp.scenarios / (name + ".yaml"),
                new_load_of_map_data=False if idx == 0 else False,
                save_scenario=True,
                save_scenario_folder=rl_dp.scenarios / "training_data" / name,
                show_plots=True,
                episode_idx_save_offset=0,
                n_episodes=100,
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
        "scenario_file_folder": [training_scenario_folders[0]],
        "merge_loaded_scenario_episodes": True,
        "max_number_of_episodes": 400,
        "simulator_config": training_sim_config,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        "rewarder_class": rewards.MPCRewarder,
        "rewarder_kwargs": {"config": rewarder_config},
        "render_update_rate": 1.0,
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "shuffle_loaded_scenario_data": True,
        "identifier": "training_env1",
        "seed": 0,
    }
    env = Monitor(gym.make(id=env_id, **env_config))
    env_config.update(
        {
            "max_number_of_episodes": 1,
            "scenario_file_folder": test_scenario_folders,
            "merge_loaded_scenario_episodes": True,
            "seed": 1,
            "simulator_config": eval_sim_config,
            "reload_map": False,
            "identifier": "eval_env1",
        }
    )
    eval_env = Monitor(gym.make(id=env_id, **env_config))

    mpc_config_file = rl_dp.config / "rlmpc.yaml"
    # actor_noise_std_dev = np.array([0.004, 0.004, 0.025])  # normalized std dev for the action space [x, y, speed]
    actor_noise_std_dev = np.array([0.005, 0.002])  # normalized std dev for the action space [course, speed]

    mpc_param_provider_kwargs = {
        "param_list": ["r_safe_do"],
        "hidden_sizes": [256],
        "activation_fn": th.nn.ELU,
    }
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "critic_arch": [258, 128],
        "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
        "mpc_config": mpc_config_file,
        "activation_fn": th.nn.ELU,
        "std_init": actor_noise_std_dev,
        "debug": False,
    }
    model = sac_rlmpc.SAC(
        rlmpc_policies.SACPolicyWithMPC,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.002,
        buffer_size=500,
        learning_starts=0,
        batch_size=8,
        gradient_steps=1,
        train_freq=(16, "step"),
        device="cpu",
        tensorboard_log=str(log_dir),
        data_path=base_dir,
        pretrain_critic_using_mpc=True,
        verbose=1,
    )
    exp_name_str = "sac_rlmpc1"
    load_model = False
    if load_model:
        model.custom_load(model_dir / "sac_rlmpc1_100")

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        log_path=base_dir / "eval_data",
        eval_freq=10000000,
        n_eval_episodes=5,
        callback_after_eval=stop_train_callback,
        experiment_name=exp_name_str,
        record=True,
        render=True,
        verbose=1,
    )
    stats_callback = CollectStatisticsCallback(
        env,
        log_dir=base_dir,
        experiment_name=exp_name_str,
        save_stats_freq=1,
        save_agent_model_freq=100,
        log_stats_freq=2,
        verbose=1,
    )
    total_training_timesteps = 10000
    model.learn(
        total_timesteps=total_training_timesteps,
        progress_bar=False,
        log_interval=2,
        callback=CallbackList([stats_callback, eval_callback]),
    )
    mean_reward, std_reward = evaluate_mpc_policy(
        model, eval_env, n_eval_episodes=10, record=True, record_path=base_dir / "eval_videos", record_name="final_eval"
    )
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    import cProfile
    import pstats

    cProfile.run("main()", sort="cumulative", filename="sac_rlmpc.prof")

    p = pstats.Stats("sac_rlmpc.prof")
    p.sort_stats("cumulative").print_stats(50)
    # main()
