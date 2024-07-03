from pathlib import Path
from typing import Tuple

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.paths as rl_dp
import rlmpc.policies as rlmpc_policies
import rlmpc.rewards as rewards
import rlmpc.sac as sac_rlmpc
import yaml
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback, evaluate_policy
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

def create_data_dirs(base_dir: Path, experiment_name: str) -> Tuple[Path, Path, Path, Path]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path.home() / "Desktop/machine_learning/rlmpc/"))
    parser.add_argument("--experiment_name", type=str, default="sac_rlmpc1")
    parser.add_argument("--n_cpus", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--train_freq", type=int, default=8)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=50)
    args = parser.parse_args(args)
    args.base_dir = Path(args.base_dir)
    print("Provided args to SAC RLMPC training:")
    print("".join(f"{k}={v}\n" for k, v in vars(args).items()))

    experiment_name = args.experiment_name
    base_dir, log_dir, model_dir, best_model_dir = create_data_dirs(
        base_dir=args.base_dir, experiment_name=experiment_name
    )

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
                n_episodes=600,
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
                n_episodes=50,
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
        "max_number_of_episodes": 250,
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
    actor_noise_std_dev = np.array([0.004, 0.004])  # normalized std dev for the action space [course, speed]

    mpc_param_provider_kwargs = {
        "param_list": ["r_safe_do"],
        "hidden_sizes": [256, 64],
        "activation_fn": th.nn.ReLU,
    }
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "critic_arch": [256, 128],
        "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
        "mpc_config": mpc_config_file,
        "activation_fn": th.nn.ReLU,
        "std_init": actor_noise_std_dev,
        "debug": False,
    }
    learning_rate = 0.004
    model = sac_rlmpc.SAC(
        rlmpc_policies.SACPolicyWithMPC,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=0,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        train_freq=(args.train_freq, "step"),
        device="cpu",
        tensorboard_log=str(log_dir),
        data_path=base_dir,
        pretrain_critic_using_mpc=False,
        tau=0.001,
        verbose=1,
    )
    load_buffer = False
    if load_buffer:
        model.load_replay_buffer(base_dir / "replay_buffer1")
    load_model = False
    if load_model:
        model.custom_load(model_dir / (experiment_name + "_700"))

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        log_path=base_dir / "eval_data",
        eval_freq=10000000,
        n_eval_episodes=5,
        callback_after_eval=stop_train_callback,
        experiment_name=experiment_name,
        record=True,
        render=True,
        verbose=1,
    )
    stats_callback = CollectStatisticsCallback(
        env,
        log_dir=base_dir,
        experiment_name=args.experiment_name,
        save_stats_freq=2,
        save_agent_model_freq=100,
        log_stats_freq=10,
        verbose=1,
    )
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False,
        log_interval=2,
        callback=CallbackList([stats_callback, eval_callback]),
    )
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.n_eval_episodes, record=True, record_path=base_dir / "eval_videos", record_name="final_eval"
    )
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    model.custom_save(model_dir / "best_model")
    print(f"{args.experiment_name} final evaluation | mean_reward: {mean_reward}, std_reward: {std_reward}")
    train_cfg = {
        "experiment_name": args.experiment_name,
        "timesteps": args.timesteps,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "batch_size": args.batch_size,
        "n_eval_episodes": args.n_eval_episodes,
        "final_mean_eval_reward": mean_reward,
        "final_std_eval_reward": std_reward,
        "n_cpus": args.n_cpus,
        "buffer_size": args.buffer_size,
    }
    with (base_dir / "train_config.yaml").open(mode="w", encoding="utf-8") as fp:
        yaml.dump(train_cfg, fp)


if __name__ == "__main__":
    import cProfile
    import pstats

    # cProfile.run("main()", sort="cumulative", filename="sac_rlmpc.prof")
    # p = pstats.Stats("sac_rlmpc.prof")
    # p.sort_stats("cumulative").print_stats(50)
    main(sys.argv[1:])
