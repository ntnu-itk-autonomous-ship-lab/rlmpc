"""Test a standard SAC DRL agent on the COLAV environment.
"""

import argparse
import copy
import sys
import tracemalloc
from pathlib import Path

import colav_simulator.common.paths as dp
import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
import yaml
from colav_simulator.gym.environment import COLAVEnvironment
from memory_profiler import profile
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback, evaluate_policy
from rlmpc.networks.feature_extractors import CombinedExtractor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


@profile
def main(args):
    hf.set_memory_limit(28_000_000_000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path.home() / "Desktop/machine_learning/rlmpc/"))
    parser.add_argument("--experiment_name", type=str, default="sac_drl1")
    parser.add_argument("--n_cpus", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--train_freq", type=int, default=8)
    parser.add_argument("--n_eval_episodes", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--sde_sample_freq", type=int, default=60)
    args = parser.parse_args(args)
    args.base_dir = Path(args.base_dir)
    print("Provided args to SAC DRL training:")
    print("".join(f"{k}={v}\n" for k, v in vars(args).items()))

    experiment_name = args.experiment_name
    base_dir, log_dir, model_dir = hf.create_data_dirs(base_dir=args.base_dir, experiment_name=experiment_name)

    scenario_names = [
        "rlmpc_scenario_ms_channel"
    ]  # ["rlmpc_scenario_ho", "rlmpc_scenario_cr_ss", "rlmpc_scenario_random_many_vessels"]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            "perception_image_observation",
            "relative_tracking_observation",
            # "navigation_3dof_state_observation",
        ]
    }

    rewarder_config = rewards.Config.from_file(rl_dp.config / "rewarder.yaml")
    training_sim_config = cs_sim.Config.from_file(rl_dp.config / "training_simulator.yaml")
    eval_sim_config = cs_sim.Config.from_file(rl_dp.config / "eval_simulator.yaml")
    scen_gen_config = cs_sg.Config.from_file(rl_dp.config / "scenario_generator.yaml")
    env_id = "COLAVEnvironment-v0"
    training_env_config = {
        "scenario_file_folder": [training_scenario_folders[0]],
        "scenario_generator_config": scen_gen_config,
        "max_number_of_episodes": 1,
        "simulator_config": training_sim_config,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        "rewarder_class": rewards.MPCRewarder,
        "rewarder_kwargs": {"config": rewarder_config},
        "render_update_rate": 0.5,
        "render_mode": "rgb_array",
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "merge_loaded_scenario_episodes": True,
        "shuffle_loaded_scenario_data": True,
        "identifier": "training_env",
        "seed": 0,
    }

    num_cpu = args.n_cpus
    # training_vec_env = SubprocVecEnv([make_env(env_id, training_env_config, i + 1) for i in range(num_cpu)])
    training_vec_env = make_vec_env(
        env_id=env_id,
        env_kwargs=training_env_config,
        n_envs=num_cpu,
        seed=0,
        monitor_dir=str(log_dir),
        vec_env_cls=SubprocVecEnv,
    )

    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "net_arch": [256, 256],
        "log_std_init": -3.0,
    }
    model = SAC(
        "MultiInputPolicy",
        training_vec_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        train_freq=(args.train_freq, "step"),
        learning_starts=10,
        tau=args.tau,
        use_sde=True,
        sde_sample_freq=args.sde_sample_freq,
        device="cpu",
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(log_dir),
        policy_kwargs=policy_kwargs,
        replay_buffer_kwargs={"handle_timeout_termination": True},
    )

    load_model = False
    if load_model:
        model.load(model_dir / "sac_drl1_240000_steps.zip")

    eval_env_config = copy.deepcopy(training_env_config)
    eval_env_config.update(
        {
            "max_number_of_episodes": 1,
            "scenario_file_folder": test_scenario_folders,
            "seed": 1,
            "simulator_config": eval_sim_config,
            "identifier": "eval_env",
        }
    )
    eval_env = Monitor(gym.make(id=env_id, **eval_env_config))

    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=40000,
        save_path=model_dir,
        name_prefix=args.experiment_name,
        verbose=1,
        save_replay_buffer=True,
    )
    # stats_callback = CollectStatisticsCallback(
    #     env=training_vec_env,
    #     log_dir=base_dir,
    #     model_dir=model_dir,
    #     experiment_name=args.experiment_name,
    #     save_stats_freq=1000,
    #     save_agent_model_freq=50000,
    #     log_freq=10,
    #     max_num_env_episodes=1000,
    #     max_num_training_stats_entries=30000,
    #     verbose=1,
    # )
    eval_callback = EvalCallback(
        eval_env,
        log_path=base_dir / "eval_data",
        eval_freq=50000,
        n_eval_episodes=1,
        # callback_after_eval=stop_train_callback,
        experiment_name=experiment_name,
        record=True,
        render=True,
        verbose=1,
    )

    n_timesteps_per_learn = 100000
    n_learn_iterations = args.timesteps // n_timesteps_per_learn

    # tracemalloc.start(20)
    # t_start = tracemalloc.take_snapshot()
    for i in range(n_learn_iterations):
        model.learn(
            total_timesteps=n_timesteps_per_learn,
            log_interval=4,
            tb_log_name=args.experiment_name,
            reset_num_timesteps=False,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
        )
        model.save(model_dir / f"{args.experiment_name}_{(i + 1) * n_timesteps_per_learn}_steps")
        model.save_replay_buffer(model_dir / f"{args.experiment_name}__replay_buffer")
        print(
            f"[SAC DRL] Finished learning iteration {i + 1}. Progress: {(i + 1) * n_timesteps_per_learn}/{args.timesteps} steps. "
        )

    # tm_end = tracemalloc.take_snapshot()
    # stats = tm_end.compare_to(t_start, "lineno")
    # for stat in stats[:10]:
    #     print(stat)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        record=True,
        record_path=base_dir / "eval_data" / "final_eval_videos",
        record_name=experiment_name + "_final_eval",
    )
    print(f"{args.experiment_name} final evaluation | mean_reward: {mean_reward}, std_reward: {std_reward}")
    train_cfg = {
        "n_timsteps_per_learn": n_timesteps_per_learn,
        "n_learn_iterations": n_learn_iterations,
        "experiment_name": args.experiment_name,
        "timesteps": args.timesteps,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "batch_size": args.batch_size,
        "n_eval_episodes": args.n_eval_episodes,
        "tau": args.tau,
        "sde_sample_freq": args.sde_sample_freq,
        "final_mean_eval_reward": np.mean(mean_reward),
        "final_std_eval_reward": np.mean(std_reward),
        "n_cpus": args.n_cpus,
        "buffer_size": args.buffer_size,
    }
    with (base_dir / "train_config.yaml").open(mode="w", encoding="utf-8") as fp:
        yaml.dump(train_cfg, fp)


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # cProfile.run("main()", sort="cumulative", filename="sac_drl.prof")
    # p = pstats.Stats("sac_drl.prof")
    # p.sort_stats("cumulative").print_stats(100)

    main(sys.argv[1:])
