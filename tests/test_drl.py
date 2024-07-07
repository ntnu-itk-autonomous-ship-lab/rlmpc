"""Test a standard SAC DRL agent on the COLAV environment.
"""

import argparse
import copy
import sys
import tracemalloc
from pathlib import Path

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import numpy as np
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
import yaml
from colav_simulator.gym.environment import COLAVEnvironment
from memory_profiler import profile
from rlmpc.common.callbacks import evaluate_policy
from rlmpc.networks.feature_extractors import CombinedExtractor
from rlmpc.train_drl_sac import train_sac
from stable_baselines3.common.monitor import Monitor


@profile
def main(args):
    hf.set_memory_limit(28_000_000_000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path.home() / "Desktop/machine_learning/rlmpc/"))
    parser.add_argument("--experiment_name", type=str, default="sac_drl2")
    parser.add_argument("--n_cpus", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--train_freq", type=int, default=8)
    parser.add_argument("--n_eval_episodes", type=int, default=2)
    parser.add_argument("--n_experiments", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--sde_sample_freq", type=int, default=60)
    args = parser.parse_args(args)
    args.base_dir = Path(args.base_dir)
    print("Provided args to SAC DRL training:")
    print("".join(f"{k}={v}\n" for k, v in vars(args).items()))

    base_dir, log_dir, model_dir = hf.create_data_dirs(base_dir=args.base_dir, experiment_name=args.experiment_name)

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

    model_kwargs = {
        "policy": "MultiInputPolicy",
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "gradient_steps": args.gradient_steps,
        "train_freq": (args.train_freq, "step"),
        "learning_starts": 0,
        "tau": args.tau,
        "sde_sample_freq": args.sde_sample_freq,
        "device": "cpu",
        "ent_coef": "auto",
        "verbose": 1,
        "tensorboard_log": str(log_dir),
        "policy_kwargs": {
            "features_extractor_class": CombinedExtractor,
            "net_arch": [512, 512],
            "log_std_init": -3.0,
            "use_sde": True,
        },
        "replay_buffer_kwargs": {"handle_timeout_termination": True},
    }

    # tracemalloc.start(20)
    # t_start = tracemalloc.take_snapshot()
    load_model = False
    load_model_name = args.experiment_name + "_1000000_steps"
    n_timesteps_per_learn = 10
    n_learn_iterations = args.timesteps // n_timesteps_per_learn

    for e in range(args.n_experiments):
        seed = e
        training_env_config["seed"] = seed
        eval_env_config["seed"] = seed + 1
        if e > 0:
            load_model = True
            load_model_name = args.experiment_name + f"_exp{e}_{args.timesteps}_steps"
        for i in range(n_learn_iterations):
            if i > 0:
                load_model = True
                load_model_name = args.experiment_name + f"_exp{e + 1}_{i * n_timesteps_per_learn}_steps"

            model = train_sac(
                model_kwargs=model_kwargs,
                n_timesteps=n_timesteps_per_learn,
                env_id=env_id,
                training_env_config=copy.deepcopy(training_env_config),
                n_training_envs=args.n_cpus,
                eval_env_config=copy.deepcopy(eval_env_config),
                base_dir=base_dir,
                log_dir=log_dir,
                model_dir=model_dir,
                experiment_name=args.experiment_name,
                load_model=load_model,
                load_model_name=load_model_name,
                load_rb_name=args.experiment_name + "_replay_buffer",
                seed=seed,
                iteration=i + 1,
            )
            model.save(model_dir / f"{args.experiment_name}_exp{e + 1}_{(i + 1) * n_timesteps_per_learn}_steps")
            model.save_replay_buffer(model_dir / f"{args.experiment_name}_replay_buffer")
            print(
                f"[SAC DRL] Finished learning iteration {i + 1}. Progress: {100.0 * (i + 1) * n_timesteps_per_learn / args.timesteps}%"
            )
        print(f"[SAC DRL] Finished experiment {e + 1}/{args.n_experiments}")

    # tm_end = tracemalloc.take_snapshot()
    # stats = tm_end.compare_to(t_start, "lineno")
    # for stat in stats[:10]:
    #     print(stat)
    eval_env = Monitor(gym.make(id=env_id, **eval_env_config))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        record=True,
        record_path=base_dir / "eval_data" / "final_eval_videos",
        record_name=args.experiment_name + "_final_eval",
    )
    print(f"{args.experiment_name} final evaluation | mean_reward: {mean_reward}, std_reward: {std_reward}")
    train_cfg = {
        "n_timsteps_per_learn": n_timesteps_per_learn,
        "n_learn_iterations": n_learn_iterations,
        "n_experiments": args.n_experiments,
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
