import argparse
import copy
import pickle
import sys
import warnings
from pathlib import Path

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.policies as rlmpc_policies
import rlmpc.rewards as rewards
import torch as th
import yaml
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from rlmpc.common.callbacks import CollectStatisticsCallback, EvalCallback, evaluate_policy
from rlmpc.networks.feature_extractors import CombinedExtractor
from rlmpc.train_rlmpc_sac import train_rlmpc_sac
from stable_baselines3.common.monitor import Monitor

# Supressing futurewarning to speed up execution time
warnings.simplefilter(action="ignore", category=FutureWarning)


# tuning:
# horizon
# tau/barrier param
# edge case shit
# constraint satisfaction highly dependent on tau/barrier
# if ship gets too much off path/course it will just continue off course
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path.home() / "Desktop/machine_learning/rlmpc/"))
    parser.add_argument("--experiment_name", type=str, default="sac_rlmpc2")
    parser.add_argument("--n_cpus", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--train_freq", type=int, default=2)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=20000)
    args = parser.parse_args(args)
    args.base_dir = Path(args.base_dir)
    print("Provided args to SAC RLMPC training:")
    print("".join(f"{k}={v}\n" for k, v in vars(args).items()))

    base_dir, log_dir, model_dir = hf.create_data_dirs(base_dir=args.base_dir, experiment_name=args.experiment_name)

    scenario_names = [
        "rlmpc_scenario_ms_channel"
    ]  # ["rlmpc_scenario_ho", "rlmpc_scenario_cr_ss", "rlmpc_scenario_random_many_vessels"]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    generate = False
    if generate:
        scenario_generator = cs_sg.ScenarioGenerator(config_file=rl_dp.config / "scenario_generator.yaml")
        for idx, name in enumerate(scenario_names):

            scenario_generator.seed(idx)
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
                n_episodes=20,
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
    scen_gen_config = cs_sg.Config.from_file(rl_dp.config / "scenario_generator.yaml")
    env_id = "COLAVEnvironment-v0"
    training_env_config = {
        "scenario_file_folder": [training_scenario_folders[0]],
        "scenario_generator_config": scen_gen_config,
        "max_number_of_episodes": 200,
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
            "reload_map": False,
            "max_number_of_episodes": 20,
            "scenario_file_folder": test_scenario_folders,
            "seed": 1,
            "simulator_config": eval_sim_config,
            "identifier": "eval_env",
        }
    )

    mpc_config_file = rl_dp.config / "rlmpc.yaml"
    # actor_noise_std_dev = np.array([0.004, 0.004, 0.025])  # normalized std dev for the action space [x, y, speed]
    actor_noise_std_dev = np.array([0.004, 0.004])  # normalized std dev for the action space [course, speed]
    mpc_param_provider_kwargs = {
        "param_list": ["Q_p", "r_safe_do"],
        "hidden_sizes": [1399, 1316, 662],
        "activation_fn": th.nn.ReLU,
        "model_file": Path.home()
        / "Desktop/machine_learning/rlmpc/dnn_pp/pretrained_dnn_pp_HD_1399_1316_662_ReLU/best_model.pth",
    }
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "critic_arch": [1500, 1000, 500],
        "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
        "mpc_config": mpc_config_file,
        "activation_fn": th.nn.ReLU,
        "std_init": actor_noise_std_dev,
        "disable_parameter_provider": True,
        "debug": False,
    }
    model_kwargs = {
        "policy": rlmpc_policies.SACPolicyWithMPC,
        "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "gradient_steps": args.gradient_steps,
        "train_freq": (args.train_freq, "step"),
        "learning_starts": 10,
        "tau": 0.005,
        "device": "cpu",
        "ent_coef": "auto",
        "verbose": 1,
        "tensorboard_log": str(log_dir),
    }
    with (base_dir / "model_kwargs.pkl").open(mode="wb") as fp:
        pickle.dump(model_kwargs, fp)

    load_model = False
    load_model_name = "sac_rlmpc1_1000"
    n_timesteps_per_learn = 2000
    n_learn_iterations = args.timesteps // n_timesteps_per_learn
    for i in range(n_learn_iterations):
        if i > 0:
            load_model = True
            load_model_name = args.experiment_name + f"_{i * n_timesteps_per_learn}"

        model = train_rlmpc_sac(
            model_kwargs=model_kwargs,
            n_timesteps=n_timesteps_per_learn,
            env_id=env_id,
            training_env_config=copy.deepcopy(training_env_config),
            n_training_envs=args.n_cpus,
            eval_env_config=copy.deepcopy(eval_env_config),
            n_eval_episodes=args.n_eval_episodes,
            eval_freq=args.eval_freq,
            base_dir=base_dir,
            model_dir=model_dir,
            experiment_name=args.experiment_name,
            load_model=load_model,
            load_model_name=load_model_name,
            load_rb_name=args.experiment_name + "_replay_buffer",
            seed=0,
            iteration=i + 1,
        )
        model.custom_save(model_dir / f"{args.experiment_name}_{(i + 1) * n_timesteps_per_learn}")
        model.save_replay_buffer(model_dir / f"{args.experiment_name}_replay_buffer")
        print(
            f"[SAC RLMPC] Finished learning iteration {i + 1}. Progress: {100.0 * (i + 1) * n_timesteps_per_learn}/{args.timesteps}%"
        )

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
        "policy_kwargs": policy_kwargs,
        "n_cpus": args.n_cpus,
        "buffer_size": args.buffer_size,
    }
    with (base_dir / "train_config.yaml").open(mode="w", encoding="utf-8") as fp:
        yaml.dump(train_cfg, fp)


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # cProfile.run("main()", sort="cumulative", filename="sac_rlmpc.prof")
    # p = pstats.Stats("sac_rlmpc.prof")
    # p.sort_stats("cumulative").print_stats(50)
    main(sys.argv[1:])
