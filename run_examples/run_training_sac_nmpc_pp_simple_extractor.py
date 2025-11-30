"""Training a custom SAC-RLMPC agent with an NMPC parameter provider DNN policy and a simpler feature extractor.

Note, you need to generate the scenario episode data first,
e.g. using the generate_scenario_episodes.py script.
"""
import argparse
import copy
import pickle
import sys
import warnings
from pathlib import Path

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import numpy as np
import torch as th
import yaml
from stable_baselines3.common.monitor import Monitor

import rlmpc.action as rlmpc_actions
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.policies as rlmpc_policies
import rlmpc.rewards as rewards
from rlmpc.common.callbacks import evaluate_policy
from rlmpc.networks.feature_extractors import SimpleCombinedExtractor
from rlmpc.scripts.train_rlmpc_sac import train_rlmpc_sac

# Supressing futurewarning to speed up execution time
warnings.simplefilter(action="ignore", category=FutureWarning)


# @profile
def main(args):
    # hf.set_memory_limit(28_000_000_000)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(Path.home() / "machine_learning/rlmpc/"),
    )
    parser.add_argument("--experiment_name", type=str, default="snmpc_pp1")
    parser.add_argument("--n_training_envs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_steps", type=int, default=2)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--n_eval_episodes", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--n_eval_envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_timesteps_per_learn", type=int, default=20000)
    parser.add_argument("--disable_parameter_provider", type=bool, default=False)
    parser.add_argument("--max_num_loaded_train_scen_episodes", type=int, default=1)
    parser.add_argument("--max_num_loaded_eval_scen_episodes", type=int, default=4)
    parser.add_argument("--load_model_name", type=str, default="")
    parser.add_argument("--load_critics", default=False, action="store_true")
    parser.add_argument("--reset_num_timesteps", default=True, action="store_true")
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args(args)
    args.base_dir = Path(args.base_dir)
    print("Provided args to training SAC with NMPC parameter provider DNN:")
    print("".join(f"{k}={v}\n" for k, v in vars(args).items()))

    base_dir, log_dir, model_dir = hf.create_data_dirs(
        base_dir=args.base_dir,
        experiment_name=args.experiment_name,
        remove_log_files=False,
    )

    scenario_names = [
        "rlmpc_scenario_ms_channel"
    ]  # ["rlmpc_scenario_ho", "rlmpc_scenario_cr_ss", "rlmpc_scenario_random_many_vessels"]
    training_scenario_folders = [
        rl_dp.scenarios / "training_data" / name for name in scenario_names
    ]
    test_scenario_folders = [
        rl_dp.scenarios / "test_data" / name for name in scenario_names
    ]

    # map_size: [4000.0, 4000.0]
    # map_origin_enu: [-33524.0, 6572500.0]
    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            "perception_image_observation",
            "relative_tracking_observation",
            "time_observation",
            # "mpc_parameter_observation",
        ]
    }
    env_id = "COLAVEnvironment-v0"
    rewarder_config = rewards.Config.from_file(rl_dp.config / "rewarder.yaml")
    training_sim_config = cs_sim.Config.from_file(
        rl_dp.config / "training_simulator.yaml"
    )
    eval_sim_config = cs_sim.Config.from_file(rl_dp.config / "eval_simulator.yaml")
    scen_gen_config = cs_sg.Config.from_file(rl_dp.config / "scenario_generator.yaml")
    mpc_config_path = rl_dp.config / "rlmpc.yaml"
    mpc_param_list = ["Q_p", "K_app_course", "K_app_speed", "w_colregs", "r_safe_do"]
    n_mpc_params = 3 + 1 + 1 + 3 + 1

    # action_noise_std_dev = np.array([0.004, 0.004, 0.025])  # normalized std dev for the action space [x, y, speed]
    action_noise_std_dev = np.array(
        [0.002, 0.002]
    )  # normalized std dev for the action space [course, speed]
    param_action_noise_std_dev = np.array([0.5 for _ in range(n_mpc_params)])
    action_kwargs = {
        "mpc_config_path": mpc_config_path,
        "debug": False,
        "mpc_param_list": mpc_param_list,
        "std_init": action_noise_std_dev,
        "deterministic": False,
        "recompile_on_reset": False,
        "acados_code_gen_path": str(base_dir.parents[0])
        + f"/{args.experiment_name}/acados_code_gen",
    }
    training_env_config = {
        "scenario_file_folder": [training_scenario_folders[0]],
        "scenario_generator_config": scen_gen_config,
        "max_number_of_episodes": args.max_num_loaded_train_scen_episodes,
        "simulator_config": training_sim_config,
        "action_type_class": rlmpc_actions.MPCParameterSettingAction,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        "action_kwargs": action_kwargs,
        "rewarder_class": rewards.MPCRewarder,
        "rewarder_kwargs": {"config": rewarder_config},
        "render_update_rate": 0.5,
        "render_mode": "rgb_array",
        "observation_type": observation_type,
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "merge_loaded_scenario_episodes": True,
        "shuffle_loaded_scenario_data": False,
        "identifier": "training_env_" + args.experiment_name,
        "seed": args.seed,
        "verbose": True,
    }

    eval_env_config = copy.deepcopy(training_env_config)
    eval_env_config.update(
        {
            "reload_map": False,
            "max_number_of_episodes": args.max_num_loaded_eval_scen_episodes,
            "scenario_file_folder": test_scenario_folders,
            "seed": args.seed + 1,
            "simulator_config": eval_sim_config,
            "identifier": "eval_env_" + args.experiment_name,
        }
    )

    load_model = True if not args.load_model_name == "" else False

    load_model_name = (
        args.load_model_name
        if not args.load_model_name
        else "snmpc_db_200te_5ee_16cpus"
    )
    rb_load_name = (
        args.load_model_name
        if not args.load_model_name
        else "snmpc_db_200te_5ee_16cpus"
    )
    model_path = (
        str(base_dir.parents[0])
        + f"/{load_model_name}/models/{load_model_name}_71888_steps"
    )
    load_rb_path = (
        str(base_dir.parents[0])
        + f"/{rb_load_name}/models/{rb_load_name}_replay_buffer.pkl"
    )

    load_critic = False  # args.load_critics
    load_critic_path = (
        str(base_dir.parents[0])
        + "/sac_critics/pretrained_sac_critics_HD_495_498_ReLU/models/best_model"
    )
    #     load_critic_path = model_path

    mpc_param_provider_kwargs = {
        "param_list": mpc_param_list,
        "hidden_sizes": [500, 500],  # [458, 242, 141],
        "activation_fn": th.nn.ReLU,
        # "model_file": Path.home()
        # / "machine_learning/rlmpc/dnn_pp/pretrained_dnn_pp_HD_458_242_141_ReLU/best_model.pth",
    }
    policy_kwargs = {
        "features_extractor_class": SimpleCombinedExtractor,
        "critic_arch": [495, 498],
        "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
        "activation_fn": th.nn.ReLU,
        "std_init": param_action_noise_std_dev,
        "mpc_std_init": action_noise_std_dev,
        "disable_parameter_provider": args.disable_parameter_provider,
    }
    model_kwargs = {
        "policy": rlmpc_policies.SACPolicyWithMPCParameterProvider,
        "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "gradient_steps": args.gradient_steps,
        "train_freq": (args.train_freq, "step"),
        "learning_starts": 100 if not load_model else 0,
        "tau": 0.009,
        "device": args.device,
        "ent_coef": "auto",
        "verbose": 1,
        "tensorboard_log": str(log_dir),
        "replay_buffer_kwargs": {
            "handle_timeout_termination": True,
            "disable_action_storage": False,
        },
    }
    with (base_dir / "model_kwargs.pkl").open(mode="wb") as fp:
        pickle.dump(model_kwargs, fp)

    n_learn_iterations = args.timesteps // args.n_timesteps_per_learn
    vecenv_failed = False
    timesteps_completed = 0
    episodes_completed = 0
    reset_num_timesteps = args.reset_num_timesteps
    for i in range(n_learn_iterations):
        if i > 0:
            load_critic = False
            load_model = True
            load_rb_path = (
                str(model_dir) + "/" + args.experiment_name + "_replay_buffer.pkl"
            )
            model_kwargs["learning_starts"] = 0
            reset_num_timesteps = False

        model, vecenv_failed = train_rlmpc_sac(
            model_kwargs=model_kwargs,
            n_timesteps=args.n_timesteps_per_learn,
            env_id=env_id,
            training_env_config=copy.deepcopy(training_env_config),
            n_training_envs=args.n_training_envs,
            eval_env_config=copy.deepcopy(eval_env_config),
            n_eval_envs=args.n_eval_envs,
            n_eval_episodes=args.n_eval_episodes,
            eval_freq=args.eval_freq,
            base_dir=base_dir,
            model_dir=model_dir,
            experiment_name=args.experiment_name,
            load_critics=load_critic,
            load_critics_path=load_critic_path,
            load_model=load_model,
            load_model_path=model_path,
            load_rb_path=load_rb_path,
            seed=args.seed,
            iteration=i + 1,
            reset_num_timesteps=reset_num_timesteps,
        )
        timesteps_completed = model.num_timesteps
        episodes_completed = model.num_episodes
        model_path = model_dir / f"{args.experiment_name}_{timesteps_completed}_steps"
        model.save(model_path)
        model.save_replay_buffer(model_dir / f"{args.experiment_name}_replay_buffer")
        print(
            f"[SAC RLMPC] Replay buffer size: {model.replay_buffer.size()} | Current num timesteps: {timesteps_completed} | Current num episodes: {episodes_completed}"
        )
        print(
            f"[SAC RLMPC] Finished learning iteration {i + 1}. Progress: {100.0 * timesteps_completed / args.timesteps:.1f}% | VecEnv failed: {vecenv_failed}"
        )
        if i < n_learn_iterations - 1:
            del model

    eval_env = Monitor(gym.make(id=env_id, **eval_env_config))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        record=True,
        record_path=base_dir / "eval_data" / "final_eval_videos",
        record_name=args.experiment_name + "_final_eval",
    )
    print(
        f"{args.experiment_name} final evaluation | mean_reward: {mean_reward}, std_reward: {std_reward}"
    )
    train_cfg = {
        "n_timsteps_per_learn": args.n_timesteps_per_learn,
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
