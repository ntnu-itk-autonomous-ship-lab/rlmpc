"""Pretraining standard SAC critics with HPO. You need to have a replay buffer dataset for this to work."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

import rlmpc.action as rlmpc_actions
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.policies as rlmpc_policies
from rlmpc.networks.feature_extractors import CombinedExtractor
from rlmpc.scripts.train_critics import train_critics
from rlmpc.standard_sac import SAC


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for the critic hyperparameter optimization.

    Args:
        trial (optuna.Trial): The optuna trial object used for optimization

    Returns:
        float: The loss value to minimize
    """
    base_dir = Path.home() / "machine_learning/rlmpc/standard_sac_critics"
    log_dir = base_dir / "logs"

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    env_id = "COLAVEnvironment-v0"
    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            "perception_image_observation",
            "relative_tracking_observation",
            "time_observation",
            "mpc_parameter_observation",
        ]
    }
    mpc_config_path = rl_dp.config / "rlmpc.yaml"
    mpc_param_list = ["Q_p", "K_app_course", "K_app_speed", "w_colregs", "r_safe_do"]
    action_noise_std_dev = np.array(
        [0.0001, 0.0001]
    )  # normalized std dev for the action space [course, speed]
    n_mpc_params = 3 + 1 + 1 + 3 + 1
    param_action_noise_std_dev = np.array([0.005 for _ in range(n_mpc_params)])
    action_kwargs = {
        "mpc_config_path": mpc_config_path,
        "debug": False,
        "mpc_param_list": mpc_param_list,
        "std_init": action_noise_std_dev,
        "deterministic": True,
        "recompile_on_reset": False,
        "disable_mpc_info_storage": True,
        "acados_code_gen_path": str(base_dir.parents[0]) + "/acados_code_gen",
    }
    scenario_names = ["rlmpc_scenario_ms_channel"]
    scenario_folders = [
        rl_dp.scenarios / "training_data" / name for name in scenario_names
    ]
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": [scenario_folders[0]],
        "max_number_of_episodes": 1,
        "render_update_rate": 0.5,
        "observation_type": observation_type,
        "action_type_class": rlmpc_actions.MPCParameterSettingAction,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        "action_kwargs": action_kwargs,
        "reload_map": False,
        "identifier": "env",
        "seed": 0,
    }

    n_cpus_used = 16
    if n_cpus_used == 1:
        env = Monitor(gym.make(id=env_id, **env_config))
    else:
        env = SubprocVecEnv(
            [hf.make_env(env_id, env_config, i + 1) for i in range(n_cpus_used)]
        )

    save_interval = 20
    batch_size = 32  # trial.suggest_int("batch_size", 1, 64)
    buffer_size = 150000
    tau = 0.008
    learning_rate = 0.0002
    num_epochs = 15  # trial.suggest_int("num_epochs", 10, 100)
    actfn_str = "ReLU"

    n_layers = trial.suggest_int("n_layers", 2, 3)
    hidden_dims = []
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 128, 500)
        hidden_dims.append(out_features)

    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)

    mpc_param_provider_kwargs = {
        "param_list": mpc_param_list,
        "hidden_sizes": [256, 256],  # [458, 242, 141],
        "activation_fn": th.nn.ReLU,
        # "model_file": Path.home()
        # / "machine_learning/rlmpc/dnn_pp/pretrained_dnn_pp_HD_458_242_141_ReLU/best_model.pth",
    }
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "critic_arch": hidden_dims,
        "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
        "activation_fn": th.nn.ReLU,
        "std_init": param_action_noise_std_dev,
        "use_sde": True,
        "full_std": True,
        "use_expln": False,
        "clip_mean": False,
    }
    model_kwargs = {
        "policy": rlmpc_policies.SACPolicyWithMPCParameterProviderStandard,
        "policy_kwargs": policy_kwargs,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gradient_steps": 2,
        "train_freq": (8, "step"),
        "learning_starts": 0,
        "tau": tau,
        "device": "cpu",
        "ent_coef": "auto",
        "verbose": 1,
        "tensorboard_log": str(log_dir),
        "replay_buffer_kwargs": {
            "handle_timeout_termination": True,
            "disable_action_storage": False,
        },
    }

    experiment_name = "standard_snmpc_1te_4ee_seed1_jid20787312"
    model_path = (
        Path.home()
        / "Desktop"
        / "machine_learning"
        / "rlmpc"
        / experiment_name
        / "models"
        / "standard_snmpc_1te_4ee_seed1_jid20787312_63987120_steps.zip"
    )
    rb_data_path = (
        Path.home()
        / "Desktop"
        / "machine_learning"
        / "rlmpc"
        / experiment_name
        / "models"
        / (experiment_name + "_replay_buffer")
    )

    model = SAC(env=env, **model_kwargs)
    model.load_replay_buffer(path=rb_data_path)

    print(f"Replay buffer size: {model.replay_buffer.size()}")

    hidden_dims_str = "_".join([str(hd) for hd in hidden_dims])
    name = (
        "pretrained_ssac_critics_HD_"
        + hidden_dims_str
        + f"_{actfn_str}"
        + f"_ent_coef{ent_coef:5f}"
    )
    experiment_path = base_dir / name
    if not experiment_path.exists():
        experiment_path.mkdir(parents=True)

    writer = SummaryWriter(log_dir=log_dir / name)

    model, opt_train_loss, opt_epoch = train_critics(
        model=model,
        writer=writer,
        n_epochs=num_epochs,
        batch_size=batch_size,
        experiment_path=experiment_path,
        save_interval=save_interval,
        save_intermittent_models=False,
        early_stopping_patience=6,
        verbose=False,
        ent_coef=ent_coef,
        optuna_trial=trial,
    )
    return opt_train_loss


def main(args):
    study_name = "sac_critics_hpo2"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name,
        storage=storage_name,
        sampler=optuna.samplers.RandomSampler(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1000)

    print(f"Best objective value: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")


if __name__ == "__main__":
    main(sys.argv[1:])
