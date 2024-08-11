import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna
import rlmpc.common.paths as rl_dp
import rlmpc.policies as rlmpc_policies
import torch as th
from rlmpc.networks.feature_extractors import CombinedExtractor
from rlmpc.sac import SAC
from rlmpc.scripts.train_critics import train_critics
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for the critic hyperparameter optimization.

    Args:
        trial (optuna.Trial): The optuna trial object used for optimization

    Returns:
        float: The loss value to minimize
    """
    base_dir = Path.home() / "Desktop/machine_learning/rlmpc/sac_critics"
    log_dir = base_dir / "logs"

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    env_id = "COLAVEnvironment-v0"
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
    scenario_names = ["rlmpc_scenario_ms_channel"]
    scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": [scenario_folders[0]],
        "max_number_of_episodes": 1,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        "render_update_rate": 0.5,
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": False,
        "identifier": "env",
        "seed": 0,
    }
    env = Monitor(gym.make(id=env_id, **env_config))

    save_interval = 5
    batch_size = 64  # trial.suggest_int("batch_size", 1, 32)
    buffer_size = 40000
    tau = 0.01
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    num_epochs = 200  # trial.suggest_int("num_epochs", 10, 100)
    actfn = th.nn.ReLU
    actfn_str = "ReLU"

    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = []
    input_dim = 64 + 12 + 5 + 2  # enc + tracking + nav + action
    prev_input_dim = input_dim
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 64, 3000)
        prev_input_dim = out_features
        hidden_dims.append(out_features)
    # hidden_dims = [1500, 1000, 500]

    mpc_config_file = rl_dp.config / "rlmpc.yaml"
    # actor_noise_std_dev = np.array([0.004, 0.004, 0.025])  # normalized std dev for the action space [x, y, speed]
    actor_noise_std_dev = np.array([0.004, 0.004])  # normalized std dev for the action space [course, speed]
    mpc_param_provider_kwargs = {
        "param_list": ["Q_p", "r_safe_do"],
        "hidden_sizes": [1315, 1579],
        "activation_fn": th.nn.ReLU,
        "model_file": Path.home()
        / "Desktop/machine_learning/rlmpc/dnn_pp/pretrained_dnn_pp_HD_1315_1579_ReLU/best_model.pth",
    }
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "critic_arch": hidden_dims,
        "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
        "mpc_config": mpc_config_file,
        "activation_fn": actfn,
        "std_init": actor_noise_std_dev,
        "disable_parameter_provider": False,
        "optimizer_class": th.optim.Adam,
        # "optimizer_kwargs": {"weight_decay": 5e-5},
        "debug": False,
    }
    model_kwargs = {
        "policy": rlmpc_policies.SACPolicyWithMPC,
        "policy_kwargs": policy_kwargs,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gradient_steps": 1,
        "train_freq": (2, "step"),
        "learning_starts": 0,
        "tau": tau,
        "device": "cpu",
        "ent_coef": "auto",
        "verbose": 1,
        "tensorboard_log": str(log_dir),
    }

    data_path = (
        Path.home() / "Desktop" / "machine_learning" / "rlmpc" / "sac_rlmpc3" / "models" / "sac_rlmpc1_replay_buffer"
    )

    model = SAC(env=env, **model_kwargs)
    model.load_replay_buffer(path=data_path)

    hidden_dims_str = "_".join([str(hd) for hd in hidden_dims])
    name = "pretrained_sac_critics_HD_" + hidden_dims_str + f"_{actfn_str}"
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
    )
    return opt_train_loss


def main(args):
    study_name = "sac_critics_hpo"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name,
        storage=storage_name,
        sampler=optuna.samplers.RandomSampler(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=200)

    print(f"Best objective value: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")


if __name__ == "__main__":
    main(sys.argv[1:])
