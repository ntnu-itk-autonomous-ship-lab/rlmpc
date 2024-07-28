import sys
from pathlib import Path

import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.policies as rlmpc_policies
import rlmpc.rewards as rewards
import torch as th
from rlmpc.networks.feature_extractors import CombinedExtractor
from rlmpc.sac import SAC
from rlmpc.train_critics import train_critics
from rlmpc.train_mpc_param_provider import train_mpc_param_dnn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for the critic hyperparameter optimization.

    Args:
        trial (optuna.Trial): The optuna trial object used for optimization

    Returns:
        float: The loss value to minimize
    """

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    input_dim = 64 + 12 + 5

    save_interval = 10
    batch_size = 16
    num_epochs = 30
    learning_rate = 5e-5

    data_dir = Path.home() / "Desktop" / "machine_learning" / "rlmpc" / "sac_rlmpc1"
    data_filename_list = []
    for i in range(1, 2):
        data_filename = "sac_rlmpc1_final_eval_env_data"
        data_filename_list.append(data_filename)

    # print(f"Training dataset length: {len(train_dataset)} | Test dataset length: {len(test_dataset)}")
    # print(f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}")

    base_dir = Path.home() / "Desktop/machine_learning/rlmpc/sac_critics"
    log_dir = base_dir / "logs"

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    batch_size = 8  # trial.suggest_int("batch_size", 1, 32)
    buffer_size = 50000
    train_freq = 2
    tau = 0.005
    gradient_steps = 1
    learning_rate = 5e-5  # trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    num_epochs = 30  # trial.suggest_int("num_epochs", 10, 100)
    actfn = th.nn.ReLU
    actfn_str = "ReLU"

    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dims = []
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 32, 2048)
        hidden_dims.append(out_features)

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
        "critic_arch": hidden_dims,
        "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
        "mpc_config": mpc_config_file,
        "activation_fn": actfn,
        "std_init": actor_noise_std_dev,
        "disable_parameter_provider": False,
        "debug": False,
    }
    model_kwargs = {
        "policy": rlmpc_policies.SACPolicyWithMPC,
        "policy_kwargs": policy_kwargs,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gradient_steps": gradient_steps,
        "train_freq": (train_freq, "step"),
        "learning_starts": 0,
        "tau": tau,
        "device": "cpu",
        "ent_coef": "auto",
        "verbose": 1,
        "tensorboard_log": str(log_dir),
    }
    model = SAC(env=env, **model_kwargs)

    hidden_dims_str = "_".join([str(hd) for hd in hidden_dims])
    name = "pretrained_sac_critics_HD_" + hidden_dims_str + f"_{actfn_str}"
    experiment_path = base_dir / name

    writer = SummaryWriter(log_dir=log_dir / name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)

    if not experiment_path.exists():
        experiment_path.mkdir(parents=True)

    model, opt_loss, opt_train_loss, opt_epoch = train_sac_critics(
        model=model,
        writer=writer,
        n_epochs=num_epochs,
        batch_size=batch_size,
        save_interval=save_interval,
        device=device,
        experiment_path=experiment_path,
    )

    return opt_loss


def main(args):
    study_name = "dnn_mpc_param_provider_hpo"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name,
        storage=storage_name,
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, n_trials=50000)

    print(f"Best objective value: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")


if __name__ == "__main__":
    main(sys.argv[1:])
