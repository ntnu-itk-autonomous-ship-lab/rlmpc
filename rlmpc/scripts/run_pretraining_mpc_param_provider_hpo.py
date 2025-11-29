"""Pretraining MPC parameter provider with HPO. You need to have a replay buffer dataset for this to work."""

import sys
from pathlib import Path

import optuna
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import rlmpc.common.datasets as rl_ds
from rlmpc.policies import MPCParameterDNN
from rlmpc.scripts.train_mpc_param_provider import train_mpc_param_dnn


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for the DNN hyperparameter optimization.

    Args:
        trial (optuna.Trial): The optuna trial object used for optimization

    Returns:
        float: The loss value to minimize
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mpc_param_list = ["Q_p", "K_app_course", "K_app_speed", "w_colregs", "r_safe_do"]

    input_dim = 40 + 12 + 5 + 9
    batch_size = 32
    learning_rate = 0.0002  # trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    num_epochs = 40
    save_interval = 10

    experiment_name = "sac_rlmpc_pp_eval1"
    data_dir = (
        Path.home()
        / "Desktop"
        / "machine_learning"
        / "rlmpc"
        / experiment_name
        / "final_eval"
    )
    data_filename_list = []
    for i in range(1, 2):
        data_filename = f"{experiment_name}_final_eval_env_data"
        data_filename_list.append(data_filename)

    dataset = torch.utils.data.ConcatDataset(
        [
            rl_ds.ParameterProviderDataset(
                env_data_pkl_file=df,
                data_dir=data_dir,
                param_list=mpc_param_list,
                transform=None,
            )
            for df in data_filename_list
        ]
    )

    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if trial.number == 0:
        print(
            f"Training dataset length: {len(train_dataset)} | Test dataset length: {len(test_dataset)}"
        )
        print(
            f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}"
        )

    base_dir = Path.home() / "machine_learning/rlmpc/dnn_pp"
    log_dir = base_dir / "logs"

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    n_layers = trial.suggest_int("n_layers", 2, 3)
    actfn_str = "ReLU"  # trial.suggest_categorical("activation_fn", ["ReLU"])
    actfn = getattr(torch.nn, actfn_str)
    hidden_dims = []
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 100, 500)
        hidden_dims.append(out_features)

    model = MPCParameterDNN(
        param_list=mpc_param_list,
        hidden_sizes=hidden_dims,
        activation_fn=actfn,
        features_dim=input_dim,
    ).to(device)

    hidden_dims_str = "_".join([str(hd) for hd in hidden_dims])
    name = "pretrained_dnn_pp_HD_" + hidden_dims_str + f"_{actfn_str}"
    experiment_path = base_dir / name

    writer = SummaryWriter(log_dir=log_dir / name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)

    if not experiment_path.exists():
        experiment_path.mkdir(parents=True)

    model, opt_loss, opt_train_loss, opt_epoch = train_mpc_param_dnn(
        model=model,
        training_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        writer=writer,
        n_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        save_interval=save_interval,
        device=device,
        early_stopping_patience=3,
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
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1000)

    print(f"Best objective value: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")


if __name__ == "__main__":
    main(sys.argv[1:])
