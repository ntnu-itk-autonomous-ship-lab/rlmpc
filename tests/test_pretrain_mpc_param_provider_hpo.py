import sys
from pathlib import Path

import optuna
import rlmpc.common.datasets as rl_ds
import torch
from rlmpc.policies import MPCParameterDNN
from rlmpc.train_mpc_param_provider import train_mpc_param_dnn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for the DNN hyperparameter optimization.

    Args:
        trial (optuna.Trial): The optuna trial object used for optimization

    Returns:
        float: The loss value to minimize
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_dim = 64 + 12 + 5  # this is, excluding the MPC parameters

    save_interval = 10
    batch_size = 4
    num_epochs = 30
    learning_rate = 5e-5

    data_dir = Path.home() / "Desktop" / "machine_learning" / "rlmpc" / "sac_rlmpc1" / "final_eval_baseline2"
    data_filename_list = []
    for i in range(1, 2):
        data_filename = "sac_rlmpc1_final_eval_env_data"
        data_filename_list.append(data_filename)

    dataset = torch.utils.data.ConcatDataset(
        [
            rl_ds.ParameterProviderDataset(env_data_pkl_file=df, data_dir=data_dir, transform=None)
            for df in data_filename_list
        ]
    )

    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # print(f"Training dataset length: {len(train_dataset)} | Test dataset length: {len(test_dataset)}")
    # print(f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}")

    base_dir = Path.home() / "Desktop/machine_learning/rlmpc/dnn_pp"
    log_dir = base_dir / "logs"

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    batch_size = 8  # trial.suggest_int("batch_size", 1, 32)
    learning_rate = 5e-5  # trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    num_epochs = 30  # trial.suggest_int("num_epochs", 10, 100)

    n_layers = trial.suggest_int("n_layers", 1, 4)
    actfn_str = trial.suggest_categorical("activation_fn", ["ReLU", "SiLU", "ELU"])
    actfn = getattr(torch.nn, actfn_str)
    hidden_dims = []
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 32, 2048)
        hidden_dims.append(out_features)

    model = MPCParameterDNN(
        param_list=["Q_p", "r_safe_do"], hidden_sizes=hidden_dims, activation_fn=actfn, features_dim=input_dim
    )

    hidden_dims_str = "_".join([str(hd) for hd in hidden_dims])
    name = "pretrained_dnn_pp_HD_" + hidden_dims_str + f"_{actfn_str}"
    experiment_path = base_dir / name

    writer = SummaryWriter(log_dir=log_dir / name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
    )
    study.optimize(objective, n_trials=50000)

    print(f"Best objective value: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")


if __name__ == "__main__":
    main(sys.argv[1:])
