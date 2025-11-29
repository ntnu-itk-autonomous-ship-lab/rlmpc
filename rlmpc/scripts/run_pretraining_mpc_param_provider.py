"""Pretraining MPC parameter provider. You need to have a replay buffer dataset for this to work."""

import sys
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import rlmpc.common.datasets as rl_ds
from rlmpc.policies import MPCParameterDNN
from rlmpc.scripts.train_mpc_param_provider import train_mpc_param_dnn


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_dims_list = [
        [512],
        [128, 32],
        [128, 64],
        [256, 64],
        [256, 128],
        [256, 256],
        [512, 256],
        [512, 512],
        [1024, 512],
        [512, 256, 64],
        [512, 256, 128],
        [512, 256, 256],
        [512, 512, 512],
        [1024, 512, 256],
        [1024, 512, 512],
        [1024, 1024, 512],
        [1024, 1024, 1024],
        [2048, 1024, 512],
        [2048, 1024, 1024],
        [2048, 2048, 1024],
        [2048, 2048, 2048],
        [512, 512, 512, 512],
        [1024, 1024, 1024, 1024],
        [2048, 2048, 2048, 2048],
    ]
    activation_fn_list = [
        torch.nn.ReLU,
        torch.nn.SiLU,
        torch.nn.ELU,
        torch.nn.LeakyReLU,
    ]
    input_dim = 64 + 12 + 5  # this is, excluding the MPC parameters

    load_model = False
    save_interval = 10
    batch_size = 4
    num_epochs = 30
    learning_rate = 5e-5

    data_dir = (
        Path.home()
        / "Desktop"
        / "machine_learning"
        / "rlmpc"
        / "sac_rlmpc1"
        / "final_eval_baseline2"
    )
    data_filename_list = []
    for i in range(1, 2):
        data_filename = "sac_rlmpc1_final_eval_env_data"
        data_filename_list.append(data_filename)

    dataset = torch.utils.data.ConcatDataset(
        [
            rl_ds.ParameterProviderDataset(
                env_data_pkl_file=df, data_dir=data_dir, transform=None
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

    test_model = True
    if test_model:
        model_file = (
            Path.home()
            / "machine_learning/rlmpc/dnn_pp/pretrained_dnn_pp_HD_1399_1316_662_ReLU/best_model.pth"
        )
        model = MPCParameterDNN(
            param_list=["Q_p", "r_safe_do"],
            hidden_sizes=[1399, 1316, 662],
            activation_fn=torch.nn.ReLU,
            features_dim=input_dim,
        ).to(device)
        model.load_state_dict(
            torch.load(
                str(model_file),
                map_location=torch.device("cpu"),
            )
        )
        model.eval()
        dataset.datasets[0].test_model_on_episode_data(model)

    best_experiment = ""
    best_loss_sofar = 1e20
    opt_losses = []
    opt_train_losses = []
    exp_counter = 0
    for hidden_dims in hidden_dims_list:
        for actfn in activation_fn_list:
            dnn_pp = MPCParameterDNN(
                param_list=["Q_p", "r_safe_do"],
                hidden_sizes=hidden_dims,
                activation_fn=actfn,
                features_dim=input_dim,
            ).to(device)

            hidden_dims_str = "_".join([str(hd) for hd in hidden_dims])
            name = (
                f"pretrained_dnn_pp{exp_counter + 1}_HD_"
                + hidden_dims_str
                + f"_{actfn.__name__}"
            )
            experiment_path = base_dir / name

            writer = SummaryWriter(log_dir=log_dir / name)
            optimizer = torch.optim.Adam(
                dnn_pp.parameters(), lr=learning_rate, weight_decay=1e-5
            )
            # T_max = len(train_dataloader) * num_epochs
            # lr_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2, eta_min=1e-5)
            lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)
            # lr_schedule = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

            if not experiment_path.exists():
                experiment_path.mkdir(parents=True)

            training_config = {
                "name": name,
                "hidden_dims": hidden_dims,
                "activation_fn": actfn.__name__,
                "input_dim": input_dim,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "save_interval": save_interval,
                "load_model": load_model,
            }
            with Path(experiment_path / "config.yaml").open(
                mode="w", encoding="utf-8"
            ) as fp:
                yaml.dump(training_config, fp)

            model, opt_loss, opt_train_loss, opt_epoch = train_mpc_param_dnn(
                model=dnn_pp,
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

            opt_losses.append(opt_loss)
            opt_train_losses.append(opt_train_loss)
            print(
                f"[EXPERIMENT: {exp_counter + 1}]: HD="
                + hidden_dims_str
                + f", AF={actfn.__name__} | Optimal loss: {opt_loss} at epoch {opt_epoch}"
            )

            if opt_loss < best_loss_sofar:
                best_loss_sofar = opt_loss
                best_experiment = name

            exp_counter += 1

    print(f"BEST EXPERIMENT: {best_experiment} WITH LOSS: {best_loss_sofar}")
    print("Optimal losses: ", opt_losses)


if __name__ == "__main__":
    main(sys.argv[1:])
