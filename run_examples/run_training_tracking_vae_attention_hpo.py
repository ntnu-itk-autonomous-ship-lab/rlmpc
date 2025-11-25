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
from rlmpc.networks.tracking_vae_attention.vae import VAE
from rlmpc.scripts.train_tracking_vae_attention import train_vae


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for the DNN hyperparameter optimization.

    Args:
        trial (optuna.Trial): The optuna trial object used for optimization

    Returns:
        float: The loss value to minimize
    """
    BASE_PATH: Path = Path.home() / "machine_learning/tracking_vae/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    latent_dim = trial.suggest_categorical("latent_dim", [8, 10, 12])
    rnn_hidden_dim = trial.suggest_categorical(
        "rnn_hidden_dim", [8, 16, 32, 64, 128, 256]
    )
    num_rnn_layers_decoder = trial.suggest_categorical("num_rnn_layers_decoder", [1, 2])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    beta = 0.01  # trial.suggest_float("beta", 0.05, 1.0, step=0.1)
    n_iter = 50000
    for i in range(n_iter):
        # embedding_dim = trial.suggest_categorical("embedding_dim", [5, 10, 16, 20, 24, 32, 40, 48, 64])
        # embedding_dim *= num_heads
        embedding_dim = trial.suggest_categorical(
            "embedding_dim", [8, 16, 32, 64, 128, 256, 512]
        )
        if embedding_dim % num_heads == 0:
            break
    # if embedding_dim < 30:
    #     embedding_dim *= num_heads
    print(f"Embedding dim: {embedding_dim}")
    learning_rate = 0.0002

    input_dim = 4
    save_interval = 20
    batch_size = 256
    num_epochs = 20

    data_dir = Path.home() / "machine_learning/tracking_vae/data"
    data_filename_list = []
    for i in range(1, 199):
        training_data_filename = f"tracking_vae_training_data_rogaland_new{i}.npy"
        data_filename_list.append(training_data_filename)

    for i in range(199):
        data_filename_list.append(f"tracking_vae_test_data_rogaland_new{i}.npy")

    full_dataset = torch.utils.data.ConcatDataset(
        [
            rl_ds.TrackingObservationDataset(data_file, data_dir)
            for data_file in data_filename_list
        ]
    )

    training_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [int(0.85 * len(full_dataset)), int(0.15 * len(full_dataset))]
    )
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(
        f"Training dataset length: {len(training_dataset)} | Test dataset length: {len(test_dataset)}"
    )
    print(
        f"Training dataloader length: {len(train_dataloader)} | Test dataloader length: {len(test_dataloader)}"
    )

    log_dir = BASE_PATH / "logs"

    bidirectional = False
    vae = VAE(
        latent_dim=latent_dim,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_rnn_layers_decoder,
        rnn_type=torch.nn.GRU,
        rnn_hidden_dim=rnn_hidden_dim,
        bidirectional=bidirectional,
    ).to(device)

    print(f"Model size: {sum(p.numel() for p in vae.parameters() if p.requires_grad)}")
    name = f"tracking_avae_mdnew_beta001_{trial.number}_NL_{num_rnn_layers_decoder}_nonbi_HD_{rnn_hidden_dim}_LD_{latent_dim}_NH_{num_heads}_ED_{embedding_dim}"
    experiment_path = BASE_PATH / name

    writer = SummaryWriter(log_dir=log_dir / name)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    # T_max = len(train_dataloader) * num_epochs
    # lr_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2, eta_min=3e-5)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)
    # lr_schedule = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if not experiment_path.exists():
        experiment_path.mkdir(parents=True)

    try:
        model, opt_loss, opt_train_loss, opt_epoch = train_vae(
            model=vae,
            training_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            writer=writer,
            n_epochs=num_epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            save_interval=save_interval,
            device=device,
            early_stopping_patience=10,  # num_epochs,
            experiment_path=experiment_path,
            verbose=False,
            save_intermittent_models=False,
            beta=beta,
            optuna_trial=trial,
        )
    except Exception as e:
        print(f"Exception caught: {e}")
        opt_loss = 1000.0
    return opt_loss


def main(args):
    study_name = "tracking_vae_hpo_moredata4"
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
