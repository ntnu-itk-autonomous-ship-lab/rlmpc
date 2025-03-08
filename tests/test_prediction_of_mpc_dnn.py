import sys
from pathlib import Path

import optuna
import rlmpc.common.datasets as rl_ds
import torch
from rlmpc.policies import MPCParameterDNN
from torch.utils.data import DataLoader


def test_prediction_of_mpc_dnn() -> None:
    input_dim = 40 + 12 + 5 + 9

    batch_size = 8
    mpc_param_list = ["Q_p", "K_app_course", "K_app_speed", "w_colregs", "r_safe_do"]

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

    actfn_str = "ReLU"
    actfn = getattr(torch.nn, actfn_str)
    hidden_dims = [458, 242, 141]

    base_dir = Path.home() / "Desktop/machine_learning/rlmpc/dnn_pp"
    model = MPCParameterDNN(
        param_list=mpc_param_list,
        hidden_sizes=hidden_dims,
        activation_fn=actfn,
        features_dim=input_dim,
        model_file=base_dir / "pretrained_dnn_pp_HD_458_242_141_ReLU/best_model.pth",
    )
    model.eval()
    test_dataset.dataset.datasets[0].test_model_on_episode_data(model)
    print("Model tested on episode data")


if __name__ == "__main__":
    test_prediction_of_mpc_dnn()
