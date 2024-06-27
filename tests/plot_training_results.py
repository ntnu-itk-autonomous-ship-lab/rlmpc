import pickle
import platform
from pathlib import Path

import rlmpc.common.helper_functions as hf
import rlmpc.common.plotters as plotters

if __name__ == "__main__":
    base_dir: Path = Path.home() / "Desktop/machine_learning/rlmpc/"

    base_experiment_name = "sac_rlmpc1"
    experiment_names = ["4"]
    env_data_list = []
    training_stats_list = []
    reward_data_list = []
    training_stats_list = []

    plot_training_results = True
    plot_env_snapshots = True
    plot_reward_curves = True
    for experiment_name in experiment_names:
        env_data_info_path = (
            base_dir / base_experiment_name / (base_experiment_name + f"_env_training_data{experiment_name}.pkl")
        )
        training_stats_data_path = (
            base_dir / base_experiment_name / (base_experiment_name + f"_training_stats{experiment_name}.pkl")
        )
        with open(training_stats_data_path, "rb") as f:
            training_stats = pickle.load(f)
        smoothed_training_stats = hf.process_rl_training_data(training_stats, ma_window_size=5)
        training_stats_list.append(smoothed_training_stats)

        if plot_env_snapshots or plot_reward_curves:
            with open(env_data_info_path, "rb") as f:
                env_data = pickle.load(f)
            env_data_list.append(env_data)

            reward_data = hf.extract_reward_data(env_data)
            reward_data_list.append(reward_data)

        if plot_env_snapshots:
            plotters.plot_single_model_training_enc_snapshots(
                env_data[0], nrows=5, ncols=3, save_fig=True, save_path=base_dir / base_experiment_name / "figures"
            )

    if plot_reward_curves:
        plotters.plot_multiple_model_reward_curves(
            reward_data_list,
            model_names=experiment_names,
            save_fig=True,
            save_path=base_dir / base_experiment_name / "figures",
        )
    if plot_training_results:
        plotters.plot_multiple_model_training_stats(
            training_stats_list,
            model_names=experiment_names,
            save_fig=True,
            save_path=base_dir / base_experiment_name / "figures",
        )

    print("Done")
