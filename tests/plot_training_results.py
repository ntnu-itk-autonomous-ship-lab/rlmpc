import pickle
import platform
from pathlib import Path

import colav_simulator.gym.logger as csenv_logger
import rlmpc.common.helper_functions as hf
import rlmpc.common.logger as rlmpc_logger
import rlmpc.common.plotters as plotters

if __name__ == "__main__":
    base_dir: Path = Path.home() / "Desktop/machine_learning/rlmpc"
    experiment_names = ["sac_rlmpc1"]

    env_data_list = []
    training_stats_list = []
    reward_data_list = []
    training_stats_list = []

    plot_training_results = True
    plot_env_snapshots = True
    plot_reward_curves = True
    for experiment_name in experiment_names:
        log_dir = base_dir / experiment_name

        rl_data_logger = rlmpc_logger.Logger(experiment_name=experiment_name, log_dir=log_dir)
        rl_data_logger.load_from_pickle(f"{experiment_name}_training_stats")
        smoothed_training_stats = hf.process_rl_training_data(rl_data_logger.rl_data, ma_window_size=5)
        training_stats_list.append(smoothed_training_stats)

        if plot_env_snapshots or plot_reward_curves:
            env_logger = csenv_logger.Logger(experiment_name=experiment_name, log_dir=log_dir, save_freq=10)
            env_logger.load_from_pickle(f"{experiment_name}_env_training_data")
            env_data_list.append(env_logger.env_data)

            reward_data = hf.extract_reward_data(env_logger.env_data)
            reward_data_list.append(reward_data)

        if plot_env_snapshots:
            plotters.plot_single_model_training_enc_snapshots(
                env_logger.env_data,
                nrows=5,
                ncols=3,
                save_fig=True,
                save_path=base_dir / experiment_name / "figures",
            )

    if plot_reward_curves:
        plotters.plot_multiple_model_reward_curves(
            reward_data_list,
            model_names=experiment_names,
            save_fig=True,
            save_path=base_dir / experiment_name / "figures",
        )
    if plot_training_results:
        plotters.plot_multiple_model_training_stats(
            training_stats_list,
            model_names=experiment_names,
            save_fig=True,
            save_path=base_dir / experiment_name / "figures",
        )

    print("Done")
