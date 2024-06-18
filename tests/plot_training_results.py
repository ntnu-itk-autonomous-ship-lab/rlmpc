import pickle
import platform
from pathlib import Path

import rlmpc.common.helper_functions as hf
import rlmpc.common.plotters as plotters

if __name__ == "__main__":
    if platform.system() == "Linux":
        base_dir = Path("/home/doctor/Desktop/machine_learning/rlmpc/")
    elif platform.system() == "Darwin":
        base_dir = Path("/Users/trtengesdal/Desktop/machine_learning/rlmpc/")

    experiment_names = ["sac_rlmpc1"]
    env_data_list = []
    training_stats_list = []
    reward_data_list = []
    training_stats_list = []

    for experiment_name in experiment_names:
        env_data_info_path = base_dir / experiment_name / (experiment_name + "_env_training_data.pkl")
        training_stats_data_path = base_dir / experiment_name / (experiment_name + "_training_stats.pkl")

        with open(env_data_info_path, "rb") as f:
            env_data = pickle.load(f)

        with open(training_stats_data_path, "rb") as f:
            training_stats = pickle.load(f)

        env_data_list.append(env_data)

        smoothed_training_stats = hf.process_rl_training_data(training_stats, ma_window_size=5)
        training_stats_list.append(smoothed_training_stats)

        reward_data = hf.extract_reward_data(env_data)
        reward_data_list.append(reward_data)

        plotters.plot_single_model_training_enc_snapshots(env_data[0], nrows=5, ncols=3)

    plotters.plot_multiple_model_reward_curves(reward_data_list, model_names=experiment_names)
    plotters.plot_multiple_model_training_stats(training_stats_list, model_names=experiment_names)

    print("Done")
