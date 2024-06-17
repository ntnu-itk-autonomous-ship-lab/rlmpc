import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_results() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))


if __name__ == "__main__":
    experiment_name = "sac_rlmpc"
    data_dir = Path("/home/doctor/Desktop/machine_learning/rlmpc/sac_rlmpc/eval_data/")
    eval_infos = None
    eval_info_path = data_dir / experiment_name / "evaluation_results.pkl"
    with open("evaluation_results.pkl", "rb") as f:
        eval_infos = pickle.load(f)

    plot_evaluation_results()
