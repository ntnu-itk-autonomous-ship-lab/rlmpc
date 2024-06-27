import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_results() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))


if __name__ == "__main__":
    base_dir: Path = Path.home() / "Desktop/machine_learning/rlmpc/"
    experiment_name = "sac_rlmpc"
    data_dir = base_dir / experiment_name / "eval_data"
    eval_infos = None
    eval_info_path = data_dir / experiment_name / "evaluation_results.pkl"
    with open("evaluation_results.pkl", "rb") as f:
        eval_infos = pickle.load(f)

    plot_evaluation_results(eval_infos)
