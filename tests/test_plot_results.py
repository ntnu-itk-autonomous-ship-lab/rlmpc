from pathlib import Path

import rlmpc.common.plotters as plotters

if __name__ == "__main__":
    base_dir: Path = Path.home() / "Desktop/machine_learning/rlmpc"
    experiment_names = ["sac_rlmpc1"]
    plotters.plot_training_results(base_dir=base_dir, experiment_names=experiment_names)
    plotters.plot_evaluation_results(base_dir=base_dir, experiment_names=experiment_names)
