from pathlib import Path

import colav_simulator.gym.logger as logger
import numpy as np

if __name__ == "__main__":
    experiment_name = "sac_rlmpc1"
    log_dir = Path.home() / "Desktop" / "machine_learning" / "rlmpc" / experiment_name
    lgr = logger.Logger(experiment_name=experiment_name, log_dir=log_dir)
    lgr.load_from_pickle(f"{experiment_name}_env_training_data")
