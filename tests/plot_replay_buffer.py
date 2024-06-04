from pathlib import Path
from sys import platform
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import rlmpc.buffers as buffers
import stable_baselines3.common.save_util as sb3_sutils

if __name__ == "__main__":
    if platform == "linux" or platform == "linux2":
        BASE_PATH: Path = Path("/home/doctor/Desktop/machine_learning/rlmpc/sac_rlmpc/")
    elif platform == "darwin":
        BASE_PATH: Path = Path("/Users/trtengesdal/Desktop/machine_learning/rlmpc/sac_rlmpc/")

    rb_file = BASE_PATH / "replay_buffer.pkl"

    replay_buffer = sb3_sutils.load_from_pkl(rb_file)

    assert isinstance(replay_buffer, buffers.DictReplayBuffer)

    obs = replay_buffer.observations

    map_origin_enu = [31924.0, 6573700.0]
    map_size = [1400.0, 1500.0]

    rw_fig, rw_ax = plt.subplots()
    rw_ax.plot(replay_buffer.rewards, label="Reward")
    rw_ax.set_xlabel("Step")
    rw_ax.set_ylabel("Reward")

    plt.show(block=False)
    print("Done")
