from pathlib import Path

import matplotlib.pyplot as plt
import stable_baselines3.common.save_util as sb3_sutils

import rlmpc.buffers as buffers

if __name__ == "__main__":
    base_path: Path = Path.home() / "machine_learning/rlmpc/sac_rlmpc/"
    rb_file = base_path / "replay_buffer.pkl"

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
