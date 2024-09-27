import os
from pathlib import Path
from typing import Callable, Tuple

import colav_simulator.behavior_generator as cs_bg
import colav_simulator.scenario_generator as cs_sg
import colav_simulator.simulator as cs_sim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
import torch
from colav_simulator.gym.environment import COLAVEnvironment
from rlmpc.networks.tracking_vae.vae import VAE
from rlmpc.networks.tracking_vae_attention.vae import VAE


def test_tracking_vae_attention() -> None:
    input_dim = 40 + 12 + 5 + 9
    TRACKINGOBS_DATADIR: Path = Path.home() / "Desktop/machine_learning/tracking_vae/data"

    scenario_names = [
        "rlmpc_scenario_ms_channel",
        # "rlmpc_scenario_random_many_vessels",
    ]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    # map_size: [4000.0, 4000.0]
    # map_origin_enu: [-33524.0, 6572500.0]
    observation_type = {
        "dict_observation": [
            "path_relative_navigation_observation",
            # "perception_image_observation",
            "relative_tracking_observation",
            "tracking_observation",
            "time_observation",
        ]
    }

    rewarder_config = rewards.Config.from_file(rl_dp.config / "rewarder.yaml")
    training_sim_config = cs_sim.Config.from_file(rl_dp.config / "training_simulator.yaml")
    eval_sim_config = cs_sim.Config.from_file(rl_dp.config / "eval_simulator.yaml")
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": training_scenario_folders,  # [training_scenario_folders[0]],
        "merge_loaded_scenario_episodes": True,
        "max_number_of_episodes": 1,
        "simulator_config": training_sim_config,
        "action_sample_time": 1.0 / 0.5,  # from rlmpc.yaml config file
        # "rewarder_class": rewards.MPCRewarder,
        # "rewarder_kwargs": {"config": rewarder_config},
        "test_mode": False,
        "render_update_rate": 1.0,
        "observation_type": observation_type,
        "action_type": "relative_course_speed_reference_sequence_action",
        "reload_map": False,
        "show_loaded_scenario_data": False,
        "shuffle_loaded_scenario_data": True,
        "identifier": "training_env",
        "seed": 0,
    }
    data_dir = Path.home() / "Desktop/machine_learning/tracking_vae/data"
    name = "tracking_avae1_NL_3_nonbi_HD_2048_LD_8_NH_3_ED_30"
    model = VAE(
        embedding_dim=30,
        num_heads=3,
        latent_dim=8,
        input_dim=7,
        num_layers=3,
        rnn_hidden_dim=2048,
        bidirectional=False,
        rnn_type=torch.nn.GRU,
    ).to("cpu")
    model.load_state_dict(torch.load(data_dir / f"../{name}/{name}_best.pth"))
    model.eval()
    env = gym.make(id=env_id, **env_config)
    env.reset()
    threshold_dist = 0.0
    for k in range(500):
        actions = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(actions)

        tracking_obs = obs["RelativeTrackingObservation"]

        tracking_obs = torch.tensor(tracking_obs.copy()).unsqueeze(0)
        mask = tracking_obs[:, 0, :] < threshold_dist
        seq_lengths = (
            torch.sum(tracking_obs[:, 0, :] < threshold_dist, dim=1).to("cpu").type(torch.int64)
        )  # idx 0 is normalized distance, where vals = 1.0 is max dist of 1e4++ and thus not valid
        tracking_obs = tracking_obs.permute(0, 2, 1)  # permute to (batch, max_seq_len, input_dim)

        recon_obs, means, log_vars, _ = model(tracking_obs, seq_lengths)
        recon_obs = recon_obs.permute(0, 2, 1).squeeze(0).detach().numpy()
        unnorm_obs = env.unwrapped.observation_type.unnormalize(obs)

        new_obs = obs.copy()
        new_obs["RelativeTrackingObservation"] = recon_obs
        new_unnorm_obs = env.unwrapped.observation_type.unnormalize(new_obs)
        new_unnorm_tracking_obs = new_unnorm_obs["RelativeTrackingObservation"]
        unnorm_tracking_obs = unnorm_obs["RelativeTrackingObservation"]

        mask = mask.squeeze(0).numpy()
        print(f"Diff unnorm: {(unnorm_tracking_obs- new_unnorm_tracking_obs) * mask}")

        if terminated or truncated:
            env.reset()
            continue


if __name__ == "__main__":
    test_tracking_vae_attention()
