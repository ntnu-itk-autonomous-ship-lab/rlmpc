from pathlib import Path

import colav_simulator.simulator as cs_sim
import gymnasium as gym
import numpy as np
import torch

import rlmpc.common.paths as rl_dp
import rlmpc.rewards as rewards
from rlmpc.networks.tracking_vae.vae import VAE
from rlmpc.networks.tracking_vae_attention.vae import VAE


def test_tracking_vae_attention() -> None:
    scenario_names = [
        "rlmpc_scenario_ms_channel",
        # "rlmpc_scenario_random_many_vessels",
    ]
    test_scenario_folders = [
        rl_dp.scenarios / "test_data" / name for name in scenario_names
    ]

    # map_size: [4000.0, 4000.0]
    # map_origin_enu: [-33524.0, 6572500.0]
    observation_type = {
        "dict_observation": [
            # "path_relative_navigation_observation",
            # "perception_image_observation",
            "relative_tracking_observation",
            # "tracking_observation",
            # "time_observation",
        ]
    }

    rewarder_config = rewards.Config.from_file(rl_dp.config / "rewarder.yaml")
    training_sim_config = cs_sim.Config.from_file(
        rl_dp.config / "training_simulator.yaml"
    )
    eval_sim_config = cs_sim.Config.from_file(rl_dp.config / "eval_simulator.yaml")
    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_file_folder": test_scenario_folders,  # [training_scenario_folders[0]],
        "merge_loaded_scenario_episodes": True,
        "max_number_of_episodes": 10,
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
        "seed": 123124014,
    }
    vae_dir = Path.home() / "machine_learning/tracking_vae/chosen"
    name = "tracking_avae_mdnew_beta001_3_NL_1_nonbi_HD_64_LD_12_NH_8_ED_16"
    input_dim = 4
    model = VAE(
        embedding_dim=16,
        num_heads=8,
        latent_dim=12,
        input_dim=input_dim,
        num_layers=1,
        rnn_hidden_dim=64,
        bidirectional=False,
        rnn_type=torch.nn.GRU,
    ).to("cpu")
    model.load_state_dict(torch.load(vae_dir / f"{name}/{name}_best.pth"))
    model.eval()
    env = gym.make(id=env_id, **env_config)
    env.reset()
    print(f"Model size: {sum(p.numel() for p in model.parameters())}")
    threshold_dist = -0.25
    n_steps = 500
    errors = np.zeros((input_dim, n_steps))
    error_divisors = np.zeros(n_steps)
    for k in range(n_steps):
        actions = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(actions)

        tracking_obs = obs["RelativeTrackingObservation"]

        tracking_obs = torch.tensor(tracking_obs.copy()).unsqueeze(0)
        tracking_obs = tracking_obs[:, :input_dim, :]
        mask = tracking_obs[:, 0, :] < threshold_dist
        seq_lengths = (
            torch.sum(tracking_obs[:, 0, :] < threshold_dist, dim=1)
            .to("cpu")
            .type(torch.int64)
        )  # idx 0 is normalized distance, where vals = 1.0 is max dist of 1e4++ and thus not valid
        tracking_obs = tracking_obs.permute(
            0, 2, 1
        )  # permute to (batch, max_seq_len, input_dim)

        recon_obs, means, log_vars, _ = model(tracking_obs, seq_lengths)
        recon_obs = recon_obs.permute(0, 2, 1).squeeze(0).detach().numpy()
        unnorm_obs = env.unwrapped.observation_type.unnormalize(obs)

        new_obs = obs.copy()
        new_obs["RelativeTrackingObservation"] = recon_obs
        new_unnorm_obs = env.unwrapped.observation_type.unnormalize(new_obs)
        new_unnorm_tracking_obs = new_unnorm_obs["RelativeTrackingObservation"]
        unnorm_tracking_obs = unnorm_obs["RelativeTrackingObservation"]

        mask = mask.squeeze(0).numpy()

        if seq_lengths[0] > 0:
            error_divisors[k] = 1.0 / seq_lengths[0]
            diff_unnorm = (unnorm_tracking_obs - new_unnorm_tracking_obs) * mask
            # print(f"z_enc otherwise: {means}")
            # print(f"Recon x: {new_unnorm_tracking_obs[:, -seq_lengths[0]:]}")
            # print(f"Diff unnorm: {diff_unnorm[:, -seq_lengths[0]:]}")
            errors[:, k] = (
                np.sum((unnorm_tracking_obs - new_unnorm_tracking_obs) * mask, axis=1)
                / seq_lengths[0]
            )
        # else:
        #     print(f"z_enc when seq_len == 0: {means}")
        #
        # print(f"Progress: {100.0 * k / n_steps}")
        if terminated or truncated:
            env.reset()
            continue

    # get errors where they are not zero
    errors = errors[:, error_divisors > 0]
    avg_error = np.mean(errors, axis=1)
    avg_rel_dist_error = avg_error[0]
    avg_rel_bearing_error = avg_error[1]
    avg_rel_vx_error = avg_error[2]
    avg_rel_vy_error = avg_error[3]
    print(
        f"Average distance error: {avg_rel_dist_error} | std: {np.std(errors[0, :])} | max: {np.max(errors[0, :])}"
    )
    print(
        f"Average bearing error: {avg_rel_bearing_error} | std: {np.std(errors[1, :])} | max: {np.max(errors[1, :])}"
    )
    print(
        f"Average vx error: {avg_rel_vx_error} | std: {np.std(errors[2, :])} | max: {np.max(errors[2, :])}"
    )
    print(
        f"Average vy error: {avg_rel_vy_error} | std: {np.std(errors[3, :])} | max: {np.max(errors[3, :])}"
    )


if __name__ == "__main__":
    test_tracking_vae_attention()
