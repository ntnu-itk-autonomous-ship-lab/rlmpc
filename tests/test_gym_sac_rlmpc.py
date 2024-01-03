from pathlib import Path

import colav_simulator.behavior_generator as cs_bg
import colav_simulator.common.paths as dp
import colav_simulator.scenario_management as cs_sm
import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.paths as rrm_dp
import rl_rrt_mpc.sac as sac_rlmpc
import stable_baselines3.common.vec_env as sb3_vec_env
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from matplotlib import animation
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

# Depending on your OS, you might need to change these paths
plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


def save_frames_as_gif(frame_list: list, filename: Path) -> None:
    # Mess with this to change frame size
    fig = plt.figure(figsize=(frame_list[0].shape[1] / 72.0, frame_list[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frame_list[0], aspect="auto")
    plt.axis("off")

    def init():
        patch.set_data(frame_list[0])
        return (patch,)

    def animate(i):
        patch.set_data(frame_list[i])
        return (patch,)

    anim = animation.FuncAnimation(
        fig=fig, func=animate, init_func=init, blit=True, frames=len(frame_list), interval=50, repeat=True
    )
    anim.save(
        filename=filename.as_posix(),
        writer=animation.PillowWriter(fps=20),
        progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"),
    )


if __name__ == "__main__":
    config_file = dp.scenarios / "rlmpc_scenario.yaml"
    sg_config = cs_sm.Config()
    sg_config.behavior_generator.ownship_method = cs_bg.BehaviorGenerationMethod.ConstantSpeedRandomWaypoints
    sg_config.behavior_generator.target_ship_method = cs_bg.BehaviorGenerationMethod.ConstantSpeedRandomWaypoints

    env_id = "COLAVEnvironment-v0"
    env_config = {
        "scenario_config_file": config_file,
        "scenario_generator_config": sg_config,
        "test_mode": True,
        "reload_map": False,
        "seed": 3,
    }
    env = gym.make(id=env_id, **env_config)

    mpc_config_file = rrm_dp.config / "rlmpc.yaml"
    policy = sac_rlmpc.SACPolicyWithMPC
    policy_kwargs = {
        "critic_arch": [256, 256],
        "mpc_config": mpc_config_file,
        "activation_fn": th.nn.ReLU,
        "use_sde": False,
        "log_std_init": -3.0,
        "use_expln": False,
        "clip_mean": 2.0,
    }
    model = sac_rlmpc.SAC(policy, env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=10_000, progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    record = True
    if record:
        video_path = dp.animation_output / "demo.mp4"
        env = gym.wrappers.RecordVideo(env, video_path.as_posix(), episode_trigger=lambda x: x == 0)

    env.reset(seed=1)
    frames = []
    for i in range(250):
        obs, reward, terminated, truncated, info = env.step(np.array([-0.2, 0.0]))

        frames.append(env.render())

        if terminated or truncated:
            env.reset()

    env.close()

    save_gif = True
    if save_gif:
        save_frames_as_gif(frames, dp.animation_output / "demo2.gif")

    print("done")

    # Vil kunne
    # 1: lese inn mappe med AIS data som kan parses til ScenarioConfig-objekter med n_episodes og moglegheit for å adde randomgenererte båtar
    # 2: Generere/hente ut shapefiler for alle kart for alle scenario-config objekt som skal brukast
    # 3: Simulere n_episodar for kvart scenario, der own-ship har random starttilstand og sluttilstand, og alle andre båtar har AIS-trajectory eller varierande random trajectory. Skal kunne velge random control policy eller spesifikk feks RLRRTMPC policy.
    # 4: Legg til moglegheit for å terminere simuleringa viss OS kræsjer
    # 5: Lagre simuleringsdata (s_k, a_k, r_k+1, s_k+1, done_k+1) for alle episodar, og lagre i ein mappe med navn som er unikt for scenarioet.
    # 6: Last opp eller direkte bruk simuleringsdata i ein replay buffer for å trene policyen til konvergens.
    # 7: Test trent policy på testdata frå tilsvarande scenario (samme geografi som brukt i trening) og lagre resultatet i ein mappe med navn som er unikt for scenarioet.
