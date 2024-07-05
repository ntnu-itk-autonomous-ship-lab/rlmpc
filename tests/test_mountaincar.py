from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


@profile
def main():
    env_id = "MountainCarContinuous-v0"
    num_envs = 4
    env = make_vec_env(env_id, n_envs=num_envs)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=40000,
        batch_size=32,
        learning_starts=1000,
        train_freq=8,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        tau=0.005,
        gamma=0.99,
        learning_rate=0.0003,
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu",
    )

    # base_dir = Path.home() / "Desktop/machine_learning/rlmpc/cartpole_test"
    # if not base_dir.exists():
    #     base_dir.mkdir(parents=True)

    # checkpoint_cb = CheckpointCallback(save_freq=50000, save_path=base_dir, name_prefix="sac_test")
    # eval_cb = EvalCallback(env, best_model_save_path=base_dir / "logs", log_path=base_dir / "logs", eval_freq=50000)
    # model.learn(
    #     total_timesteps=500000, callback=CallbackList([checkpoint_cb, eval_cb]), log_interval=4, progress_bar=True
    # )
    # model.save("sac_cartpole")

    obs = env.reset()
    timesteps = 1_000_000
    for _ in range(timesteps):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
