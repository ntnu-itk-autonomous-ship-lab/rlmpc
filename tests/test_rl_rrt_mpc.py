import colav_simulator.common.paths as dp
import gymnasium as gym
import numpy as np
import rl_rrt_mpc.agents as agents
import stable_baselines3.common.vec_env as sb3_vec_env
from colav_simulator.gym.environment import COLAVEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":

    print(*gym.envs.registry.keys())
    config_file = dp.scenarios / "rl_scenario.yaml"

    env_id = "COLAVEnvironment-v0"
    env_config = {"scenario_config_file": config_file, "test_mode": True}
    env = gym.make(id=env_id, **env_config)
    model_name = "PPO"
    num_cpu = 2
    if model_name == "DDPG" or model_name == "TD3" or model_name == "SAC":
        vec_env = sb3_vec_env.DummyVecEnv([lambda: gym.make(id=env_id, **env_config)])
    else:
        vec_env = sb3_vec_env.SubprocVecEnv([lambda: gym.make(env_id, **env_config) for i in range(num_cpu)])
    vec_env.seed(0)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=10_000, progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    converged = False
    env.reset()
    for i in range(10000):
        obs, reward, done, info = vec_env.step(np.array([0.0, 0.0]))

        vec_env.render()

    print("done")

    # Vil kunne
    # 1: lese inn mappe med AIS data som kan parses til ScenarioConfig-objekter med n_episodes og moglegheit for å adde randomgenererte båtar
    # 2: Generere/hente ut shapefiler for alle kart for alle scenario-config objekt som skal brukast
    # 3: Simulere n_episodar for kvart scenario, der own-ship har random starttilstand og sluttilstand, og alle andre båtar har AIS-trajectory eller varierande random trajectory. Skal kunne velge random control policy eller spesifikk feks RLRRTMPC policy.
    # 4: Legg til moglegheit for å terminere simuleringa viss OS kræsjer
    # 5: Lagre simuleringsdata (s_k, a_k, r_k+1, s_k+1, done_k+1) for alle episodar, og lagre i ein mappe med navn som er unikt for scenarioet.
    # 6: Last opp eller direkte bruk simuleringsdata i ein replay buffer for å trene policyen til konvergens.
    # 7: Test trent policy på testdata frå tilsvarande scenario (samme geografi som brukt i trening) og lagre resultatet i ein mappe med navn som er unikt for scenarioet.
