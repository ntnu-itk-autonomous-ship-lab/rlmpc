import colav_simulator.common.paths as dp
import gymnasium as gym
import numpy as np
import rl_rrt_mpc.agents as agents
from colav_simulator.gym.environment import COLAVEnvironment

if __name__ == "__main__":

    rlrrtmpc = agents.RLRRTMPC()

    config_file = dp.scenarios / "simple_planning_example.yaml"

    env = COLAVEnvironment()
    env = gym.make("COLAVSimulator-v1")

    # Vil kunne
    # 1: lese inn mappe med AIS data som kan parses til ScenarioConfig-objekter med n_episodes og moglegheit for å adde randomgenererte båtar
    # 2: Generere/hente ut shapefiler for alle kart for alle scenario-config objekt som skal brukast
    # 3: Simulere n_episodar for kvart scenario, der own-ship har random starttilstand og sluttilstand, og alle andre båtar har AIS-trajectory eller varierande random trajectory. Skal kunne velge random control policy eller spesifikk feks RLRRTMPC policy.
    # 4: Legg til moglegheit for å terminere simuleringa viss OS kræsjer
    # 5: Lagre simuleringsdata (s_k, a_k, r_k+1, s_k+1, done_k+1) for alle episodar, og lagre i ein mappe med navn som er unikt for scenarioet.
    # 6: Last opp eller direkte bruk simuleringsdata i ein replay buffer for å trene policyen til konvergens.
    # 7: Test trent policy på testdata frå tilsvarande scenario (samme geografi som brukt i trening) og lagre resultatet i ein mappe med navn som er unikt for scenarioet.

    converged = False
    while not converged:
        ep_execution_times = []

        if converged:
            break

    # apply on test data

    print("done")
