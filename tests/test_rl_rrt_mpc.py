import time

import colav_simulator.common.paths as dp
import numpy as np
import rl_rrt_mpc.agents as agents
from colav_evaluation_tool.evaluator import Evaluator
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    rlrrtmpc = agents.RLRRTMPC()

    config_file = dp.scenarios / "simple_planning_example.yaml"
    scenario_generator = ScenarioGenerator()
    scenario_episode_list, enc = scenario_generator.generate(config_file=config_file)

    # Vil kunne
    # 1: lese inn mappe med AIS data som kan parses til ScenarioConfig-objekter med n_episodes og moglegheit for å adde randomgenererte båtar
    # 2: Generere/hente ut shapefiler for alle kart for alle scenario-config objekt som skal brukast
    # 3: Simulere n_episodar for kvart scenario, der own-ship har random starttilstand og sluttilstand, og alle andre båtar har AIS-trajectory eller varierande random trajectory. Skal kunne velge random control policy eller spesifikk feks RLRRTMPC policy.
    # 4: Legg til moglegheit for å terminere simuleringa viss OS kræsjer
    # 5: Lagre simuleringsdata (s_k, a_k, r_k+1, s_k+1, done_k+1) for alle episodar, og lagre i ein mappe med navn som er unikt for scenarioet.
    # 6: Last opp eller direkte bruk simuleringsdata i ein replay buffer for å trene policyen til konvergens.
    # 7: Test trent policy på testdata frå tilsvarande scenario (samme geografi som brukt i trening) og lagre resultatet i ein mappe med navn som er unikt for scenarioet.

    simulator = Simulator()
    evaluator = Evaluator()
    evaluator.set_enc(enc)

    converged = False
    while not converged:
        start_time = time.time()
        ep_execution_times = []
        for e, scenario_episode in enumerate(scenario_episode_list):
            ep_start_time = time.time()
            # get out (s_k, a_k, r_k+1, s_k+1, done_k+1) from simdata
            simdata = simulator.run(ownship_colav_system=rlrrtmpc)
            vessels = simdata["vessel_data"]
            print("Evaluating scenario episode" + str(e) + " with " + str(len(vessels)) + " vessels...")

            evaluator.set_vessel_data(vessels)
            results = evaluator.evaluate()
            # rlrrtmpc.train(scenario_episode=scenario_episode, enc=enc)

            ep_execution_time = time.time() - ep_start_time
            print("Episode simulation+performance assessment execution time in seconds: " + str(ep_execution_time))
            ep_execution_times.append(ep_execution_time)

        # Update agent parameters on training data

        ep_execution_times_arr = np.array(ep_execution_times)
        execution_time = ep_execution_times_arr.sum()
        print("Batch learning execution time in seconds: " + str(execution_time))

        if converged:
            break

    # apply on test data

    print("done")
