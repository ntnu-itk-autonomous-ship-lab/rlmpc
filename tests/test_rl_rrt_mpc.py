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
            # get out rewards from evaluator + other reward terms
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

    print("done")
