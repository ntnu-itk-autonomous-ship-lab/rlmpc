import rl_rrt_mpc.agents as agents
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    rlmpc = agents.RLMPC()
    scenario_generator = ScenarioGenerator()
    scenario_data_list = scenario_generator.generate_configured_scenarios()
    simulator = Simulator()
    output = simulator.run(scenario_data_list, ownship_colav_system=rlmpc)
    print("done")
