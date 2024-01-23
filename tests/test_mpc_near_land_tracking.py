import colav_simulator.common.paths as dp
import rl_rrt_mpc.agents as agents
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    planning_scenario = dp.scenarios / "simple_planning_example.yaml"
    rlmpc = agents.RLMPC()
    scenario_generator = ScenarioGenerator()
    simulator = Simulator()
    scenario_data = scenario_generator.generate(config_file=planning_scenario, new_load_of_map_data=False)
    output = simulator.run([scenario_data], colav_systems=[(0, rlmpc)])
    print("done")
