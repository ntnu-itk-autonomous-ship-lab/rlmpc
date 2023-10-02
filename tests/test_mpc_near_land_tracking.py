import colav_simulator.common.paths as dp
import rl_rrt_mpc.rlmpc as rlmpc
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    planning_scenario = dp.scenarios / "simple_planning_example.yaml"
    rlmpc = rlmpc.RLMPC()
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=planning_scenario)
    simulator = Simulator()
    output = simulator.run([scenario_data], ownship_colav_system=rlmpc)
    print("done")
