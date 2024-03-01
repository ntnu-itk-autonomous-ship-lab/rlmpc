import colav_simulator.common.paths as dp
import rlmpc.rlmpc as rlmpc
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    # rlmpc = agents.RLMPC()
    rlrrtmpc = rlmpc.RLRRTMPC()
    scenario_file = dp.scenarios / "rl_scenario.yaml"
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=scenario_file)
    simulator = Simulator()
    output = simulator.run([scenario_data], colav_systems=[(0, rlrrtmpc)])
    print("done")
