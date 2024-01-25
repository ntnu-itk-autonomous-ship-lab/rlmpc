import colav_simulator.common.paths as dp
import rl_rrt_mpc.rlmpc as rlmpc
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    rlmpc_obj = rlmpc.RLMPC()
    scenario_file = dp.scenarios / "rlmpc_scenario.yaml"
    scenario_generator = ScenarioGenerator(seed=7)
    scenario_data = scenario_generator.generate(config_file=scenario_file, new_load_of_map_data=False)
    simulator = Simulator()
    simulator.toggle_liveplot_visibility(True)
    output = simulator.run([scenario_data], colav_systems=[(0, rlmpc_obj)])
    print("done")
