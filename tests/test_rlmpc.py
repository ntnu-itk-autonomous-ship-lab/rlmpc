import colav_simulator.common.paths as dp
import rl_rrt_mpc.rlmpc as rlmpc
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    # rlmpc = agents.RLMPC()
    rlmpc_obj = rlmpc.RLMPC()
    scenario_file = dp.scenarios / "rlmpc_scenario.yaml"
    scenario_generator = ScenarioGenerator(seed=1)
    scenario_data = scenario_generator.generate(config_file=scenario_file, new_load_of_map_data=False)
    simulator = Simulator()
    simulator.toggle_liveplot_visibility(False)
    output = simulator.run([scenario_data], ownship_colav_system=rlmpc_obj)
    print("done")
