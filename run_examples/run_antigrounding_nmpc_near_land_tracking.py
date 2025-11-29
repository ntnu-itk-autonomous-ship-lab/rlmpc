import colav_simulator.common.paths as dp
import colav_simulator.simulator as cssim
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator

import rlmpc.common.paths as rlmpc_dp
import rlmpc.trajectory_tracking_mpc as trajectory_tracking_mpc

if __name__ == "__main__":
    planning_scenario = dp.scenarios / "simple_planning_example.yaml"
    ttmpc_obj = trajectory_tracking_mpc.TrajectoryTrackingMPC()
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=planning_scenario)

    simconfig = cssim.Config.from_file(rlmpc_dp.config / "training_simulator.yaml")
    simconfig.visualizer.matplotlib_backend = "Agg"
    simulator = Simulator(config=simconfig)
    output = simulator.run([scenario_data], colav_systems=[(0, ttmpc_obj)])
    print("done")
