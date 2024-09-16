import colav_simulator.common.paths as dp
import rlmpc.trajectory_tracking_mpc as trajectory_tracking_mpc
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    planning_scenario = dp.scenarios / "simple_planning_example.yaml"
    ttmpc_obj = trajectory_tracking_mpc.TrajectoryTrackingMPC()
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=planning_scenario)
    simulator = Simulator()
    output = simulator.run([scenario_data], ownship_colav_system=[(0, ttmpc_obj)])
    print("done")
