import unittest
import warnings

import colav_simulator.simulator as cssim
import rlmpc.common.paths as dp
import rlmpc.rlmpc_cas as rlmpc_cas
from colav_simulator.scenario_generator import ScenarioGenerator

warnings.filterwarnings("ignore", module="pandas")


class TestRLMPC(unittest.TestCase):
    def setUp(self) -> None:
        self.rlmpc_obj = rlmpc_cas.RLMPC()
        csconfig = cssim.Config.from_file(dp.config / "training_simulator.yaml")
        csconfig.visualizer.matplotlib_backend = "TkAgg"
        csconfig.visualizer.show_results = True
        csconfig.visualizer.show_trajectory_tracking_results = True
        csconfig.visualizer.show_target_tracking_results = True
        self.simulator = cssim.Simulator(config=csconfig)
        self.scenario_generator = ScenarioGenerator(seed=7)

    def tearDown(self) -> None:
        return super().tearDown()

    def test_rlmpc(self):
        scenario_name = "rlmpc_scenario_ms_channel"
        scenario_data = self.scenario_generator.load_scenario_from_folders(
            folder=dp.scenarios / "training_data" / scenario_name,
            scenario_name=scenario_name,
            reload_map=False,
            max_number_of_episodes=1,
        )

        self.simulator.toggle_liveplot_visibility(True)
        output = self.simulator.run(
            [scenario_data],
            colav_systems=[(0, self.rlmpc_obj)],
            terminate_on_collision_or_grounding=True,
            save_results=True,
        )
        print("done")


if __name__ == "__main__":
    unittest.main()
