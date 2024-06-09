import unittest
import warnings

import rlmpc.common.paths as dp
import rlmpc.rlmpc as rlmpc
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator

warnings.filterwarnings("ignore", module="pandas")


class TestRLMPC(unittest.TestCase):

    def setUp(self) -> None:
        self.rlmpc_obj = rlmpc.RLMPC()
        self.scenario_generator = ScenarioGenerator(seed=7)
        self.simulator = Simulator()

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
            [scenario_data], colav_systems=[(0, self.rlmpc_obj)], terminate_on_collision_or_grounding=True
        )
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
