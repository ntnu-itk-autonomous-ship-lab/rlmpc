import unittest

import rlmpc.common.paths as dp
import rlmpc.rlmpc as rlmpc
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator


class TestRLMPC(unittest.TestCase):

    def setUp(self) -> None:
        self.rlmpc_obj = rlmpc.RLMPC()
        self.scenario_generator = ScenarioGenerator(seed=7)
        self.simulator = Simulator()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_rlmpc(self):
        scenario_file = dp.scenarios / "test_rlmpc.yaml"
        scenario_data = self.scenario_generator.generate(
            config_file=scenario_file, new_load_of_map_data=False, show_plots=False
        )
        self.simulator.toggle_liveplot_visibility(True)
        output = self.simulator.run([scenario_data], colav_systems=[(0, self.rlmpc_obj)])
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
