import warnings

import pytest

import colav_simulator.simulator as cssim
from colav_simulator.scenario_generator import ScenarioGenerator

import rlmpc.common.paths as dp
import rlmpc.rlmpc_cas as rlmpc_cas
from rlmpc.scripts.generate_scenario_episodes import generate_scenario_episodes
warnings.filterwarnings("ignore", module="pandas")


@pytest.fixture
def rlmpc_setup() -> tuple[rlmpc_cas.RLMPC, cssim.Simulator, ScenarioGenerator]:
    rlmpc_obj = rlmpc_cas.RLMPC()
    csconfig = cssim.Config.from_file(dp.config / "training_simulator.yaml")
    csconfig.visualizer.matplotlib_backend = "TkAgg"
    csconfig.visualizer.show_results = True
    csconfig.visualizer.show_trajectory_tracking_results = True
    csconfig.visualizer.show_target_tracking_results = True
    simulator = cssim.Simulator(config=csconfig)
    scenario_generator = ScenarioGenerator(seed=7)
    return rlmpc_obj, simulator, scenario_generator


def test_rlmpc(rlmpc_setup: tuple[rlmpc_cas.RLMPC, cssim.Simulator, ScenarioGenerator]) -> None:
    """Test RLMPC with MS channel scenario."""
    rlmpc_obj, simulator, scenario_generator = rlmpc_setup

    scenario_name = "rlmpc_scenario_ms_channel"
    data_folder = dp.scenarios / "training_data" / scenario_name
    if not any(file in data_folder.iterdir() for file in data_folder.iterdir()):
        generate_scenario_episodes([scenario_name], new_load_of_map_data=True)
    scenario_data = scenario_generator.load_scenario_from_folders(
        folder=dp.scenarios / "training_data" / scenario_name,
        scenario_name=scenario_name,
        reload_map=False,
        max_number_of_episodes=1,
    )

    simulator.toggle_liveplot_visibility(True)
    output = simulator.run(
        [scenario_data],
        colav_systems=[(0, rlmpc_obj)],
        terminate_on_collision_or_grounding=True,
    )
    print("done")
