import os
import pathlib

import rlmpc.common.paths as dp
import colav_simulator.simulator as cssim
import colav_simulator.behavior_generator as cs_bg
from colav_simulator.scenario_generator import ScenarioGenerator

import rlmpc.rlmpc_cas as rlmpc_cas
from rlmpc.scripts.generate_scenario_episodes import generate_scenario_episodes
    
if __name__ == "__main__":
    rlmpc_obj = rlmpc_cas.RLMPC()
    csconfig = cssim.Config.from_file(dp.config / "training_simulator.yaml")
    csconfig.visualizer.matplotlib_backend = "TkAgg"
    csconfig.visualizer.show_results = True
    csconfig.visualizer.show_trajectory_tracking_results = True
    csconfig.visualizer.show_target_tracking_results = True
    simulator = cssim.Simulator(config=csconfig)
    scenario_generator = ScenarioGenerator(seed=7)

    scenario_name = "rlmpc_scenario_ms_channel"
    data_folder = dp.scenarios / "training_data" / scenario_name
    if not data_folder.exists() or not any(file in data_folder.iterdir() for file in data_folder.iterdir()):
        generate_scenario_episodes(
            scenario_names=[scenario_name], 
            new_load_of_map_data=False, 
            ownship_behavior_generation_method=cs_bg.BehaviorGenerationMethod.PQRRTStar)
    scenario_data = scenario_generator.load_scenario_from_folders(
        folder=dp.scenarios / "training_data" / scenario_name,
        scenario_name=scenario_name,
        reload_map=True,
        max_number_of_episodes=1,
    )

    simulator.toggle_liveplot_visibility(True)
    output = simulator.run(
        [scenario_data],
        colav_systems=[(0, rlmpc_obj)],
        terminate_on_collision_or_grounding=True,
    )
    print("done")