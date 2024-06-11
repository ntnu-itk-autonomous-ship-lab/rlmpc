import rlmpc.common.paths as dp
from colav_simulator.scenario_generator import ScenarioGenerator

if __name__ == "__main__":
    scenario_generator = ScenarioGenerator(seed=0)

    scenario_name = "base_scenario_medium_head_on"

    scenario_data_list = scenario_generator.generate(
        config_file=dp.scenarios / (scenario_name + ".yaml"),
        new_load_of_map_data=True,
        show_plots=True,
        save_scenario=True,
        save_scenario_folder=dp.scenarios / scenario_name,
        delete_existing_files=True,
        episode_idx_save_offset=0,
        n_episodes=10,
    )

    print("done")
