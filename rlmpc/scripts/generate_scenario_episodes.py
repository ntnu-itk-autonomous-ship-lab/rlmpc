"""
Script for generating scenario episode definitions, i.e.
actual ship starting states, nominal paths, disturbances, etc. for training and testing.
"""

import warnings

import colav_simulator.behavior_generator as cs_bg
import colav_simulator.scenario_generator as cs_sg

import rlmpc.common.paths as rl_dp

# Supressing futurewarning to speed up execution time
warnings.simplefilter(action="ignore", category=FutureWarning)

def generate_scenario_episodes(scenario_names: list[str], 
        new_load_of_map_data: bool = True, 
        ownship_behavior_generation_method: cs_bg.BehaviorGenerationMethod = cs_bg.BehaviorGenerationMethod.PQRRTStar,
    ) -> None:

    # map_size: [4000.0, 4000.0]
    # map_origin_enu: [-33524.0, 6572500.0]
    generate = True
    if generate:
        scenario_generator = cs_sg.ScenarioGenerator(
            config_file=rl_dp.config / "scenario_generator.yaml"
        )
        for idx, name in enumerate(scenario_names):
            scenario_generator.behavior_generator.set_ownship_method(
                method=ownship_behavior_generation_method
            )
            scenario_generator.seed(idx + 0)
            _ = scenario_generator.generate(
                config_file=rl_dp.scenarios / (name + ".yaml"),
                new_load_of_map_data=new_load_of_map_data,
                save_scenario=True,
                save_scenario_folder=rl_dp.scenarios / "training_data" / name,
                show_plots=False,
                episode_idx_save_offset=0,
                n_episodes=10,
                delete_existing_files=True,
            )

            scenario_generator.seed(idx + 102)
            _ = scenario_generator.generate(
                config_file=rl_dp.scenarios / (name + ".yaml"),
                new_load_of_map_data=False,
                save_scenario=True,
                save_scenario_folder=rl_dp.scenarios / "test_data" / name,
                show_plots=False,
                episode_idx_save_offset=0,
                n_episodes=10,
                delete_existing_files=True,
            )

    
    
def main():
    scenario_names = [
        "rlmpc_scenario_ms_channel",
        "rlmpc_scenario_random_many_vessels",
    ]
    generate_scenario_episodes(scenario_names)


if __name__ == "__main__":
    main()
