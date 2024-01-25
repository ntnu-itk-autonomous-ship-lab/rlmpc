from pathlib import Path

import colav_simulator.behavior_generator as cs_bg
import colav_simulator.scenario_generator as cs_sm
import rl_rrt_mpc.common.paths as rl_dp

if __name__ == "__main__":
    scenario_choice = 0
    if scenario_choice == 0:
        scenario_name = "rlmpc_scenario_easy_cr_ss"
        config_file = rl_dp.scenarios / "rlmpc_scenario_easy_cr_ss.yaml"
    elif scenario_choice == 1:
        scenario_name = "rlmpc_scenario_easy_headon_no_hazards"
        config_file = rl_dp.scenarios / "rlmpc_scenario_easy_headon_no_hazards.yaml"

    scenario_generator = cs_sm.ScenarioGenerator(seed=0)
    scenario_data = scenario_generator.generate(
        config_file=config_file,
        new_load_of_map_data=False,
        save_scenario=True,
        save_scenario_folder=rl_dp.scenarios / "training_data" / scenario_name,
    )
    print(f"Length of scenario: {len(scenario_data)}")
    print("done")
