from pathlib import Path

import colav_simulator.behavior_generator as cs_bg
import colav_simulator.scenario_generator as cs_sm
import rl_rrt_mpc.common.paths as rl_dp

if __name__ == "__main__":
    scenario_choice = 0
    if scenario_choice == 0:
        scenario_name = "rlmpc_scenario_cr_ss"
        config_file = rl_dp.scenarios / (scenario_name + ".yaml")
    elif scenario_choice == 1:
        scenario_name = "rlmpc_scenario_head_on_channel"
        config_file = rl_dp.scenarios / "rlmpc_scenario_easy_headon_no_hazards.yaml"

    scenario_generator = cs_sm.ScenarioGenerator(seed=0)

    scen = scenario_generator.load_scenario_from_folder(
        rl_dp.scenarios / "training_data" / scenario_name, scenario_name
    )

    scenario_data = scenario_generator.generate(
        config_file=config_file,
        new_load_of_map_data=False,
        save_scenario=True,
        save_scenario_folder=rl_dp.scenarios / "training_data" / scenario_name,
    )
    print(f"Length of scenario: {len(scenario_data)}")
    print("done")
