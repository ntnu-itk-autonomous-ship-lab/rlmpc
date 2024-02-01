import colav_simulator.common.paths as cs_dp
import colav_simulator.scenario_generator as cs_sg
import rl_rrt_mpc.common.paths as rl_dp

if __name__ == "__main__":
    scenario_choice = 0
    if scenario_choice == 0:
        scenario_name = "rlmpc_scenario_cr_ss"
        config_file = rl_dp.scenarios / (scenario_name + ".yaml")
    elif scenario_choice == 1:
        scenario_name = "rlmpc_scenario_head_on_channel"
        config_file = rl_dp.scenarios / "rlmpc_scenario_easy_headon_no_hazards.yaml"
    elif scenario_choice == 2:
        scenario_name = "rogaland_random_rl"
        config_file = cs_dp.scenarios / "rogaland_random_rl.yaml"
    elif scenario_choice == 3:
        scenario_name = "rogaland_random_rl_2"
        config_file = rl_dp.scenarios / "rogaland_random_rl_2.yaml"
    elif scenario_choice == 4:
        scenario_name = "rl_scenario"
        config_file = rl_dp.scenarios / "rl_scenario.yaml"

    scenario_generator = cs_sg.ScenarioGenerator(seed=0)

    scenario_episode_list, scenario_enc = scenario_generator.load_scenario_from_folder(
        rl_dp.scenarios / "training_data" / scenario_name, scenario_name, show=True
    )
