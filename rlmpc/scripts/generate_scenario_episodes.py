import warnings

import colav_simulator.behavior_generator as cs_bg
import colav_simulator.scenario_generator as cs_sg
import rlmpc.common.paths as rl_dp

# Supressing futurewarning to speed up execution time
warnings.simplefilter(action="ignore", category=FutureWarning)


# tuning:
# horizon
# tau/barrier param
# edge case shit
# constraint satisfaction highly dependent on tau/barrier
# if ship gets too much off path/course it will just continue off course
def main():
    scenario_names = [
        "rlmpc_scenario_ms_channel",
        "rlmpc_scenario_random_many_vessels",
    ]  # ["rlmpc_scenario_ho", "rlmpc_scenario_cr_ss"]
    training_scenario_folders = [rl_dp.scenarios / "training_data" / name for name in scenario_names]
    test_scenario_folders = [rl_dp.scenarios / "test_data" / name for name in scenario_names]

    generate = True
    if generate:
        scenario_generator = cs_sg.ScenarioGenerator(config_file=rl_dp.config / "scenario_generator.yaml")
        for idx, name in enumerate(scenario_names):

            # if idx == 0:
            #     continue

            if idx > 0:
                scenario_generator.behavior_generator.set_ownship_method(
                    method=cs_bg.BehaviorGenerationMethod.ConstantSpeedRandomWaypoints
                )

            scenario_generator.seed(idx)
            # _ = scenario_generator.generate(
            #     config_file=rl_dp.scenarios / (name + ".yaml"),
            #     new_load_of_map_data=True if idx == 0 else True,
            #     save_scenario=True,
            #     save_scenario_folder=rl_dp.scenarios / "training_data" / name,
            #     show_plots=True,
            #     episode_idx_save_offset=0,
            #     n_episodes=500,
            #     delete_existing_files=True,
            # )

            scenario_generator.seed(idx + 103)
            _ = scenario_generator.generate(
                config_file=rl_dp.scenarios / (name + ".yaml"),
                new_load_of_map_data=False,
                save_scenario=True,
                save_scenario_folder=rl_dp.scenarios / "test_data" / name,
                show_plots=True,
                episode_idx_save_offset=0,
                n_episodes=50,
                delete_existing_files=True,
            )

    # map_size: [4000.0, 4000.0]
    # map_origin_enu: [-33524.0, 6572500.0]


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # cProfile.run("main()", sort="cumulative", filename="sac_rlmpc.prof")
    # p = pstats.Stats("sac_rlmpc.prof")
    # p.sort_stats("cumulative").print_stats(50)
    main()
