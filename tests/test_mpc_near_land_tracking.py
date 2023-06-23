from colav_simulator.simulator import Simulator

import autotuning.rl_rrt_mpc.rl_rrt_mpc.agent as agent

if __name__ == "__main__":
    rlmpc = agent.RLMPC()
    simulator = Simulator()
    data = simulator.run(ownship_colav_system=rlmpc)
    print("done")
