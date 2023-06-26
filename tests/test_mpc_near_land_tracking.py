import rl_rrt_mpc.agents as agents
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    rlmpc = agents.RLMPC()
    simulator = Simulator()
    data = simulator.run(ownship_colav_system=rlmpc)
    print("done")
