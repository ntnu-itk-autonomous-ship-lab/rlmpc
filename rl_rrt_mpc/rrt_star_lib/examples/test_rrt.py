"""
    Demonstrates how to use the RRT algorithm with the colav-simulator.

    Remember to install the dependencies (COLAV-simulator) before running this script.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.paths as dp
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.stochasticity as stochasticity
import matplotlib.pyplot as plt
import numpy as np
import rrt_star_lib
import seacharts.enc as senc
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.simulator import Simulator
from shapely import strtree


@dataclass
class RRTParams:
    max_nodes: int = 2000
    max_iter: int = 10000
    iter_between_direct_goal_growth: int = 500
    min_node_dist: float = 15.0
    goal_radius: float = 300.0
    step_size: float = 0.5
    min_steering_time: float = 5.0
    max_steering_time: float = 30.0
    steering_acceptance_radius: float = 5.0
    max_nn_node_dist: float = 150.0
    gamma: float = 1200.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


def shift_nominal_plan(nominal_trajectory: np.ndarray, nominal_inputs: np.ndarray, ownship_state: np.ndarray, N: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Updates the nominal trajectory and inputs based on the current ownship state. This is done by
    find closest point on nominal trajectory to the current state and then shifting the nominal trajectory to this point

    Args:
        - nominal_trajectory (np.ndarray): The nominal trajectory.
        - nominal_inputs (np.ndarray): The nominal inputs.
        - ownship_state (np.ndarray): The ownship state.
        - N (int): Horizon length in samples

    Returns:
        Tuple[np.ndarray, np.ndarray]: The shifted nominal trajectory and inputs.
    """
    #
    nx = ownship_state.size
    nu = nominal_inputs.shape[0]
    closest_idx = int(np.argmin(np.linalg.norm(nominal_trajectory[:2, :] - np.tile(ownship_state[:2], (len(nominal_trajectory[0, :]), 1)).T, axis=0)))
    shifted_nominal_trajectory = nominal_trajectory[:, closest_idx:]
    shifted_nominal_inputs = nominal_inputs[:, closest_idx:]
    n_samples = shifted_nominal_trajectory.shape[1]
    if n_samples == 0:  # Done with following nominal trajectory, stop
        shifted_nominal_trajectory = np.tile(np.array([ownship_state[0], ownship_state[1], ownship_state[2], 0.0, 0.0, 0.0]), (N + 1, 1)).T
        shifted_nominal_inputs = np.zeros((nu, N))
    elif n_samples < N + 1:
        shifted_nominal_trajectory = np.zeros((nx, N + 1))
        shifted_nominal_trajectory[:, :n_samples] = nominal_trajectory[:, closest_idx : closest_idx + n_samples]
        shifted_nominal_trajectory[:, n_samples:] = np.tile(nominal_trajectory[:, closest_idx + n_samples - 1], (N + 1 - n_samples, 1)).T
        shifted_nominal_inputs = np.zeros((nu, N))
        shifted_nominal_inputs[:, : n_samples - 1] = nominal_inputs[:, closest_idx : closest_idx + n_samples - 1]
        shifted_nominal_inputs[:, n_samples - 1 :] = np.tile(nominal_inputs[:, closest_idx + n_samples - 2], (N - n_samples + 1, 1)).T
    else:
        shifted_nominal_trajectory = shifted_nominal_trajectory[:, : N + 1]
        shifted_nominal_inputs = shifted_nominal_inputs[:, :N]
    return shifted_nominal_trajectory, shifted_nominal_inputs


class RRT(ci.ICOLAV):
    def __init__(self, params: RRTParams) -> None:
        self._rrt_params = params
        self._rrt = rrt_star_lib.RRT(params)

        self._rrt_inputs: np.ndarray = np.empty(3)
        self._rrt_trajectory: np.ndarray = np.empty(6)
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])

        self._min_depth: int = 0

        self._map_origin: np.ndarray = np.array([])
        self._references = np.array([])
        self._initialized = False
        self._t_prev: float = 0.0

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[senc.ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        w: Optional[stochasticity.DisturbanceData] = None,
        **kwargs,
    ) -> np.ndarray:
        assert goal_state is not None, "Goal state must be provided to the RRT"
        assert enc is not None, "ENC must be provided to the RRT"
        if not self._initialized:
            self._min_depth = mapf.find_minimum_depth(kwargs["os_draft"], enc)
            self._t_prev = t
            self._map_origin = ownship_state[:2]
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(self._min_depth, enc, buffer=None)
            self._geometry_tree, _ = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)
            safe_sea_triangulation = mapf.create_safe_sea_triangulation(enc, self._min_depth, show_plots=False)
            self._rrt.transfer_enc_hazards(relevant_grounding_hazards[0])
            self._rrt.transfer_safe_sea_triangulation(safe_sea_triangulation)
            self._rrt.set_init_state(ownship_state.tolist())
            self._rrt.set_goal_state(goal_state.tolist())

            U_d = ownship_state[3]  # Constant desired speed given by the initial own-ship speed
            rrt_solution: dict = self._rrt.grow_towards_goal(ownship_state.tolist(), U_d, [])
            print("RRT-run completed")
            if not rrt_solution:
                raise RuntimeError("RRT did not find a solution")

            # rrt_solution = hf.load_rrt_solution()
            times = np.array(rrt_solution["times"])
            n_samples = len(times)
            self._rrt_trajectory = np.zeros((6, n_samples))
            self._rrt_inputs = np.zeros((3, n_samples - 1))
            for k in range(n_samples):
                self._rrt_trajectory[:, k] = np.array(rrt_solution["states"][k])
                if k < n_samples - 1:
                    self._rrt_inputs[:, k] = np.array(rrt_solution["inputs"][k])

        nominal_trajectory, nominal_inputs = shift_nominal_plan(self._rrt_trajectory, self._rrt_inputs[:2, :], ownship_state, N=len(self._rrt_trajectory[0, :]) - 1)
        psi_nom = nominal_trajectory[2, :]
        nominal_trajectory[2, :] = np.unwrap(np.concatenate(([psi_nom[0]], psi_nom)))[1:]

        # Plots for debugging. Set breakpoint here to inspect
        if enc is not None:
            mapf.plot_rrt_tree(self._rrt.get_tree_as_list_of_dicts(), enc)
            mapf.plot_trajectory(nominal_trajectory, enc, "magenta")
            mapf.plot_dynamic_obstacles(do_list, enc, 100.0, self._rrt_params.step_size)
            ship_poly = mapf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.0, 1.0)
            enc.draw_polygon(ship_poly, color="pink")

        self._t_prev = t
        # Alternative 1: Use LOS-guidance to track the trajectory
        # self._references = self._los.compute_references(
        #     nominal_trajectory[:2, :], speed_plan=nominal_trajectory[3, :], times=None, xs=ownship_state, dt=t - self._t_prev
        # )
        # self._references = np.zeros((9, len(nominal_trajectory[0, :])))
        # self._references[:nx, :] = nominal_trajectory

        # Alternative 2: Apply inputs directly to the ownship (bad todo in practice)
        self._references = np.zeros((9, len(nominal_inputs[0, :])))
        self._references[:2, :] = nominal_inputs[:2, :]
        return self._references

    def get_current_plan(self) -> np.ndarray:
        return self._references

    def get_colav_data(self) -> dict:
        if not self._initialized:
            return {
                "nominal_trajectory": np.zeros((6, 1)),
                "nominal_inputs": np.zeros((3, 1)),
                "params": self._rrt_params,
                "t": self._t_prev,
            }
        else:
            return {
                "nominal_trajectory": self._rrt_trajectory,
                "nominal_inputs": self._rrt_inputs,
                "params": self._rrt_params,
                "t": self._t_prev,
            }

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:

        if self._rrt_trajectory.size > 6:
            plt_handles["colav_nominal_trajectory"].set_xdata(self._rrt_trajectory[1, 0:-1:10] + self._map_origin[1])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._rrt_trajectory[0, 0:-1:10] + self._map_origin[0])

        return plt_handles


if __name__ == "__main__":
    rrt = RRT(RRTParams())

    scenario_file = dp.scenarios / "rl_scenario.yaml"
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=scenario_file)
    simulator = Simulator()
    output = simulator.run([scenario_data], ownship_colav_system=rrt)
    print("done")
