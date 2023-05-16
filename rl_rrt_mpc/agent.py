"""
    rl_rrt_mpc.py

    Summary:
        Contains the main RL-RRT-MPC (top and mid-level COLAV planner) and RL-MPC (mid-level COLAV planner) system class.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.integrators as sim_integrators
import colav_simulator.core.models as sim_models
import informed_rrt_star_rust as rrt
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.config_parsing as cp
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.mpc.models as mpc_models
import rl_rrt_mpc.mpc.mpc as mpc
import rl_rrt_mpc.mpc.parameters as mpc_params
import rl_rrt_mpc.mpc.set_generator as sg
import rl_rrt_mpc.rl as rl
import seacharts.enc as senc
from scipy.interpolate import interp1d
from shapely import strtree


@dataclass
class RRTParams:
    max_nodes: int = 2000
    max_iter: int = 10000
    iter_between_direct_goal_growth: int = 100
    min_node_dist: float = 5.0
    goal_radius: float = 100.0
    step_size: float = 0.1
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 5.0
    max_nn_node_dist: float = 300.0
    gamma: float = 1000.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


@dataclass
class RLRRTMPCParams:
    rl_method: rl.RLParams
    rrt: RRTParams
    mpc: mpc.Config

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RLRRTMPCParams(
            rl_method=rl.RLParams.from_dict(config_dict["rl"]),
            rrt=RRTParams.from_dict(config_dict["rrt"]),
            mpc=mpc.Config.from_dict(config_dict["mpc"]),
        )
        return config


class RLRRTMPCBuilder:
    @classmethod
    def build(cls, config: RLRRTMPCParams) -> Tuple[rl.RL, rrt.InformedRRTStar, mpc.MPC]:
        rl_obj = rl.RL(config.rl_method)
        rrt_obj = rrt.InformedRRTStar(config.rrt)
        rlmpc_obj = mpc.MPC(mpc_models.Telemetron(), config.mpc)
        return rl_obj, rrt_obj, rlmpc_obj


class RLRRTMPC(ci.ICOLAV):
    """The RL-RRT-MPC both plans a trajectory from a start to a goal state through the RRT, and plans solutions for tracking this trajectory through the MPC while avoiding static and dynamic obstacles. RL is used to update planner parameters. The RL-RRT-MPC is a top and mid-level COLAV planner."""

    def __init__(self, config: Optional[RLRRTMPCParams] = None, config_file: Optional[Path] = dp.rl_rrt_mpc_config) -> None:

        if config:
            self._config: RLRRTMPCParams = config
        else:
            self._config = cp.extract(RLRRTMPCParams, config_file, dp.rl_rrt_mpc_schema)

        self._rl, self._rrt, self._mpc = RLRRTMPCBuilder.build(self._config)

        self._references = np.empty(9)
        self._initialized = False
        self._t_prev = 0.0
        self._min_depth: int = 0
        self._mpc_rel_polygons: list = []
        self._rrt_inputs: np.ndarray = np.empty(3)
        self._rrt_trajectory: np.ndarray = np.empty(6)
        self._rrt_references: np.ndarray = np.empty(2)
        self._rel_rrt_trajectory: np.ndarray = np.empty(6)
        self._mpc_trajectory: np.ndarray = np.empty(6)
        self._mpc_inputs: np.ndarray = np.empty(3)
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[senc.ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Implements the ICOLAV plan interface function. Relies on getting the own-ship minimum depth
        in order to extract relevant grounding hazards.
        """
        assert goal_state is not None, "Goal state must be provided to the RL-RRT-MPC"
        assert enc is not None, "ENC must be provided to the RL-RRT-MPC"
        rel_do_list = hf.shift_dynamic_obstacle_coordinates(do_list, enc.origin[0], enc.origin[1])
        if not self._initialized:
            self._min_depth = mapf.find_minimum_depth(kwargs["os_draft"], enc)
            self._t_prev = t
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards(self._min_depth, enc)
            self._geometry_tree, poly_list = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

            self._rrt.transfer_enc_data(relevant_grounding_hazards)
            self._rrt.set_init_state(ownship_state.tolist())
            self._rrt.set_goal_state(goal_state.tolist())

            U_d = ownship_state[3]  # Constant desired speed given by the initial own-ship speed
            rrt_solution: dict = self._rrt.grow_towards_goal(ownship_state.tolist(), U_d, [])
            hf.save_rrt_solution(rrt_solution)
            if enc is not None:
                enc.start_display()
                for hazard in relevant_grounding_hazards:
                    enc.draw_polygon(hazard, color="red", fill=False)
                # tree_list = self._rrt.get_tree_as_list_of_dicts()
                # hf.plot_rrt_tree(tree_list, enc)
                ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 5, 2)
                enc.draw_circle((ownship_state[1], ownship_state[0]), radius=40, color="yellow", alpha=0.4)
                enc.draw_polygon(ship_poly, color="pink")
                enc.draw_circle((goal_state[1], goal_state[0]), radius=40, color="cyan", alpha=0.4)

            # rrt_solution = hf.load_rrt_solution()
            rrt_solution["references"] = [[r[0], r[1]] for r in rrt_solution["references"]]
            times = np.array(rrt_solution["times"])
            n_samples = len(times)
            self._rrt_trajectory = np.zeros((6, n_samples))
            self._rrt_inputs = np.zeros((3, n_samples))
            self._rrt_references = np.zeros((2, n_samples))
            for k in range(n_samples):
                self._rrt_trajectory[:, k] = np.array(rrt_solution["states"][k])
                self._rrt_inputs[:, k] = np.array(rrt_solution["inputs"][k])
                self._rrt_references[:, k] = np.array(rrt_solution["references"][k])

            self._rel_rrt_trajectory = self._rrt_trajectory.copy()
            self._rel_rrt_trajectory[0, :] -= enc.origin[1]
            self._rel_rrt_trajectory[1, :] -= enc.origin[0]

            if enc is not None:
                hf.plot_trajectory(self._rrt_trajectory, enc, color="magenta")

            poly_tuple_list, enveloping_polygon = mapf.extract_polygons_near_trajectory(
                self._ktp_trajectory, self._geometry_tree, buffer=self._mpc.params.reference_traj_bbox_buffer, enc=enc
            )
            self._mpc_rel_polygons = []
            for poly_tuple in poly_tuple_list:
                self._mpc_rel_polygons.extend(poly_tuple[0])

            # triangle_polygons = mapf.extract_boundary_polygons_inside_envelope(poly_tuple_list, enveloping_polygon, enc=enc)
            self._mpc.construct_ocp(nominal_trajectory=self._rel_rrt_trajectory, do_list=rel_do_list, so_list=self._mpc_rel_polygons, enc=enc)

        rel_ownship_state = ownship_state.copy()
        rel_ownship_state[0] -= enc.origin[1]
        rel_ownship_state[1] -= enc.origin[0]
        self._mpc_trajectory, self._mpc_inputs = self._mpc.plan(
            nominal_trajectory=self._rel_rrt_trajectory,
            nominal_inputs=self._rrt_inputs,
            xs=rel_ownship_state,
            do_list=rel_do_list,
            so_list=self._mpc_rel_polygons,
            enc=enc,
        )
        self._mpc_trajectory[0, :] += enc.origin[1]
        self._mpc_trajectory[1, :] += enc.origin[0]

        hf.plot_dynamic_obstacles(do_list, enc, self._mpc.params.T, self._mpc.params.dt)
        hf.plot_trajectory(self._mpc_trajectory, enc, color="cyan")
        self._references = np.zeros((9, len(self._mpc_trajectory[0, :])))
        self._references[:6, :] = self._mpc_trajectory
        return self._references

    def get_current_plan(self) -> np.ndarray:
        return self._references

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:
        return []


@dataclass
class RLMPCParams:
    rl: rl.RLParams
    ktp: guidances.KTPGuidanceParams
    los: guidances.LOSGuidanceParams
    mpc: mpc.Config

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RLMPCParams(
            rl=rl.RLParams.from_dict(config_dict["rl"]),
            ktp=guidances.KTPGuidanceParams.from_dict(config_dict["ktp"]),
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
            mpc=mpc.Config.from_dict(config_dict["mpc"]),
        )
        return config


class RLMPC(ci.ICOLAV):
    """The RL-MPC is a mid-level planner, using the MPC to plan a solution for tracking a nominal trajectory while avoiding obstacles.
    RL is used to update parameters online. Path-following/trajectory tracking can both be used. A Kinematic Trajectory Planner is used to generate the nominal trajectory.
    """

    def __init__(self, config: Optional[RLMPCParams] = None, config_file: Optional[Path] = dp.rl_mpc_config) -> None:

        if config:
            self._config: RLMPCParams = config
        else:
            self._config = cp.extract(RLMPCParams, config_file, dp.rl_mpc_schema)

        self._rl = rl.RL(self._config.rl)
        self._ktp = guidances.KinematicTrajectoryPlanner(self._config.ktp)
        self._los = guidances.LOSGuidance(self._config.los)
        self._mpc = mpc.MPC(mpc_models.Telemetron(), self._config.mpc)

        self._references = np.array([])
        self._initialized = False
        self._t_prev: float = 0.0
        self._t_prev_mpc: float = 0.0
        self._min_depth: int = 0
        self._nominal_trajectory: np.ndarray = np.array([])
        self._mpc_trajectory: np.ndarray = np.array([])
        self._mpc_inputs: np.ndarray = np.array([])
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])
        self._mpc_rel_polygons: list = []
        self._rel_polygons: list = []
        self._original_poly_list: list = []
        self._set_generator: Optional[sg.SetGenerator] = None

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[senc.ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Implements the ICOLAV plan interface function. Relies on getting the own-ship minimum depth
        in order to extract relevant grounding hazards.
        """
        assert goal_state is not None, "Goal state must be provided to the RL-MPC"
        assert enc is not None, "ENC must be provided to the RL-MPC"
        nx = ownship_state.size
        if not self._initialized:
            self._initialized = True
            # x_spline, y_spline, psi_spline, speed_spline = self._ktp.compute_splines(waypoints, speed_plan, None)
            # self._nominal_trajectory = self._ktp.compute_reference_trajectory(self._mpc.params.dt)
            self._nominal_trajectory = create_los_based_trajectory(ownship_state, waypoints, speed_plan, self._los, self._mpc.params.dt)

            self._setup_mpc_static_obstacle_input(ownship_state, enc, self._mpc.params.debug, **kwargs)
            self._mpc.construct_ocp(nominal_trajectory=self._nominal_trajectory, xs=ownship_state, do_list=do_list, so_list=self._mpc_rel_polygons, enc=enc)

        self._update_mpc_so_polygon_input(ownship_state, enc, self._mpc.params.debug)
        if t == 0 or t - self._t_prev_mpc >= 1.0 / self._mpc.params.rate:
            # Find closest point on nominal trajectory to the current state
            closest_idx = int(np.argmin(np.linalg.norm(self._nominal_trajectory[:2, :] - np.tile(ownship_state[:2], (len(self._nominal_trajectory[0, :]), 1)).T, axis=0)))
            shifted_nominal_trajectory = self._nominal_trajectory[:, closest_idx:]
            N = int(self._mpc.params.T / self._mpc.params.dt)
            n_samples = shifted_nominal_trajectory.shape[1]
            if n_samples == 0:  # Done with following nominal trajectory, stop
                shifted_nominal_trajectory = np.tile(np.array([ownship_state[0], ownship_state[1], ownship_state[2], 0.0, 0.0, 0.0]), (N + 1, 1)).T
            elif n_samples < N + 1:
                shifted_nominal_trajectory = np.zeros((nx, N + 1))
                shifted_nominal_trajectory[:, :n_samples] = self._nominal_trajectory
                shifted_nominal_trajectory[:, n_samples:] = np.tile(self._nominal_trajectory[:, -1], (N + 1 - n_samples, 1)).T
            else:
                shifted_nominal_trajectory = shifted_nominal_trajectory[:, : N + 1]

            # self._ktp.compute_reference_trajectory(self._mpc.params.dt)
            # self._ktp.update_path_variable(t - self._t_prev)
            self._mpc_trajectory, self._mpc_inputs = self._mpc.plan(
                t,
                nominal_trajectory=shifted_nominal_trajectory,
                nominal_inputs=None,
                xs=ownship_state,
                do_list=do_list,
                so_list=self._mpc_rel_polygons,
                enc=enc,
            )
            self._t_prev_mpc = t
            if enc is not None and self._mpc.params.debug:
                hf.plot_trajectory(shifted_nominal_trajectory, enc, "magenta")
                hf.plot_dynamic_obstacles(do_list, enc, self._mpc.params.T, self._mpc.params.dt)
                hf.plot_trajectory(self._mpc_trajectory, enc, color="cyan")
                ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.0, 1.0)
                enc.draw_polygon(ship_poly, color="pink")
            self._mpc_trajectory, self._mpc_inputs = self._interpolate_solution(t)
        else:
            self._mpc_trajectory = self._mpc_trajectory[:, 1:]
            self._mpc_inputs = self._mpc_inputs[:, 1:]

        self._references = np.zeros((nx + 3, len(self._mpc_trajectory[0, :])))
        self._references[:nx, :] = self._mpc_trajectory
        # print(f"RLMPC references: {self._references[:, 0]}")
        self._t_prev = t
        return self._references

    def _setup_mpc_static_obstacle_input(self, ownship_state: np.ndarray, enc: Optional[senc.ENC] = None, show_plots: bool = False, **kwargs) -> None:
        """Sets up the fixed static obstacle parameters for the MPC.

        Args:
            - ownship_state (np.ndarray): The ownship state.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - show_plots (bool): Whether to show plots or not.
            - **kwargs: Additional keyword arguments.
        """
        self._min_depth = mapf.find_minimum_depth(kwargs["os_draft"], enc)
        relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards(self._min_depth, enc)
        self._geometry_tree, self._original_poly_list = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

        poly_tuple_list, enveloping_polygon = mapf.extract_polygons_near_trajectory(
            self._nominal_trajectory, self._geometry_tree, buffer=self._mpc.params.reference_traj_bbox_buffer, enc=enc, show_plots=self._mpc.params.debug
        )
        for poly_tuple in poly_tuple_list:
            self._rel_polygons.extend(poly_tuple[0])

        self._mpc_rel_polygons = self._rel_polygons.copy()
        if enc is not None and show_plots:
            enc.start_display()
            # hf.plot_trajectory(waypoints, enc, color="green")
            hf.plot_trajectory(self._nominal_trajectory, enc, color="magenta")
            for hazard in relevant_grounding_hazards:
                enc.draw_polygon(hazard, color="red", fill=False)
            ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.5, 1.5)
            # enc.draw_circle((ownship_state[1], ownship_state[0]), radius=40, color="yellow", alpha=0.4)
            enc.draw_polygon(ship_poly, color="pink")
            # enc.draw_circle((goal_state[1], goal_state[0]), radius=40, color="cyan", alpha=0.4)

        if self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.CIRCULAR:
            self._mpc_rel_polygons = mapf.compute_smallest_enclosing_circle_for_polygons(self._rel_polygons, enc)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.ELLIPSOIDAL:
            self._mpc_rel_polygons = mapf.compute_multi_ellipsoidal_approximations_from_polygons(poly_tuple_list, enveloping_polygon, enc)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            P1, P2 = mapf.create_point_list_from_polygons(self._rel_polygons)
            self._set_generator = sg.SetGenerator(P1, P2)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.PARAMETRICSURFACE:
            self._mpc_rel_polygons = self._rel_polygons  # mapf.extract_boundary_polygons_inside_envelope(poly_tuple_list, enveloping_polygon, enc)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.TRIANGULARBOUNDARY:
            self._mpc_rel_polygons = mapf.extract_boundary_polygons_inside_envelope(poly_tuple_list, enveloping_polygon, enc)

    def _update_mpc_so_polygon_input(self, state: np.ndarray, enc: Optional[senc.ENC] = None, show_plots: bool = False) -> None:
        """Updates the static obstacle constraint parameters to the MPC, based on the constraint type used.

        Args:
            state (np.ndarray): _description_
            so_list (list): _description_
        """
        if self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            A_full, b_full = self._set_generator(state[0:2])
            A_reduced, b_reduced = sg.reduce_constraints(A_full, b_full, self._mpc.params.max_num_so_constr)
            if show_plots:
                sg.plot_constraints(A_reduced, b_reduced, state[0:2], "black", enc)
            self._mpc_rel_polygons = [A_reduced, b_reduced]

    def _interpolate_solution(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates the solution from the MPC to the time step in the simulation.

        Args:
            t (float): The current time step.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The interpolated solution state trajectory and input trajectory.
        """
        if self._mpc.params.dt > t - self._t_prev:
            dt_sim = np.max([t - self._t_prev, 0.5])
            sim_times = np.arange(0.0, self._mpc.params.T, dt_sim)
            mpc_times = np.arange(0.0, self._mpc.params.T, self._mpc.params.dt)
            x_d = interp1d(mpc_times, self._mpc_trajectory[0, :], kind="linear", bounds_error=False)
            y_d = interp1d(mpc_times, self._mpc_trajectory[1, :], kind="linear", bounds_error=False)
            psi_d = interp1d(mpc_times, self._mpc_trajectory[2, :], kind="linear", bounds_error=False)
            u_d = interp1d(mpc_times, self._mpc_trajectory[3, :], kind="linear", bounds_error=False)
            v_d = interp1d(mpc_times, self._mpc_trajectory[4, :], kind="linear", bounds_error=False)
            r_d = interp1d(mpc_times, self._mpc_trajectory[5, :], kind="linear", bounds_error=False)
            self._mpc_trajectory = np.zeros((6, len(sim_times)))
            self._mpc_trajectory[0, :] = x_d(sim_times)
            self._mpc_trajectory[1, :] = y_d(sim_times)
            self._mpc_trajectory[2, :] = psi_d(sim_times)
            self._mpc_trajectory[3, :] = u_d(sim_times)
            self._mpc_trajectory[4, :] = v_d(sim_times)
            self._mpc_trajectory[5, :] = r_d(sim_times)
            X_d = interp1d(mpc_times, self._mpc_inputs[0, :], kind="linear", bounds_error=False)
            Y_d = interp1d(mpc_times, self._mpc_inputs[1, :], kind="linear", bounds_error=False)
            self._mpc_inputs = np.zeros((2, len(sim_times)))
            self._mpc_inputs[0, :] = X_d(sim_times)
            self._mpc_inputs[1, :] = Y_d(sim_times)
        return self._mpc_trajectory, self._mpc_inputs

    def get_current_plan(self) -> np.ndarray:
        return self._references

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:

        if self._nominal_trajectory.size > 6:
            plt_handles["colav_nominal_trajectory"].set_xdata(self._nominal_trajectory[1, 0:-1:10])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._nominal_trajectory[0, 0:-1:10])

        if self._mpc_trajectory.size > 6:
            plt_handles["colav_predicted_trajectory"].set_xdata(self._mpc_trajectory[1, 0:-1:2])
            plt_handles["colav_predicted_trajectory"].set_ydata(self._mpc_trajectory[0, 0:-1:2])

        # plot convex safe set or relevant static obstacles

        # plot dynamic obstacles
        return plt_handles


def create_los_based_trajectory(
    xs: np.ndarray,
    waypoints: np.ndarray,
    speed_plan: np.ndarray,
    los: guidances.LOSGuidance,
    dt: float,
) -> np.ndarray:
    """Creates a trajectory based on the provided LOS guidance, controller and model.

    Args:
        - xs (np.ndarray): State vector
        - waypoints (np.ndarray): Waypoints
        - speed_plan (np.ndarray): Speed plan
        - los (guidances.LOSGuidance): LOS guidance object
        - dt (float): Time step

    Returns:
        np.ndarray: Trajectory
    """
    controller = controllers.FLSH()
    model = sim_models.Telemetron()
    trajectory = []
    xs_k = xs
    t = 0.0
    while t < 2000.0:
        trajectory.append(xs_k)
        references = los.compute_references(waypoints, speed_plan, None, xs_k, dt)
        u = controller.compute_inputs(references, xs_k, dt, model)
        xs_k = sim_integrators.erk4_integration_step(model.dynamics, model.bounds, xs_k, u, dt)

        dist2goal = np.linalg.norm(xs_k[0:2] - waypoints[:, -1])
        t += dt
        if dist2goal < 10.0:
            break
    return np.array(trajectory).T
