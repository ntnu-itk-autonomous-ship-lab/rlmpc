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

        self._rrt_inputs: np.ndarray = np.empty(3)
        self._rrt_trajectory: np.ndarray = np.empty(6)
        self._rrt_references: np.ndarray = np.empty(2)
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])

        self._mpc = mpc.MPC(mpc_models.Telemetron(), self._config.mpc)

        self._map_origin: np.ndarray = np.array([])
        self._references = np.array([])
        self._initialized = False
        self._t_prev: float = 0.0
        self._t_prev_mpc: float = 0.0
        self._min_depth: int = 0
        self._mpc_soln: dict = {}
        self._mpc_trajectory: np.ndarray = np.array([])
        self._mpc_inputs: np.ndarray = np.array([])
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
        assert goal_state is not None, "Goal state must be provided to the RL-RRT-MPC"
        assert enc is not None, "ENC must be provided to the RL-RRT-MPC"
        if not self._initialized:
            self._min_depth = mapf.find_minimum_depth(kwargs["os_draft"], enc)
            self._t_prev = t
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards(self._min_depth, enc)
            self._geometry_tree, _ = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)
            self._rrt.transfer_enc_data(relevant_grounding_hazards)
            self._rrt.set_init_state(ownship_state.tolist())
            self._rrt.set_goal_state(goal_state.tolist())

            U_d = ownship_state[3]  # Constant desired speed given by the initial own-ship speed
            rrt_solution: dict = self._rrt.grow_towards_goal(ownship_state.tolist(), U_d, [])
            hf.save_rrt_solution(rrt_solution)
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

            self._setup_mpc_static_obstacle_input(ownship_state, goal_state, enc, **kwargs)
            self._rrt_trajectory[:2, :] -= self._map_origin.reshape((2, 1))
            translated_do_list = hf.translate_dynamic_obstacle_coordinates(do_list, self._map_origin[1], self._map_origin[0])
            self._mpc.construct_ocp(
                nominal_trajectory=self._rrt_trajectory,
                nominal_inputs=self._rrt_inputs,
                xs=ownship_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]),
                do_list=translated_do_list,
                so_list=self._mpc_rel_polygons,
                enc=enc,
                map_origin=self._map_origin,
                min_depth=self._min_depth,
            )

        translated_do_list = hf.translate_dynamic_obstacle_coordinates(do_list, self._map_origin[1], self._map_origin[0])
        self._update_mpc_so_polygon_input(ownship_state, enc, self._mpc.params.debug)

        if t == 0 or t - self._t_prev_mpc >= 1.0 / self._mpc.params.rate:
            N = int(self._mpc.params.T / self._mpc.params.dt)
            nominal_trajectory, nominal_inputs = shift_nominal_plan(
                self._rrt_trajectory, self._rrt_inputs, ownship_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]), N
            )
            psi_nom = nominal_trajectory[2, :]
            nominal_trajectory[2, :] = np.unwrap(np.concatenate(([psi_nom[0]], psi_nom)))[1:]
            self._mpc_soln = self._mpc.plan(
                t,
                nominal_trajectory=nominal_trajectory,
                nominal_inputs=nominal_inputs,
                xs=ownship_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]),
                do_list=translated_do_list,
                so_list=self._mpc_rel_polygons,
                enc=enc,
            )
            self._mpc_trajectory = self._mpc_soln["trajectory"][:, :N]
            self._mpc_inputs = self._mpc_soln["inputs"]
            self._mpc_trajectory[:2, :] += self._map_origin.reshape((2, 1))

            if enc is not None and self._mpc.params.debug:
                hf.plot_trajectory(nominal_trajectory + np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]).reshape(6, 1), enc, "magenta")
                hf.plot_dynamic_obstacles(do_list, enc, self._mpc.params.T, self._mpc.params.dt)
                hf.plot_trajectory(self._mpc_trajectory, enc, color="cyan")
                ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.0, 1.0)
                enc.draw_polygon(ship_poly, color="pink")
            self._mpc_trajectory, self._mpc_inputs = interpolate_solution(
                self._mpc_trajectory, self._mpc_inputs, t, self._t_prev_mpc, self._mpc.params.T, self._mpc.params.dt
            )
            d2last_ref = np.linalg.norm(nominal_trajectory[:2, -1] - ownship_state[:2])
            # self._los.reset_wp_counter()
            self._t_prev_mpc = t

        else:
            self._mpc_trajectory = self._mpc_trajectory[:, 1:]
            self._mpc_inputs = self._mpc_inputs[:, 1:]

        self._t_prev = t
        # Alternative 1: Use LOS-guidance to track the MPC trajectory
        # self._references = self._los.compute_references(
        #     self._mpc_trajectory[:2, :], speed_plan=self._mpc_trajectory[3, :], times=None, xs=ownship_state, dt=t - self._t_prev
        # )
        # self._references = np.zeros((9, len(self._mpc_trajectory[0, :])))
        # self._references[:nx, :] = self._mpc_trajectory

        # Alternative 2: Apply MPC inputs directly to the ownship
        self._references = np.zeros((9, len(self._mpc_inputs[0, :])))
        self._references[:2, :] = self._mpc_inputs

        return self._references

    def _setup_mpc_static_obstacle_input(
        self, ownship_state: np.ndarray, goal_state: np.ndarray, enc: Optional[senc.ENC] = None, show_plots: bool = False, **kwargs
    ) -> None:
        """Sets up the fixed static obstacle parameters for the MPC.

        Args:
            - ownship_state (np.ndarray): The ownship state.
            - goal_state (np.ndarray): The goal state.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - show_plots (bool): Whether to show plots or not.
            - **kwargs: Additional keyword arguments.
        """
        poly_tuple_list, enveloping_polygon = mapf.extract_polygons_near_trajectory(
            self._rrt_trajectory, self._geometry_tree, buffer=self._mpc.params.reference_traj_bbox_buffer, enc=enc, show_plots=self._mpc.params.debug
        )
        for poly_tuple in poly_tuple_list:
            self._rel_polygons.extend(poly_tuple[0])

        if enc is not None and show_plots:
            enc.start_display()
            # hf.plot_trajectory(waypoints, enc, color="green")
            hf.plot_trajectory(self._rrt_trajectory, enc, color="magenta")
            for hazard in self._rel_polygons:
                enc.draw_polygon(hazard, color="red", fill=False)

            ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.0, 1.0)
            # enc.draw_circle((ownship_state[1], ownship_state[0]), radius=40, color="yellow", alpha=0.4)
            enc.draw_polygon(ship_poly, color="pink")
            enc.draw_circle((goal_state[1], goal_state[0]), radius=40, color="cyan", alpha=0.4)

        # Translate the polygons to the origin of the map
        translated_rel_polygons = hf.translate_polygons(self._rel_polygons.copy(), self._map_origin[1], self._map_origin[0])
        translated_poly_tuple_list = []
        for polygons, original_polygon in poly_tuple_list:
            translated_poly_tuple_list.append(
                (
                    hf.translate_polygons(polygons, self._map_origin[1], self._map_origin[0]),
                    hf.translate_polygons([original_polygon], self._map_origin[1], self._map_origin[0])[0],
                )
            )
        translated_enveloping_polygon = hf.translate_polygons([enveloping_polygon], self._map_origin[1], self._map_origin[0])[0]

        # enc.save_image(name="enc_hazards", path=dp.figures, extension="pdf")
        if self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.CIRCULAR:
            self._mpc_rel_polygons = mapf.compute_smallest_enclosing_circle_for_polygons(translated_rel_polygons, enc, self._map_origin)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.ELLIPSOIDAL:
            self._mpc_rel_polygons = mapf.compute_multi_ellipsoidal_approximations_from_polygons(
                translated_poly_tuple_list, translated_enveloping_polygon, enc, self._map_origin
            )
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            P1, P2 = mapf.create_point_list_from_polygons(translated_rel_polygons)
            self._set_generator = sg.SetGenerator(P1, P2)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.PARAMETRICSURFACE:
            self._mpc_rel_polygons = translated_poly_tuple_list  # mapf.extract_boundary_polygons_inside_envelope(poly_tuple_list, enveloping_polygon, enc)
        else:
            raise ValueError(f"Unknown static obstacle constraint type: {self._mpc.params.so_constr_type}")

    def _update_mpc_so_polygon_input(self, ownship_state: np.ndarray, enc: Optional[senc.ENC] = None, show_plots: bool = False) -> None:
        """Updates the static obstacle constraint parameters to the MPC, based on the constraint type used.

        Args:
            - state (np.ndarray): The current ownship state.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - show_plots (bool): Whether to show plots or not.
        """
        if self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            A_full, b_full = self._set_generator(ownship_state[0:2] - self._map_origin)
            A_reduced, b_reduced = sg.reduce_constraints(A_full, b_full, self._mpc.params.max_num_so_constr)
            if show_plots:
                sg.plot_constraints(A_reduced, b_reduced, ownship_state[0:2] - self._map_origin, "black", enc, self._map_origin)
            self._mpc_rel_polygons = [A_reduced, b_reduced]

    def get_current_plan(self) -> np.ndarray:
        return self._references

    def get_colav_data(self) -> dict:
        output = {}
        if self._t_prev_mpc == self._t_prev:
            output = {
                "nominal_trajectory": self._rrt_trajectory + np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]).reshape(6, 1),
                "nominal_inputs": self._rrt_inputs,
                "time_of_last_plan": self._t_prev_mpc,
                "mpc_soln": self._mpc_soln,
                "mpc_trajectory": self._mpc_trajectory,
                "mpc_inputs": self._mpc_inputs,
                "params": self._config,
                "t": self._t_prev,
            }
        return output

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:

        if self._rrt_trajectory.size > 6:
            plt_handles["colav_nominal_trajectory"].set_xdata(self._rrt_trajectory[1, 0:-1:10] + self._map_origin[1])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._rrt_trajectory[0, 0:-1:10] + self._map_origin[0])

        if self._mpc_trajectory.size > 6:
            plt_handles["colav_predicted_trajectory"].set_xdata(self._mpc_trajectory[1, 0:-1:10])
            plt_handles["colav_predicted_trajectory"].set_ydata(self._mpc_trajectory[0, 0:-1:10])

        return plt_handles


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
    RL is used to update parameters online. Path-following/trajectory tracking can both be used. LOS-guidance is used to generate the nominal trajectory.
    """

    def __init__(self, config: Optional[RLMPCParams] = None, config_file: Optional[Path] = dp.rl_mpc_config) -> None:

        if config:
            self._config: RLMPCParams = config
        else:
            self._config = cp.extract(RLMPCParams, config_file, dp.rl_mpc_schema)

        self._rl = rl.RL(self._config.rl)
        self._los = guidances.LOSGuidance(self._config.los)
        self._mpc = mpc.MPC(mpc_models.Telemetron(), self._config.mpc)

        self._map_origin: np.ndarray = np.array([])
        self._references = np.array([])
        self._initialized: bool = False
        self._t_prev: float = 0.0
        self._t_prev_mpc: float = 0.0
        self._min_depth: int = 0
        self._nominal_trajectory: np.ndarray = np.array([])
        self._nominal_inputs: np.ndarray = np.array([])
        self._mpc_soln: dict = {}
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
        assert enc is not None, "ENC must be provided to the RL-MPC"
        N = int(self._mpc.params.T / self._mpc.params.dt)
        if not self._initialized:
            self._map_origin = ownship_state[:2]
            self._initialized = True
            self._nominal_trajectory, self._nominal_inputs = create_los_based_trajectory(ownship_state, waypoints, speed_plan, self._los, self._mpc.params.dt)
            self._setup_mpc_static_obstacle_input(ownship_state, enc, self._mpc.params.debug, **kwargs)
            self._nominal_trajectory[:2, :] -= self._map_origin.reshape((2, 1))
            translated_do_list = hf.translate_dynamic_obstacle_coordinates(do_list, self._map_origin[1], self._map_origin[0])
            self._mpc.construct_ocp(
                nominal_trajectory=self._nominal_trajectory,
                nominal_inputs=self._nominal_inputs,
                xs=ownship_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]),
                do_list=translated_do_list,
                so_list=self._mpc_rel_polygons,
                enc=enc,
                map_origin=self._map_origin,
                min_depth=self._min_depth,
            )
        translated_do_list = hf.translate_dynamic_obstacle_coordinates(do_list, self._map_origin[1], self._map_origin[0])
        self._update_mpc_so_polygon_input(ownship_state, enc, self._mpc.params.debug)

        if t == 0 or t - self._t_prev_mpc >= 1.0 / self._mpc.params.rate:
            nominal_trajectory, nominal_inputs = shift_nominal_plan(
                self._nominal_trajectory, self._nominal_inputs, ownship_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]), N
            )
            psi_nom = nominal_trajectory[2, :]
            nominal_trajectory[2, :] = np.unwrap(np.concatenate(([psi_nom[0]], psi_nom)))[1:]
            self._mpc_soln = self._mpc.plan(
                t,
                nominal_trajectory=nominal_trajectory,
                nominal_inputs=nominal_inputs,
                xs=ownship_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]),
                do_list=translated_do_list,
                so_list=self._mpc_rel_polygons,
                enc=enc,
            )
            self._mpc_trajectory = self._mpc_soln["trajectory"][:, :N]
            self._mpc_inputs = self._mpc_soln["inputs"]
            self._mpc_trajectory[:2, :] += self._map_origin.reshape((2, 1))

            if enc is not None and self._mpc.params.debug:
                hf.plot_trajectory(nominal_trajectory + np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]).reshape(6, 1), enc, "magenta")
                hf.plot_dynamic_obstacles(do_list, enc, self._mpc.params.T, self._mpc.params.dt)
                hf.plot_trajectory(self._mpc_trajectory, enc, color="cyan")
                ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.0, 1.0)
                enc.draw_polygon(ship_poly, color="pink")
            self._mpc_trajectory, self._mpc_inputs = interpolate_solution(
                self._mpc_trajectory, self._mpc_inputs, t, self._t_prev, self._mpc.params.T, self._mpc.params.dt
            )
            d2last_ref = np.linalg.norm(nominal_trajectory[:2, -1] - ownship_state[:2])
            # self._los.reset_wp_counter()
            self._t_prev_mpc = t
            if t == 190.0 or t == 400.0 or t == 550.0:
                print("here")

        else:
            self._mpc_trajectory = self._mpc_trajectory[:, 1:]
            self._mpc_inputs = self._mpc_inputs[:, 1:]

        self._t_prev = t
        # Alternative 1: Use LOS-guidance to track the MPC trajectory
        # self._references = self._los.compute_references(
        #     self._mpc_trajectory[:2, :], speed_plan=self._mpc_trajectory[3, :], times=None, xs=ownship_state, dt=t - self._t_prev
        # )
        # self._references = np.zeros((9, len(self._mpc_trajectory[0, :])))
        # self._references[:nx, :] = self._mpc_trajectory

        # Alternative 2: Apply MPC inputs directly to the ownship
        self._references = np.zeros((9, len(self._mpc_inputs[0, :])))
        self._references[:2, :] = self._mpc_inputs

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

        if enc is not None and show_plots:
            enc.start_display()
            # hf.plot_trajectory(waypoints, enc, color="green")
            hf.plot_trajectory(self._nominal_trajectory, enc, color="magenta")
            for hazard in self._rel_polygons:
                enc.draw_polygon(hazard, color="red", fill=False)

            ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.0, 1.0)
            # enc.draw_circle((ownship_state[1], ownship_state[0]), radius=40, color="yellow", alpha=0.4)
            enc.draw_polygon(ship_poly, color="pink")
            # enc.draw_circle((goal_state[1], goal_state[0]), radius=40, color="cyan", alpha=0.4)

        # Translate the polygons to the origin of the map
        translated_rel_polygons = hf.translate_polygons(self._rel_polygons.copy(), self._map_origin[1], self._map_origin[0])
        translated_poly_tuple_list = []
        for polygons, original_polygon in poly_tuple_list:
            translated_poly_tuple_list.append(
                (
                    hf.translate_polygons(polygons, self._map_origin[1], self._map_origin[0]),
                    hf.translate_polygons([original_polygon], self._map_origin[1], self._map_origin[0])[0],
                )
            )
        translated_enveloping_polygon = hf.translate_polygons([enveloping_polygon], self._map_origin[1], self._map_origin[0])[0]

        # enc.save_image(name="enc_hazards", path=dp.figures, extension="pdf")
        if self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.CIRCULAR:
            self._mpc_rel_polygons = mapf.compute_smallest_enclosing_circle_for_polygons(translated_rel_polygons, enc, self._map_origin)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.ELLIPSOIDAL:
            self._mpc_rel_polygons = mapf.compute_multi_ellipsoidal_approximations_from_polygons(
                translated_poly_tuple_list, translated_enveloping_polygon, enc, self._map_origin
            )
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            P1, P2 = mapf.create_point_list_from_polygons(translated_rel_polygons)
            self._set_generator = sg.SetGenerator(P1, P2)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.PARAMETRICSURFACE:
            self._mpc_rel_polygons = translated_poly_tuple_list  # mapf.extract_boundary_polygons_inside_envelope(poly_tuple_list, enveloping_polygon, enc)
        else:
            raise ValueError(f"Unknown static obstacle constraint type: {self._mpc.params.so_constr_type}")

    def _update_mpc_so_polygon_input(self, ownship_state: np.ndarray, enc: Optional[senc.ENC] = None, show_plots: bool = False) -> None:
        """Updates the static obstacle constraint parameters to the MPC, based on the constraint type used.

        Args:
            - state (np.ndarray): The current ownship state.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - show_plots (bool): Whether to show plots or not.
        """
        if self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            A_full, b_full = self._set_generator(ownship_state[0:2] - self._map_origin)
            A_reduced, b_reduced = sg.reduce_constraints(A_full, b_full, self._mpc.params.max_num_so_constr)
            if show_plots:
                sg.plot_constraints(A_reduced, b_reduced, ownship_state[0:2] - self._map_origin, "black", enc, self._map_origin)
            self._mpc_rel_polygons = [A_reduced, b_reduced]

    def get_current_plan(self) -> np.ndarray:
        return self._references

    def get_colav_data(self) -> dict:
        output = {}
        if self._t_prev_mpc == self._t_prev:
            output = {
                "nominal_trajectory": self._nominal_trajectory + np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]).reshape(6, 1),
                "nominal_inputs": self._nominal_inputs,
                "time_of_last_plan": self._t_prev_mpc,
                "mpc_soln": self._mpc_soln,
                "mpc_trajectory": self._mpc_trajectory,
                "mpc_inputs": self._mpc_inputs,
                "params": self._config,
                "t": self._t_prev,
            }
        return output

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:

        if self._nominal_trajectory.size > 6:
            plt_handles["colav_nominal_trajectory"].set_xdata(self._nominal_trajectory[1, 0:-1:10] + self._map_origin[1])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._nominal_trajectory[0, 0:-1:10] + self._map_origin[0])

        if self._mpc_trajectory.size > 6:
            plt_handles["colav_predicted_trajectory"].set_xdata(self._mpc_trajectory[1, 0:-1:10])
            plt_handles["colav_predicted_trajectory"].set_ydata(self._mpc_trajectory[0, 0:-1:10])

        return plt_handles


def create_los_based_trajectory(
    xs: np.ndarray,
    waypoints: np.ndarray,
    speed_plan: np.ndarray,
    los: guidances.LOSGuidance,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
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
    inputs = []
    xs_k = xs
    t = 0.0
    reached_goal = False
    t_braking = 30.0
    t_brake_start = 0.0
    while t < 2000.0:
        trajectory.append(xs_k)
        references = los.compute_references(waypoints, speed_plan, None, xs_k, dt)
        if reached_goal:
            references[3:] = np.tile(0.0, (references[3:].size, 1))
        u = controller.compute_inputs(references, xs_k, dt, model)
        inputs.append(u)
        xs_k = sim_integrators.erk4_integration_step(model.dynamics, model.bounds, xs_k, u, dt)

        dist2goal = np.linalg.norm(xs_k[0:2] - waypoints[:, -1])
        t += dt
        if dist2goal < 70.0 and not reached_goal:
            reached_goal = True
            t_brake_start = t

        if reached_goal and t - t_brake_start > t_braking:
            break

    return np.array(trajectory).T, np.array(inputs)[:, :2].T


def interpolate_solution(trajectory: np.ndarray, inputs: np.ndarray, t: float, t_prev: float, T_mpc: float, dt_mpc: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolates the solution from the MPC to the time step in the simulation.

    Args:
        - trajectory (np.ndarray): The solution state trajectory.
        - inputs (np.ndarray): The solution input trajectory.
        - t (float): The current time step.
        - t_prev (float): The previous time step.
        - T_mpc (float): The MPC horizon.
        - dt_mpc (float): The MPC time step.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The interpolated solution state trajectory and input trajectory.
    """
    intp_trajectory = trajectory
    intp_inputs = inputs
    if dt_mpc > t - t_prev or dt_mpc < t - t_prev:
        dt_sim = np.max([t - t_prev, 0.5])
        sim_times = np.arange(0.0, T_mpc, dt_sim)
        mpc_times = np.arange(0.0, T_mpc, dt_mpc)
        x_d = interp1d(mpc_times, trajectory[0, :], kind="linear", bounds_error=False)
        y_d = interp1d(mpc_times, trajectory[1, :], kind="linear", bounds_error=False)
        psi_d = interp1d(mpc_times, trajectory[2, :], kind="linear", bounds_error=False)
        u_d = interp1d(mpc_times, trajectory[3, :], kind="linear", bounds_error=False)
        v_d = interp1d(mpc_times, trajectory[4, :], kind="linear", bounds_error=False)
        r_d = interp1d(mpc_times, trajectory[5, :], kind="linear", bounds_error=False)
        intp_trajectory = np.zeros((6, len(sim_times)))
        intp_trajectory[0, :] = x_d(sim_times)
        intp_trajectory[1, :] = y_d(sim_times)
        intp_trajectory[2, :] = psi_d(sim_times)
        intp_trajectory[3, :] = u_d(sim_times)
        intp_trajectory[4, :] = v_d(sim_times)
        intp_trajectory[5, :] = r_d(sim_times)
        X_d = interp1d(mpc_times, inputs[0, :], kind="linear", bounds_error=False)
        Y_d = interp1d(mpc_times, inputs[1, :], kind="linear", bounds_error=False)
        intp_inputs = np.zeros((2, len(sim_times)))
        intp_inputs[0, :] = X_d(sim_times)
        intp_inputs[1, :] = Y_d(sim_times)
    return intp_trajectory, intp_inputs


def shift_nominal_plan(nominal_trajectory: np.ndarray, nominal_inputs: np.ndarray, ownship_state: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Updates the nominal trajectory and inputs to the MPC based on the current ownship state. This is done by
    find closest point on nominal trajectory to the current state and then shifting the nominal trajectory to this point

    Args:
        - nominal_trajectory (np.ndarray): The nominal trajectory.
        - nominal_inputs (np.ndarray): The nominal inputs.
        - ownship_state (np.ndarray): The ownship state.
        - N (int): MPC horizon length in samples

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
