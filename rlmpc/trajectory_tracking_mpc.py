"""
    antigrounding_mpc.py

    Summary:
        COLAV-simulator planner wrapper for the anti-grounding MPC.

    Author: Trym Tengesdal
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import colav_simulator.common.map_functions as mapf
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.guidances as guidances
import colav_simulator.core.stochasticity as stochasticity
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.config_parsing as cp
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as dp
import rlmpc.common.set_generator as sg
import rlmpc.mpc.parameters as mpc_params
import rlmpc.mpc.trajectory_tracking.ttmpc as ttmpc
import seacharts.enc as senc
from shapely import strtree


@dataclass
class TrajectoryTrackingMPCParams:
    los: guidances.LOSGuidanceParams
    mpc: ttmpc.Config

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = TrajectoryTrackingMPCParams(
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
            mpc=ttmpc.Config.from_dict(config_dict["ttmpc"]),
        )
        return config


class TrajectoryTrackingMPC(ci.ICOLAV):
    """MPC for COLAV with anti-grounding functionality."""

    def __init__(
        self, config: Optional[TrajectoryTrackingMPCParams] = None, config_file: Optional[Path] = dp.ttmpc_config
    ) -> None:

        if config:
            self._config: TrajectoryTrackingMPCParams = config
        else:
            self._config = cp.extract(TrajectoryTrackingMPCParams, config_file, dp.ttmpc_schema)

        self._los = guidances.LOSGuidance(self._config.los)
        self._mpc = ttmpc.TTMPC(self._config.mpc)

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
        w: Optional[stochasticity.DisturbanceData] = None,
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
            self._nominal_trajectory, self._nominal_inputs = hf.create_los_based_trajectory(
                ownship_state, waypoints, speed_plan, self._los, self._mpc.params.dt
            )
            self._setup_mpc_static_obstacle_input(ownship_state, enc, self._mpc.params.debug, **kwargs)
            self._nominal_trajectory[:2, :] -= self._map_origin.reshape((2, 1))
            translated_do_list = hf.translate_dynamic_obstacle_coordinates(
                do_list, self._map_origin[1], self._map_origin[0]
            )
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
        translated_do_list = hf.translate_dynamic_obstacle_coordinates(
            do_list, self._map_origin[1], self._map_origin[0]
        )
        self._update_mpc_so_polygon_input(ownship_state, enc, self._mpc.params.debug)

        if t == 0 or t - self._t_prev_mpc >= 1.0 / self._mpc.params.rate:
            nominal_trajectory, nominal_inputs = hf.shift_nominal_plan(
                self._nominal_trajectory,
                self._nominal_inputs,
                ownship_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]),
                N,
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
                hf.plot_trajectory(
                    nominal_trajectory
                    + np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]).reshape(6, 1),
                    enc,
                    "yellow",
                )
                hf.plot_dynamic_obstacles(do_list, enc, self._mpc.params.T, self._mpc.params.dt, color="red")
                hf.plot_trajectory(self._mpc_trajectory, enc, color="cyan")
                ship_poly = hf.create_ship_polygon(
                    ownship_state[0],
                    ownship_state[1],
                    ownship_state[2],
                    kwargs["os_length"],
                    kwargs["os_width"],
                    1.0,
                    1.0,
                )
                enc.draw_polygon(ship_poly, color="pink")
            self._mpc_trajectory, self._mpc_inputs = hf.interpolate_solution(
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

    def _setup_mpc_static_obstacle_input(
        self, ownship_state: np.ndarray, enc: Optional[senc.ENC] = None, show_plots: bool = False, **kwargs
    ) -> None:
        """Sets up the fixed static obstacle parameters for the MPC.

        Args:
            - ownship_state (np.ndarray): The ownship state.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - show_plots (bool): Whether to show plots or not.
            - **kwargs: Additional keyword arguments.
        """
        self._min_depth = mapf.find_minimum_depth(kwargs["os_draft"], enc)
        relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(self._min_depth, enc)
        self._geometry_tree, self._original_poly_list = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

        poly_tuple_list, enveloping_polygon = mapf.extract_polygons_near_trajectory(
            self._nominal_trajectory,
            self._geometry_tree,
            buffer=self._mpc.params.reference_traj_bbox_buffer,
            enc=enc,
            show_plots=self._mpc.params.debug,
        )
        for poly_tuple in poly_tuple_list:
            self._rel_polygons.extend(poly_tuple[0])

        if enc is not None and show_plots:
            enc.start_display()
            # hf.plot_trajectory(waypoints, enc, color="green")
            hf.plot_trajectory(self._nominal_trajectory, enc, color="yellow")
            for hazard in self._rel_polygons:
                enc.draw_polygon(hazard, color="red", fill=False)

            ship_poly = hf.create_ship_polygon(
                ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.0, 1.0
            )
            # enc.draw_circle((ownship_state[1], ownship_state[0]), radius=40, color="yellow", alpha=0.4)
            enc.draw_polygon(ship_poly, color="pink")
            # enc.draw_circle((goal_state[1], goal_state[0]), radius=40, color="cyan", alpha=0.4)

        # Translate the polygons to the origin of the map
        translated_rel_polygons = hf.translate_polygons(
            self._rel_polygons.copy(), self._map_origin[1], self._map_origin[0]
        )
        translated_poly_tuple_list = []
        for polygons, original_polygon in poly_tuple_list:
            translated_poly_tuple_list.append(
                (
                    hf.translate_polygons(polygons, self._map_origin[1], self._map_origin[0]),
                    hf.translate_polygons([original_polygon], self._map_origin[1], self._map_origin[0])[0],
                )
            )
        translated_enveloping_polygon = hf.translate_polygons(
            [enveloping_polygon], self._map_origin[1], self._map_origin[0]
        )[0]

        # enc.save_image(name="enc_hazards", path=dp.figures, extension="pdf")
        if self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.CIRCULAR:
            self._mpc_rel_polygons = mapf.compute_smallest_enclosing_circle_for_polygons(
                translated_rel_polygons, enc, self._map_origin
            )
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

    def _update_mpc_so_polygon_input(
        self, ownship_state: np.ndarray, enc: Optional[senc.ENC] = None, show_plots: bool = False
    ) -> None:
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
                sg.plot_constraints(
                    A_reduced, b_reduced, ownship_state[0:2] - self._map_origin, "black", enc, self._map_origin
                )
            self._mpc_rel_polygons = [A_reduced, b_reduced]

    def get_current_plan(self) -> np.ndarray:
        return self._references

    def get_colav_data(self) -> dict:
        output = {}
        if self._t_prev_mpc == self._t_prev:
            output = {
                "nominal_trajectory": self._nominal_trajectory
                + np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]).reshape(6, 1),
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
            plt_handles["colav_nominal_trajectory"].set_xdata(self._nominal_trajectory[1, 0:-1:5] + self._map_origin[1])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._nominal_trajectory[0, 0:-1:5] + self._map_origin[0])

        if self._mpc_trajectory.size > 6:
            plt_handles["colav_predicted_trajectory"].set_xdata(self._mpc_trajectory[1, 0:-1:2])
            plt_handles["colav_predicted_trajectory"].set_ydata(self._mpc_trajectory[0, 0:-1:2])

        return plt_handles
