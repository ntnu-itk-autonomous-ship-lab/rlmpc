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
import colav_simulator.core.guidances as guidances
import informed_rrt_star_rust as rrt
import numpy as np
import rl_rrt_mpc.common.config_parsing as cp
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.mpc.models as mpc_models
import rl_rrt_mpc.mpc.mpc as mpc
import rl_rrt_mpc.rl as rl
import seacharts.enc as senc
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
        **kwargs
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
            tree_list = self._rrt.get_tree_as_list_of_dicts()
            if enc is not None:
                enc.start_display()
                for hazard in relevant_grounding_hazards:
                    enc.draw_polygon(hazard, color="red", fill=False)
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
                hf.plot_trajectory(self._rrt_trajectory, times, enc, color="magenta")

            polygons_considered_in_mpc = mapf.extract_polygons_near_trajectory(
                self._rrt_trajectory, self._geometry_tree, buffer=self._config.mpc.reference_traj_bbox_buffer, enc=enc
            )
            triangle_polygons = mapf.extract_triangle_boundaries_from_polygons(polygons_considered_in_mpc, enc=enc)
            self._mpc_rel_polygons = hf.shift_polygon_coordinates(polygons_considered_in_mpc, enc.origin[0], enc.origin[1])
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
        )
        self._mpc_trajectory[0, :] += enc.origin[1]
        self._mpc_trajectory[1, :] += enc.origin[0]

        hf.plot_dynamic_obstacles(do_list, enc, self._mpc.params.T, self._mpc.params.dt)
        hf.plot_trajectory(self._mpc_trajectory, times, enc, color="cyan")
        references = np.zeros((9, len(self._mpc_trajectory[0, :])))
        references[:6, :] = self._mpc_trajectory
        return references

    def get_current_plan(self) -> np.ndarray:
        return self._references


@dataclass
class RLMPCParams:
    rl: rl.RLParams
    ktp: guidances.KTPGuidanceParams
    mpc: mpc.Config

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RLMPCParams(
            rl=rl.RLParams.from_dict(config_dict["rl"]),
            ktp=guidances.KTPGuidanceParams.from_dict(config_dict["ktp"]),
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
        self._mpc = mpc.MPC(mpc_models.Telemetron(), self._config.mpc)

        self._references = np.empty(9)
        self._initialized = False
        self._t_prev = 0.0
        self._min_depth: int = 0
        self._ktp_trajectory: np.ndarray = np.empty(6)
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
        **kwargs
    ) -> np.ndarray:
        """Implements the ICOLAV plan interface function. Relies on getting the own-ship minimum depth
        in order to extract relevant grounding hazards.
        """
        assert goal_state is not None, "Goal state must be provided to the RL-RRT-MPC"
        assert enc is not None, "ENC must be provided to the RL-RRT-MPC"

        x_spline, y_spline, psi_spline, speed_spline = self._ktp.compute_splines(waypoints, speed_plan, None)
        # spline_ktp_trajectory = [x_spline, y_spline, psi_spline, speed_spline]

        self._ktp_trajectory = self._ktp.compute_reference_trajectory(self._mpc.params.dt)
        if enc is not None:
            hf.plot_trajectory(self._ktp_trajectory, np.array([]), enc, color="magenta")

        if not self._initialized:
            self._initialized = True
            self._min_depth = mapf.find_minimum_depth(kwargs["os_draft"], enc)
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards(self._min_depth, enc)
            self._geometry_tree, poly_list = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

            if enc is not None:
                enc.start_display()
                # for hazard in relevant_grounding_hazards:
                #     enc.draw_polygon(hazard, color="red", fill=False)
                ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 3, 1.5)
                enc.draw_circle((ownship_state[1], ownship_state[0]), radius=40, color="yellow", alpha=0.4)
                enc.draw_polygon(ship_poly, color="pink")
                enc.draw_circle((goal_state[1], goal_state[0]), radius=40, color="cyan", alpha=0.4)

            triangle_polygons = mapf.extract_boundary_polygons_near_trajectory(
                self._ktp_trajectory, self._geometry_tree, buffer=self._mpc.params.reference_traj_bbox_buffer, enc=enc
            )
            self._mpc.construct_ocp(nominal_trajectory=self._ktp_trajectory, do_list=do_list, so_list=poly_list, enc=enc)

        self._mpc_trajectory, self._mpc_inputs = self._mpc.plan(
            nominal_trajectory=self._ktp_trajectory,
            nominal_inputs=None,
            xs=ownship_state,
            do_list=do_list,
            so_list=triangle_polygons,
            enc=enc,
        )

        hf.plot_dynamic_obstacles(do_list, enc, self._mpc.params.T, self._mpc.params.dt)
        hf.plot_trajectory(self._mpc_trajectory, np.array([]), enc, color="cyan")
        references = np.zeros((9, len(self._mpc_trajectory[0, :])))
        references[:6, :] = self._mpc_trajectory
        self._t_prev = t
        return references

    def get_current_plan(self) -> np.ndarray:
        return self._references
