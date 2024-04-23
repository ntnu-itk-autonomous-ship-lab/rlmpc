"""
    rlmpc.py

    Summary:
        COLAV-simulator wrapper for the RL-MPC algorithm.

    Author: Trym Tengesdal
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as cs_mhm
import colav_simulator.common.plotters as plotters
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.guidances as guidances
import colav_simulator.core.stochasticity as stochasticity
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.colregs_handler as ch
import rlmpc.common.config_parsing as cp
import rlmpc.common.helper_functions as hf
import rlmpc.common.map_functions as rl_mapf
import rlmpc.common.paths as dp
import rlmpc.common.set_generator as sg
import rlmpc.mpc.common as mpc_common
import rlmpc.mpc.mid_level.mid_level_mpc as mlmpc
import rlmpc.mpc.parameters as mpc_params
import rrt_star_lib
import scipy.interpolate as interp
import seacharts.enc as senc
import yaml
from colav_simulator.behavior_generator import RRTConfig
from shapely import strtree


@dataclass
class RLMPCParams:
    mpc: mlmpc.Config
    los: guidances.LOSGuidanceParams
    rrtstar: RRTConfig
    colregs_handler: ch.COLREGSHandlerParams
    ship_length: float = 8.0
    ship_width: float = 3.0
    ship_draft: float = 0.5

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RLMPCParams(
            mpc=mlmpc.Config.from_dict(config_dict["midlevel_mpc"]),
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
            colregs_handler=ch.COLREGSHandlerParams.from_dict(config_dict["colregs_handler"]),
            rrtstar=RRTConfig.from_dict(config_dict["rrtstar"]),
        )
        return config

    def save(self, file: Path) -> None:
        """Saves the parameters to a YAML file.

        Args:
            file (Path): Path to the YAML file.
        """
        param_dict = {
            "mpc": self.mpc.to_dict(),
            "los": self.los.to_dict(),
            "colregs_handler": self.colregs_handler.to_dict(),
            "rrtstar": self.rrtstar.to_dict(),
            "ship_length": self.ship_length,
            "ship_width": self.ship_width,
            "ship_draft": self.ship_draft,
        }
        with file.open(mode="w") as f:
            yaml.dump(param_dict, f)

    @classmethod
    def from_file(cls, filename: Path):
        """Loads the parameters from a YAML file.

        Args:
            filename (Path): Path to the YAML file.

        Returns:
            RLMPCParams: The parameters.
        """
        assert filename.exists(), f"File {filename} does not exist"
        with open(filename, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class RLMPC(ci.ICOLAV):
    """The RL-MPC is a mid-level planner, using the MPC to plan a solution for tracking a nominal trajectory while avoiding obstacles.
    RL is used to update parameters online. Path-following/trajectory tracking can both be used. LOS-guidance is used to generate the nominal trajectory.
    """

    def __init__(self, config: RLMPCParams | Path = dp.rlmpc_config) -> None:
        if isinstance(config, RLMPCParams):
            self._config: RLMPCParams = config
        elif isinstance(config, Path):
            self._config = cp.extract(RLMPCParams, config, dp.rlmpc_schema)

        self._lookahead_sample: int = 3

        self._los = guidances.LOSGuidance(self._config.los)
        self._ktp = guidances.KinematicTrajectoryPlanner()
        self._mpc = mlmpc.MidlevelMPC(self._config.mpc)
        self._ma_filter = stochasticity.MovingAverageFilter()
        self._colregs_handler = ch.COLREGSHandler(self._config.colregs_handler)
        self._dt_sim: float = 0.5  # get from scenario config, typically alway s0 .5
        self._rrtstar = None
        self._map_origin: np.ndarray = np.array([])
        self._references = np.array([])
        self._initialized: bool = False
        self._t_prev: float = 0.0
        self._t_prev_mpc: float = 0.0
        self._min_depth: int = 0
        self._mpc_soln: dict = {}
        self._mpc_trajectory: np.ndarray = np.array([])
        self._mpc_inputs: np.ndarray = np.array([])
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])
        self._mpc_rel_polygons: list = []
        self._rel_polygons: list = []
        self._original_poly_list: list = []
        self._set_generator: Optional[sg.SetGenerator] = None
        self._debug: bool = True
        self._disturbance_handles: list = []

        self._goal_state: np.ndarray = np.array([])

        self._waypoints: np.ndarray = np.array([])
        self._speed_plan: np.ndarray = np.array([])
        self._enc: Optional[senc.ENC] = None
        self._nominal_path = None

    @property
    def lookahead_sample(self) -> int:
        return self._lookahead_sample

    def _clear_disturbance_handles(self) -> None:
        if self._disturbance_handles:
            for handle in self._disturbance_handles:
                handle.remove()
            self._disturbance_handles = []

    def get_nominal_path(
        self,
    ) -> Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.PchipInterpolator, float]:
        return self._nominal_path

    def initialize(
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
    ) -> None:
        """Initialize the planner by setting up the nominal path, static obstacle inputs and constructing
        the OCP"""
        self._waypoints = waypoints
        self._speed_plan = speed_plan
        self._speed_plan[self._speed_plan > 7.0] = 7.0
        self._speed_plan[self._speed_plan < 2.0] = 2.0
        self._mpc_soln: dict = {}
        self._mpc_trajectory: np.ndarray = np.array([])
        self._mpc_inputs: np.ndarray = np.array([])
        self._colregs_handler.reset()

        enc.close_display()
        self._enc = copy.deepcopy(enc)
        self._enc.start_display()
        ownship_csog_state = cs_mhm.convert_3dof_state_to_sog_cog_state(ownship_state)
        state_copy = ownship_csog_state.copy()
        ownship_csog_state[2] = state_copy[3]
        ownship_csog_state[3] = state_copy[2]
        ownship_csog_state[3] = ownship_state[3]
        speed_plan[-1] = 0.0
        self._map_origin = ownship_state[:2]
        self._nominal_path = self._ktp.compute_splines(
            waypoints=waypoints - np.array([self._map_origin[0], self._map_origin[1]]).reshape(2, 1),
            speed_plan=speed_plan,
            arc_length_parameterization=True,
        )
        self._setup_mpc_static_obstacle_input(ownship_csog_state, self._enc, self._debug, **kwargs)
        self._mpc.construct_ocp(
            nominal_path=self._nominal_path,
            xs=ownship_csog_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0]),
            so_list=self._mpc_rel_polygons,
            enc=self._enc,
            map_origin=self._map_origin,
            min_depth=self._min_depth,
        )

        self._goal_state = np.array([waypoints[0, -1], waypoints[1, -1], 0.0, 0.0, 0.0, 0.0])
        bbox = mapf.create_bbox_from_points(self._enc, ownship_csog_state[:2], self._goal_state[:2], buffer=300.0)
        planning_cdt = mapf.create_safe_sea_triangulation(
            self._enc,
            vessel_min_depth=5,
            buffer=self._mpc.params.r_safe_so,
            bbox=bbox,
            show_plots=False,
        )
        relevant_hazards = mapf.extract_hazards_within_bounding_box(
            self._rel_polygons, bbox, self._enc, show_plots=False
        )
        self._hazards = relevant_hazards[0]
        self._rrtstar = rrt_star_lib.RRTStar(
            los=self._config.rrtstar.los, model=self._config.rrtstar.model, params=self._config.rrtstar.params
        )
        self._rrtstar.reset(0)
        self._rrtstar.transfer_bbox(bbox)
        self._rrtstar.transfer_enc_hazards(relevant_hazards[0])
        self._rrtstar.transfer_safe_sea_triangulation(planning_cdt)
        self._rrtstar.set_goal_state(self._goal_state.tolist())

        if self._debug:
            self.plot_path(self._enc)

        os_poly = mapf.create_ship_polygon(
            ownship_csog_state[0], ownship_csog_state[1], ownship_csog_state[2], 8.0, 3.0, 1.0, 1.0
        )
        self._enc.draw_polygon(os_poly, color="pink")
        self._enc.draw_circle((self._goal_state[1], self._goal_state[0]), radius=5.0, color="black")

        # self.plot_surfaces(ownship_state)
        self._initialized = True
        # print("RL-MPC initialized!")

    def plot_hazards(self):
        """Plot the grounding hazards."""
        for poly in self._hazards:
            self._enc.draw_polygon(poly, color="red", alpha=0.7)

    def plot_surfaces(self, ownship_state: np.ndarray, npoints: int = 300):
        """Plot surface interpolations of the static obstacles."""
        so_surfaces = self._mpc.get_antigrounding_surface_functions()
        fig, ax = plt.subplots()
        center = ownship_state[:2] - np.array([self._map_origin[0], self._map_origin[1]])
        npx = npoints
        npy = npoints
        x = np.linspace(center[0] - 150, center[0] + 150, npx)
        y = np.linspace(center[1] - 150, center[1] + 150, npy)
        z = np.zeros((npy, npx))
        for idy, y_val in enumerate(y):
            for idx, x_val in enumerate(x):
                for surface in so_surfaces:
                    surfval = min(1.0, surface(np.array([x_val, y_val]).reshape(1, 2)).full()[0][0])
                    z[idy, idx] += max(0.0, surfval)
        pc = ax.pcolormesh(x, y, z, shading="gouraud", rasterized=True)
        ax.scatter(center[1], center[0], color="red", s=30, marker="x")
        cbar = fig.colorbar(pc)
        cbar.set_label("Surface value capped to +-1.0")
        ax.set_xlabel("North [m]")
        ax.set_ylabel("East [m]")
        plt.show(block=False)

    def act(
        self,
        t: float,
        ownship_state: np.ndarray,
        do_list: list,
        w: Optional[stochasticity.DisturbanceData] = None,
        prev_soln: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Act function for the RL-MPC. Calls the plan function and returns the reference trajectory.

        Args:
            t (float): Current time.
            ownship_state (np.ndarray): Current ownship state on the form [x, y, psi, u, v, r]^T.
            do_list (list): List of dynamic obstacles.
            w (Optional[stochasticity.DisturbanceData]): Stochastic disturbance data.
            prev_soln (Optional[dict]): Previous MPC solution.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The reference action and the most recent MPC solution.
        """
        if prev_soln:
            self._mpc_trajectory = prev_soln["trajectory"]
            self._mpc_inputs = prev_soln["inputs"]
            self._t_prev = prev_soln["t_prev"]
            self._t_prev_mpc = prev_soln["t_prev_mpc"]

        _ = self.plan(
            t,
            self._waypoints,
            self._speed_plan,
            ownship_state,
            do_list,
            self._enc,
            w=w,
            goal_state=None,
            prev_soln=prev_soln,
            **kwargs,
        )
        mpc_output = self._mpc_soln

        U = np.sqrt(ownship_state[3] ** 2 + ownship_state[4] ** 2)  # absolute speed / COG
        chi = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])  # COg
        chi_0_ref = self._mpc_trajectory[2, 1]  # starting from 1 since the first sample is the current state
        chi_1_ref = self._mpc_trajectory[2, 2]
        U_0_ref = self._mpc_trajectory[3, 1]
        U_1_ref = self._mpc_trajectory[3, 2]
        action = np.array([chi_0_ref - chi, U_0_ref - U, chi_1_ref - chi, U_1_ref - U])

        print(
            f"t: {t} | U_mpc: {U_0_ref} | U: {U} | chi_mpc: {180.0 * chi_0_ref / np.pi} | chi: {180.0 * chi / np.pi} | r_mpc: {0.0} | r: {ownship_state[5]}"
        )
        # double check action indices:
        # nx, nu, ns, _ = self._mpc.dims()
        # n_samples = self._mpc.params.T / self._mpc.params.dt
        # action_indices = [
        #     int(nu * n_samples + (1 * nx) + 2),  # chi 2
        #     int(nu * n_samples + (1 * nx) + 3),  # speed 2
        #     int(nu * n_samples + (2 * nx) + 2),  # chi 3
        #     int(nu * n_samples + (2 * nx) + 3),  # speed 3
        # ]
        # action_vals = self._mpc_soln["soln"]["x"].full().flatten()[action_indices]

        return action, mpc_output

    def get_mpc_params(self) -> mpc_params.MidlevelMPCParams:
        """Returns the MPC parameters.

        Returns:
            mpc_params.MidlevelMPCParams: The MPC parameters.
        """
        return self._mpc.params

    def get_mpc_model_dims(self) -> Tuple[int, int]:
        """Returns the model dimensions.

        Returns:
            Tuple[int, int]: The MPC state and input dimensions.
        """
        return self._mpc.model_dims

    def get_adjustable_mpc_params(self) -> np.ndarray:
        """Returns the adjustable parameters of the MPC."""
        return self._mpc.adjustable_params

    def get_fixed_mpc_params(self) -> np.ndarray:
        """Returns the fixed parameters of the (casadi) MPC NLP."""
        return self._mpc.fixed_params

    def update_adjustable_mpc_params(self, delta: np.ndarray) -> None:
        """Updates the adjustable parameters of the MPC."""
        self._mpc.update_adjustable_params(delta)

    def set_action_indices(self, action_indices: list):
        """Sets the indices of the action variables used for calculating the sensitivity da_dp.

        Args:
            action_indices (list): List of indices of the action variables in the decision vector.
        """
        self._mpc.set_action_indices(action_indices)

    def set_mpc_params(self, params: mpc_params.MidlevelMPCParams) -> None:
        """Sets the MPC parameters.

        Args:
            params (mpc_params.MidlevelMPCParams): The MPC parameters.
        """
        self._mpc.set_params(params)

    def save_params(self, filename: Path) -> None:
        """Saves the parameters to a YAML file.

        Args:
            filename (Path): Path to the YAML file.
        """
        self._config.save(filename)

    def load_params(self, filename: Path) -> None:
        """Loads the parameters from a YAML file. Overwrites the current parameters.

        NOTE: You should now re-initialize the planner.

        Args:
            filename (Path): Path to the YAML file.
        """
        self._config = RLMPCParams.from_file(filename)

    def visualize_disturbance(self, ddata: stochasticity.DisturbanceData | None) -> None:
        """Visualizes the disturbance object.

        Args:
            disturbance (stoch.Disturbance | None): Disturbance object.
        """
        if ddata is None:
            return

        self._clear_disturbance_handles()

        handles = []
        if ddata.currents is not None and ddata.currents["speed"] > 0.0:
            speed = ddata.currents["speed"]
            handles.extend(
                plotters.plot_disturbance(
                    magnitude=90.0,
                    direction=ddata.currents["direction"],
                    name=f"current: {speed:.2f} m/s",
                    enc=self._enc,
                    color="white",
                    linewidth=1.0,
                    location="topright",
                    text_location_offset=(0.0, 0.0),
                )
            )

        if ddata.wind is not None and ddata.wind["speed"] > 0.0:
            speed = ddata.wind["speed"]
            handles.extend(
                plotters.plot_disturbance(
                    magnitude=90.0,
                    direction=ddata.wind["direction"],
                    name=f"wind: {speed:.2f} m/s",
                    enc=self._enc,
                    color="peru",
                    linewidth=1.0,
                    location="topright",
                    text_location_offset=(0.0, -20.0),
                )
            )
        self._disturbance_handles = handles

    def plot_path(self, enc: senc.ENC) -> None:
        """Plot the nominal path."""
        enc.start_display()
        nominal_trajectory = self._ktp.compute_reference_trajectory(2.0)
        nominal_trajectory = nominal_trajectory + np.array(
            [self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ).reshape(9, 1)
        plotters.plot_waypoints(
            self._waypoints[:2, :],
            draft=1.0,
            enc=enc,
            color="orange",
            point_buffer=3.0,
            disk_buffer=6.0,
            hole_buffer=3.0,
            alpha=0.4,
        )
        plotters.plot_trajectory(
            nominal_trajectory[:2, :],
            enc,
            "yellow",
        )

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
        assert waypoints.size > 2, "Waypoints and speed plan must be provided to the RLMPC"
        if not self._initialized:
            self.initialize(t, waypoints, speed_plan, ownship_state, do_list, enc, goal_state, w, **kwargs)

        if t == 0 or t - self._t_prev_mpc >= 1.0 / self._mpc.params.rate:
            translated_do_list = hf.translate_dynamic_obstacle_coordinates(
                do_list, self._map_origin[1], self._map_origin[0]
            )

            if self._debug:
                plotters.plot_dynamic_obstacles(do_list, "red", enc, self._mpc.params.dt, self._mpc.params.dt)
                ship_poly = mapf.create_ship_polygon(
                    ownship_state[0],
                    ownship_state[1],
                    ownship_state[2],
                    self._config.ship_length,
                    self._config.ship_width,
                    1.0,
                    1.0,
                )
                enc.draw_polygon(ship_poly, color="pink")
            csog_state = cs_mhm.convert_3dof_state_to_sog_cog_state(ownship_state)
            csog_state_cpy = csog_state.copy()
            csog_state[2] = csog_state_cpy[3]
            csog_state[3] = csog_state_cpy[2]
            do_cr_list, do_ho_list, do_ot_list = self._colregs_handler.handle(
                csog_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0]), translated_do_list
            )

            # if self._debug:
            #     plotters.plot_dynamic_obstacles(
            #         do_cr_list, "blue", enc, self._mpc.params.T, self._mpc.params.dt, map_origin=self._map_origin
            #     )
            #     plotters.plot_dynamic_obstacles(
            #         do_ho_list, "orange", enc, self._mpc.params.T, self._mpc.params.dt, map_origin=self._map_origin
            #     )
            #     plotters.plot_dynamic_obstacles(
            #         do_ot_list, "magenta", enc, self._mpc.params.T, self._mpc.params.dt, map_origin=self._map_origin
            #     )

            warm_start = self.create_mpc_warm_start(ownship_state, enc, **kwargs)

            self._mpc_soln = self._mpc.plan(
                t,
                xs=ownship_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]),
                do_cr_list=do_cr_list,
                do_ho_list=do_ho_list,
                do_ot_list=do_ot_list,
                so_list=self._mpc_rel_polygons,
                enc=enc,
                warm_start=warm_start,
                **kwargs,
            )
            self._mpc_trajectory = self._mpc_soln["trajectory"]
            self._mpc_inputs = self._mpc_soln["inputs"]
            self._mpc_trajectory[:2, :] += self._map_origin.reshape((2, 1))

            if self._debug:
                plotters.plot_trajectory(self._mpc_trajectory, enc, color="cyan")

            self._t_prev_mpc = t
            self._mpc_soln["trajectory"] = self._mpc_trajectory
            self._mpc_soln["inputs"] = self._mpc_inputs
            self._mpc_soln["t_prev_mpc"] = t
            self._mpc_soln["xs_prev"] = ownship_state - np.array(
                [self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]
            )
            self._mpc_soln["u_prev"] = self._mpc_inputs[:, 0]
        #
        # lookahead_sample = 4  # 5s=dt_mpc between each sample, => 30 * 0.5 = 15s => for 2m/s speed, 30m ahead
        waypoints = np.column_stack((self._mpc_trajectory[:2, 0], self._mpc_trajectory[:2, self.lookahead_sample]))
        speed_plan = np.array(
            [self._mpc_trajectory[3, self.lookahead_sample], self._mpc_trajectory[3, self.lookahead_sample]]
        )
        # if self._debug:
        #     plotters.plot_waypoints(
        #         waypoints, enc, color="magenta", point_buffer=3.0, disk_buffer=6.0, hole_buffer=3.0, alpha=0.4
        #     )

        self._references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        self._mpc_soln["t_prev"] = t
        U = np.sqrt(ownship_state[3] ** 2 + ownship_state[4] ** 2)
        chi = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])
        # print(
        #     f"t: {t} | U_mpc: {self._references[3, 0]} | U: {U} | chi_mpc: {180.0 * self._references[2, 0] / np.pi} | chi: {180.0 * chi / np.pi} | r_mpc: {self._references[5, 0]} | r: {ownship_state[5]}"
        # )
        return self._references

    def create_mpc_warm_start(
        self, ownship_state: np.ndarray, enc: senc.ENC, **kwargs
    ) -> Optional[Dict[str, np.ndarray]]:
        """Creates a warm start for the MPC by growing an RRT from the terminal MPC state towards the goal state. If t == 0, the RRT is grown from the current ownship state.

        Args:
            ownship_state (np.ndarray): Ownship state on the form [x, y, psi, u, v, r]^T.
            enc (senc.ENC): Electronic navigational chart.

        Returns:
            Dict[str, np.ndarray]: Warm start dictionary containing the trajectory, inputs, waypoints and times.
        """
        nx, nu, ns, dim_g = self._mpc.dims()
        prev_soln = self._mpc_soln if "soln" in self._mpc_soln else None
        is_prev_soln = True if "prev_soln" in kwargs and kwargs["prev_soln"] else False
        prev_soln = kwargs["prev_soln"] if is_prev_soln else None

        os_state_csog = ownship_state.copy()
        os_state_csog[2] = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])
        os_state_csog[3] = np.sqrt(ownship_state[3] ** 2 + ownship_state[4] ** 2)
        start_state = self._mpc_trajectory[:, -1] if self._mpc_trajectory.size > 0 else os_state_csog
        start_state = prev_soln["trajectory"][:, -1] if is_prev_soln else start_state
        last_mpc_input = self._mpc_inputs[:, -1] if self._mpc_inputs.size > 0 else np.zeros((nu, 1))
        last_mpc_input = prev_soln["inputs"][:, -1] if is_prev_soln else last_mpc_input
        path_var, path_var_dot = self._mpc.compute_path_variable_info(
            start_state[:4] - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0])
        )
        nominal_speed_ref = float(self._nominal_path[3](path_var))

        self._rrtstar.reset(0)
        self._rrtstar.set_goal_state(self._goal_state.tolist())
        start_state_copy = start_state.copy()
        start_state_copy[4:] = np.array([0.0, 0.0])
        rrt_soln = self._rrtstar.grow_towards_goal(
            ownship_state=start_state_copy.tolist(),
            U_d=nominal_speed_ref,
            initialized=False,
            return_on_first_solution=True if prev_soln else False,
        )
        _, rrt_trajectory, rrt_inputs, rrt_times = cs_mhm.parse_rrt_solution(rrt_soln)

        N = int(self._mpc.params.T / self._mpc.params.dt)
        if rrt_trajectory.size == 0:
            rrt_trajectory = self._mpc.model_prediction(start_state, last_mpc_input.reshape(nu, 1), N + 1)
            rrt_times = np.arange(0, N + 1) * self._mpc.params.dt
            rrt_inputs = np.tile(last_mpc_input.reshape(3, 1), (1, N))

        # plotters.plot_rrt_tree(self._rrtstar.get_tree_as_list_of_dicts(), self._enc)
        plotters.plot_trajectory(rrt_trajectory, enc, color="black")
        rrt_trajectory -= np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]).reshape(6, 1)

        if self._config.rrtstar.params.step_size < self._mpc.params.dt:
            step = int(self._mpc.params.dt / self._config.rrtstar.params.step_size)
            rrt_trajectory = rrt_trajectory[:, ::step][:, : N + 1]
            rrt_times = rrt_times[::step][: N + 1]
            num_rrt_samples = rrt_trajectory.shape[1]
            rrt_inputs = rrt_inputs[:, ::step][:, : num_rrt_samples - 1]

        if rrt_trajectory.shape[1] < N + 1:
            # use model prediction to extend the trajectory
            sample_diff = N + 1 - rrt_trajectory.shape[1]
            model_traj = self._mpc.model_prediction(
                rrt_trajectory[:, -1], rrt_inputs[:, -1].reshape(nu, 1), sample_diff + 1
            )
            rrt_trajectory = np.concatenate((rrt_trajectory, model_traj[:, 1:]), axis=1)
            rrt_times = np.concatenate((rrt_times, rrt_times[-1] + np.arange(1, sample_diff + 1) * self._mpc.params.dt))
            rrt_inputs = np.concatenate(
                (rrt_inputs, np.tile(rrt_inputs[:, -1].reshape(nu, 1), (1, sample_diff))), axis=1
            )

        if prev_soln:
            rrt_trajectory[4:, 0] = start_state[4:]
        else:
            rrt_trajectory[4:, 0] = np.array([path_var, path_var_dot])

        last_mpc_input = self._mpc_inputs[:, -1] if self._mpc_inputs.size > 0 else np.array([0.0, 0.0, -0.04])
        last_mpc_input = prev_soln["inputs"][:, -1] if is_prev_soln else last_mpc_input
        rrt_inputs[2, :] = np.tile(last_mpc_input[2], (1, rrt_inputs.shape[1]))

        # Add path timing dynamics to warm start trajectory
        mpc_model_traj = self._mpc.model_prediction(rrt_trajectory[:, 0], rrt_inputs, rrt_trajectory.shape[1])
        rrt_trajectory[4:, :] = mpc_model_traj[4:, :]
        chi = rrt_trajectory[2, :]
        rrt_trajectory[2, :] = np.unwrap(chi)

        # 2s mellom kvar mpc iter
        # 4s MPC step time => må shifte MPC trajectory 2s fram i tid
        warm_start = {}
        if prev_soln:
            U, X, Sigma = self._mpc.decision_trajectories(prev_soln["soln"])
            warm_start.update(
                {"xs_prev": prev_soln["xs_prev"], "u_prev": prev_soln["u_prev"], "sigma_prev": Sigma[:, 0]}
            )
            U = np.concatenate((U[:, 1:], rrt_inputs[:, 0].reshape(nu, 1)), axis=1)
            X = np.concatenate((X[:, 1:], rrt_trajectory[:, 1].reshape(nx, 1)), axis=1)
            Sigma = np.concatenate((Sigma[:, 1:], np.zeros((Sigma.shape[0], 1))), axis=1)
            w = self._mpc.decision_variables(U, X, Sigma)
            lam_g = prev_soln["soln"]["lam_g"]
            lam_g = np.concatenate((lam_g[1:], lam_g[-1].reshape(1, 1)))
            lam_x = prev_soln["soln"]["lam_x"]
            lam_x = np.concatenate((lam_x[1:], lam_x[-1].reshape(1, 1)))
        else:
            U = rrt_inputs[:, :N]
            X = rrt_trajectory[:, : N + 1]
            Sigma = np.zeros((ns, N + 1))
            w = self._mpc.decision_variables(U, X, Sigma)
            lam_g = np.zeros((dim_g, 1))
            lam_x = np.zeros((w.shape[0], 1))

        warm_start = {"x": w, "lam_g": lam_g, "lam_x": lam_x, "X": X, "U": U, "Sigma": Sigma}
        return warm_start

    def compute_disturbance_velocity_estimate(self, w: Optional[stochasticity.DisturbanceData] = None) -> np.ndarray:
        """Computes the disturbance velocity estimate for the given time.

        Args:
            w (Optional[stochasticity.DisturbanceData]): Stochastic disturbance data.

        Returns:
            np.ndarray: The disturbance velocity estimate.
        """
        v_disturbance = np.array([0.0, 0.0])
        if w is None:
            return v_disturbance
        if "speed" in w.wind:
            V_w = w.wind["speed"]
            beta_w = w.wind["direction"]
            v_w = np.array([V_w * np.cos(beta_w), V_w * np.sin(beta_w)])
            v_disturbance += v_w
        if "speed" in w.currents:
            V_c = w.currents["speed"]
            beta_c = w.currents["direction"]
            v_c = np.array([V_c * np.cos(beta_c), V_c * np.sin(beta_c)])
            v_disturbance += v_c
        return self._ma_filter.update(v_disturbance)

    def build_sensitivities(self, tau: Optional[float] = None) -> mpc_common.NLPSensitivities:
        """Builds the sensitivity of the KKT matrix function underlying the MPC NLP with respect to the decision variables and parameters.

        Args:
            tau (float, optional): Barrier parameter used in the primal-dual formulation. Defaults to None.

        Returns:
            mpc_common.NLPSensitivities: Class containing the sensitivity functions necessary for computing the score function gradient in RL context.
        """
        return self._mpc.build_sensitivities(tau)

    def _setup_mpc_static_obstacle_input(
        self, ownship_state: np.ndarray, enc: Optional[senc.ENC] = None, show_plots: bool = False, **kwargs
    ) -> None:
        """Sets up the fixed static obstacle parameters for the MPC.

        Args:
            - ownship_state (np.ndarray): The ownship state on the form [x, y, psi, u, v, r]^T.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - show_plots (bool): Whether to show plots or not.
            - **kwargs: Additional keyword arguments.
        """
        self._min_depth = mapf.find_minimum_depth(self._config.ship_draft, enc)
        relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
            self._min_depth, enc, buffer=self._mpc.params.r_safe_so, show_plots=show_plots
        )
        self._geometry_tree, self._original_poly_list = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

        nominal_trajectory = self._ktp.compute_reference_trajectory(self._mpc.params.dt)
        nominal_trajectory = nominal_trajectory + np.array(
            [self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ).reshape(9, 1)
        poly_tuple_list, enveloping_polygon = mapf.extract_polygons_near_trajectory(
            nominal_trajectory,
            self._geometry_tree,
            buffer=self._mpc.params.reference_traj_bbox_buffer,
            enc=enc,
            show_plots=self._debug,
        )
        for poly_tuple in poly_tuple_list:
            self._rel_polygons.extend(poly_tuple[0])

        if enc is not None and show_plots:
            enc.start_display()
            # hf.plot_trajectory(waypoints, enc, color="green")
            plotters.plot_trajectory(nominal_trajectory, enc, color="yellow")
            for hazard in self._rel_polygons:
                enc.draw_polygon(hazard, color="red", fill=True)

            ship_poly = mapf.create_ship_polygon(
                ownship_state[0],
                ownship_state[1],
                ownship_state[2],
                self._config.ship_length,
                self._config.ship_width,
                1.0,
                1.0,
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
            self._mpc_rel_polygons = rl_mapf.compute_smallest_enclosing_circle_for_polygons(
                translated_rel_polygons, enc, self._map_origin
            )
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.ELLIPSOIDAL:
            self._mpc_rel_polygons = rl_mapf.compute_multi_ellipsoidal_approximations_from_polygons(
                translated_poly_tuple_list, translated_enveloping_polygon, enc, self._map_origin
            )
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            P1, P2 = mapf.create_point_list_from_polygons(translated_rel_polygons)
            self._set_generator = sg.SetGenerator(P1, P2)
        elif self._mpc.params.so_constr_type == mpc_params.StaticObstacleConstraint.PARAMETRICSURFACE:
            self._mpc_rel_polygons = translated_poly_tuple_list  # mapf.extract_boundary_polygons_inside_envelope(poly_tuple_list, enveloping_polygon, enc)
        else:
            raise ValueError(f"Unknown static obstacle constraint type: {self._mpc.params.so_constr_type}")

    def get_mpc_antigrounding_surface_functions(self) -> list:
        """Returns the surface interpolation functions for the anti-grounding constraints.

        Returns:
            list: List of interpolation functions.
        """
        return self._mpc.get_antigrounding_surface_functions()

    def get_current_plan(self) -> np.ndarray:
        return self._mpc_trajectory

    def get_colav_data(self) -> dict:
        output = {}
        if self._t_prev_mpc == self._t_prev:
            output = {
                "time_of_last_plan": self._t_prev_mpc,
                "mpc_soln": self._mpc_soln,
                "mpc_trajectory": self._mpc_trajectory,
                "mpc_inputs": self._mpc_inputs,
                "params": self._config,
                "t": self._t_prev,
            }
        return output

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:
        """NOTE: Must use the "colav_nominal_trajectory" and "colav_predicted_trajectory" keys in the plt_handles dictionary.

        Args:
            ax_map (plt.Axes): Matplotlib axes object.
            enc (senc.ENC): ENC object.
            plt_handles (dict): Dictionary containing the matplotlib handles.

        Returns:
            dict: Updated matplotlib handles.
        """
        if self._mpc_trajectory.size > 8:
            plt_handles["colav_predicted_trajectory"].set_xdata(self._mpc_trajectory[1, ::2])
            plt_handles["colav_predicted_trajectory"].set_ydata(self._mpc_trajectory[0, ::2])

        return plt_handles
