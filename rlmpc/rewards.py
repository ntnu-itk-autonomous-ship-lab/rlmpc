"""
    rewards.py

    Summary:
        Reward functions for the RL agent.

    Author: Trym Tengesdal
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as cs_mhm
import colav_simulator.common.plotters as plotters
import colav_simulator.core.guidances as guidances
import colav_simulator.gym.action as csgym_action
import colav_simulator.gym.observation as csgym_obs
import colav_simulator.gym.reward as cs_reward
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.colregs_handler as ch
import rlmpc.common.helper_functions as hf
import rlmpc.common.map_functions as rl_mapf
import rlmpc.mpc.common as mpc_common
import yaml

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


@dataclass
class AntiGroundingRewarderParams:
    rho_anti_grounding: float = 100.0
    r_safe: float = 5.0
    reference_traj_bbox_buffer: float = 200.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            rho_anti_grounding=config_dict["rho_anti_grounding"],
            r_safe=config_dict["r_safe"],
            reference_traj_bbox_buffer=config_dict["reference_traj_bbox_buffer"],
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollisionAvoidanceRewarderParams:
    rho_colav: float = 100.0  # collision avoidance reward weight
    r_safe: float = 30.0  # safe distance to nearby dynamic obstacles/vessels

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(rho_colav=config_dict["rho_colav"], r_safe=config_dict["r_safe"])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class COLREGRewarderParams:
    alpha_cr: np.ndarray = np.array([1.0 / 500.0, 1.0 / 500.0])  # Crossing potential function parameters
    y_0_cr: float = 100.0  # Crossing potential function parameters
    alpha_ho: np.ndarray = np.array([1.0 / 500.0, 1.0 / 500.0])  # Head-on potential function parameters
    x_0_ho: float = 200.0  # Head-on potential function parameters
    alpha_ot: np.ndarray = np.array([1.0 / 500.0, 1.0 / 500.0])  # Overtaking potential function parameters
    x_0_ot: float = 200.0  # Overtaking potential function parameters
    y_0_ot: float = 100.0  # Overtaking potential function parameters
    d_attenuation: float = 400.0  # attenuation distance for the COLREGS potential functions
    w_colregs: np.ndarray = np.array([1.0, 1.0, 1.0])  # Weights for the COLREGS potential functions

    colregs_handler: ch.COLREGSHandlerParams = field(default_factory=lambda: ch.COLREGSHandlerParams())

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = cls()
        params.alpha_cr = np.array(config_dict["alpha_cr"])
        params.y_0_cr = config_dict["y_0_cr"]
        params.alpha_ho = np.array(config_dict["alpha_ho"])
        params.x_0_ho = config_dict["x_0_ho"]
        params.alpha_ot = np.array(config_dict["alpha_ot"])
        params.x_0_ot = config_dict["x_0_ot"]
        params.y_0_ot = config_dict["y_0_ot"]
        params.d_attenuation = config_dict["d_attenuation"]
        params.w_colregs = np.array(config_dict["w_colregs"])
        params.colregs_handler = ch.COLREGSHandlerParams.from_dict(config_dict["colregs_handler"])
        return params

    def to_dict(self) -> dict:
        return {
            "alpha_cr": self.alpha_cr.tolist(),
            "y_0_cr": self.y_0_cr,
            "alpha_ho": self.alpha_ho.tolist(),
            "x_0_ho": self.x_0_ho,
            "alpha_ot": self.alpha_ot.tolist(),
            "x_0_ot": self.x_0_ot,
            "y_0_ot": self.y_0_ot,
            "d_attenuation": self.d_attenuation,
            "w_colregs": self.w_colregs.tolist(),
            "colregs_handler": self.colregs_handler.to_dict(),
        }


@dataclass
class TrajectoryTrackingRewarderParams:
    rho_path_dev: float = 0.5  # path deviation reward weight
    rho_speed_dev: float = 10.0  # speed deviation reward weight

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = cls()
        params.rho_path_dev = config_dict["rho_path_dev"]
        params.rho_speed_dev = config_dict["rho_speed_dev"]
        return params

    def to_dict(self) -> dict:
        return {
            "rho_path_dev": self.rho_path_dev,
            "rho_speed_dev": self.rho_speed_dev,
        }


@dataclass
class ReadilyApparentManeuveringRewarderParams:
    K_app_course: float = 15.0  # rate cost weight for turn rate
    K_app_speed: float = 10.0  # rate cost weight for speed
    alpha_app_course: np.ndarray = field(
        default_factory=lambda: np.array([112.5, 0.00006])
    )  # Rate cost function parameters for turn rate
    alpha_app_speed: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 0.00025])
    )  # Rate cost function parameters for speed
    r_max: float = 6.0  # Maximum turn rate
    a_max: float = 2.0  # Maximum acceleration

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = cls()
        params.K_app_course = config_dict["K_app_course"]
        params.K_app_speed = config_dict["K_app_speed"]
        params.alpha_app_course = np.array(config_dict["alpha_app_course"])
        params.alpha_app_speed = np.array(config_dict["alpha_app_speed"])
        params.r_max = np.deg2rad(config_dict["r_max"])
        params.a_max = config_dict["a_max"]
        return params

    def to_dict(self) -> dict:
        return {
            "K_app_course": self.K_app_course,
            "K_app_speed": self.K_app_speed,
            "alpha_app_course": self.alpha_app_course.tolist(),
            "alpha_app_speed": self.alpha_app_speed.tolist(),
            "r_max": np.rad2deg(self.r_max),
            "a_max": self.a_max,
        }


@dataclass
class Config:

    trajectory_tracking: TrajectoryTrackingRewarderParams = field(
        default_factory=lambda: TrajectoryTrackingRewarderParams()
    )
    anti_grounding: AntiGroundingRewarderParams = field(default_factory=lambda: AntiGroundingRewarderParams())
    collision_avoidance: CollisionAvoidanceRewarderParams = field(
        default_factory=lambda: CollisionAvoidanceRewarderParams()
    )
    colreg: COLREGRewarderParams = field(default_factory=lambda: COLREGRewarderParams())
    readily_apparent_maneuvering: ReadilyApparentManeuveringRewarderParams = field(
        default_factory=lambda: ReadilyApparentManeuveringRewarderParams()
    )

    @classmethod
    def from_dict(cls, config_dict: dict):
        return Config(
            trajectory_tracking=TrajectoryTrackingRewarderParams.from_dict(config_dict["trajectory_tracking"]),
            anti_grounding=AntiGroundingRewarderParams.from_dict(config_dict["anti_grounding"]),
            collision_avoidance=CollisionAvoidanceRewarderParams.from_dict(config_dict["collision_avoidance"]),
            colreg=COLREGRewarderParams.from_dict(config_dict["colreg"]),
            readily_apparent_maneuvering=ReadilyApparentManeuveringRewarderParams.from_dict(
                config_dict["readily_apparent_maneuvering"]
            ),
        )

    @classmethod
    def from_file(cls, config_file: Path) -> "Config":
        with open(config_file, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls.from_dict(config_dict)


class AntiGroundingRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: AntiGroundingRewarderParams) -> None:
        super().__init__(env)
        self._config: AntiGroundingRewarderParams = config
        self._ktp: guidances.KinematicTrajectoryPlanner = guidances.KinematicTrajectoryPlanner()
        self._polygons: list = []
        self._rel_polygons: list = []
        self._show_plots: bool = False
        self._so_surfaces: list = []
        self._map_origin: np.ndarray = np.zeros(2)
        self._initialized: bool = False

    def create_so_surfaces(self):
        self._map_origin = self.env.ownship.state[:2]
        self._nominal_path = self._ktp.compute_splines(
            waypoints=self.env.ownship.waypoints - np.array([self._map_origin[0], self._map_origin[1]]).reshape(2, 1),
            speed_plan=self.env.ownship.speed_plan,
            arc_length_parameterization=True,
        )

        self._setup_static_obstacle_input()
        self._so_surfaces, _ = rl_mapf.compute_surface_approximations_from_polygons(
            self._rel_polygons,
            self.env.enc,
            safety_margins=[0.0],
            map_origin=self._map_origin,
            show_plots=False,
        )
        self._so_surfaces = self._so_surfaces[0]
        self._initialized = True

    def _setup_static_obstacle_input(self) -> None:
        """Setup the static obstacle input for the anti-grounding rewarder."""
        self._polygons = []
        self._rel_polygons = []
        self._min_depth = mapf.find_minimum_depth(self.env.ownship.draft, self.env.enc)
        relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
            self._min_depth, self.env.enc, buffer=self._config.r_safe, show_plots=False
        )
        self._geometry_tree, _ = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

        nominal_trajectory = self._ktp.compute_reference_trajectory(dt=5.0)
        nominal_trajectory = nominal_trajectory + np.array(
            [self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ).reshape(9, 1)
        poly_tuple_list, enveloping_polygon = mapf.extract_polygons_near_trajectory(
            nominal_trajectory,
            self._geometry_tree,
            buffer=self._config.reference_traj_bbox_buffer,  # should match the mpc reference trajectory buffer
            enc=self.env.enc,
            show_plots=self._show_plots,
        )
        for poly_tuple in poly_tuple_list:
            self._polygons.extend(poly_tuple[0])

        translated_poly_tuple_list = []
        for polygons, original_polygon in poly_tuple_list:
            translated_poly_tuple_list.append(
                (
                    hf.translate_polygons(polygons, self._map_origin[1], self._map_origin[0]),
                    hf.translate_polygons([original_polygon], self._map_origin[1], self._map_origin[0])[0],
                )
            )
        self._rel_polygons = translated_poly_tuple_list

    def plot_surfaces(self, npoints: int = 300):
        """Plot surface interpolations of the static obstacles."""
        ownship_state = self.env.ownship.state
        fig, ax = plt.subplots()
        center = ownship_state[:2] - np.array([self._map_origin[0], self._map_origin[1]])
        npx = npoints
        npy = npoints
        x = np.linspace(center[0] - 150, center[0] + 150, npx)
        y = np.linspace(center[1] - 150, center[1] + 150, npy)
        z = np.zeros((npy, npx))
        for idy, y_val in enumerate(y):
            for idx, x_val in enumerate(x):
                for surface in self._so_surfaces:
                    surfval = min(1.0, surface(np.array([x_val, y_val]).reshape(1, 2)).full()[0][0])
                    z[idy, idx] += max(0.0, surfval)
        pc = ax.pcolormesh(x, y, z, shading="gouraud", rasterized=True)
        ax.scatter(center[1], center[0], color="red", s=30, marker="x")
        cbar = fig.colorbar(pc)
        cbar.set_label("Surface value capped to +-1.0")
        ax.set_xlabel("North [m]")
        ax.set_ylabel("East [m]")
        plt.show(block=False)

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.time < 0.0001:
            self.create_so_surfaces()
        p_os = (self.env.ownship.state[:2] - self._map_origin).reshape(1, 2)
        g_so = np.zeros(len(self._so_surfaces))
        for j, surface in enumerate(self._so_surfaces):
            surf_val = surface(p_os).full()[0][0]
            g_so[j] = np.clip(surf_val, 0.0, 1.0)
            if g_so[j] > 0.0:
                d2so = np.linalg.norm(
                    mapf.compute_distance_vectors_to_grounding(self.env.ownship.state, self._min_depth, self.env.enc)
                )
                print(f"Static obstacle {j} is too close to the ownship! g_so[i]={g_so[j]} | d2so={d2so}.")
        grounding_cost = self._config.rho_anti_grounding * g_so.sum()

        # Add extra cost if the ship is grounded in the simulator (i.e. the ship is on land)
        # The above cost only gives a penalty if the ship CG is inside the grounding polygon approximations.
        if self.env.simulator.determine_ship_grounding():
            grounding_cost += self._config.rho_anti_grounding * 1.0

        return -grounding_cost


class CollisionAvoidanceRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: CollisionAvoidanceRewarderParams) -> None:
        super().__init__(env)
        self._config = config

    def compute_dynamic_obstacle_constraint(self, do_tuple: Tuple[int, np.ndarray, np.ndarray, float, float]) -> float:
        """Compute the dynamic obstacle constraint for the given dynamic obstacle.

        Args:
            p_os (np.ndarray): The ownship position.
            do_tuple (Tuple[int, np.ndarray, np.ndarray, float, float]): Tuple containing the dynamic obstacle index, state, covariance, length, and width.

        Returns:
            float: The dynamic obstacle constraint value.
        """
        do_idx, do_state, do_length, do_width = do_tuple
        do_course = np.arctan2(do_state[3], do_state[2])
        Rchi_do_i = mf.Rmtrx2D(do_course)
        p_diff_do_frame = Rchi_do_i.T @ (self.env.ownship.state[:2] - do_state[0:2])
        weights = np.diag(
            [
                1.0 / (0.5 * do_length + self._config.r_safe) ** 2,
                1.0 / (0.5 * do_length + self._config.r_safe) ** 2,
            ],
        )
        epsilon = 1e-6
        return np.log(1.0 + epsilon) - np.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon)

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        true_ship_states = cs_mhm.extract_do_states_from_ship_list(self.env.time, self.env.ship_list)
        do_list = cs_mhm.get_relevant_do_states(true_ship_states, idx=0)
        g_do = np.zeros(len(do_list))
        for i, do_tup in enumerate(do_list):
            d2do = np.linalg.norm(self.env.ownship.state[:2] - do_tup[1][:2])
            g_do[i] = self.compute_dynamic_obstacle_constraint(do_tup)
            g_do[i] = np.clip(g_do[i], 0.0, 1.0)
            if g_do[i] > 0.0:
                print(f"Dynamic obstacle {i} is too close to the ownship! g_do[i] = {g_do[i]} | distance = {d2do}.")

        colav_cost = self._config.rho_colav * np.sum(g_do)
        return -colav_cost


class ReadilyApparentManeuveringRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: ReadilyApparentManeuveringRewarderParams) -> None:
        super().__init__(env)
        self._config = config
        self._config.r_max = self.env.ownship.max_turn_rate
        self.K_app = np.array([self._config.K_app_course, self._config.K_app_speed])
        self.alpha_app = np.concatenate([self._config.alpha_app_course, self._config.alpha_app_speed])
        self._prev_speed = self.env.ownship.speed

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.time < 0.0001:
            self._prev_speed = self.env.ownship.speed
        turn_rate = self.env.ownship.state[5]
        speed = self.env.ownship.speed

        acceleration = (speed - self._prev_speed) / self.env.dt_action
        rate_cost, _, _ = mpc_common.rate_cost(
            r=turn_rate,
            a=acceleration,
            K_app=self.K_app,
            alpha_app=self.alpha_app,
            r_max=self._config.r_max,
            a_max=self._config.a_max,
        )
        self._prev_speed = speed
        return -rate_cost


class COLREGRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: COLREGRewarderParams) -> None:
        super().__init__(env)
        self._config = config
        self._debug: bool = False
        self._map_origin: np.ndarray = np.zeros(2)
        self._colregs_handler: ch.COLREGSHandler = ch.COLREGSHandler(config.colregs_handler)
        self._nx_do: int = 6
        self._all_polygons: list = []
        self._r_safe: float = 10.0
        self._min_depth: int = 0
        self._geometry_tree: Any = None

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.time < 0.0001:
            self._colregs_handler.reset()
            self._map_origin = self.env.ownship.csog_state[:2]
            self._min_depth = mapf.find_minimum_depth(self.env.ownship.draft, self.env.enc)
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
                self._min_depth, self.env.enc, buffer=self._r_safe, show_plots=False
            )
            self._geometry_tree, self._all_polygons = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

        do_arr = state["GroundTruthTrackingObservation"]
        do_list = cs_mhm.extract_do_list_from_do_array(do_arr)
        ownship_state = self.env.ownship.state
        translated_do_list = hf.translate_dynamic_obstacle_coordinates(
            do_list, self._map_origin[1], self._map_origin[0]
        )
        on_land_indices = []
        for i, do_tup in enumerate(do_list):
            p_do = do_tup[1][:2]
            if mapf.point_in_polygon_list(p_do, self._all_polygons):
                # print(f"Dynamic obstacle {i} is on land, i.e. not relevant")
                on_land_indices.append(i)
        translated_do_list = [
            translated_do_list[i] for i in range(len(do_list)) if do_list[i][0] not in on_land_indices
        ]

        csog_state = cs_mhm.convert_3dof_state_to_sog_cog_state(ownship_state)
        csog_state_cpy = csog_state.copy()
        csog_state[2] = csog_state_cpy[3]
        csog_state[3] = csog_state_cpy[2]
        do_cr_list, do_ho_list, do_ot_list = self._colregs_handler.handle(
            csog_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0]), translated_do_list
        )
        # if self.__debug:
        #     plotters.plot_dynamic_obstacles(
        #         do_cr_list, "blue", enc, self._mpc.params.T, self._mpc.params.dt, map_origin=self._map_origin
        #     )
        #     plotters.plot_dynamic_obstacles(
        #         do_ho_list, "orange", enc, self._mpc.params.T, self._mpc.params.dt, map_origin=self._map_origin
        #     )
        #     plotters.plot_dynamic_obstacles(
        #         do_ot_list, "magenta", enc, self._mpc.params.T, self._mpc.params.dt, map_origin=self._map_origin
        #     )

        n_do_cr = len(do_cr_list)
        n_do_ho = len(do_ho_list)
        n_do_ot = len(do_ot_list)
        max_n_do = max(n_do_cr, n_do_ho, n_do_ot)
        if max_n_do == 0:
            return 0.0

        x_do_inactive = np.array([0.0 - 1e10, 0.0 - 1e10, 0.0, 0.0, 10.0, 2.0])
        X_do_cr = np.vstack([x_do_inactive for _ in range(max_n_do)]).T.reshape(-1)
        X_do_ho = np.vstack([x_do_inactive for _ in range(max_n_do)]).T.reshape(-1)
        X_do_ot = np.vstack([x_do_inactive for _ in range(max_n_do)]).T.reshape(-1)
        for i, (_, do_state, _, do_length, do_width) in enumerate(do_cr_list):
            X_do_cr[i * self._nx_do : (i + 1) * self._nx_do] = np.array(
                [do_state[0], do_state[1], do_state[2], do_state[3], do_length, do_width]
            )
        for i, (_, do_state, _, do_length, do_width) in enumerate(do_ho_list):
            X_do_ho[i * self._nx_do : (i + 1) * self._nx_do] = np.array(
                [do_state[0], do_state[1], do_state[2], do_state[3], do_length, do_width]
            )
        for i, (_, do_state, _, do_length, do_width) in enumerate(do_ot_list):
            X_do_ot[i * self._nx_do : (i + 1) * self._nx_do] = np.array(
                [do_state[0], do_state[1], do_state[2], do_state[3], do_length, do_width]
            )

        colreg_cost, _, _, _ = mpc_common.colregs_cost(
            x=csog_state - np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0]),
            X_do_cr=X_do_cr,
            X_do_ho=X_do_ho,
            X_do_ot=X_do_ot,
            nx_do=self._nx_do,
            alpha_cr=self._config.alpha_cr,
            y_0_cr=self._config.y_0_cr,
            alpha_ho=self._config.alpha_ho,
            x_0_ho=self._config.x_0_ho,
            alpha_ot=self._config.alpha_ot,
            x_0_ot=self._config.x_0_ot,
            y_0_ot=self._config.y_0_ot,
            d_attenuation=self._config.d_attenuation,
            weights=self._config.w_colregs,
        )
        return -colreg_cost.full()[0][0]


class TrajectoryTrackingRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: TrajectoryTrackingRewarderParams) -> None:
        super().__init__(env)
        self._config = config

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        unnormalized_obs = self.env.observation_type.unnormalize(state)
        path_obs = unnormalized_obs["PathRelativeNavigationObservation"]  # [path_dev, speed_dev, u, v, r]
        huber_loss = mpc_common.huber_loss(path_obs[0] ** 2, 1.0)
        tt_cost = self._config.rho_path_dev * huber_loss + self._config.rho_speed_dev * path_obs[1] ** 2
        return -tt_cost


class MPCRewarder(cs_reward.IReward):
    """The MPC rewarder class. The sub-reward classes compute the RL stage cost, but
    return the negative of the cost to be consistent with the RL literature on reward maximization.
    """

    def __init__(self, env: "COLAVEnvironment", config: Config = Config()) -> None:
        super().__init__(env)
        self.reward_scale: float = 10.0
        self._config = config
        self.anti_grounding_rewarder = AntiGroundingRewarder(env, config.anti_grounding)
        self.collision_avoidance_rewarder = CollisionAvoidanceRewarder(env, config.collision_avoidance)
        self.colreg_rewarder = COLREGRewarder(env, config.colreg)
        self.trajectory_tracking_rewarder = TrajectoryTrackingRewarder(env, config.trajectory_tracking)
        self.readily_apparent_maneuvering_rewarder = ReadilyApparentManeuveringRewarder(
            env, config.readily_apparent_maneuvering
        )
        self.r_antigrounding: float = 0.0
        self.r_collision_avoidance: float = 0.0
        self.r_colreg: float = 0.0
        self.r_trajectory_tracking: float = 0.0
        self.r_readily_apparent_maneuvering: float = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        self.r_antigrounding = self.anti_grounding_rewarder(state, action, **kwargs)
        self.r_collision_avoidance = self.collision_avoidance_rewarder(state, action, **kwargs)
        self.r_colreg = self.colreg_rewarder(state, action, **kwargs)
        self.r_trajectory_tracking = self.trajectory_tracking_rewarder(state, action, **kwargs)
        self.r_readily_apparent_maneuvering = self.readily_apparent_maneuvering_rewarder(state, action, **kwargs)

        reward = (
            self.r_antigrounding
            + self.r_collision_avoidance
            + self.r_colreg
            + self.r_trajectory_tracking
            + self.r_readily_apparent_maneuvering
        )
        reward = reward / self.reward_scale
        # print(
        #     f"r_scaled: {reward} | r_antigrounding: {self.r_antigrounding:.2f} | r_collision_avoidance: {self.r_collision_avoidance:.2f} | r_colreg: {self.r_colreg:.2f} | r_trajectory_tracking: {self.r_trajectory_tracking:.2f} | r_readily_apparent_maneuvering: {self.r_readily_apparent_maneuvering:.2f}"
        # )
        return reward
