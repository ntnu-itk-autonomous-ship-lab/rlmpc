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
import colav_simulator.core.guidances as guidances
import colav_simulator.gym.action as csgym_action
import colav_simulator.gym.observation as csgym_obs
import colav_simulator.gym.reward as cs_reward
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.action as mpc_action
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
    alpha_cr: np.ndarray = field(
        default_factory=lambda: np.array([1.0 / 500.0, 1.0 / 500.0])
    )  # Crossing potential function parameters
    y_0_cr: float = 100.0  # Crossing potential function parameters
    alpha_ho: np.ndarray = field(
        default_factory=lambda: np.array([1.0 / 500.0, 1.0 / 500.0])
    )  # Head-on potential function parameters
    x_0_ho: float = 200.0  # Head-on potential function parameters
    alpha_ot: np.ndarray = field(
        default_factory=lambda: np.array([1.0 / 500.0, 1.0 / 500.0])
    )  # Overtaking potential function parameters
    x_0_ot: float = 200.0  # Overtaking potential function parameters
    y_0_ot: float = 100.0  # Overtaking potential function parameters
    d_attenuation: float = 400.0  # attenuation distance for the COLREGS potential functions
    w_colregs: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0])
    )  # Weights for the COLREGS potential functions

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
    rho_d2path: float = 1.0  # path deviation reward weight
    rho_speed_dev: float = 10.0  # speed deviation reward weight
    rho_d2goal: float = 0.1  # final path deviation reward weight
    rho_course_dev: float = 0.0  # course deviation reward weight
    rho_turn_rate: float = 0.0  # turn rate reward weight
    rho_goal: float = 100.0  # penalty for not reaching the goal
    goal_radius: float = 30.0  # radius around the goal point where the goal is considered reached

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = cls(**config_dict)
        return params

    def to_dict(self) -> dict:
        return asdict(self)


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
class DNNParameterRewarderParams:
    rho_solver_time: float = 0.0
    rho_non_optimal_solution: float = 0.0
    rho_non_relevant_safety_param_change: float = 0.0
    disable: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = cls(**config_dict)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionChatterRewarderParams:
    rho_chatter: np.ndarray = field(default_factory=lambda: np.diag([0.0] * 9))

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = cls(**config_dict)
        cfg.rho_chatter = np.diag(cfg.rho_chatter)
        return cfg

    def to_dict(self) -> dict:
        out = asdict(self)
        out["rho_chatter"] = self.rho_chatter.diagonal().tolist()
        return out


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
    dnn_parameter_provider: DNNParameterRewarderParams = field(default_factory=lambda: DNNParameterRewarderParams())
    action_chatter: ActionChatterRewarderParams = field(default_factory=lambda: ActionChatterRewarderParams())

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = Config()
        cfg.trajectory_tracking = TrajectoryTrackingRewarderParams.from_dict(config_dict["trajectory_tracking"])
        cfg.anti_grounding = AntiGroundingRewarderParams.from_dict(config_dict["anti_grounding"])
        cfg.collision_avoidance = CollisionAvoidanceRewarderParams.from_dict(config_dict["collision_avoidance"])
        cfg.colreg = COLREGRewarderParams.from_dict(config_dict["colreg"])
        cfg.readily_apparent_maneuvering = ReadilyApparentManeuveringRewarderParams.from_dict(
            config_dict["readily_apparent_maneuvering"]
        )
        cfg.dnn_parameter_provider = DNNParameterRewarderParams.from_dict(config_dict["dnn_parameter_provider"])
        cfg.action_chatter = ActionChatterRewarderParams.from_dict(config_dict["action_chatter"])
        return cfg

    @classmethod
    def from_file(cls, config_file: Path) -> "Config":
        with open(config_file, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return cls.from_dict(config_dict)


class AntiGroundingRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: AntiGroundingRewarderParams) -> None:
        super().__init__(env)
        self._config: AntiGroundingRewarderParams = config
        self._ktp: guidances.KinematicTrajectoryPlanner = guidances.KinematicTrajectoryPlanner()
        self._polygons: list = []
        self._rel_polygons: list = []
        self._min_depth: int = 0
        self._show_plots: bool = False
        self._so_surfaces: list = []
        self._map_origin: np.ndarray = np.zeros(2)
        self._initialized: bool = False
        self._nominal_path = None
        self.last_reward: float = 0.0
        self._map_origin = np.zeros(2)

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
                print(
                    f"[{self.env.env_id.upper()}] Static obstacle {j} is too close to the ownship! g_so[i]={g_so[j]} | d2so={d2so}."
                )
        grounding_cost = self._config.rho_anti_grounding * g_so.sum()

        # Add extra cost if the ship is grounded in the simulator (i.e. the ship is on land)
        # The above cost only gives a penalty if the ship CG is inside the grounding polygon approximations.
        if self.env.simulator.determine_ship_grounding():
            grounding_cost = self._config.rho_anti_grounding
        self.last_reward = -grounding_cost
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_anti_grounding": self.last_reward}


class CollisionAvoidanceRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: CollisionAvoidanceRewarderParams) -> None:
        super().__init__(env)
        self._config = config
        self.last_reward = 0.0

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
                print(
                    f"[{self.env.env_id.upper()}] Dynamic obstacle {i} is too close to the ownship! g_do[i] = {g_do[i]} | distance = {d2do}."
                )

        colav_cost = self._config.rho_colav * g_do.sum()

        # Add extra cost if the ship collides with a dynamic obstacle
        # The above cost only gives a penalty if a dynamic obstacle is inside the set own-ship safety zone
        if self.env.simulator.determine_ship_collision(ship_idx=0):
            colav_cost = self._config.rho_colav

        self.last_reward = -colav_cost
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_colav": self.last_reward}


class ReadilyApparentManeuveringRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: ReadilyApparentManeuveringRewarderParams) -> None:
        super().__init__(env)
        self._config = config
        self.last_reward = 0.0
        self._config.r_max = self.env.ownship.max_turn_rate
        self.K_app = np.array([self._config.K_app_course, self._config.K_app_speed])
        self.alpha_app = np.concatenate([self._config.alpha_app_course, self._config.alpha_app_speed])
        self._prev_speed = self.env.ownship.speed
        self._distance_threshold = 500.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.time < 0.0001:
            self._prev_speed = self.env.ownship.speed
        turn_rate = self.env.ownship.state[5]
        speed = self.env.ownship.speed

        true_ship_states = cs_mhm.extract_do_states_from_ship_list(self.env.time, self.env.ship_list)
        do_list = cs_mhm.get_relevant_do_states(true_ship_states, idx=0, add_empty_cov=True)
        distances_to_obstacles = hf.compute_distances_to_dynamic_obstacles(
            ownship_state=self.env.ownship.state, do_list=do_list
        )
        if distances_to_obstacles[0][1] > self._distance_threshold:
            self.last_reward = 0.0
            return self.last_reward

        acceleration = (speed - self._prev_speed) / self.env.dt_action
        acceleration = np.clip(acceleration, -self._config.a_max, self._config.a_max)
        rate_cost, _, _ = mpc_common.rate_cost(
            r=turn_rate,
            a=acceleration,
            K_app=self.K_app,
            alpha_app=self.alpha_app,
            r_max=self._config.r_max,
            a_max=self._config.a_max,
        )
        self._prev_speed = speed
        self.last_reward = -rate_cost
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_ra_maneuvering": self.last_reward}


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
        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.time < 0.0001:
            self._colregs_handler.reset()
            self._map_origin = self.env.ownship.csog_state[:2]
            self._min_depth = mapf.find_minimum_depth(self.env.ownship.draft, self.env.enc)
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
                self._min_depth, self.env.enc, buffer=self._r_safe, show_plots=False
            )
            self._geometry_tree, self._all_polygons = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

        do_list, _ = self.env.ownship.get_do_track_information()
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

        csog_state_rel = cs_mhm.convert_3dof_state_to_sog_cog_state(ownship_state) - np.array(
            [self._map_origin[0], self._map_origin[1], 0.0, 0.0]
        )
        csog_state_rel_cpy = csog_state_rel.copy()
        csog_state_rel[2] = csog_state_rel_cpy[3]
        csog_state_rel[3] = csog_state_rel_cpy[2]
        do_cr_list, do_ho_list, do_ot_list = self._colregs_handler.handle(csog_state_rel, translated_do_list)

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
            x=csog_state_rel,
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
        self.last_reward = -colreg_cost.full()[0][0]
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_colregs": self.last_reward}


class TrajectoryTrackingRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: TrajectoryTrackingRewarderParams) -> None:
        super().__init__(env)
        self.last_reward = 0.0
        self._config = config
        self._last_course_error = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.time < 0.0001:
            self._last_course_error = 0.0
        truncated = kwargs.get("truncated", False)
        goal_reached = self.env.simulator.determine_ship_goal_reached()
        # Positive reward if the goal is reached
        if goal_reached:
            self.last_reward = self._config.rho_goal
            print(f"[{self.env.env_id.upper()}] Goal reached! Rewarding +rho_goal.")
            return self.last_reward

        d2goal = np.linalg.norm(self.env.ownship.state[:2] - self.env.ownship.waypoints[:, -1])

        ownship_state = self.env.ownship.state
        do_list, _ = self.env.ownship.get_do_track_information()

        d2dos = np.array([1e12])
        if len(do_list) > 0:
            d2dos = hf.compute_distances_to_dynamic_obstacles(ownship_state, do_list)
        no_dos_in_the_way = d2dos[0][1] > 100.0
        if truncated and not goal_reached and no_dos_in_the_way:
            self.last_reward = -self._config.rho_goal  # * d2goal
            print(f"[{self.env.env_id.upper()}] Goal not reached! Rewarding -rho_goal.")
            return self.last_reward

        unnormalized_obs = self.env.observation_type.unnormalize(state)
        path_obs = unnormalized_obs["PathRelativeNavigationObservation"]
        huber_loss_d2path = mpc_common.huber_loss(path_obs[0] ** 2, 1.0)
        huber_loss_d2goal = mpc_common.huber_loss(path_obs[1] ** 2, 1.0)
        unwrapped_course_error = mf.unwrap_angle(self._last_course_error, path_obs[2])
        self._last_course_error = path_obs[2]
        tt_cost = (
            self._config.rho_d2path * huber_loss_d2path
            + self._config.rho_d2goal * huber_loss_d2goal
            + self._config.rho_course_dev * unwrapped_course_error**2
            + self._config.rho_speed_dev * path_obs[3] ** 2
            + self._config.rho_turn_rate * path_obs[4] ** 2
        )
        self.last_reward = -tt_cost
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_trajectory_tracking": self.last_reward}


class DNNParameterRewarder(cs_reward.IReward):

    def __init__(self, env: "COLAVEnvironment", config: DNNParameterRewarderParams) -> None:
        super().__init__(env)
        self.last_reward = 0.0
        self._config = config

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:

        if self._config.disable:
            return 0.0

        # t = self.env.time
        # tspan = self.env.simulator.t_end - self.env.simulator.t_start
        colav_info = self.env.ownship.get_colav_data()
        if colav_info is None:
            return 0.0

        r_param_dnn = 0.0

        mpc_sampling_time = 1.0 / colav_info["mpc_rate"]
        mpc_solve_time = colav_info["t_solve"]
        if mpc_solve_time > mpc_sampling_time:
            r_param_dnn += -self._config.rho_solver_time * (mpc_solve_time - mpc_sampling_time)

        if not colav_info["optimal"]:
            r_param_dnn += -self._config.rho_non_optimal_solution

        ownship_state = self.env.ownship.state
        do_list, _ = self.env.ownship.get_do_track_information()

        if len(do_list) > 0:
            d2dos = hf.compute_distances_to_dynamic_obstacles(ownship_state, do_list)
            if isinstance(self.env.action_type, mpc_action.MPCParameterSettingAction):
                unnorm_param_increments = self.env.action_type.unnormalize(action)
                r_safe_incr = unnorm_param_increments[-1]
            else:
                r_safe_incr = colav_info["new_mpc_params"][-1] - colav_info["old_mpc_params"][-1]
            if d2dos[0][1] > 400.0 and abs(r_safe_incr) > 0.0:
                r_param_dnn += -self._config.rho_non_relevant_safety_param_change * abs(r_safe_incr)

        self.last_reward = r_param_dnn
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_dnn": self.last_reward}


class ActionChatterRewarder(cs_reward.IReward):
    """Used to penalize chattering actions for a relative course+speed action."""

    def __init__(self, env: "COLAVEnvironment", config: ActionChatterRewarderParams) -> None:
        super().__init__(env)
        self._config = config
        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if action is None:
            return 0.0
        unnorm_action = self.env.action_type.unnormalize(action)
        chatter_cost = unnorm_action.T @ self._config.rho_chatter @ unnorm_action

        # mpc_param_obs = state["MPCParameterObservation"]

        self.last_reward = -chatter_cost
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_action_chatter": self.last_reward}


class MPCRewarder(cs_reward.IReward):
    """The MPC rewarder class. The sub-reward classes compute the RL stage cost, but
    return the negative of the cost to be consistent with the RL literature on reward maximization.
    """

    def __init__(self, env: "COLAVEnvironment", config: Config = Config()) -> None:
        super().__init__(env)
        self.reward_scale: float = 10.0
        self.last_reward: float = 0.0
        self._config = config
        self.anti_grounding_rewarder = AntiGroundingRewarder(env, config.anti_grounding)
        self.collision_avoidance_rewarder = CollisionAvoidanceRewarder(env, config.collision_avoidance)
        self.colreg_rewarder = COLREGRewarder(env, config.colreg)
        self.trajectory_tracking_rewarder = TrajectoryTrackingRewarder(env, config.trajectory_tracking)
        self.readily_apparent_maneuvering_rewarder = ReadilyApparentManeuveringRewarder(
            env, config.readily_apparent_maneuvering
        )
        self.action_chatter_rewarder = ActionChatterRewarder(env, config.action_chatter)
        self.dnn_parameter_provider_rewarder = DNNParameterRewarder(env, config.dnn_parameter_provider)
        self.r_antigrounding: float = 0.0
        self.r_collision_avoidance: float = 0.0
        self.r_colreg: float = 0.0
        self.r_trajectory_tracking: float = 0.0
        self.r_readily_apparent_maneuvering: float = 0.0
        self.r_action_chatter: float = 0.0
        self.r_dnn_parameters: float = 0.0
        self.verbose: bool = False

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        self.r_antigrounding = self.anti_grounding_rewarder(state, action, **kwargs)
        self.r_collision_avoidance = self.collision_avoidance_rewarder(state, action, **kwargs)
        self.r_colreg = self.colreg_rewarder(state, action, **kwargs)
        self.r_trajectory_tracking = self.trajectory_tracking_rewarder(state, action, **kwargs)
        self.r_readily_apparent_maneuvering = self.readily_apparent_maneuvering_rewarder(state, action, **kwargs)
        self.r_action_chatter = self.action_chatter_rewarder(state, action, **kwargs)
        self.r_dnn_parameters = self.dnn_parameter_provider_rewarder(state, action, **kwargs)
        reward = (
            self.r_antigrounding
            + self.r_collision_avoidance
            + self.r_colreg
            + self.r_trajectory_tracking
            + self.r_readily_apparent_maneuvering
            + self.r_action_chatter
            + self.r_dnn_parameters
        )
        reward = reward / self.reward_scale
        if self.verbose:
            print(
                f"[MPC-REWARDER | {self.env.env_id.upper()}]:\n\t- r_scaled: {reward:.4f} \n\t- r_antigrounding: {self.r_antigrounding:.4f} \n\t- r_collision_avoidance: {self.r_collision_avoidance:.4f} \n\t- r_colreg: {self.r_colreg:.4f} \n\t- r_trajectory_tracking: {self.r_trajectory_tracking:.4f} \n\t- r_readily_apparent_maneuvering: {self.r_readily_apparent_maneuvering:.4f} \n\t- r_action_chatter: {self.r_action_chatter:.4f} \n\t- r_dnn_parameters: {self.r_dnn_parameters:.4f}"
            )
        self.last_reward = reward
        return reward

    def get_last_rewards_as_dict(self) -> dict:
        return {
            "r_scaled": self.last_reward,
            "r_antigrounding": self.r_antigrounding,
            "r_collision_avoidance": self.r_collision_avoidance,
            "r_colreg": self.r_colreg,
            "r_trajectory_tracking": self.r_trajectory_tracking,
            "r_readily_apparent_maneuvering": self.r_readily_apparent_maneuvering,
            "r_action_chatter": self.r_action_chatter,
            "r_dnn_parameters": self.r_dnn_parameters,
        }
