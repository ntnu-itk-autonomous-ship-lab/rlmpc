"""
    action.py

    Summary:
        This file contains various action type definitions for setting the parameters of an MPC CAS agent in the colav-simulator.

    Author: Trym Tengesdal
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import colav_simulator.core.stochasticity as stochasticity
import colav_simulator.gym.action as csgym_action
import gymnasium as gym
import numpy as np
import rlmpc.common.buffers as rlmpc_buffers
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as rl_dp
import rlmpc.rlmpc_cas as rlmpc_cas
import torch as th

Action = Union[list, np.ndarray]
import colav_simulator.common.math_functions as mf

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


class MPCParameterSettingAction(csgym_action.ActionType):
    """(Continuous) Action consisting of setting the MPC parameter values and solving its MPC to update the own-ship references."""

    def __init__(
        self,
        env: "COLAVEnvironment",
        sample_time: Optional[float] = None,
        mpc_param_list: List[str] = ["Q_p", "r_safe_do"],
        **kwargs,
    ) -> None:
        """Create a continuous action space for setting the own-ship autopilot references in speed and course."""
        super().__init__(env, sample_time)
        assert self.env.ownship is not None, "Ownship must be set before using the action space"
        self.course_range = (-np.pi / 4.0, np.pi / 4.0)
        self.speed_range = (-self.env.ownship.max_speed / 4.0, self.env.ownship.max_speed / 4.0)
        self.name = "MPCParameterSettingAction"
        self.mpc = rlmpc_cas.RLMPC(config=rl_dp.config / "rlmpc.yaml")
        self.mpc_sensitivities = None
        self.infeasible_solutions: int = 0
        self.mpc.set_adjustable_param_str_list(mpc_param_list)
        self.mpc_params = self.mpc.get_mpc_params()
        self.mpc_adjustable_params_init = self.mpc.get_adjustable_mpc_params()
        self.mpc_action_dim = 2

        nx, nu = self.mpc.get_mpc_model_dims()
        n_samples = int(self.mpc_params.T / self.mpc_params.dt)
        self.action_indices = [
            int(nu * n_samples + (2 * nx) + 2),  # chi 2
            int(nu * n_samples + (2 * nx) + 3),  # speed 2
        ]
        self.mpc.set_action_indices(self.action_indices)
        self.course_ref: float = 0.0
        self.speed_ref: float = 0.0
        self.debug: bool = kwargs.get("debug", False)

    def compute_mpc_sensitivities(self, info: Dict[str, Any]) -> None:
        """Compute the MPC sensitivities for the given solution info.

        Args:
            info (Dict[str, Any]): The solution info
        """
        if self.mpc_sensitivities is None:
            self.mpc_sensitivities = self.mpc.build_sensitivities()

        da_dp_mpc = np.zeros((self.mpc_action_dim, self.mpc_adjustable_params_init.size))
        if info["optimal"]:
            soln = info["soln"]
            p = info["p"]
            p_fixed = info["p_fixed"]
            z = np.concatenate((soln["x"], soln["lam_g"]), axis=0).astype(np.float32)
            da_dp_mpc = self.mpc_sensitivities.da_dp(z, p_fixed, p).full()
        return da_dp_mpc

    def initialize(
        self,
        env: "COLAVEnvironment",
        build_sensitivities: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the planner by setting up the nominal path, static obstacle inputs and constructing
        the OCP

        Args:
            env (csenv.COLAVEnvironment): The environment
            build_sensitivities (bool, optional): Whether to build the sensitivities.

        """
        t = env.unwrapped.time
        waypoints = env.unwrapped.ownship.waypoints
        speed_plan = env.unwrapped.ownship.speed_plan
        ownship_state = env.unwrapped.ownship.state
        enc = env.unwrapped.enc
        do_list = env.unwrapped.ownship.get_do_track_information()
        goal_state = env.unwrapped.ownship.goal_state
        w = env.unwrapped.disturbance.get() if env.unwrapped.disturbance is not None else None

        self.mpc.initialize(
            t=t,
            waypoints=waypoints,
            speed_plan=speed_plan,
            ownship_state=ownship_state,
            do_list=do_list,
            enc=enc,
            goal_state=goal_state,
            w=w,
            debug=self.debug,
            **kwargs,
        )
        self.mpc.set_action_indices(self.action_indices)
        self.mpc.set_mpc_param_subset(self.mpc_adjustable_params_init)
        if build_sensitivities:
            self.mpc_sensitivities = self.mpc.build_sensitivities()
        print(f"MPC initialized! Built sensitivities? {build_sensitivities}")

    def extract_mpc_observation_features(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray], rlmpc_buffers.TensorDict]
    ) -> Tuple[float, np.ndarray, List, stochasticity.DisturbanceData]:
        """Extract features from the observation at a given index in the batch.

        Args:
            observation (Union[np.ndarray, Dict[str, np.ndarray], rlmpc_buffers.TensorDict]): The input observation

        Returns:
            Tuple[float, np.ndarray, List, stoch.DisturbanceData]: Time, ownship state, DO list and disturbance data
        """
        t = observation["TimeObservation"].flatten()[0]
        do_list = hf.extract_do_list_from_tracking_observation(observation["TrackingObservation"])

        ownship_state = self.env.ownship.state
        ownship_course = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])
        disturbance_vector = np.array([0.0, 0.0, 0.0, 0.0])
        w = stochasticity.DisturbanceData()
        w.currents = {"speed": disturbance_vector[0], "direction": (disturbance_vector[1] + ownship_course)}
        w.wind = {"speed": disturbance_vector[2], "direction": disturbance_vector[3] + ownship_course}
        return t, ownship_state, do_list, w

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, shape=(self.mpc_adjustable_params_init.size,), dtype=np.float32)

    def normalize(self, action: list | np.ndarray) -> list | np.ndarray:
        action_norm = np.zeros(self.mpc_action_dim)
        action_norm[0] = mf.linear_map(action[0], self.course_range, (-1.0, 1.0))
        action_norm[1] = mf.linear_map(action[1], self.speed_range, (-1.0, 1.0))
        return action_norm

    def unnormalize(self, action: list | np.ndarray) -> list | np.ndarray:
        action_unnorm = np.zeros(self.mpc_action_dim)
        action_unnorm[0] = mf.linear_map(action[0], (-1.0, 1.0), self.course_range)
        action_unnorm[1] = mf.linear_map(action[1], (-1.0, 1.0), self.speed_range)
        return action_unnorm

    def sample_collision_seeking_action(
        self,
        ownship_state: np.ndarray,
        closest_do: Tuple[int, np.ndarray, np.ndarray, float, float],
        norm_mpc_action: np.ndarray,
    ) -> th.Tensor:
        """Sample a collision-seeking action from the policy.

        Args:
            ownship_state (np.ndarray): The ownship state
            closest_do (Tuple[int, np.ndarray, np.ndarray, float, float]): The closest dynamic obstacle info
            norm_mpc_action (np.ndarray): The normalized MPC action

        Returns:
            np.ndarray: The collision-seeking action, normalized
        """
        do_state = closest_do[1]
        os_course = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])
        bearing_to_do = np.arctan2(do_state[1] - ownship_state[1], do_state[0] - ownship_state[0]) - os_course

        action = np.array([bearing_to_do, 0.0])
        norm_collision_seeking_action = self.normalize(action)
        norm_collision_seeking_action[1] = norm_mpc_action[1]
        return th.from_numpy(norm_collision_seeking_action)

    def get_exploratory_action(
        self,
        norm_action: np.ndarray,
        t: float,
        ownship_state: np.ndarray,
        do_list: List,
    ) -> np.ndarray:
        """Get the exploratory action from the policy.

        Args:
            norm_action (np.ndarray): The normalized action
            t (float): The current time
            ownship_state (np.ndarray): The ownship state
            do_list (List): The DO list
        """
        distances2do = hf.compute_distances_to_dynamic_obstacles(ownship_state, do_list)

        norm_action = th.from_numpy(norm_action)
        sampled_collision_seeking_action = False
        if t < 0.0001 or t - self.t_prev > self.noise_application_duration:
            if distances2do[0][1] < 400.0:
                self.prev_noise_action = self.sample_collision_seeking_action(
                    ownship_state, do_list[distances2do[0][0]], norm_action.numpy()
                )
                sampled_collision_seeking_action = True
            else:
                self.prev_noise_action = self.sample_action(norm_action)
            self.t_prev = t

        # Check if probability of the previous noise action is too low
        # given the current mean mpc action
        self.action_dist = self.action_dist.proba_distribution(mean_actions=norm_action, log_std=self.log_std)
        log_prob_noise_action = self.action_dist.log_prob(self.prev_noise_action)
        if log_prob_noise_action < LOG_PROB_MIN and not sampled_collision_seeking_action:
            self.prev_noise_action = self.sample_action(norm_action)
        expln_action = self.prev_noise_action
        norm_action = norm_action.numpy()
        return expln_action.numpy()

    def sample_action(self, mpc_actions: np.ndarray | th.Tensor) -> np.ndarray:
        """Sample an action from the policy distribution with mean from the input MPC action


        Args:
            mpc_actions (np.ndarray | th.Tensor): The input MPC action (normalized)

        Returns:
            np.ndarray: The sampled action (normalized)
        """
        if isinstance(mpc_actions, np.ndarray):
            mpc_actions = th.from_numpy(mpc_actions)
        self.action_dist = self.action_dist.proba_distribution(mean_actions=mpc_actions, log_std=self.log_std)
        norm_actions = self.action_dist.get_actions()
        norm_actions = th.clamp(norm_actions, -1.0, 1.0)
        return norm_actions

    def act(self, action: Action, **kwargs) -> None:
        """Execute the action on the own-ship, which is to set new MPC parameters and solve the MPC problem to set new autopilot references for the ship course and speed.

        Args:
            action (Action): New MPC parameter increments within [-1, 1].
        """
        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        unnorm_action = self.unnormalize(action)  # u
        # ship references in general is a 9-entry array consisting of 3DOF pose, velocity and acceleartion
        course = self.env.ownship.course
        speed = self.env.ownship.speed
        self.course_ref = mf.wrap_angle_to_pmpi(unnorm_action[0] + course)
        self.speed_ref = unnorm_action[1] + speed

        t, ownship_state, do_list, w = self.extract_mpc_observation_features(kwargs["observation"])

        mpc_action = self.mpc.act(t, ownship_state, do_list, w)
        self.course_ref = mpc_action[0] + course
        self.speed_ref = mpc_action[1] + speed

        refs = np.array([0.0, 0.0, self.course_ref, self.speed_ref, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.env.ownship.set_references(refs)
