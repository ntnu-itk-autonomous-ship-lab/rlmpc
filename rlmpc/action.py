"""
    action.py

    Summary:
        This file contains various action type definitions for setting the parameters of an MPC CAS agent in the colav-simulator.

    Author: Trym Tengesdal
"""

from pathlib import Path
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
from stable_baselines3.common.distributions import DiagGaussianDistribution

Action = Union[list, np.ndarray]
import colav_simulator.common.math_functions as mf

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment

LOG_PROB_MIN = -15.0


class MPCParameterSettingAction(csgym_action.ActionType):
    """(Continuous) Action consisting of setting the MPC parameter values and solving its MPC to update the own-ship references."""

    def __init__(
        self,
        env: "COLAVEnvironment",
        sample_time: Optional[float] = None,
        mpc_param_list: List[str] = ["Q_p", "r_safe_do"],
        mpc_config_path: Path = rl_dp.config / "rlmpc.yaml",
        std_init: np.ndarray | float = np.array([2.0, 2.0]),
        deterministic: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(env, sample_time)
        assert self.env.ownship is not None, "Ownship must be set before using the action space"
        self.course_range = (-np.pi / 4.0, np.pi / 4.0)
        self.speed_range = (-self.env.ownship.max_speed / 4.0, self.env.ownship.max_speed / 4.0)
        self.name = "MPCParameterSettingAction"

        self.mpc = rlmpc_cas.RLMPC(config=mpc_config_path, identifier=self.env.env_id + "_mpc")
        self.build_sensitivities = True if "train" in self.env.env_id else False
        self.mpc_param_list = mpc_param_list
        self.mpc.set_adjustable_param_str_list(mpc_param_list)
        self.mpc_params = self.mpc.get_mpc_params()
        self.mpc_action_dim = 2
        self.mpc_sensitivities = None

        nx, nu = self.mpc.get_mpc_model_dims()
        n_samples = int(self.mpc_params.T / self.mpc_params.dt)
        self.action_indices = [
            int(nu * n_samples + (2 * nx) + 2),  # chi 2
            int(nu * n_samples + (2 * nx) + 3),  # speed 2
        ]
        self.mpc.set_action_indices(self.action_indices)

        if isinstance(std_init, float):
            std_init = np.array([std_init] * self.mpc_action_dim)
        log_std_init = th.log(th.from_numpy(std_init)).to(th.float32)
        self.log_std = log_std_init
        self.action_dist = DiagGaussianDistribution(self.mpc_action_dim).proba_distribution(
            th.zeros(self.mpc_action_dim), log_std=self.log_std
        )

        self.mpc_parameter_ranges, self.mpc_parameter_incr_ranges, self.mpc_parameter_lengths = (
            self.mpc_params.get_adjustable_parameter_info()
        )
        offset = 0
        self.mpc_parameter_indices = {}
        for param in mpc_param_list:
            self.mpc_parameter_indices[param] = offset
            offset += self.mpc_parameter_lengths[param]
        self.num_adjustable_mpc_params = offset
        self.mpc_param_list = mpc_param_list
        self.mpc_adjustable_params_init = self.mpc.get_adjustable_mpc_params()
        self.mpc_adjustable_params_init = hf.map_mpc_param_incr_array_to_parameter_dict(
            x=np.zeros(self.num_adjustable_mpc_params),
            current_params=self.mpc_adjustable_params_init,
            param_list=self.mpc_param_list,
            parameter_ranges=self.mpc_parameter_ranges,
            parameter_incr_ranges=self.mpc_parameter_incr_ranges,
            parameter_lengths=self.mpc_parameter_lengths,
            parameter_indices=self.mpc_parameter_indices,
        )

        self.prev_noise_action: th.Tensor = th.zeros(self.mpc_action_dim)
        self.t_prev: float = 0.0
        self.noise_application_duration: float = 10.0

        self.non_optimal_solutions: int = 0
        self.debug: bool = debug
        self.deterministic: bool = deterministic
        self.last_action: np.ndarray = np.zeros(self.mpc_action_dim)
        self.action_result: csgym_action.ActionResult = csgym_action.ActionResult(success=False, info={})

    def normalize_mpc_action(self, mpc_action: np.ndarray) -> np.ndarray:
        mpc_action_norm = np.zeros(self.mpc_action_dim)
        mpc_action_norm[0] = mf.linear_map(mpc_action[0], self.course_range, (-1.0, 1.0))
        mpc_action_norm[1] = mf.linear_map(mpc_action[1], self.speed_range, (-1.0, 1.0))
        return mpc_action_norm

    def unnormalize_mpc_action(self, mpc_action: np.ndarray) -> np.ndarray:
        mpc_action_unnorm = np.zeros(self.mpc_action_dim)
        mpc_action_unnorm[0] = mf.linear_map(mpc_action[0], (-1.0, 1.0), self.course_range)
        mpc_action_unnorm[1] = mf.linear_map(mpc_action[1], (-1.0, 1.0), self.speed_range)
        return mpc_action_unnorm

    def compute_mpc_sensitivities(self, info: Dict[str, Any]) -> np.ndarray:
        """Compute the MPC sensitivities for the given solution info.

        Args:
            info (Dict[str, Any]): The mpc solution info

        Returns:
            np.ndarray: The MPC sensitivities
        """
        assert self.build_sensitivities, "Sensitivities must be built before computing them"
        da_dp_mpc = np.zeros((self.mpc_action_dim, self.num_adjustable_mpc_params))
        if info["optimal"]:
            soln = info["soln"]
            p = info["p"]
            p_fixed = info["p_fixed"]
            z = np.concatenate((soln["x"], soln["lam_g"]), axis=0).astype(np.float32)
            da_dp_mpc = self.mpc_sensitivities.da_dp(z, p_fixed, p).full()
        return da_dp_mpc

    def initialize(
        self,
        build_sensitivities: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the planner by setting up the nominal path, static obstacle inputs and constructing
        the OCP

        Args:
            build_sensitivities (bool, optional): Whether to build the sensitivities.
        """
        self.t_prev = 0.0
        self.action_result = csgym_action.ActionResult(success=True, info={})
        self.prev_noise_action = th.zeros(self.mpc_action_dim)
        t = self.env.time
        waypoints = self.env.ownship.waypoints
        speed_plan = self.env.ownship.speed_plan
        ownship_state = self.env.ownship.state
        enc = self.env.enc
        do_list, _ = self.env.ownship.get_do_track_information()

        self.mpc.set_action_indices(self.action_indices)
        self.mpc.set_mpc_param_subset(self.mpc_adjustable_params_init)
        self.mpc.initialize(
            t=t,
            waypoints=waypoints,
            speed_plan=speed_plan,
            ownship_state=ownship_state,
            do_list=do_list,
            enc=enc,
            debug=self.debug,
            **kwargs,
        )
        self.build_sensitivities = build_sensitivities
        if build_sensitivities:
            self.mpc_sensitivities = self.mpc.build_sensitivities()
        print(f"[{self.env.env_id.upper()}] MPC initialized! Built sensitivities? {build_sensitivities}")

    def extract_mpc_observation_features(self) -> Tuple[float, np.ndarray, List, stochasticity.DisturbanceData]:
        """Extract features from the observation at a given index in the batch.

        Args:
            observation (Union[np.ndarray, Dict[str, np.ndarray], rlmpc_buffers.TensorDict]): The input observation

        Returns:
            Tuple[float, np.ndarray, List, stoch.DisturbanceData]: Time, ownship state, DO list and disturbance data
        """
        t = self.env.time
        do_list, _ = self.env.ownship.get_do_track_information()

        ownship_state = self.env.ownship.state
        ownship_course = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])
        disturbance_vector = np.array([0.0, 0.0, 0.0, 0.0])
        w = stochasticity.DisturbanceData()
        w.currents = {"speed": disturbance_vector[0], "direction": (disturbance_vector[1] + ownship_course)}
        w.wind = {"speed": disturbance_vector[2], "direction": disturbance_vector[3] + ownship_course}
        return t, ownship_state, do_list, w

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, shape=(self.num_adjustable_mpc_params,), dtype=np.float32)

    def normalize(self, action: Action) -> Action:
        action_norm = hf.normalize_mpc_param_increment_tensor(
            th.from_numpy(action).detach().clone(),
            self.mpc_param_list,
            self.mpc_parameter_incr_ranges,
            self.mpc_parameter_lengths,
            self.mpc_parameter_indices,
        )
        return action_norm

    def unnormalize(self, action: Action) -> Action:
        action_unnorm = hf.unnormalize_mpc_param_increment_tensor(
            th.from_numpy(action).detach().clone(),
            self.mpc_param_list,
            self.mpc_parameter_incr_ranges,
            self.mpc_parameter_lengths,
            self.mpc_parameter_indices,
        )
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
        norm_collision_seeking_action = self.normalize_mpc_action(action)
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
                self.prev_noise_action = self.sample_mpc_action(norm_action)
            self.t_prev = t

        # Check if probability of the previous noise action is too low
        # given the current mean mpc action
        self.action_dist = self.action_dist.proba_distribution(mean_actions=norm_action, log_std=self.log_std)
        log_prob_noise_action = self.action_dist.log_prob(self.prev_noise_action)
        if log_prob_noise_action < LOG_PROB_MIN and not sampled_collision_seeking_action:
            self.prev_noise_action = self.sample_mpc_action(norm_action)
        expln_action = self.prev_noise_action
        norm_action = norm_action.numpy()
        return expln_action.numpy()

    def sample_mpc_action(self, mpc_actions: np.ndarray | th.Tensor) -> np.ndarray:
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

    def act(self, action: Action, **kwargs) -> csgym_action.ActionResult:
        """Execute the action on the own-ship, which is to set new MPC parameters and solve the MPC problem to set new autopilot references for the ship course and speed.

        Args:
            action (Action): New MPC parameter increments within [-1, 1].
        """
        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        if self.env.time < 0.0001:
            self.initialize(build_sensitivities=self.build_sensitivities, **kwargs)

        if kwargs["applied"]:
            return self.action_result

        current_params = self.mpc.get_adjustable_mpc_params()
        param_dict = hf.map_mpc_param_incr_array_to_parameter_dict(
            x=action,
            current_params=current_params,
            param_list=self.mpc_param_list,
            parameter_ranges=self.mpc_parameter_ranges,
            parameter_incr_ranges=self.mpc_parameter_incr_ranges,
            parameter_lengths=self.mpc_parameter_lengths,
            parameter_indices=self.mpc_parameter_indices,
        )
        self.mpc.set_mpc_param_subset(param_subset=param_dict)
        # print(f"[{self.env.env_id.upper()}] Setting MPC parameters: {self.mpc.get_adjustable_mpc_params()}")

        t, ownship_state, do_list, w = self.extract_mpc_observation_features()
        mpc_action, mpc_info = self.mpc.act(t, ownship_state, do_list, w)
        self.last_action = mpc_action

        self.env.ownship.set_colav_data(mpc_info)
        self.env.ownship.set_remote_actor_predicted_trajectory(mpc_info["trajectory"])
        success = not mpc_info["qp_failure"]

        if not mpc_info["optimal"]:
            self.non_optimal_solutions += 1

        norm_mpc_action = self.normalize_mpc_action(mpc_action)
        expl_action = norm_mpc_action
        if not self.deterministic:
            expl_action = self.get_exploratory_action(norm_mpc_action, t, ownship_state, do_list)

        mpc_info.update(
            {
                "da_dp_mpc": self.compute_mpc_sensitivities(mpc_info) if self.build_sensitivities else None,
                "non_optimal_solutions": self.non_optimal_solutions,
                "norm_mpc_action": norm_mpc_action,
                "unnorm_mpc_action": mpc_action,
                "expl_action": expl_action,
                "new_mpc_params": self.mpc.get_adjustable_mpc_params(),
            }
        )

        course = self.env.ownship.course
        speed = self.env.ownship.speed
        course_ref = mf.wrap_angle_to_pmpi(mpc_action[0] + course)
        speed_ref = np.clip(mpc_action[1] + speed, self.env.ownship.min_speed + 0.5, self.env.ownship.max_speed)
        refs = np.array([0.0, 0.0, course_ref, speed_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.env.ownship.set_references(refs)

        self.action_result = csgym_action.ActionResult(success=success, info=mpc_info)
        return self.action_result
