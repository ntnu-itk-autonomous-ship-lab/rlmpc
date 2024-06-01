"""
    policies.py

    Summary:
        Soft Actor-Critic (SAC) policies (and actors) implementation for the mid-level MPC.


    Author: Trym Tengesdal
"""

import pathlib
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import colav_simulator.common.math_functions as csmf
import colav_simulator.core.stochasticity as stochasticity
import numpy as np
import rlmpc.buffers as rlmpc_buffers
import rlmpc.common.paths as dp
import rlmpc.mpc.common as mpc_common
import rlmpc.networks.feature_extractors as rlmpc_fe
import rlmpc.rlmpc as rlmpc
import stable_baselines3.common.noise as sb3_noise
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import BasePolicy, ContinuousCritic
from torch.nn import functional as F

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MPCParameterDNN(th.nn.Module):
    """The DNN for predicting the MPC parameter increments, based on the
    current situation, parameters and action. The DNN outputs are normalized to [-1, 1] and then mapped to the
    actual parameter ranges.
    """

    def __init__(
        self,
        param_list: List[str],
        hidden_sizes: List[int] = [128, 64],
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        features_dim: int = 124,
        action_dim: int = 2,
    ):
        super().__init__()
        self.out_parameter_ranges = {
            "Q_p": [0.001, 200.0],
            "K_app_course": [0.1, 200.0],
            "K_app_speed": [0.1, 200.0],
            "d_attenuation": [10.0, 1000.0],
            "w_colregs": [0.1, 500.0],
            "r_safe_do": [2.5, 100.0],
        }
        self.out_parameter_lengths = {
            "Q_p": 3,
            "K_app_course": 1,
            "K_app_speed": 1,
            "d_attenuation": 1,
            "w_colregs": 3,
            "r_safe_do": 1,
        }
        self.parameter_weights = 10.0 * th.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
        self.action_weight = 100.0 * th.diag([1.0, 1.0])
        self.mpc_cost_val_scaling = 0.0001
        self.human_preference_cost_val_scaling = 0.0001

        offset = 0
        self.parameter_indices = {}
        for param in param_list:
            self.parameter_indices[param] = offset
            offset += self.out_parameter_lengths[param]

        self.num_params = offset
        self.param_list = param_list
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn()
        self.layers = th.nn.ModuleList()

        self.input_dim = (
            features_dim + offset + action_dim
        )  # Add the number of parameters to the input features, and the action dim
        prev_size = self.input_dim
        for size in hidden_sizes:
            self.layers.append(th.nn.Linear(prev_size, size))
            prev_size = size
        self.layers.append(th.nn.Linear(prev_size, self.num_params))
        self.tanh = th.nn.Tanh()

    def forward(self, features: th.Tensor, current_params: th.Tensor, prev_action: th.Tensor) -> th.Tensor:
        input_tensor = th.cat([features, current_params, prev_action], dim=-1)
        x = input_tensor
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        # x = self.tanh(x)
        return x

    def map_to_parameter_dict(self, x: np.ndarray, current_params: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Maps the DNN output tensor to a dictionary of unnormalized parameters, given the current parameters.

        Args:
            x (np.ndarray): The DNN output, consisting of normalized parameter increments.
            current_params (np.ndarray): The current parameters

        Returns:
            Dict[str, Union[float, np.ndarray]]: The dictionary of unnormalized parameters
        """
        params = {}
        x_np = x.copy()
        current_params_np = current_params.copy()
        for param in self.param_list:
            param_range = self.out_parameter_ranges[param]
            param_length = self.out_parameter_lengths[param]
            pindx = self.parameter_indices[param]
            x_param_current = current_params_np[pindx : pindx + param_length]
            x_param_new = (
                x_np[pindx : pindx + param_length] + x_param_current
            )  # Add the current parameter value to the increment
            x_param_new = np.clip(x_param_new, -1.0, 1.0)

            for j in range(len(x_param_new)):  # pylint: disable=consider-using-enumerate
                x_param_new[j] = csmf.linear_map(x_param_new[j], (-1.0, 1.0), tuple(param_range))
            params[param] = x_param_new
        return params

    def unnormalize(self, x: th.Tensor) -> np.ndarray:
        """Unnormalize the DNN output tensor.

        Args:
            x (th.Tensor): The DNN output tensor

        Returns:
            np.ndarray: The unnormalized output as a numpy array
        """
        x_unnorm = th.zeros_like(x)
        for param in self.param_list:
            param_range = self.out_parameter_ranges[param]
            param_length = self.out_parameter_lengths[param]
            pindx = self.parameter_indices[param]
            x_param = x[pindx : pindx + param_length]

            for j in range(len(x_param)):  # pylint: disable=consider-using-enumerate
                x_param[j] = csmf.linear_map(x_param[j], (-1.0, 1.0), tuple(param_range))
            x_unnorm[pindx : pindx + param_length] = x_param

        return x_unnorm.detach().numpy()

    def normalize(self, p: th.Tensor) -> th.Tensor:
        """Normalize the input parameter (not increment) tensor.

        Args:
            p (th.Tensor): The parameter tensor

        Returns:
            th.Tensor: The normalized parameter tensor
        """
        p_norm = th.zeros_like(p)
        for param in self.param_list:
            param_range = self.out_parameter_ranges[param]
            param_length = self.out_parameter_lengths[param]
            pindx = self.parameter_indices[param]
            p_param = p[pindx : pindx + param_length]

            for j in range(len(p_param)):  # pylint: disable=consider-using-enumerate
                p_param[j] = csmf.linear_map(p_param[j], tuple(param_range), (-1.0, 1.0))
            p_norm[pindx : pindx + param_length] = p_param
        return p_norm

    def loss_function(
        self,
        param_increment: th.Tensor,
        prev_action: th.Tensor,
        new_action: th.Tensor,
        mpc_cost_val: th.Tensor,
        human_tuned_cost_val: th.Tensor,
    ) -> th.Tensor:
        """Compute the loss of the MPC parameter provider.

        Args:
            param_increment (th.Tensor): The increment of the MPC parameters.
            prev_action (th.Tensor): The previous action.
            new_action (th.Tensor): The new action.
            mpc_cost_val (th.Tensor): The MPC cost value.
            human_tuned_cost_val (th.Tensor): The human-tuned cost value.

        Returns:
            th.Tensor: The loss of the MPC parameter provider.
        """
        Q_param = 1.0 * th.ones_like(param_increment)
        action_diff = th.sum(th.pow(new_action - prev_action, 2))
        loss = th.pow(param_increment - action_diff, 2) + th.pow(mpc_cost_val - human_tuned_cost_val, 2)
        return loss


class SACMPCActor(BasePolicy):
    """
    MPC-Actor (policy) for SAC.

    Args:
        - observation_space (spaces.Space): Observation space
        - action_space (spaces.Box): Action space
        - observation_type (Any): Observation type
        - action_type (Any): Action type
        - mpc_config (rlmpc.RLMPCParams | pathlib.Path): MPC configuration
        - features_extractor_class (Type[BaseFeaturesExtractor], optional): Features extractor to use. Defaults to FlattenExtractor.
        - features_extractor_kwargs (Optional[Dict[str, Any]]): Keyword arguments
        - use_sde (bool, optional): Whether to use State Dependent Exploration or not. Defaults to False.
        - log_std_init (float, optional): Initial value for the log standard deviation. Defaults to -3.
        - use_expln (bool, optional): Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough. Defaults to False.
        - clip_mean (float, optional): Clip the mean output when using gSDE to avoid numerical instability. Defaults to 2.0.
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        observation_type: Any,
        action_type: Any,
        mpc_param_provider_kwargs: Dict[str, Any],
        mpc_config: rlmpc.RLMPCParams | pathlib.Path = dp.config / "rlmpc.yaml",
        use_sde: bool = False,
        std_init: np.ndarray | float = np.array([2.0, 2.0]),
        use_expln: bool = False,
        clip_mean: float = 2.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=0,
            normalize_images=False,
            squash_output=True,
        )

        self.observation_type = observation_type
        self.action_type = action_type

        action_dim = get_action_dim(self.action_space)
        if isinstance(std_init, float):
            std_init = np.array([std_init] * action_dim)
        log_std_init = th.log(th.from_numpy(std_init)).to(th.float32)
        self.log_std_dev = th.nn.Parameter(log_std_init)
        self.log_std = log_std_init

        unnorm_std_init = self.action_type.unnormalize(std_init)
        print(f"SAC MPC Actor gaussian std dev: {unnorm_std_init}")

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.log_std_init = self.log_std
        self.use_expln = use_expln
        self.clip_mean = clip_mean
        self.infeasible_solutions: int = 0

        action_dim = get_action_dim(self.action_space)
        self.action_dist = DiagGaussianDistribution(action_dim).proba_distribution(
            th.zeros(action_dim), log_std=self.log_std_init
        )

        # mpc_param_provider_kwargs = mpc_param_provider_kwargs.update[
        #     {
        #         "features_extractor_class": features_extractor_class,
        #         "features_extractor_kwargs": features_extractor_kwargs,
        #     }
        # ]
        self.mpc_param_provider = MPCParameterDNN(**mpc_param_provider_kwargs)
        self.mpc = rlmpc.RLMPC(mpc_config)
        self.mpc_sensitivities = None
        nx, nu = self.mpc.get_mpc_model_dims()
        self.mpc.set_adjustable_param_str_list(self.mpc_param_provider.param_list)
        self.mpc_params = self.mpc.get_mpc_params()
        self.num_params = self.mpc_param_provider.num_params
        n_samples = int(self.mpc_params.T / self.mpc_params.dt)
        # lookahead_sample = self.mpc.lookahead_sample
        # Indices for the RLMPC action a = [x_LD, y_LD, speed_0]
        # where LD is the lookahead sample (3)
        # self.action_indices = [
        #     nu * n_samples + lookahead_sample * nx,
        #     nu * n_samples + (lookahead_sample * nx + 1),
        #     nu * n_samples + (lookahead_sample * nx + 3),
        # ]

        # second option, sequence of course and speed refs
        self.action_indices = [
            int(nu * n_samples + (1 * nx) + 2),  # chi 1
            int(nu * n_samples + (1 * nx) + 3),  # speed 1
            # int(nu * n_samples + (2 * nx) + 2),  # chi 2
            # int(nu * n_samples + (2 * nx) + 3),  # speed 2
        ]

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def action_log_prob(
        self,
        obs: rlmpc_buffers.TensorDict,
        actions: Optional[th.Tensor] = None,
        infos: Optional[List[Dict[str, Any]]] = None,
        is_next_action: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Computes the log probability of the policy distribution for the given observation.

        Args:
            obs (th.Tensor): Observations
            actions (th.Tensor): (MPC) Actions to evaluate the log probability for
            infos (Optional[List[Dict[str, Any]]], optional): Additional information. Defaults to None.
            is_next_action (bool, optional): Whether the action is the next action in the SARSA tuple. Used for extracting the correct (mpc) mean action.

        Returns:
            Tuple[th.Tensor, th.Tensor]:
        """
        # obs = obs.to("cpu").detach()
        # action = action.to("cpu").detach()

        # If a proper stochastic policy is used, we need to solve a perturbed MPC problem
        # action = self.mpc.act(t, ownship_state, do_list, w, prev_soln, perturb=True)
        # log_prob = self.compute SPG machinery using the perturbed action, solution and mpc sensitivities
        #
        # If the ad hoc stochastic policy is used, we just add noise to the input (MPC) action
        assert infos is not None, "Infos must be provided when using ad hoc stochastic policy"
        actor_str = "actor_info" if not is_next_action else "next_actor_info"
        if infos is not None:
            # Extract mean of the policy distribution = MPC action for the given observation
            norm_mpc_actions = np.array([info[0][actor_str]["norm_mpc_action"] for info in infos], dtype=np.float32)
            norm_mpc_actions = th.from_numpy(norm_mpc_actions)

        if isinstance(actions, np.ndarray):
            actions = th.from_numpy(actions)

        self.action_dist = self.action_dist.proba_distribution(mean_actions=norm_mpc_actions, log_std=self.log_std)
        log_prob = self.action_dist.log_prob(actions)

        return log_prob

    def _convert_obs_tensor_to_numpy(self, obs: Dict[str, th.Tensor]) -> Dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in obs.items()}

    def _convert_obs_numpy_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, th.Tensor]:
        return {k: th.from_numpy(v) for k, v in obs.items()}

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        obs = self._convert_obs_tensor_to_numpy(observation)
        return self.predict_with_mpc(obs, deterministic)

    def predict_with_mpc(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: List[Dict[str, Any]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Get the policy action from an observation (and optional actor state).

        Args:
            - observation (Union[np.ndarray, Dict[str, np.ndarray]]): the input observation
            - state (List[Dict[str, Any]]): The MPC internal state (current solution info, etc.)
            - deterministic (bool): Whether or not to return deterministic actions.

        Returns:
            - Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]: the MPC unnormalized action, normalized action and the MPC internal state
        """
        batch_size = observation["TrackingObservation"].shape[0]
        normalized_actions = np.zeros((batch_size, self.action_space.shape[0]), dtype=np.float32)
        unnormalized_actions = np.zeros((batch_size, self.action_space.shape[0]), dtype=np.float32)
        actor_infos = [{} for _ in range(batch_size)]
        obs_tensor = self._convert_obs_numpy_to_tensor(observation)
        preprocessed_obs = self.preprocess_obs_for_dnn(obs_tensor, self.observation_space)
        features = self.features_extractor(preprocessed_obs)

        old_mpc_params = self.mpc.get_adjustable_mpc_params()
        current_mpc_params = th.from_numpy(old_mpc_params).float()
        current_mpc_params = self.mpc_param_provider.normalize(current_mpc_params)

        for idx in range(batch_size):
            prev_action = th.zeros(self.action_space.shape[0]) if state is None else state[idx]["norm_mpc_action"]
            mpc_param_increment = (
                self.mpc_param_provider(features[idx], current_mpc_params, prev_action).detach().numpy()
            )

            mpc_param_subset_dict = self.mpc_param_provider.map_to_parameter_dict(
                mpc_param_increment, current_mpc_params.detach().numpy()
            )
            print(f"Provided MPC parameters: {mpc_param_subset_dict} | Old: {old_mpc_params.tolist()}")
            # self.mpc.set_mpc_param_subset(mpc_param_subset_dict)
            t, ownship_state, do_list, w = self.extract_mpc_observation_features(observation, idx)
            # w.print()
            prev_soln = state[idx] if state is not None else None
            action, info = self.mpc.act(t=t, ownship_state=ownship_state, do_list=do_list, w=w, prev_soln=prev_soln)
            norm_action = self.action_type.normalize(action)
            info.update({"unnorm_mpc_action": action, "norm_mpc_action": norm_action})
            info.update({"dnn_input_features": features[idx].detach().cpu().numpy()})
            info.update(
                {
                    "mpc_param_increment": self.mpc_param_provider.unnormalize(mpc_param_increment),
                    "new_mpc_params": self.mpc.get_adjustable_mpc_params(),
                    "old_mpc_params": old_mpc_params,
                }
            )
            if not deterministic:
                norm_action = self.sample_action(norm_action)
            unnormalized_actions[idx, :] = self.action_type.unnormalize(norm_action)
            normalized_actions[idx, :] = norm_action

            actor_infos[idx] = info
            if not info["optimal"]:
                self.infeasible_solutions += 1

        return unnormalized_actions, normalized_actions, actor_infos

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

    def extract_mpc_observation_features(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray], rlmpc_buffers.TensorDict], idx: int
    ) -> Tuple[float, np.ndarray, List, stochasticity.DisturbanceData]:
        """Extract features from the observation at a given index in the batch.

        Args:
            observation (Union[np.ndarray, Dict[str, np.ndarray], rlmpc_buffers.TensorDict]): The input observation
            idx (int): The index of the observation in the batch

        Returns:
            Tuple[float, np.ndarray, List, stoch.DisturbanceData]: Time, ownship state, DO list and disturbance data
        """
        obs_b = {k: v[idx].numpy() if isinstance(v[idx], th.Tensor) else v[idx] for k, v in observation.items()}
        do_arr = obs_b["TrackingObservation"]
        t = obs_b["TimeObservation"].flatten()[0]
        unnorm_obs_b = self.observation_type.unnormalize(obs_b)
        ownship_state = unnorm_obs_b["Navigation3DOFStateObservation"].flatten()
        disturbance_vector = unnorm_obs_b["DisturbanceObservation"].flatten()
        max_num_do = do_arr.shape[1]
        do_list = []
        for i in range(max_num_do):
            if np.sum(do_arr[:, i]) > 1.0:  # A proper DO entry has non-zeros in its vector
                cov = do_arr[6:, i].reshape(4, 4)
                do_list.append((i, do_arr[0:4, i], cov, do_arr[4, i], do_arr[5, i]))

        ownship_course = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])
        w = stochasticity.DisturbanceData()
        w.currents = {"speed": disturbance_vector[0], "direction": (disturbance_vector[1] + ownship_course)}
        w.wind = {"speed": disturbance_vector[2], "direction": disturbance_vector[3] + ownship_course}
        return t, ownship_state, do_list, w

    def preprocess_obs_for_dnn(
        self,
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
        observation_space: spaces.Space,
        normalize_images: bool = True,
    ) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        """
        Preprocess observation to be to a neural network.
        For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
        For discrete observations, it create a one hot vector.

        Args:
            obs (Union[th.Tensor, Dict[str, th.Tensor]]): Observation
            observation_space (spaces.Space):
            normalize_images (bool): Whether to normalize images or not (True by default)

        Returns:
            Union[th.Tensor, Dict[str, th.Tensor]]: The preprocessed observation
        """
        if isinstance(observation_space, spaces.Dict):
            # Do not modify by reference the original observation
            assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
            preprocessed_obs = {}
            for key, _obs in obs.items():
                preprocessed_obs[key] = self.preprocess_obs_for_dnn(
                    _obs, observation_space[key], normalize_images=normalize_images
                )
                # print(
                #     f"Preprocessed observation: {key} shape: {preprocessed_obs[key].shape} | (min, max): ({preprocessed_obs[key].min()}, {preprocessed_obs[key].max()})"
                # )
            return preprocessed_obs  # type: ignore[return-value]

        assert isinstance(obs, th.Tensor), f"Expecting a torch Tensor, but got {type(obs)}"

        if isinstance(observation_space, spaces.Box):
            if normalize_images and is_image_space(observation_space):
                return obs.float() / 255.0
            return obs.float()

        else:
            raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")

    def initialize(
        self,
        env: COLAVEnvironment,
        **kwargs,
    ) -> None:
        """Initialize the planner by setting up the nominal path, static obstacle inputs and constructing
        the OCP"""
        t = env.unwrapped.time
        waypoints = env.unwrapped.ownship.waypoints
        speed_plan = env.unwrapped.ownship.speed_plan
        ownship_state = env.unwrapped.ownship.state
        enc = env.unwrapped.enc
        do_list = env.unwrapped.ownship.get_do_track_information()
        goal_state = env.unwrapped.ownship.goal_state
        w = env.unwrapped.disturbance.get() if env.unwrapped.disturbance is not None else None

        self.mpc.initialize(t, waypoints, speed_plan, ownship_state, do_list, enc, goal_state, w, **kwargs)
        self.mpc.set_action_indices(self.action_indices)
        self.mpc_sensitivities = self.mpc.build_sensitivities()
        print("SAC MPC Actor initialized!")

    def update_params(self, step: th.Tensor) -> None:
        """Update the parameters of the actor policy (if any)."""
        step = step.detach().cpu().numpy()
        self.mpc.update_adjustable_mpc_params(step)


class SACPolicyWithMPC(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    Args:
        - observation_space (spaces.Space): Observation space
        - action_space (spaces.Box): Action space
        - learning_rate: Union[float, Schedule] = 3e-4,
        - critic_arch (Optional[List[int]], optional): Architecture of the critic network. Defaults to [256, 256].
        - mpc_config (rlmpc.RLMPCParams | pathlib.Path): MPC configuration
        - activation_fn (Type[nn.Module], optional): Activation function. Defaults to nn.ReLU.
        - use_sde (bool, optional): Whether to use State Dependent Exploration or not. Defaults to False.
        - log_std_init (float, optional): Initial value for the log standard deviation. Defaults to -3.
        - use_expln (bool, optional): Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough. Defaults to False.
        - clip_mean (float, optional): Clip the mean output when using gSDE to avoid numerical instability. Defaults to 2.0.
        - features_extractor_class (Type[BaseFeaturesExtractor], optional): Features extractor to use. Defaults to FlattenExtractor.
        - features_extractor_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments
            to pass to the features extractor. Defaults to None.
        - normalize_images (bool, optional): Whether to normalize images or not,
            dividing by 255.0 (True by default). Defaults to True.
        - optimizer_class (Type[th.optim.Optimizer], optional): The optimizer to use,
            ``th.optim.Adam`` by default. Defaults to th.optim.Adam.
        - optimizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer. Defaults to None.
        - n_critics (int, optional): Number of critic networks to create. Defaults to 2.
    """

    actor: SACMPCActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        observation_type: Any,
        action_type: Any,
        lr_schedule: Schedule,
        critic_arch: Optional[List[int]] = [256, 256],
        mpc_param_provider_kwargs: Dict[str, Any] = {},
        mpc_config: rlmpc.RLMPCParams | pathlib.Path = dp.config / "rlmpc.yaml",
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        use_sde: bool = False,
        std_init: np.ndarray | float = np.array([2.0, 2.0, 0.5]),
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = rlmpc_fe.CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )
        self.observation_type = observation_type
        self.action_type = action_type
        self.activation_fn = activation_fn
        self.critic_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": critic_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "n_critics": n_critics,
            "share_features_extractor": False,
        }

        self.actor_kwargs = {
            "mpc_param_provider_kwargs": mpc_param_provider_kwargs,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "mpc_config": mpc_config,
            "observation_type": self.observation_type,
            "action_type": self.action_type,
        }
        sde_kwargs = {
            "use_sde": use_sde,
            "std_init": std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)

        self._build_critic(lr_schedule)

        self._build_actor(lr_schedule)
        print("Actor and critic built!")

    def _build_critic(self, lr_schedule: Schedule) -> None:
        # Create a separate features extractor for the critic
        # this requires more memory and computation
        self.critic = self.make_critic(features_extractor=None)
        critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _build_actor(self, lr_schedule: Schedule) -> None:
        self.actor = SACMPCActor(**self.actor_kwargs)
        self.actor.features_extractor = self.critic.features_extractor  # share features extractor with critic
        self.actor.optimizer = self.optimizer_class(
            self.actor.mpc_param_provider.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def initialize_mpc_actor(
        self,
        env: COLAVEnvironment,
        **kwargs,
    ) -> None:
        self.actor.initialize(env, **kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.critic_kwargs["net_arch"],
                activation_fn=self.critic_kwargs["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def _predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: List[Dict[str, Any]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> th.Tensor:
        return self.actor.predict_with_mpc(observation, state, deterministic=deterministic)

    def sensitivities(self) -> mpc_common.NLPSensitivities:
        return self.actor.mpc_sensitivities

    def predict_with_mpc(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: List[Dict[str, Any]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Get the policy action from an observation (and optional actor state).

        Args:
            - observation (Union[np.ndarray, Dict[str, np.ndarray]]): the input observation
            - state (List[Dict[str, Any]]): The MPC internal state (current solution info, etc.)
            - episode_start (Optional[np.ndarray]): The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            - deterministic (bool): Whether or not to return deterministic actions.

        Returns:
            - Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]: the MPC unnormalized action, normalized action and the MPC internal state
        """
        return self.actor.predict_with_mpc(observation, state, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode
