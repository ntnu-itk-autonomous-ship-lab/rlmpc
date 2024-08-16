"""
    policies.py

    Summary:
        Soft Actor-Critic (SAC) policies (and actors) implementation for the mid-level MPC.


    Author: Trym Tengesdal
"""

import pathlib
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import colav_simulator.core.stochasticity as stochasticity
import colav_simulator.gym.environment as csenv
import numpy as np
import rlmpc.common.buffers as rlmpc_buffers
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as dp
import rlmpc.mpc.common as mpc_common
import rlmpc.mpc.parameters as mpc_params
import rlmpc.networks.feature_extractors as rlmpc_fe
import rlmpc.rlmpc_cas as rlmpc_cas
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import BaseModel, ContinuousCritic
from stable_baselines3.common.preprocessing import (get_action_dim,
                                                    is_image_space)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import BasePolicy

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_PROB_MIN = -15.0


class CustomContinuousCritic(BaseModel):
    """
    CUSTOMIZED CRITIC NETWORK FOR SAC

    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: rlmpc_fe.CombinedExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: rlmpc_fe.CombinedExtractor,
        features_dim: int,
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[th.nn.Module] = []
        for idx in range(n_critics):
            q_net_list = self.create_mlp(
                input_dim=features_dim + action_dim, output_dim=1, net_arch=net_arch, activation_fn=activation_fn
            )
            q_net = th.nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        for q_net in self.q_networks:
            q_net.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, th.nn.Conv2d):
            th.nn.init.xavier_uniform_(m.weight, gain=th.nn.init.calculate_gain("linear"))
            th.nn.init.zeros_(m.bias)
        elif isinstance(m, th.nn.Linear):
            th.nn.init.xavier_uniform_(m.weight, gain=th.nn.init.calculate_gain("linear"))
            th.nn.init.zeros_(m.bias)

    def create_mlp(
        self,
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        squash_output: bool = False,
        with_bias: bool = True,
    ) -> List[th.nn.Module]:
        """
        Create a multi layer perceptron (MLP), which is
        a collection of fully-connected layers with intermittent activation functions
        each followed by an end activation function.

        Args:
            - input_dim (int): Dimension of the input vector
            - output_dim (int): Dimension of the output vector
            - net_arch (List[int]): Architecture of the neural net
                It represents the number of units per layer.
                The length of this list is the number of layers.
            - activation_fn (Type[th.nn.Module]): The activation function to use in between layers.
            - squash_output (bool): Whether to squash the output or not
            - with_bias (bool): Whether to use bias in the layers or not

        Returns:
            - List[th.nn.Module]: The layers of the MLP
        """
        modules = []
        feature_dim = input_dim
        for arch in net_arch:
            modules.append(th.nn.Linear(feature_dim, arch, bias=with_bias))
            modules.append(activation_fn())
            feature_dim = arch
        modules.append(th.nn.Linear(feature_dim, output_dim, bias=with_bias))
        if squash_output:
            modules.append(th.nn.Tanh())
        return modules

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class MPCParameterDNN(th.nn.Module):
    """The DNN for predicting the MPC parameter increments, based on the
    current situation, parameters and action. The DNN outputs are normalized to [-1, 1] and then mapped to the
    actual parameter ranges. 64+12+5 = 81 features (64 latent enc, 12 latent tracking obs, 5 path rel obs)
    """

    def __init__(
        self,
        param_list: List[str],
        hidden_sizes: List[int] = [128, 64],
        activation_fn: Type[th.nn.Module] = th.nn.ELU,
        features_dim: int = 57,
        model_file: Optional[pathlib.Path] = None,
    ):
        super().__init__()
        self.out_parameter_ranges, self.out_parameter_incr_ranges, self.out_parameter_lengths = (
            mpc_params.MidlevelMPCParams.get_adjustable_parameter_info()
        )
        offset = 0
        self.out_parameter_indices = {}
        for param in param_list:
            self.out_parameter_indices[param] = offset
            offset += self.out_parameter_lengths[param]

        self.num_output_params = offset
        self.param_list = param_list
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn()
        self.layers = th.nn.ModuleList()

        self.input_dim = features_dim + offset  # Add the number of parameters to the input features
        prev_size = self.input_dim
        for size in hidden_sizes:
            self.layers.append(th.nn.Linear(prev_size, size))
            prev_size = size
        self.layers.append(th.nn.Linear(prev_size, self.num_output_params))
        self.tanh = th.nn.Tanh()

        # The DNN parameters
        self.parameter_lengths = [p.numel() for p in self.parameters()]
        self.num_params = sum(self.parameter_lengths)

        for layer in self.layers:
            self.weights_init(layer)

        if model_file is not None:
            self.load_state_dict(th.load(model_file))

    def weights_init(self, m):
        if isinstance(m, th.nn.Conv2d):
            th.nn.init.xavier_uniform_(m.weight, gain=th.nn.init.calculate_gain("linear"))
            th.nn.init.zeros_(m.bias)
        elif isinstance(m, th.nn.Linear):
            th.nn.init.xavier_uniform_(m.weight, gain=th.nn.init.calculate_gain("linear"))
            th.nn.init.zeros_(m.bias)

    def forward(self, x: th.Tensor) -> th.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        x = self.tanh(x)
        return x

    def set_gradients(self, grads: th.Tensor) -> None:
        """Update the parameters of the actor DNN mpc parameter provider policy.

        Args:
            - grads (th.Tensor): The parameter gradient tensor 1 x num_params.

        """
        idx = 0
        for param_name, param in self.named_parameters():
            param.grad = grads[idx : idx + param.numel()].reshape(param.shape)
            # print(f"Parameter: {param_name} | Gradient: {param.grad}")
            idx += param.numel()

    def map_to_parameter_dict(self, x: np.ndarray, current_params: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Maps the DNN output tensor to a dictionary of unnormalized parameters, given the current parameters.

        Args:
            x (np.ndarray): The DNN output, consisting of normalized parameter increments.
            current_params (np.ndarray): The current parameters, unnormalized.

        Returns:
            Dict[str, Union[float, np.ndarray]]: The dictionary of unnormalized parameters
        """
        output = hf.map_mpc_param_incr_array_to_parameter_dict(
            x=x,
            current_params=current_params,
            param_list=self.param_list,
            parameter_ranges=self.out_parameter_ranges,
            parameter_incr_ranges=self.out_parameter_incr_ranges,
            parameter_lengths=self.out_parameter_lengths,
            parameter_indices=self.out_parameter_indices,
        )
        return output

    def save(self, path: pathlib.Path) -> None:
        """Save the DNN model to a file.

        Args:
            path (pathlib.Path): The path to save the model to
        """
        th.save(self.state_dict(), path)

    def unnormalize_increment(self, x: th.Tensor) -> np.ndarray:
        """Unnormalize the input parameter increment tensor.

        Args:
            x (th.Tensor): The normalized parameter increment tensor

        Returns:
            np.ndarray: The unnormalized output increment as a numpy array
        """
        return hf.unnormalize_mpc_param_increment_tensor(
            x=x,
            param_list=self.param_list,
            parameter_incr_ranges=self.out_parameter_incr_ranges,
            parameter_lengths=self.out_parameter_lengths,
            parameter_indices=self.out_parameter_indices,
        )

    def unnormalize(self, x: th.Tensor | np.ndarray) -> np.ndarray:
        """Unnormalize the input parameter tensor.

        Args:
            x (th.Tensor | np.ndarray): The normalized parameter tensor

        Returns:
            np.ndarray: The unnormalized output as a numpy array
        """
        x_in = x.detach().clone() if isinstance(x, th.Tensor) else x.copy()
        return hf.unnormalize_mpc_param_tensor(
            x_in, self.param_list, self.out_parameter_ranges, self.out_parameter_lengths, self.out_parameter_indices
        )

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize the input parameter tensor.

        Args:
            x (th.Tensor): The unnormalized parameter tensor

        Returns:
            th.Tensor: The normalized parameter tensor
        """
        x_in = x.detach().clone() if isinstance(x, th.Tensor) else x.copy()
        return hf.normalize_mpc_param_tensor(
            x_in, self.param_list, self.out_parameter_ranges, self.out_parameter_lengths, self.out_parameter_indices
        )

    def parameter_jacobian(self, x: th.Tensor) -> th.Tensor:
        """Compute the Jacobian of the DNN output wrt its parameters.

        Args:
            x (th.Tensor): The input tensor

        Returns:
            th.Tensor: The Jacobian of the DNN output wrt its parameters
        """
        params = {k: v.detach() for k, v in self.named_parameters()}
        buffers = {k: v.detach() for k, v in self.named_buffers()}

        # n_dnn_params = sum(p.numel() for p in params.values())
        # print(f"Total number of DNN MPC provider parameters: {n_dnn_params}")
        def compute_sample_jacobian(sample):
            # this will calculate the gradients for a single sample
            # we want the gradients for each output wrt to the parameters
            # this is the same as the jacobian of the model wrt the parameters

            call = lambda x: th.func.functional_call(self, (x, buffers), sample)

            # calculate the jacobian of the self.actor.mpc_param_provider wrt the parameters
            J = th.func.jacrev(call)(params)

            # J is a dictionary with keys the names of the parameters and values the gradients
            # we want a tensor
            grads = th.cat([v.flatten(1) for v in J.values()], -1)
            return grads

        # no we can use vmap to calculate the gradients for all samples at once
        dnn_jacobians = th.vmap(compute_sample_jacobian)(x)
        return dnn_jacobians


class SACMPCParameterProviderActor(BasePolicy):
    """
    MPC-Actor (policy) for SAC, which uses the MPC parameter provider DNN to predict the MPC parameter increments.

    Args:
        - observation_space (spaces.Space): Observation space
        - action_space (spaces.Box): Action space
        - observation_type (Any): Observation type
        - action_type (Any): Action type
        - mpc_config (rlmpc.RLMPCParams | pathlib.Path): MPC configuration
        - features_extractor_class (Type[rlmpc_fe.CombinedExtractor], optional): Features extractor to use. Defaults to FlattenExtractor.
        - features_extractor_kwargs (Optional[Dict[str, Any]]): Keyword arguments
        - use_sde (bool, optional): Whether to use State Dependent Exploration or not. Defaults to False.
        - log_std_init (float, optional): Initial value for the log standard deviation. Defaults to -3.
        - use_expln (bool, optional): Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough. Defaults to False.
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        mpc_param_provider_kwargs: Dict[str, Any],
        std_init: Union[float, np.ndarray] = -3.0,
        mpc_std_init: Union[float, np.ndarray] = -3.0,
        disable_parameter_provider: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=0,
            normalize_images=False,
            squash_output=True,
        )

        self.disable_parameter_provider = disable_parameter_provider
        self.mpc_param_provider = MPCParameterDNN(**mpc_param_provider_kwargs)
        if isinstance(std_init, float):
            std_init = np.array([std_init] * self.action_space.shape[0])
        log_std_init = th.log(th.from_numpy(std_init)).to(th.float32)
        self.log_std = log_std_init
        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0]).proba_distribution(
            th.zeros(self.action_space.shape[0]), log_std=self.log_std
        )

        self.mpc_action_dim = 2
        if isinstance(mpc_std_init, float):
            mpc_std_init = np.array([mpc_std_init] * self.mpc_action_dim)
        mpc_log_std_init = th.log(th.from_numpy(mpc_std_init)).to(th.float32)
        self.mpc_log_std = mpc_log_std_init
        self.mpc_action_dist = DiagGaussianDistribution(self.mpc_action_dim).proba_distribution(
            th.zeros(self.mpc_action_dim), log_std=self.mpc_log_std
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                features_extractor=self.features_extractor,
            )
        )
        return data

    def set_gradients(self, grads: th.Tensor) -> None:
        """Update the parameter gradients of the actor DNN mpc parameter provider policy.

        Args:
            - grads (th.Tensor): The parameter gradient tensor 1 x num_params.
        """
        self.mpc_param_provider.set_gradients(grads)

    def mpc_action_log_prob(
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
            infos (Optional[List[Dict[str, Any]]], optional): Additional information.
            is_next_action (bool, optional): Whether the action is the next action in the SARSA tuple. Used for extracting the correct (mpc) mean action.

        Returns:
            Tuple[th.Tensor, th.Tensor]:
        """
        # If a proper stochastic policy is used, we need to solve a perturbed MPC problem
        # action = self.mpc.act(t, ownship_state, do_list, w, prev_soln, perturb=True)
        # log_prob = self.compute SPG machinery using the perturbed action, solution and mpc sensitivities
        #
        # If the ad hoc stochastic policy is used, we just add noise to the input (MPC) action
        assert infos is not None, "Infos must be provided when using ad hoc stochastic policy"
        actor_str = "actor_info" if not is_next_action else "next_actor_info"
        if infos is not None:
            # Extract mean of the policy distribution = MPC action for the given observation
            norm_mpc_actions = np.array([info[actor_str]["norm_mpc_action"] for info in infos], dtype=np.float32)
            norm_mpc_actions = th.from_numpy(norm_mpc_actions)

        if isinstance(actions, np.ndarray):
            actions = th.from_numpy(actions)

        self.mpc_action_dist = self.mpc_action_dist.proba_distribution(
            mean_actions=norm_mpc_actions, log_std=self.mpc_log_std
        )
        log_prob = self.mpc_action_dist.log_prob(actions)

        return log_prob

    def action_log_prob(
        self,
        obs: rlmpc_buffers.TensorDict,
        actions: Optional[th.Tensor] = None,
        infos: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Computes the log probability of the policy distribution for the given observation.

        Args:
            obs (th.Tensor): Observations
            actions (th.Tensor): (MPC) Actions to evaluate the log probability for
            infos (Optional[List[Dict[str, Any]]], optional): Additional information.
            is_next_action (bool, optional): Whether the action is the next action in the SARSA tuple. Used for extracting the correct (mpc) mean action.

        Returns:
            Tuple[th.Tensor, th.Tensor]:
        """

        batch_size = obs["TrackingObservation"].shape[0]
        # obs = self._convert_obs_numpy_to_tensor(obs)
        preprocessed_obs = self.preprocess_obs_for_dnn(obs, self.observation_space)
        features = self.features_extractor(preprocessed_obs)

        log_prob = th.zeros(batch_size)
        for idx in range(batch_size):
            norm_current_mpc_params = obs["MPCParameterObservation"][idx]
            norm_current_mpc_params = th.from_numpy(norm_current_mpc_params).float()
            dnn_input = th.cat([features[idx], norm_current_mpc_params], dim=-1)
            mpc_param_increment = self.mpc_param_provider(dnn_input)
            self.action_dist = self.action_dist.proba_distribution(
                mean_actions=mpc_param_increment, log_std=self.log_std
            )
            log_prob[idx] = self.action_dist.log_prob(actions)

        return log_prob

    def _convert_obs_tensor_to_numpy(self, obs: Dict[str, th.Tensor]) -> Dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in obs.items()}

    def _convert_obs_numpy_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, th.Tensor]:
        return {k: th.from_numpy(v) for k, v in obs.items()}

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        obs = self._convert_obs_tensor_to_numpy(observation)
        return self.custom_predict(obs, deterministic)

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
        norm_collision_seeking_action = self.action_type.normalize(action)
        norm_collision_seeking_action[1] = norm_mpc_action[1]
        return th.from_numpy(norm_collision_seeking_action)

    def custom_predict(
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

        for idx in range(batch_size):
            norm_current_mpc_params = obs_tensor["MPCParameterObservation"][idx]
            dnn_input = th.cat([features[idx], norm_current_mpc_params], dim=-1)
            mpc_param_increment = np.zeros(self.action_space.shape[0])
            if not self.disable_parameter_provider:
                mpc_param_increment = self.mpc_param_provider(dnn_input).detach().numpy()

            if not deterministic:
                mpc_param_increment = self.sample_action(mean_actions=mpc_param_increment)
            unnorm_action = self.mpc_param_provider.unnormalize_increment(mpc_param_increment)
            info = {
                "dnn_input_features": dnn_input.detach().cpu().numpy(),
                "norm_mpc_param_increment": mpc_param_increment,
                "unnorm_mpc_param_increment": unnorm_action,
                "norm_old_mpc_params": norm_current_mpc_params.detach().numpy(),
                "old_mpc_params": self.mpc_param_provider.unnormalize(norm_current_mpc_params),
            }

            unnormalized_actions[idx, :] = unnorm_action
            normalized_actions[idx, :] = mpc_param_increment
            actor_infos[idx] = info

        return unnormalized_actions, normalized_actions, actor_infos

    def sample_mpc_action(self, mpc_actions: np.ndarray | th.Tensor) -> np.ndarray:
        """Sample an mpc action from the policy distribution with mean from the input MPC action.

        Args:
            mpc_actions (np.ndarray | th.Tensor): The input MPC action (normalized)

        Returns:
            np.ndarray: The sampled action (normalized)
        """
        if isinstance(mpc_actions, np.ndarray):
            mpc_actions = th.from_numpy(mpc_actions)
        self.mpc_action_dist = self.mpc_action_dist.proba_distribution(
            mean_actions=mpc_actions, log_std=self.mpc_log_std
        )
        norm_actions = self.mpc_action_dist.get_actions()
        norm_actions = th.clamp(norm_actions, -1.0, 1.0)
        return norm_actions

    def sample_action(self, mean_actions: np.ndarray) -> np.ndarray:
        """Sample an action from the policy distribution

        Args:
            mean_actions (np.ndarray): The mean action from the MPC parameter provider

        Returns:
            np.ndarray: The sampled action (normalized) from the policy distribution
        """
        norm_actions = self.action_dist.get_actions()
        norm_actions = th.clamp(norm_actions, -1.0, 1.0)
        return norm_actions

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

    def update_params(self, step: th.Tensor) -> None:
        """Update the parameters of the actor DNN mpc parameter provider policy."""
        self.mpc_param_provider.update_params(step)


class SACMPCActor(BasePolicy):
    """
    MPC-Actor (policy) for SAC.

    Args:
        - observation_space (spaces.Space): Observation space
        - action_space (spaces.Box): Action space
        - observation_type (Any): Observation type
        - action_type (Any): Action type
        - mpc_config (rlmpc.RLMPCParams | pathlib.Path): MPC configuration
        - features_extractor_class (Type[rlmpc_fe.CombinedExtractor], optional): Features extractor to use. Defaults to FlattenExtractor.
        - features_extractor_kwargs (Optional[Dict[str, Any]]): Keyword arguments
        - use_sde (bool, optional): Whether to use State Dependent Exploration or not. Defaults to False.
        - log_std_init (float, optional): Initial value for the log standard deviation. Defaults to -3.
        - use_expln (bool, optional): Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough. Defaults to False.
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        observation_type: Any,
        action_type: Any,
        mpc_param_provider_kwargs: Dict[str, Any],
        mpc_config: rlmpc_cas.RLMPCParams | pathlib.Path = dp.config / "rlmpc.yaml",
        std_init: np.ndarray | float = np.array([2.0, 2.0]),
        disable_parameter_provider: bool = False,
        debug: bool = False,
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
        self.disable_parameter_provider = disable_parameter_provider
        self.debug = debug

        action_dim = get_action_dim(self.action_space)
        if isinstance(std_init, float):
            std_init = np.array([std_init] * action_dim)
        log_std_init = th.log(th.from_numpy(std_init)).to(th.float32)
        self.log_std = log_std_init

        unnorm_std_init = self.action_type.unnormalize(std_init)
        print(f"SAC MPC Actor gaussian std dev: {unnorm_std_init}")

        # Save arguments to re-create object at loading
        self.t_prev: float = 0.0
        self.noise_application_duration: float = 10.0
        self.log_std_init = self.log_std
        self.non_optimal_solutions: int = 0
        self.prev_noise_action = None
        action_dim = get_action_dim(self.action_space)
        self.action_dist = DiagGaussianDistribution(action_dim).proba_distribution(
            th.zeros(action_dim), log_std=self.log_std_init
        )

        self.mpc_param_provider = MPCParameterDNN(**mpc_param_provider_kwargs)
        self.training_mpc = rlmpc_cas.RLMPC(mpc_config, "train")
        self.training_mpc.set_adjustable_param_str_list(self.mpc_param_provider.param_list)
        self.mpc_params = self.training_mpc.get_mpc_params()
        self.mpc_adjustable_params_init = self.training_mpc.get_adjustable_mpc_params()
        self.mpc_adjustable_params_init = self.mpc_param_provider.map_to_parameter_dict(
            np.zeros(self.mpc_param_provider.num_output_params), self.mpc_adjustable_params_init
        )
        self.mpc = None
        self.mpc_sensitivities = None

        nx, nu = self.training_mpc.get_mpc_model_dims()
        n_samples = int(self.mpc_params.T / self.mpc_params.dt)
        if self.mpc_params.dt == 1.0:
            self.action_indices = [
                int(nu * n_samples + (5 * nx) + 2),  # chi 2
                int(nu * n_samples + (5 * nx) + 3),  # speed 2
            ]
        else:
            self.action_indices = [
                int(nu * n_samples + (2 * nx) + 2),  # chi 3
                int(nu * n_samples + (2 * nx) + 3),  # speed 3
            ]

        self.training_mpc.set_action_indices(self.action_indices)

        self.eval_mpc = rlmpc_cas.RLMPC(mpc_config, identifier="eval")
        self.eval_mpc.set_adjustable_param_str_list(self.mpc_param_provider.param_list)
        self.eval_mpc.set_action_indices(self.action_indices)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                noise_application_duration=self.noise_application_duration,
                log_std_init=self.log_std_init,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """Sets the actor into training mode or not, and switches MPC mode accordingly.
        The MPC parameters are set to the initial values.

        Args:
            mode (bool): Whether to switch MPC mode to training or not.
        """
        self.training = mode
        if self.training:
            self.mpc = self.training_mpc
        else:
            self.mpc = self.eval_mpc
        self.mpc.set_mpc_param_subset(self.mpc_adjustable_params_init)

    def set_gradients(self, grads: th.Tensor) -> None:
        """Update the parameter gradients of the actor DNN mpc parameter provider policy.

        Args:
            - grads (th.Tensor): The parameter gradient tensor 1 x num_params.
        """
        self.mpc_param_provider.set_gradients(grads)

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
            infos (Optional[List[Dict[str, Any]]]): Additional information.
            is_next_action (bool): Whether the action is the next action in the SARSA tuple. Used for extracting the correct (mpc) mean action.

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
        return self.custom_predict(obs, deterministic)

    def reset(self) -> None:
        """Reset the policy."""
        self.non_optimal_solutions = 0
        self.mpc_sensitivities = None

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
        norm_collision_seeking_action = self.action_type.normalize(action)
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

    def custom_predict(
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

        if isinstance(observation["TimeObservation"], th.Tensor):
            observation = self._convert_obs_tensor_to_numpy(observation)
        obs_tensor = self._convert_obs_numpy_to_tensor(observation)

        preprocessed_obs = self.preprocess_obs_for_dnn(obs_tensor, self.observation_space)
        features = self.features_extractor(preprocessed_obs)

        unnorm_current_mpc_params = self.mpc.get_adjustable_mpc_params()
        norm_current_mpc_params = th.from_numpy(unnorm_current_mpc_params).float()
        norm_current_mpc_params = self.mpc_param_provider.normalize(norm_current_mpc_params)

        for idx in range(batch_size):
            prev_soln = state[idx] if state is not None else None

            dnn_input = th.cat([features[idx], norm_current_mpc_params], dim=-1)

            if not self.disable_parameter_provider:
                mpc_param_increment = self.mpc_param_provider(dnn_input).detach().numpy()
                mpc_param_subset_dict = self.mpc_param_provider.map_to_parameter_dict(
                    mpc_param_increment, unnorm_current_mpc_params
                )
                print(f"Provided MPC parameters: {mpc_param_subset_dict} | Old: {unnorm_current_mpc_params.tolist()}")
                self.mpc.set_mpc_param_subset(mpc_param_subset_dict)

            t, ownship_state, do_list, w = self.extract_mpc_observation_features(observation, idx)
            if t < 0.001:
                self.t_prev = t
            action, info = self.mpc.act(t=t, ownship_state=ownship_state, do_list=do_list, w=w, prev_soln=prev_soln)
            norm_action = self.action_type.normalize(action)

            info.update(
                {
                    "da_dp_mpc": self.compute_mpc_sensitivities(info) if self.training else None,
                    "unnorm_mpc_action": action,
                    "dnn_input_features": dnn_input.detach().cpu().numpy(),
                    "mpc_param_increment": mpc_param_increment if not self.disable_parameter_provider else None,
                    "new_mpc_params": self.mpc.get_adjustable_mpc_params(),
                    "old_mpc_params": unnorm_current_mpc_params,
                    "norm_old_mpc_params": norm_current_mpc_params.detach().numpy(),
                    "norm_mpc_action": norm_action,
                }
            )
            if not deterministic:
                norm_action = self.get_exploratory_action(norm_action, t, ownship_state, do_list)

            unnormalized_actions[idx, :] = self.action_type.unnormalize(norm_action)
            normalized_actions[idx, :] = norm_action

            if not info["optimal"]:
                self.non_optimal_solutions += 1

            actor_infos[idx] = info

        return unnormalized_actions, normalized_actions, actor_infos

    def compute_mpc_sensitivities(self, info: Dict[str, Any]) -> np.ndarray:
        """Compute the MPC sensitivities for the given solution info.

        Args:
            info (Dict[str, Any]): The solution info

        Returns:
            np.ndarray: The MPC sensitivities
        """
        assert self.mpc == self.training_mpc, "MPC sensitivities can only be computed for the training MPC"
        if self.mpc_sensitivities is None:
            self.mpc_sensitivities = self.mpc.build_sensitivities()

        da_dp_mpc = np.zeros((self.action_space.shape[0], self.mpc_param_provider.num_output_params))
        if info["optimal"]:
            soln = info["soln"]
            p = info["p"]
            p_fixed = info["p_fixed"]
            z = np.concatenate((soln["x"], soln["lam_g"]), axis=0).astype(np.float32)
            da_dp_mpc = self.mpc_sensitivities.da_dp(z, p_fixed, p).full()
        return da_dp_mpc

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
        t = obs_b["TimeObservation"].flatten()[0]
        unnorm_obs_b = self.observation_type.unnormalize(obs_b)
        do_list = hf.extract_do_list_from_tracking_observation(obs_b["TrackingObservation"])

        ownship_state = unnorm_obs_b["Navigation3DOFStateObservation"].flatten()
        ownship_course = ownship_state[2] + np.arctan2(ownship_state[4], ownship_state[3])
        disturbance_vector = np.array([0.0, 0.0, 0.0, 0.0])
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
        env: csenv.COLAVEnvironment,
        evaluate: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the planner by setting up the nominal path, static obstacle inputs and constructing
        the OCP

        Args:
            env (csenv.COLAVEnvironment): The environment
            evaluate (bool, optional): Whether to evaluate the MPC policy.

        """
        self.observation_type = env.unwrapped.observation_type
        self.action_type = env.unwrapped.action_type
        t = env.unwrapped.time
        waypoints = env.unwrapped.ownship.waypoints
        speed_plan = env.unwrapped.ownship.speed_plan
        ownship_state = env.unwrapped.ownship.state
        enc = env.unwrapped.enc
        do_list = env.unwrapped.ownship.get_do_track_information()

        self.training = not evaluate
        self.mpc = self.training_mpc
        if evaluate:
            self.mpc = self.eval_mpc

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
        if self.training:
            self.mpc_sensitivities = self.training_mpc.build_sensitivities()

        print(f"SAC MPC Actor initialized! Built sensitivities? {self.training}")

    def update_params(self, step: th.Tensor) -> None:
        """Update the parameters of the actor DNN mpc parameter provider policy."""
        self.mpc_param_provider.update_params(step)


class SACPolicyWithMPC(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    Args:
        - observation_space (spaces.Space): Observation space
        - action_space (spaces.Box): Action space
        - observation_type (Any): Observation type
        - action_type (Any): Action type
        - learning_rate: Union[float, Schedule] = 3e-4,
        - critic_arch (Optional[List[int]], optional): Architecture of the critic network. Defaults to [256, 256].
        - mpc_config (rlmpc.RLMPCParams | pathlib.Path): MPC configuration
        - activation_fn (Type[nn.Module], optional): Activation function. Defaults to nn.ReLU.
        - use_sde (bool, optional): Whether to use State Dependent Exploration or not. Defaults to False.
        - std_init (float, optional): Initial value for the action standard deviation
        - use_expln (bool, optional): Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough. Defaults to False.
        - features_extractor_class (Type[rlmpc_fe.CombinedExtractor], optional): Features extractor to use. Defaults to FlattenExtractor.
        - features_extractor_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments
            to pass to the features extractor.
        - normalize_images (bool, optional): Whether to normalize images or not,
            dividing by 255.0 (True by default).
        - optimizer_class (Type[th.optim.Optimizer], optional): The optimizer to use,
            ``th.optim.Adam`` by default. Defaults to th.optim.Adam.
        - optimizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer.
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
        mpc_config: rlmpc_cas.RLMPCParams | pathlib.Path = dp.config / "rlmpc.yaml",
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        std_init: np.ndarray | float = 0.0002,
        features_extractor_class: Type[rlmpc_fe.CombinedExtractor] = rlmpc_fe.CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        disable_parameter_provider: bool = False,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        debug: bool = False,
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
            "disable_parameter_provider": disable_parameter_provider,
            "std_init": std_init,
            "debug": debug,
        }

        self._build_critic(lr_schedule)

        mpc_param_provider_kwargs.update({"features_dim": self.critic.features_extractor.features_dim})
        self._build_actor(lr_schedule)

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

    def initialize_actor(
        self,
        env: csenv.COLAVEnvironment,
        evaluate: bool = False,
        **kwargs,
    ) -> None:
        self.actor.initialize(env, evaluate, **kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.critic_kwargs["net_arch"],
                activation_fn=self.critic_kwargs["activation_fn"],
                log_std_init=self.actor_kwargs["log_std_init"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def make_critic(self, features_extractor: Optional[rlmpc_fe.CombinedExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def _predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: List[Dict[str, Any]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> th.Tensor:
        return self.actor.custom_predict(observation, state, deterministic=deterministic)

    def sensitivities(self) -> mpc_common.NLPSensitivities:
        return self.actor.mpc_sensitivities

    def custom_predict(
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
        return self.actor.custom_predict(observation, state, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


class SACPolicyWithMPCParameterProvider(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    Args:
        - observation_space (spaces.Space): Observation space
        - action_space (spaces.Box): Action space
        - learning_rate: Union[float, Schedule] = 3e-4,
        - critic_arch (Optional[List[int]], optional): Architecture of the critic network. Defaults to [256, 256].
        - mpc_config (rlmpc.RLMPCParams | pathlib.Path): MPC configuration
        - activation_fn (Type[nn.Module], optional): Activation function. Defaults to nn.ReLU.
        - std_init (float, optional): Initial value for the parameter action standard deviation
        - mpc_std_init (np.ndarray | float, optional): Initial value for the MPC action standard deviation
        - features_extractor_class (Type[rlmpc_fe.CombinedExtractor], optional): Features extractor to use. Defaults to FlattenExtractor.
        - features_extractor_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments
            to pass to the features extractor.
        - disable_parameter_provider (bool, optional): Whether to disable the parameter provider or not.
        - normalize_images (bool, optional): Whether to normalize images or not,
            dividing by 255.0 (True by default).
        - optimizer_class (Type[th.optim.Optimizer], optional): The optimizer to use,
            ``th.optim.Adam`` by default. Defaults to th.optim.Adam.
        - optimizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer.
        - n_critics (int, optional): Number of critic networks to create. Defaults to 2.
    """

    actor: SACMPCParameterProviderActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        critic_arch: Optional[List[int]] = [256, 256],
        mpc_param_provider_kwargs: Dict[str, Any] = {},
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        std_init: np.ndarray | float = 0.0002,
        mpc_std_init: np.ndarray | float = np.array([0.002, 0.002]),
        features_extractor_class: Type[rlmpc_fe.CombinedExtractor] = rlmpc_fe.CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        disable_parameter_provider: bool = False,
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
        self.activation_fn = activation_fn
        self.mpc_action_dim = 2
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
            "disable_parameter_provider": disable_parameter_provider,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "std_init": std_init,
            "mpc_std_init": mpc_std_init,
        }

        self._build_critic(lr_schedule)

        mpc_param_provider_kwargs.update({"features_dim": self.critic.features_extractor.features_dim})
        self._build_actor(lr_schedule)

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
        self.actor = SACMPCParameterProviderActor(**self.actor_kwargs)
        self.actor.features_extractor = self.critic.features_extractor  # share features extractor with critic
        self.actor.optimizer = self.optimizer_class(
            self.actor.mpc_param_provider.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.critic_kwargs["net_arch"],
                activation_fn=self.critic_kwargs["activation_fn"],
                log_std_init=self.actor_kwargs["log_std_init"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def make_critic(self, features_extractor: Optional[rlmpc_fe.CombinedExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(
            {"action_space": spaces.Box(low=-1.0, high=1.0, shape=(self.mpc_action_dim,), dtype=np.float32)}
        )
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def _predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: List[Dict[str, Any]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> th.Tensor:
        return self.actor.custom_predict(observation, state, deterministic=deterministic)

    def custom_predict(
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
        return self.actor.custom_predict(observation, state, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode
