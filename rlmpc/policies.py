"""
    policies.py

    Summary:
        Soft Actor-Critic (SAC) policies (and actors) implementation for the mid-level MPC.


    Author: Trym Tengesdal
"""

import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import colav_simulator.core.stochasticity as stochasticity
import numpy as np
import rlmpc.buffers as rlmpc_buffers
import rlmpc.common.paths as dp
import rlmpc.mpc.common as mpc_common
import rlmpc.networks.feature_extractors as rlmpc_fe
import rlmpc.off_policy_algorithm as opa
import rlmpc.rlmpc as rlmpc
import seacharts.enc as senc
import stable_baselines3.common.noise as sb3_noise
import torch as th
from colav_simulator.gym.environment import COLAVEnvironment
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import BasePolicy, ContinuousCritic

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACMPCActor(BasePolicy):
    """
    MPC-Actor (policy) for SAC.

    Args:
        - observation_space (spaces.Space): Observation space
        - action_space (spaces.Box): Action space
        - mpc_config (rlmpc.RLMPCParams | pathlib.Path): MPC configuration
        - features_extractor (th.nn.Module): Network to extract features
        - features_dim (int): Dimension of the features extracted by ``features_extractor``
        - activation_fn (Type[th.nn.Module], optional): Activation function. Defaults to nn.ReLU.
        - use_sde (bool, optional): Whether to use State Dependent Exploration or not. Defaults to False.
        - log_std_init (float, optional): Initial value for the log standard deviation. Defaults to -3.
        - full_std (bool, optional): Whether to use the full standard deviation or just the diagonal one. Defaults to True.
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
        mpc_config: rlmpc.RLMPCParams | pathlib.Path = dp.config / "rlmpc.yaml",
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        use_sde: bool = False,
        std_init: np.ndarray | float = np.array([2.0, 2.0, 0.5]),
        full_std: bool = True,
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
        self.activation_fn = activation_fn
        self.log_std_init = self.log_std
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        self.infeasible_solutions: int = 0

        action_dim = get_action_dim(self.action_space)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim).proba_distribution(
            th.zeros(action_dim), log_std=self.log_std_init
        )
        self.mpc = rlmpc.RLMPC(mpc_config)
        self.mpc_sensitivities = None
        nx, nu = self.mpc.get_mpc_model_dims()
        self.mpc_params = self.mpc.get_mpc_params()
        self.num_params = self.mpc.get_adjustable_mpc_params().size
        n_samples = int(self.mpc_params.T / self.mpc_params.dt)
        lookahead_sample = 3
        # Indices for the RLMPC action a = [x_LD, y_LD, speed_0]
        # where LD is the lookahead sample (3)
        self.action_indices = [
            nu * n_samples + lookahead_sample * nx,
            nu * n_samples + (lookahead_sample * nx + 1),
            nu * n_samples + (lookahead_sample * nx + 3),
        ]

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
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

    def get_adhoc_action_dist_params(
        self, obs: rlmpc_buffers.TensorDict, action: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        Args:
            - obs (th.Tensor): Observation
            - action (th.Tensor): Action computed by the actor for the given observation

        Returns:
            - Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]: Mean, standard deviation and optional keyword arguments.
        """
        assert isinstance(self.observation_space, spaces.Dict)

        mean_actions = action
        log_std = self.log_std
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def action_log_prob(
        self,
        obs: rlmpc_buffers.TensorDict,
        action: Optional[th.Tensor] = None,
        infos: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Computes the log probability of the policy distribution for the given observation.

        Args:
            obs (th.Tensor): Observation
            action (th.Tensor): (MPC) Action for the given observation
            infos (Optional[List[Dict[str, Any]]], optional): Additional information. Defaults to None.

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
        if action is None:
            unnorm_actions, norm_actions, actor_infos = self.predict_with_mpc(obs, state=infos, deterministic=False)
            action = norm_actions
        if isinstance(action, np.ndarray):
            action = th.from_numpy(action)

        mean_actions, log_std, kwargs = self.get_adhoc_action_dist_params(obs, action)

        # return action and associated gaussian log prob if an ad hoc stochastic policy is used
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _convert_obs_tensor_to_numpy(self, obs: Dict[str, th.Tensor]) -> Dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in obs.items()}

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
        for idx in range(batch_size):
            t, ownship_state, do_list, w = self.extract_observation_features(observation, idx)
            prev_soln = state[idx] if state is not None else None
            action, info = self.mpc.act(t=t, ownship_state=ownship_state, do_list=do_list, w=w, prev_soln=prev_soln)

            norm_action = self.action_type.normalize(action)
            if not deterministic:
                norm_action = self.action_dist.actions_from_params(th.from_numpy(norm_action), self.log_std).numpy()
            unnormalized_actions[idx, :] = self.action_type.unnormalize(norm_action)
            normalized_actions[idx, :] = norm_action

            actor_infos[idx] = info
            if not info["optimal"]:
                self.infeasible_solutions += 1

        return unnormalized_actions, normalized_actions, actor_infos

    def extract_observation_features(
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

        w = stochasticity.DisturbanceData()
        w.currents = {"speed": disturbance_vector[0], "direction": disturbance_vector[1]}
        w.wind = {"speed": disturbance_vector[2], "direction": disturbance_vector[3]}
        return t, ownship_state, do_list, w

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

        self.actor = SACMPCActor(
            **self.actor_kwargs,
            activation_fn=activation_fn,
            full_std=True,
        )

        self._build_critic(lr_schedule)

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
