"""
    sac.py

    Summary:
        Soft Actor-Critic (SAC) implementation for the mid-level MPC.


    Author: Trym Tengesdal
"""
import pathlib
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import colav_simulator.core.stochasticity as stochasticity
import numpy as np
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.off_policy_algorithm as opa
import rl_rrt_mpc.rlmpc as rlmpc
import scipy.interpolate as interp
import seacharts.enc as senc
import stable_baselines3.common.buffers as sb3_buffers
import stable_baselines3.common.noise as sb3_noise
import stable_baselines3.common.vec_env as sb3_vec_env
import stable_baselines3.sac.policies as sb3_sac_policies
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor, FlattenExtractor, create_mlp
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    get_parameters_by_name,
    get_schedule_fn,
    polyak_update,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.sac.policies import Actor, BasePolicy, CnnPolicy, ContinuousCritic, MlpPolicy, MultiInputPolicy
from torch.nn import functional as F

SelfSAC = TypeVar("SelfSAC", bound="SAC")


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
        features_extractor: th.nn.Module,
        features_dim: int,
        mpc_config: rlmpc.RLMPCParams | pathlib.Path = dp.config / "rlmpc.yaml",
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3.0,
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

        action_dim = get_action_dim(self.action_space)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
        self.mu = th.nn.Linear(features_dim, action_dim)
        self.log_std_dev = th.nn.Parameter(th.ones(1, action_dim) * log_std_init)

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        # Not used, dummy
        self.net_arch = [256, 256]
        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, self.net_arch, activation_fn)
        self.latent_pi = th.nn.Sequential(*latent_pi_net)
        last_layer_dim = self.net_arch[-1] if len(self.net_arch) > 0 else features_dim

        action_dim = get_action_dim(self.action_space)
        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = th.nn.Sequential(self.mu, th.nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = th.nn.Linear(last_layer_dim, action_dim)
            self.log_std = th.nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]

        self.mu_mpc = rlmpc.RLMPC(mpc_config)
        # Consider using a stochastic Mpc with a gaussian random variable in the cost function

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
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

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        Args:
            - obs (th.Tensor): Observation

        Returns:
            - Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]: Mean, standard deviation and optional keyword arguments.
        """
        assert isinstance(self.observation_space, spaces.Dict)
        #        features = self.extract_features(obs, self.features_extractor)

        # Call mpc
        mean_actions = self.mu_mpc()

        log_std = self.log_std_dev
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

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
        """Initializes the actor MPC"""
        self.mu_mpc.initialize(
            t,
            waypoints,
            speed_plan,
            ownship_state,
            do_list,
            enc,
            goal_state,
            w,
            **kwargs,
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Computes the mean policy action (output of MPC), and samples from its stochastic
        distribution.

        Args:
            - obs (th.Tensor): Consists of the current time, own-ship state, dynamic obstacles and potentially
                disturbance data
            - deterministic (bool): Whether or not to

        Returns:
            - th.Tensor: A sampled action from the policy distribution.
        """
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # Compute mpc solution for the observation(s)
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


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
        lr_schedule: Schedule,
        critic_arch: Optional[List[int]] = [256, 256],
        mpc_config: rlmpc.RLMPCParams | pathlib.Path = dp.config / "rlmpc.yaml",
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
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
        self.activation_fn = activation_fn
        self.critic_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": critic_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "n_critics": n_critics,
            "net_arch": critic_arch,
            "share_features_extractor": False,
        }

        self.actor_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "mpc_config": mpc_config,
            "features_extractor": 0,
            "features_dim": 1,
        }
        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        # fix features_dim=0 og action_dim=1
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

    def _build_actor(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[senc.ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        w: Optional[stochasticity.DisturbanceData] = None,
    ) -> None:
        """Initialize the planner by setting up the nominal path, static obstacle inputs and constructing
        the OCP"""
        self.actor.mu_mpc.initialize(t, waypoints, speed_plan, ownship_state, do_list, enc, goal_state, w)

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

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def predict_with_mpc(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # convert observation to mpc plan inputs ()
        return self.actor.mu_mpc.plan(t, waypoints, speed_plan, ownship_state, do_list, enc)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


class SAC(opa.OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) with MPC in the actor.

    Builds on:
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    Args:
        - policy (SACPolicyWithMPC): The MPC policy model to use (SACPolicyWithMPC)
        - env (Union[GymEnv, str]): The environment to learn from (if registered in Gym, can be str)
        - learning_rate (Union[float, Schedule]): learning rate for adam optimizer,
            the same learning rate will be used for all networks (Q-Values, Actor and Value function)
            it can be a function of the current progress remaining (from 1 to 0)
        - buffer_size (int): size of the replay buffer
        - learning_starts (int): how many steps of the model to collect transitions for before learning starts
        - batch_size (int): Minibatch size for each gradient update
        - tau (float): the soft update coefficient ("Polyak update", between 0 and 1)
        - gamma (float): the discount factor
        - train_freq (Union[int, Tuple[int, str]]): Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
            like ``(5, "step")`` or ``(2, "episode")``.
        - gradient_steps (int): How many gradient steps to do after each rollout (see ``train_freq``)
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout.
        - action_noise (Optional[ActionNoise]): the action noise type (None by default), this can help
            for hard exploration problem. Cf common.noise for the different action noise type.
        - replay_buffer_class (Optional[Type[ReplayBuffer]]): Replay buffer class to use (for instance ``HerReplayBuffer``).
            If ``None``, it will be automatically selected.
        - replay_buffer_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to the replay buffer on creation.
        - optimize_memory_usage (bool): Enable a memory efficient variant of the replay buffer
            at a cost of more complexity.
        - ent_coef (Union[str, float]): Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.)
            Controlling exploration/exploitation trade-off.
            Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
        - target_update_interval (int): update the target network every ``target_network_update_freq`` gradient steps.
        - target_entropy (Union[str, float]): target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
        - use_sde (bool): Whether to use generalized State Dependent Exploration (gSDE)
            instead of action noise exploration (default: False)
        - sde_sample_freq (int): Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        - use_sde_at_warmup (bool): Whether to use gSDE instead of uniform sampling
            during the warm up phase (before learning starts)
        - stats_window_size (int): Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        - tensorboard_log (Optional[str]): the log location for tensorboard (if None, no logging)
        - policy_kwargs (Optional[Dict[str, Any]]): additional arguments to be passed to the policy on creation
        - verbose (int): Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        - seed (Optional[int]): Seed for the pseudo random generators
        - device (Union[th.device, str]): Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        - _init_setup_model (bool): Whether or not to build the network at the creation of the instance
    """

    # policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
    #     "MlpPolicy": MlpPolicy,
    #     "CnnPolicy": CnnPolicy,
    #     "MultiInputPolicy": MultiInputPolicy,
    # }
    policy: SACPolicyWithMPC
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Type[SACPolicyWithMPC],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[sb3_noise.ActionNoise] = None,
        replay_buffer_class: Optional[Type[sb3_buffers.ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor: SACMPCActor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor, using MPC-sensitivities

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
