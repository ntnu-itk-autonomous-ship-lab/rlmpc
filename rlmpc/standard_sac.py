"""
    sac.py

    Summary:
        Standard Soft Actor-Critic (SAC) implementation using only neural networks.


    Author: Trym Tengesdal
"""

import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import colav_simulator.gym.environment as csgym_env
import numpy as np
import rlmpc.common.buffers as rlmpc_buffers
import rlmpc.off_policy_algorithm as opa
import rlmpc.policies as rlmpc_policies
import stable_baselines3.common.noise as sb3_noise
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, set_random_seed, update_learning_rate
from stable_baselines3.sac.policies import ContinuousCritic
from torch.nn import functional as F

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class SAC(opa.OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)

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
        - policy (rlmpc_policies.SACPolicyWithMPC): The MPC policy model to use (rlmpc_policies.SACPolicyWithMPC)
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
        - data_path (Optional[str]): the path to save the replay buffer data
        - policy_kwargs (Optional[Dict[str, Any]]): additional arguments to be passed to the policy on creation
        - verbose (int): Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        - seed (Optional[int]): Seed for the pseudo random generators
        - device (Union[th.device, str]): Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        - _init_setup_model (bool): Whether or not to build the network at the creation of the instance
    """

    policy: rlmpc_policies.SACPolicyWithMPCParameterProviderStandard
    actor: rlmpc_policies.SACMPCParameterProviderActorStandard
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Type[rlmpc_policies.SACPolicyWithMPC | rlmpc_policies.SACMPCParameterProviderActor],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 64,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (100, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[sb3_noise.ActionNoise] = None,
        replay_buffer_class: Optional[Type[rlmpc_buffers.ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        data_path: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if policy_kwargs is None:
            policy_kwargs = {"features_extractor_kwargs": {"batch_size": batch_size}}
        else:
            policy_kwargs.update({"features_extractor_kwargs": {"batch_size": batch_size}})

        if isinstance(policy, rlmpc_policies.SACPolicyWithMPC):
            observation_type = env.unwrapped.observation_type
            action_type = env.unwrapped.action_type
            policy_kwargs.update(
                {
                    "observation_type": observation_type,
                    "action_type": action_type,
                }
            )
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
            data_path=data_path,
            verbose=verbose,
            device=device,
            seed=seed,
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
            init_value = 0.01
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

    def load_critics(self, path: pathlib.Path, critic_arch: Optional[List[int]] = None) -> None:
        """Loads the model critics.

        Args:
            - path (pathlib.Path): The path to the saved critics (2 files), includes the base model name.
            - critic_arch (Optional[List[int]]): Optional list of integers to rebuild the critic with.
        """
        if critic_arch:
            self.policy.rebuild_critic_and_actor(critic_arch=critic_arch)
            self.policy_kwargs["critic_arch"] = critic_arch
            self._create_aliases()

        self.critic.load_state_dict(th.load(pathlib.Path(str(path) + "_critic.pth")))
        self.critic_target.load_state_dict(th.load(pathlib.Path(str(path) + "_critic_target.pth")))
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def save_critics(self, path: pathlib.Path) -> None:
        """Saves only the SAC critics.

        Args:
            path (pathlib.Path): The path to save the critics (2 files), includes the base model name.
        """
        th.save(self.critic.state_dict(), pathlib.Path(str(path) + "_critic.pth"))
        th.save(self.critic_target.state_dict(), pathlib.Path(str(path) + "_critic_target.pth"))

    def custom_save(self, path: pathlib.Path) -> None:
        """Saves the model parameters (NN critic and MPC actor)

        Args:
            - path (pathlib.Path): The path to save the model, includes the base model name.
        """
        self.save_critics(path=path)
        th.save(self.log_ent_coef, pathlib.Path(str(path) + "_log_ent_coef.pth"))
        # self.actor.mpc.save_params(pathlib.Path(str(path) + "_actor.yaml"))
        th.save(self.actor.mpc_param_provider.state_dict(), pathlib.Path(str(path) + "_mpc_param_provider.pth"))

    def inplace_load(self, path: pathlib.Path) -> None:
        """Loads the model parameters (NN critic and MPC actor)

        Args:
            - path (pathlib.Path): The path to the saved model, includes the base model name.
        """
        self.load_critics(path=path)
        self.log_ent_coef = th.load(pathlib.Path(str(path) + "_log_ent_coef.pth"))
        # self.actor.mpc.load_params(pathlib.Path(str(path) + "_actor.yaml"))
        # self.actor.mpc_param_provider.load_state_dict(th.load(pathlib.Path(str(path) + "_mpc_param_provider.pth")))

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def extract_mpc_param_provider_inputs(self, replay_data: rlmpc_buffers.ReplayBufferSamples) -> th.Tensor:
        actor_dnn_feature_inputs = th.from_numpy(
            np.array([info["actor_info"]["dnn_input_features"] for info in replay_data.infos], dtype=np.float32)
        )
        dnn_input = actor_dnn_feature_inputs
        return dnn_input

    def train(self, gradient_steps: int, batch_size: int = 64, disable_log: bool = False) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        batch_start_time = time.time()
        th.autograd.set_detect_anomaly(True)

        self.policy.set_training_mode(True)
        optimizers = [self.critic.optimizer, self.actor.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        actor_grad_norms = []
        mpc_param_grad_norms = []
        mean_actor_loss = 0.0
        mean_actor_grad_norm = 0.0
        mean_mpc_param_grad_norm = 0.0
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Action by the current actor for the sampled state
            # reparameterization trick
            sampled_actions, sampled_log_prob = self.actor.action_log_prob(replay_data.observations)
            sampled_log_prob = sampled_log_prob.reshape(-1, 1)
            sampled_log_prob = th.clamp(sampled_log_prob, min=-30.0, max=1e12)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (sampled_log_prob + self.target_entropy).detach()).mean()
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

            critic_loss = self.train_critics(replay_data=replay_data, gradient_step=gradient_step, ent_coef=ent_coef)
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            actor_loss = self.train_actor(
                replay_data=replay_data,
                sampled_actions=sampled_actions,
                sampled_log_prob=sampled_log_prob,
                ent_coef=ent_coef,
            )
            actor_losses.append(actor_loss)
            actor_grad_norms.append(0.0)
            mpc_param_grad_norms.append(0.0)

        mean_actor_loss = th.mean(th.tensor(actor_losses)).item()
        mean_actor_grad_norm = np.mean(actor_grad_norms)
        mean_mpc_param_grad_norm = np.mean(mpc_param_grad_norms)

        self._n_updates += gradient_steps

        print(
            f"[TRAINING] Updates: {self._n_updates} | Timesteps: {self.num_timesteps} | Actor Loss: {mean_actor_loss:.4f} | Actor Grad Norm: {mean_actor_grad_norm:.8f} | MPC Param Grad Norm: {mean_mpc_param_grad_norm:.8f} | Critic Loss: {np.mean(critic_losses):.4f} | Ent Coeff Loss: {np.mean(ent_coef_losses):.4f} | Ent Coeff: {np.mean(ent_coefs):.4f} | Batch processing time: {time.time() - batch_start_time:.2f}s"
        )

        self.last_training_info.update(
            {
                "actor_loss": mean_actor_loss,
                "actor_grad_norm": mean_actor_grad_norm,
                "critic_loss": np.mean(critic_losses),
                "ent_coef_loss": np.mean(ent_coef_losses),
                "ent_coef": np.mean(ent_coefs),
                "batch_processing_time": time.time() - batch_start_time,
                "time_elapsed": max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon),
                "n_updates": self._n_updates,
                "non_optimal_solution_rate": self.non_optimal_solutions_per_episode.mean(),
            }
        )

        if not disable_log:
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", mean_actor_loss)
            self.logger.record("train/actor_grad_norm", mean_actor_grad_norm)
            self.logger.record("train/mpc_param_grad_norm", mean_mpc_param_grad_norm)
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            self.logger.record("train/batch_processing_time", time.time() - batch_start_time)
            self.logger.record(
                "train/time_elapsed", max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            )

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = True,
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

    def train_critics(
        self, replay_data: rlmpc_buffers.DictReplayBufferSamples, gradient_step: int, ent_coef: th.Tensor = th.ones(1)
    ) -> th.Tensor:
        """Single-step trains the SAC critics using the input data.

        Args:
            replay_data (DictReplayBufferSamples):
            gradient_step (int): The current gradient update step, used for determining target network updates.
            ent_coef (th.Tensor): The current entropy/temperature coefficient.

        Returns:
            th.Tensor: The resulting critic loss
        """
        with th.no_grad():
            next_log_prob = self.actor.action_dist.log_prob(replay_data.next_actions)

            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, replay_data.next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

            # add entropy term
            # low action probability gives very high negative entropy (log_prob) -> dominates the Q value
            # leads to insanely high critic loss
            next_log_prob = th.clamp(next_log_prob, min=-30.0, max=1e12)
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

            # td error + entropy term
            target_q_values = replay_data.rewards + (1.0 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)  # for type checker

        if critic_loss.item() < 5e4:  # avoid exploding gradients
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

        if gradient_step % self.target_update_interval == 0:
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        return critic_loss

    def train_actor(
        self,
        replay_data: rlmpc_buffers.DictReplayBufferSamples,
        sampled_actions: th.Tensor,
        sampled_log_prob: th.Tensor,
        ent_coef: th.Tensor = th.ones(1),
    ) -> th.Tensor:
        """Single-step trains the SAC actor.

        Args:
            replay_data (DictReplayBufferSamples): Batched replay data.
            sampled_actions (th.Tensor): Sampled (normalized) actions given the replay data and the MPC policy ad hoc distribution.
            sampled_log_prob (th.Tensor): Log probabilities of the sampled actions.
            ent_coef (th.Tensor, optional): The current entropy/temperature coefficient.

        Returns:
           th.Tensor: The actor loss, actor gradient and mpc param provider dnn gradient norm.
        """
        q_values_pi_sampled = th.cat(self.critic(replay_data.observations, sampled_actions), dim=1)
        min_qf_pi_sampled, _ = th.min(q_values_pi_sampled, dim=1, keepdim=True)

        actor_loss = (ent_coef * sampled_log_prob - min_qf_pi_sampled).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        return actor_loss

    def extract_action_info_from_sarsa_buffer(
        self, replay_data: rlmpc_buffers.DictReplayBufferSamples
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Extracts the actions, next actions and next log probabilities from the replay buffer, depending on the actor type.

        Args:
            replay_data (rlmpc_buffers.DictReplayBufferSamples): DictReplayBufferSamples object.

        Returns:
            Tuple[th.Tensor, th.Tensor, th.Tensor]: The actions, next actions and next log probabilities.
        """
        next_actions = replay_data.next_actions
        actions = replay_data.actions
        _, next_log_prob = self.actor.action_log_prob(
            replay_data.next_observations,
        )
        return actions, next_actions, next_log_prob
