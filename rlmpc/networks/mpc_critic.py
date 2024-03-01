"""
    mpc_critic.py

    Summary:
        Continuous action space critic for the MPC agent.

    Author: Trym Tengesdal
"""

import pathlib
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import colav_simulator.core.stochasticity as stochasticity
import numpy as np
import rlmpc.common.paths as dp
import rlmpc.off_policy_algorithm as opa
import rlmpc.rlmpc as rlmpc
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


class MPCCritic