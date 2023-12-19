"""
    replay_buffer.py

    Summary:
        Contains functionality for storing and fetching data for RL agent training.


    Author: Trym Tengesdal
"""
import random

import numpy as np
import rl_rrt_mpc.off_policy_algorithm as opa
import stable_baselines3.common.buffers as sb3_buffers
import stable_baselines3.sac.policies as sb3_sac_policies
import torch
import torch.nn.functional as F
import torch.optim as optim


class SAC(opa.OffPolicyAlgorithm):
    """Class for Soft Actor Critic algorithm"""

    def __init__(self) -> None:
        super().__init__()

    def train(n_timesteps: int, buffer: sb3_buffers.ReplayBuffer) -> None:
        pass

    def sample_action() -> np.ndarray:
        pass
