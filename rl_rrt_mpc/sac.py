"""
    replay_buffer.py

    Summary:
        Contains functionality for storing and fetching data for RL agent training.


    Author: Trym Tengesdal
"""
import random

import numpy as np
import rl_rrt_mpc.replay_buffer as rb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class SAC:
    """Class for Soft Actor Critic algorithm"""

    def __init__(self) -> None:
        pass

    def train(n_timesteps: int, buffer: rb.ReplayBuffer) -> None:
        pass

    def sample_action() -> np.ndarray:
        pass
