"""
    replay_buffer.py

    Summary:
        Contains functionality for storing and fetching data for RL agent training.


    Author: Trym Tengesdal
"""
import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Replay buffer class, enabling random batch sampling
    """

    def __init__(self, max_size: int, seed: int) -> None:
        """Initiate replay buffer

        Args:
            max_size (int): Maximum size of the replay buffer
            seed (int): Seed for random number generator
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.size_list = deque(maxlen=max_size)
        self.size = 0
        self.rng1 = np.random.default_rng(seed)  # Not thread safe
        self.rng2 = random.Random(seed)

    def push(self, rollout: list) -> None:
        """
        Add rollout to replay buffer

        Args:
            - rollout (list): Data from a full rollout
        """
        self.size_list.append(len(rollout))
        self.size = sum(self.size_list)
        self.buffer.append(rollout)

    def sample(self, batch_size: int):
        """
        Sample batch_size number of experiences from all the available ones

        Args:
            - batch_size (int): Number of experiences required to be sampled

        Returns:
            - tuple: Of lists containing the sampled experiences

        """
        state_batch = []
        observation_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_observation_batch = []
        done_batch = []
        info_batch = []

        for _ in range(batch_size):
            rollout = self.rng2.sample(self.buffer, 1)[0]
            (state, obs, action, reward, next_state, next_obs, done, info,) = self.rng2.sample(
                rollout, 1
            )[0]
            state_batch.append(state)
            observation_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_observation_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)
        return (
            state_batch,
            observation_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_observation_batch,
            done_batch,
            info_batch,
        )

    def sample_sequence(self, sample_len: int):
        """
        Sample a sequence of experiences with length = sample_len
        One rollout is randomly chosen. If length of this rollout < sample_len,
        then the returned sequence is also smaller

        Args:
            - sample_len (int): Length of the sequence to be sampled

        Returns:
            - tuple: Of lists containing the sampled experiences

        """
        # batch_size is taken to be the size of each episode
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        info_batch = []

        rollout = self.buffer[self.rng1.integers(0, len(self.buffer))]
        if len(rollout) >= sample_len:
            start = self.rng1.integers(0, len(rollout) - sample_len + 1)
            rollout_sample = rollout[start : start + sample_len]
        else:
            rollout_sample = self.buffer[self.rng1.integers(0, len(self.buffer))]

        for transition in rollout_sample:
            state, obs, action, reward, next_state, next_obs, done, info = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            info_batch.append(info)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            info_batch,
        )

    def last_rollout(self):
        """
        Sample the latest rollout in the replay buffer.

        Returns:
            - tuple: with x_batch : x is the appropriate variable containing all samples

        """
        # batch_size is taken to be the size of each episode
        state_batch = []
        observation_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_observation_batch = []
        done_batch = []
        info_batch = []

        rollout_sample = self.buffer[-1]

        for transition in rollout_sample:
            state, obs, action, reward, next_state, next_obs, done, info = transition
            state_batch.append(state)
            observation_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_observation_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)

        return (
            state_batch,
            observation_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_observation_batch,
            done_batch,
            info_batch,
        )

    def __len__(self) -> int:
        return len(self.buffer)
