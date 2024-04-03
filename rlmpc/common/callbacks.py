"""
    callbacks.py

    Summary:
        Contains callback classes for the RL training process.

    Author: Trym Tengesdal
"""

from pathlib import Path

import numpy as np
from colav_simulator.gym.environment import COLAVEnvironment
from colav_simulator.gym.logger import Logger as COLAVEnvironmentLogger
from stable_baselines3.common.callbacks import BaseCallback


class CollectStatisticsCallback(BaseCallback):
    def __init__(
        self,
        env: COLAVEnvironment,
        total_timesteps: int,
        log_dir: Path,
        experiment_name: str,
        save_stats_freq: int = 1000,
        save_agent_model_freq: int = 100,
        log_stats_freq: int = 100,
        verbose: int = 1,
    ):
        """Initializes the CollectStatisticsCallback class.

        Args:
            env (COLAVEnvironment): The environment to collect statistics from.
            total_timesteps (int): The total number of timesteps to train the agent for.
            save_stats_freq (int): Frequency to save statistics.
            record_agent_freq (int): Frequency to record the agent.
            log_dir (Path): The directory to save the models, environment datalogs++ to.
            experiment_name (str): Name of the experiment.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(CollectStatisticsCallback, self).__init__(verbose)
        self.experiment_name = experiment_name
        self.save_stats_freq = save_stats_freq
        self.save_agent_freq = save_agent_model_freq
        self.log_stats_freq = log_stats_freq
        self.save_agent_model_freq = total_timesteps // 100
        self.log_dir = log_dir
        self.model_save_path = log_dir / "models"
        self.envdata_save_path = log_dir / "envdata"
        self.n_episodes = 0
        self.vec_env = env

        self.envdata_logger: COLAVEnvironmentLogger = COLAVEnvironmentLogger(log_dir, experiment_name)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.model_save_path is not None:
            self.model_save_path.mkdir(parents=True, exist_ok=True)

        if self.envdata_save_path is not None:
            self.envdata_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Checking for both 'done' and 'dones' keywords because:
        # Some models use keyword 'done' (e.g.,: SAC, TD3, DQN, DDPG)
        # While some models use keyword 'dones' (e.g.,: A2C, PPO)
        done_array = np.array(
            self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones")
        )

        if self.num_timesteps % self.log_stats_freq == 0:
            self.envdata_logger(self.vec_env)

            self.logger.record("mpc/infeasible_solutions", self.model.actor.infeasible_solutions)
            frame = self.vec_env.render()
            if frame is not None:
                self.logger.record("env/frame", frame)

        if np.sum(done_array).item() > 0:
            self.n_episodes += np.sum(done_array).item()
            self.logger.record("time/episodes", self.n_episodes)

        if self.num_timesteps % self.save_stats_freq == 0:
            self.logger.dump()

        if self.num_timesteps % self.save_agent_model_freq == 0:
            print("Saving agent after", self.num_timesteps, "timesteps")
            assert hasattr(
                self.model, "custom_save"
            ), "Model must have a custom_save method, i.e. be a custom SAC model with an MPC actor."
            self.model.custom_save(self.model_save_path / f"model_{self.experiment_name}_{self.num_timesteps}")

        return True
