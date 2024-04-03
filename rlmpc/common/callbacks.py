"""
    callbacks.py

    Summary:
        Contains callback classes for the RL training process.

    Author: Trym Tengesdal
"""

from pathlib import Path

import numpy as np
from colav_simulator.gym.environment import COLAVEnvironment
from stable_baselines3.common.callbacks import BaseCallback


class CollectStatisticsCallback(BaseCallback):
    def __init__(
        self,
        env: COLAVEnvironment,
        total_timesteps: int,
        save_stats_freq: int,
        record_agent_freq: int,
        log_dir: Path,
        verbose=1,
    ):
        super(CollectStatisticsCallback, self).__init__(verbose)
        self.save_stats_freq = save_stats_freq
        self.record_agent_freq = record_agent_freq
        self.save_agent_freq = total_timesteps // 100
        self.log_dir = log_dir
        self.model_save_path = log_dir / "models"
        self.report_save_path = log_dir / "reports"
        self.n_episodes = 0
        self.vec_env = env

        self.report = self.vec_env.get_attr("history")[0]
        for stat in self.report.keys():
            self.report[stat] = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.model_save_path is not None:
            self.model_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Checking for both 'done' and 'dones' keywords because:
        # Some models use keyword 'done' (e.g.,: SAC, TD3, DQN, DDPG)
        # While some models use keyword 'dones' (e.g.,: A2C, PPO)
        done_array = np.array(
            self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones")
        )

        if np.sum(done_array).item() > 0:
            self.n_episodes += np.sum(done_array).item()
            self.logger.record("time/episodes", self.n_episodes)

            # Fetch stats from history attribute and log to tensorboard
            stats = np.array(self.vec_env.get_attr("history"))[done_array]
            for _env in stats:
                for stat in _env.keys():
                    # self.logger.record('stats/'+stat, _env[stat])
                    if len(_env[stat]) > 0:
                        self.report[stat].append(_env[stat][-1])

        if self.num_timesteps % self.save_stats_freq == 0:
            gym_auv.reporting.report(self.report, report_dir=figure_folder)

        if self.num_timesteps % self.save_agent_freq == 0:
            print("Saving agent after", self.num_timesteps, "timesteps")
            agent_filepath = os.path.join(self.log_dir, str(self.num_timesteps) + ".pkl")
            self.model.save(agent_filepath)

        return True
