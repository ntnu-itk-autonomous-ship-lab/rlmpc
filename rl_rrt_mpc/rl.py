"""
    rl.py

    Summary:
        Contains a class for an RL based learning approach.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

# import colav_simulator.gym.action as csaction
# import colav_simulator.gym.observation as csobs
# import colav_simulator.gym.reward as csreward
# import numpy as np

# if TYPE_CHECKING:
#     from colav_simulator.gym.environment import COLAVEnvironment


# @dataclass
# class GroundingRiskRewardParams:
#     """Parameters for the Grounding Risk rewarder."""

#     C_g: float = -1e6

#     def __init__(self) -> None:
#         pass

#     @classmethod
#     def from_dict(cls, config_dict: dict):
#         cfg = GroundingRiskRewardParams()
#         cfg.C_g = config_dict["C_g"]
#         return cfg

#     def to_dict(self):
#         return {"C_g": self.C_g}


# class GroundingRiskRewarder(csreward.IReward):
#     def __init__(self, env: "COLAVEnvironment", params: Optional[GroundingRiskRewardParams] = GroundingRiskRewardParams()) -> None:
#         super().__init__(env)
#         self.params = params if params is not None else GroundingRiskRewardParams()

#     def reward(self, action: csaction.Action, observation: csobs.Observation) -> float:
#         """Returns the reward for the action and observation."""
#         state, _ = observation[0], observation[1]
#         # risk model computation
#         return self.params.C_g


# @dataclass
# class FuelConsumptionRewardParams:
#     """Parameters for the Grounding Risk rewarder."""

#     fuel_baserate: float = 1.0  # [l/km]
#     fuel_cost: float = 1e3  # [NOK/l]

#     def __init__(self) -> None:
#         pass

#     @classmethod
#     def from_dict(cls, config_dict: dict):
#         return cls(**config_dict)

#     def to_dict(self):
#         return asdict(self)


# class FuelConsumptionRewarder(csreward.IReward):
#     def __init__(self, env: "COLAVEnvironment", params: Optional[FuelConsumptionRewardParams] = None) -> None:
#         super().__init__(env)
#         self.params = params if params is not None else FuelConsumptionRewardParams()

#     def fuel_rate(self, speed: float) -> float:
#         """Computes the fuel rate for the given speed, assuming a linear relationship.

#         Args:
#             speed (float): Vessel speed [m/s]

#         Returns:
#             float: Fuel rate [l/km]
#         """
#         return self.params.fuel_baserate * speed

#     def reward(self, action: csaction.Action, observation: csobs.Observation) -> float:
#         """Returns the reward for the action and observation."""
#         state, _ = observation[0], observation[1]
#         speed = float(np.linalg.norm(state[3:5]))
#         fuel_cost = self.fuel_rate(speed) * self.params.fuel_cost
#         return fuel_cost


@dataclass
class RLParams:
    learning_rate: float = 10.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RLParams(learning_rate=config_dict["learning_rate"])
        return config

    def to_dict(self):
        return {
            "learning_rate": self.learning_rate,
        }


class RL:
    def __init__(self, config: Optional[RLParams] = RLParams()) -> None:
        self._params = config

    def update(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        pass
