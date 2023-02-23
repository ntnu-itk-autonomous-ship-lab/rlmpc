"""
    rl.py

    Summary:
        Contains a class for an RL based learning approach.

    Author: Trym Tengesdal
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


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
