"""
    parameters.py

    Summary:
        Contains a dataclasses for parameters used by the RL(N)MPC.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class RLMPCParams:
    reference_traj_bbox_buffer: float = 500.0
    T: float = 10.0
    dt: float = 0.5
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R: np.ndarray = np.diag([1.0, 1.0, 1.0])
    gamma: float = 0.0
    d_safe_so: float = 5.0
    d_safe_do: float = 5.0
    spline_reference: bool = False
    path_following: bool = False

    @classmethod
    def from_dict(self, config_dict: dict):
        params = RLMPCParams(**config_dict)
        if params.path_following and params.Q.shape[0] != 2:
            raise ValueError("Q must be a 2x2 matrix when path_following is True.")

        if not params.path_following and params.Q.shape[0] != 6:
            raise ValueError("Q must be a 6x6 matrix when path_following is False (trajectory tracking).")

        return params

    def to_dict(self) -> dict:
        return asdict(self)
