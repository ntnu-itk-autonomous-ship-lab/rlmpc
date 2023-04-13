"""
    parameters.py

    Summary:
        Contains a dataclasses for a parameter interface and parameters used by the RL(N)MPC mid-level COLAV.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class IParams(ABC):
    @classmethod
    @abstractmethod
    def from_dict(self, config_dict: dict):
        """Creates a parameters object from a dictionary."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Converts the parameters to a dictionary."""

    @abstractmethod
    def adjustable(self) -> list:
        """Returns a list of the adjustable parameters by an RL scheme."""


@dataclass
class RLMPCParams(IParams):
    """Class for parameters used by the RL(N)MPC mid-level COLAV. Can be used as regular (N)MPC COLAV by setting gamma to 1.0."""

    reference_traj_bbox_buffer: float = 500.0
    T: float = 10.0
    dt: float = 0.5
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    gamma: float = 0.9
    d_safe_so: float = 5.0
    d_safe_do: float = 5.0
    spline_reference: bool = False
    path_following: bool = False

    @classmethod
    def from_dict(self, config_dict: dict):
        params = RLMPCParams(**config_dict)
        params.Q = np.diag(params.Q)
        if params.path_following and params.Q.shape[0] != 2:
            raise ValueError("Q must be a 2x2 matrix when path_following is True.")

        if not params.path_following and params.Q.shape[0] != 6:
            raise ValueError("Q must be a 6x6 matrix when path_following is False (trajectory tracking).")

        return params

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def adjustable(self) -> list:
        """Returns a list of the adjustable parameters by the RL scheme.

        Returns:
            list: List of adjustable parameters.
        """
        return [self.Q.flatten().tolist(), self.d_safe_so, self.d_safe_do]
