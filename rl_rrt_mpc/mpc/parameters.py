"""
    parameters.py

    Summary:
        Contains a dataclasses for a parameter interface and parameters used by the RL(N)MPC mid-level COLAV.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import numpy as np


class StaticObstacleConstraint(Enum):
    """Enum for the different possible static obstacle constraints

    Explanation:
        PARAMETRIC_SURFACE: Uses a surface approximation of the static obstacle CDT triangles to create a constraint.
        CIRCULAR: Uses a maximum coverage circular constraint for each static obstacle.
        ELLIPSOIDAL: Uses a maximum coverage elliptic constraint for the boundary of each static obstacle.
        APPROXCONVEXSAFESET: Uses an approximate maximum coverage convex set constraint for the own-ship to stay within.
        TRIANGULARBOUNDARY: Uses a triangular boundary constraint for each static obstacle.
    """

    PARAMETRICSURFACE = 0
    CIRCULAR = 1
    ELLIPSOIDAL = 2
    APPROXCONVEXSAFESET = 3
    TRIANGULARBOUNDARY = 4


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

    rate: float = 5.0
    reference_traj_bbox_buffer: float = 500.0
    T: float = 10.0
    dt: float = 0.5
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    gamma: float = 0.9
    d_safe_so: float = 5.0
    d_safe_do: float = 5.0
    spline_reference: bool = False
    so_constr_type: StaticObstacleConstraint = StaticObstacleConstraint.APPROXCONVEXSAFESET
    max_num_so_constr: int = 5
    max_num_do_constr: int = 0
    path_following: bool = False
    debug: bool = False

    @classmethod
    def from_dict(self, config_dict: dict):
        params = RLMPCParams(**config_dict)
        params.so_constr_type = StaticObstacleConstraint[config_dict["so_constr_type"]]
        params.Q = np.diag(params.Q)
        if params.path_following and params.Q.shape[0] != 2:
            raise ValueError("Q must be a 2x2 matrix when path_following is True.")

        if not params.path_following and params.Q.shape[0] != 6:
            raise ValueError("Q must be a 6x6 matrix when path_following is False (trajectory tracking).")

        return params

    def to_dict(self) -> dict:
        config_dict = asdict(self)
        config_dict["so_constr_type"] = self.so_constr_type.name
        config_dict["Q"] = self.Q.diagonal().tolist()
        return config_dict

    @property
    def adjustable(self) -> np.ndarray:
        """Returns an array of the adjustable parameters by the RL scheme.

        Returns:
            np.ndarray: Array of adjustable parameters.
        """
        return np.array([*self.Q.flatten().tolist(), self.d_safe_so, self.d_safe_do])
