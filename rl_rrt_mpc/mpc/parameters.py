"""
    parameters.py

    Summary:
        Contains parameter classes for MPC-based COLAV in different flavours.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np


class StaticObstacleConstraint(Enum):
    """Enum for the different possible static obstacle constraints

    Explanation:
        PARAMETRIC_SURFACE: Uses a surface approximation of the static obstacle CDT triangles to create a constraint.
        CIRCULAR: Uses a set of circular constraint types.
        ELLIPSOIDAL: Uses a set of elliptic constraint types.
        APPROXCONVEXSAFESET: Uses an approximate maximum coverage convex set constraint for the own-ship to stay within.
    """

    PARAMETRICSURFACE = 0
    CIRCULAR = 1
    ELLIPSOIDAL = 2
    APPROXCONVEXSAFESET = 3


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
    def adjustable(self) -> np.ndarray:
        """Returns an array of the adjustable parameters by an RL scheme."""


@dataclass
class TTMPCParams(IParams):
    """Class for parameters used by the lower level trajectory tracking MPC with COLAV. Can be used as regular (N)MPC COLAV by setting gamma to 1.0."""

    rate: float = 5.0  # rate of the controller
    reference_traj_bbox_buffer: float = 500.0  # buffer for the reference trajectory bounding box
    T: float = 10.0  # prediction horizon
    dt: float = 0.5  # time step
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # state cost matrix
    R: np.ndarray = np.diag([1.0, 1.0])  # input cost matrix
    w_L2: float = 1e4  # slack variable weight L2 norm
    w_L1: float = 1e2  # slack variable weight L1 norm
    gamma: float = 0.9  # discount factor in RL setting
    d_safe_so: float = 5.0  # safety distance to static obstacles
    d_safe_do: float = 5.0  # safety distance to dynamic obstacles
    so_constr_type: StaticObstacleConstraint = StaticObstacleConstraint.PARAMETRICSURFACE
    max_num_so_constr: int = 5  # maximum number of static obstacle constraints
    max_num_do_constr: int = 0  # maximum number of dynamic obstacle constraints
    path_following: bool = False  # whether to use path following or trajectory tracking
    debug: bool = False  # whether to print debug information

    @classmethod
    def from_dict(self, config_dict: dict):
        params = TTMPCParams(**config_dict)
        params.so_constr_type = StaticObstacleConstraint[config_dict["so_constr_type"]]
        params.Q = np.diag(params.Q)
        params.R = np.diag(params.R)
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

    def adjustable(self) -> np.ndarray:
        """Returns an array of the adjustable parameters by the RL scheme.

        Returns:
            np.ndarray: Array of adjustable parameters.
        """
        return np.array([*self.Q.flatten().tolist(), *self.R.flatten().tolist(), self.d_safe_so, self.d_safe_do])


@dataclass
class RiskBasedMPCParams(IParams):
    """Class for parameters used by the mid-level risk-aware Riskbased-MPC."""

    rate: float = 5.0  # rate of the controller
    reference_traj_bbox_buffer: float = 200.0  # buffer for the reference trajectory bounding box
    T: float = 100.0  # prediction horizon
    dt: float = 1.0  # time step
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0])  # path following cost matrix, position (x, y) and path timing (s, sdot)
    K_speed: float = 1.0  # speed deviation penalty
    K_fuel: float = 1.0  # fuel penalty
    w_L2: float = 1e4  # slack variable weight L2 norm
    w_L1: float = 1e2  # slack variable weight L1 norm
    gamma: float = 0.9  # discount factor in RL setting
    d_safe_so: float = 5.0  # safety distance to static obstacles
    d_safe_do: float = 5.0  # safety distance to dynamic obstacles
    so_constr_type: StaticObstacleConstraint = StaticObstacleConstraint.PARAMETRICSURFACE
    max_num_so_constr: int = 5  # maximum number of static obstacle constraints
    max_num_do_constr: int = 0  # maximum number of dynamic obstacle constraints
    debug: bool = False  # whether to print debug information

    @classmethod
    def from_dict(self, config_dict: dict):
        params = RiskBasedMPCParams(**config_dict)
        params.so_constr_type = StaticObstacleConstraint[config_dict["so_constr_type"]]
        params.Q = np.diag(params.Q)
        params.R = np.diag(params.R)
        return params

    def to_dict(self) -> dict:
        config_dict = asdict(self)
        config_dict["so_constr_type"] = self.so_constr_type.name
        config_dict["Q"] = self.Q.diagonal().tolist()
        return config_dict

    def adjustable(self) -> np.ndarray:
        """Returns an array of the adjustable parameters by the RL scheme.

        Returns:
            np.ndarray: Array of adjustable parameters.
        """
        return np.array([*self.Q.flatten().tolist(), *self.R.flatten().tolist(), self.d_safe_so, self.d_safe_do])
