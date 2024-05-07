"""
    parameters.py

    Summary:
        Contains parameter classes for MPC-based COLAV in different flavours.

    Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional

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
class MidlevelMPCParams(IParams):
    """Class for parameters used by the mid-level MPC COLAV."""

    rate: float = 5.0  # rate of the controller
    reference_traj_bbox_buffer: float = 200.0  # buffer for the reference trajectory bounding box
    T: float = 100.0  # prediction horizon
    dt: float = 1.0  # time step
    so_constr_type: StaticObstacleConstraint = StaticObstacleConstraint.PARAMETRICSURFACE
    max_num_so_constr: int = 5  # maximum number of static obstacle constraints
    max_num_do_constr_per_zone: int = 5  # maximum number of dynamic obstacle constraints

    w_L2: float = 1e4  # slack variable weight L2 norm
    w_L1: float = 1e2  # slack variable weight L1 norm
    gamma: float = 0.9  # discount factor in RL setting
    r_safe_so: float = 5.0  # safety distance radius to static obstacles

    # Adjustable
    Q_p: np.ndarray = np.diag(
        [0.1, 0.1, 1.0]
    )  # path following cost matrix, position (x, y), speed deviation and speed assignment path variable deviation.
    # R: np.ndarray = np.diag([1.0, 1.0])  # input cost matrix
    alpha_app_course: np.ndarray = np.array([112.0, 0.0006])
    alpha_app_speed: np.ndarray = np.array([8.0, 0.00025])
    K_app_course: float = 0.5  # turn rate penalty
    K_app_speed: float = 0.6  # speed deviation penalty

    alpha_cr: np.ndarray = np.array([1.0 / 500.0, 1.0 / 500.0])
    y_0_cr: float = 100.0
    alpha_ho: np.ndarray = np.array([1.0 / 500.0, 1.0 / 500.0])
    x_0_ho: float = 200.0
    alpha_ot: np.ndarray = np.array([1.0 / 500.0, 1.0 / 500.0])
    x_0_ot: float = 200.0
    y_0_ot: float = 100.0
    d_attenuation: float = 400.0  # attenuation distance for the COLREGS potential functions
    w_colregs: np.ndarray = np.array([1.0, 1.0, 1.0])  # weights for the COLREGS potential functions
    r_safe_do: float = 10.0  # safety distance radius to dynamic obstacles

    @classmethod
    def from_dict(self, config_dict: dict):
        params = MidlevelMPCParams(**config_dict)
        params.so_constr_type = StaticObstacleConstraint[config_dict["so_constr_type"]]
        params.Q_p = np.diag(params.Q_p)
        params.alpha_app_course = np.array(params.alpha_app_course)
        params.alpha_app_speed = np.array(params.alpha_app_speed)
        params.alpha_cr = np.array(params.alpha_cr)
        params.alpha_ho = np.array(params.alpha_ho)
        params.alpha_ot = np.array(params.alpha_ot)
        params.w_colregs = np.array(params.w_colregs)
        return params

    def to_dict(self) -> dict:
        config_dict = asdict(self)
        config_dict["so_constr_type"] = self.so_constr_type.name
        config_dict["Q_p"] = self.Q_p.diagonal().tolist()
        config_dict["alpha_app_course"] = self.alpha_app_course.tolist()
        config_dict["alpha_app_speed"] = self.alpha_app_speed.tolist()
        config_dict["alpha_cr"] = self.alpha_cr.tolist()
        config_dict["alpha_ho"] = self.alpha_ho.tolist()
        config_dict["alpha_ot"] = self.alpha_ot.tolist()
        config_dict["w_colregs"] = self.w_colregs.tolist()
        return config_dict

    def adjustable(self, name_list: Optional[list[str]] = None) -> np.ndarray:
        """Returns an array of the adjustable parameters by the RL scheme.

        Args:
            name_list (Optional[list[str]]): List of adjustable parameters to return. If None, all adjustable parameters are returned.

        Returns:
            np.ndarray: Array of adjustable parameters.
        """
        params = []
        if name_list is not None:
            for key in name_list:
                if key == "Q_p":
                    params.extend(self.Q_p.diagonal().tolist())
                elif key == "alpha_app_course":
                    params.extend(self.alpha_app_course.tolist())
                elif key == "alpha_app_speed":
                    params.extend(self.alpha_app_speed.tolist())
                elif key == "K_app_course":
                    params.append(self.K_app_course)
                elif key == "K_app_speed":
                    params.append(self.K_app_speed)
                elif key == "alpha_cr":
                    params.extend(self.alpha_cr.tolist())
                elif key == "y_0_cr":
                    params.append(self.y_0_cr)
                elif key == "alpha_ho":
                    params.extend(self.alpha_ho.tolist())
                elif key == "x_0_ho":
                    params.append(self.x_0_ho)
                elif key == "alpha_ot":
                    params.extend(self.alpha_ot.tolist())
                elif key == "x_0_ot":
                    params.append(self.x_0_ot)
                elif key == "y_0_ot":
                    params.append(self.y_0_ot)
                elif key == "d_attenuation":
                    params.append(self.d_attenuation)
                elif key == "w_colregs":
                    params.extend(self.w_colregs.tolist())
                elif key == "r_safe_do":
                    params.append(self.r_safe_do)
                else:
                    raise ValueError(f"Parameter {key} not in the parameter list.")
            params = np.array(params)
        else:
            params = np.array(
                [
                    *self.Q_p.diagonal().tolist(),
                    *self.alpha_app_course.tolist(),
                    *self.alpha_app_speed.tolist(),
                    self.K_app_course,
                    self.K_app_speed,
                    *self.alpha_cr.tolist(),
                    self.y_0_cr,
                    *self.alpha_ho.tolist(),
                    self.x_0_ho,
                    *self.alpha_ot.tolist(),
                    self.x_0_ot,
                    self.y_0_ot,
                    self.d_attenuation,
                    *self.w_colregs.tolist(),
                    self.r_safe_do,
                ]
            )
        return params

    def adjustable_string_list(self) -> list[str]:
        """Returns a list of adjustable parameters by the RL scheme.

        Returns:
            list[str]: List of adjustable parameters.
        """
        return [
            "Q_p",
            "alpha_app_course",
            "alpha_app_speed",
            "K_app_course",
            "K_app_speed",
            "alpha_cr",
            "y_0_cr",
            "alpha_ho",
            "x_0_ho",
            "alpha_ot",
            "x_0_ot",
            "y_0_ot",
            "d_attenuation",
            "w_colregs",
            "r_safe_do",
        ]

    def set_parameter_subset(self, param_subset: Dict[str, float | np.ndarray]) -> None:
        """Sets the adjustable parameters in the subset.

        Args:
            param_subset (Dict[str, float | np.ndarray]): Dictionary of adjustable parameters.
        """
        for key, value in param_subset.items():
            if key == "Q_p":
                self.Q_p = np.clip(np.diag(value), np.diag([1e-4, 1e-4, 1e-4]), np.diag([1e3, 1e3, 1e3]))
            elif key == "alpha_app_course":
                self.alpha_app_course = np.clip(value, np.array([1e-6, 0.000001]), np.array([1e3, 1.0]))
            elif key == "alpha_app_speed":
                self.alpha_app_speed = np.clip(value, np.array([1.0, 0.00001]), np.array([1e3, 1.0]))
            elif key == "K_app_course":
                self.K_app_course = float(np.clip(value, 0.001, 1e3))
            elif key == "K_app_speed":
                self.K_app_speed = float(np.clip(value, 0.001, 1e3))
            elif key == "alpha_cr":
                self.alpha_cr = np.clip(value, np.array([1e-6, 1e-6]), np.array([1.0, 1.0]))
            elif key == "y_0_cr":
                self.y_0_cr = float(np.clip(value, -1e4, 1e4))
            elif key == "alpha_ho":
                self.alpha_ho = np.clip(value, np.array([1e-6, 1e-6]), np.array([1.0, 1.0]))
            elif key == "x_0_ho":
                self.x_0_ho = float(np.clip(value, -1e4, 1e4))
            elif key == "alpha_ot":
                self.alpha_ot = np.clip(value, np.array([1e-6, 1e-6]), np.array([1.0, 1.0]))
            elif key == "x_0_ot":
                self.x_0_ot = float(np.clip(value, -1e4, 1e4))
            elif key == "y_0_ot":
                self.y_0_ot = float(np.clip(value, -1e4, 1e4))
            elif key == "d_attenuation":
                self.d_attenuation = float(np.clip(value, 1.0, 1e4))
            elif key == "w_colregs":
                self.w_colregs = np.clip(value, np.array([1e-4, 1e-4, 1e-4]), np.array([1e4, 1e4, 1e4]))
            elif key == "r_safe_do":
                self.r_safe_do = float(np.clip(value, 1.0, 1e4))
            else:
                raise ValueError(f"Parameter {key} not in the parameter list.")

    def set_adjustable(self, adjustable: np.ndarray) -> None:
        """Sets the adjustable parameters.

        Args:
            adjustable (np.ndarray): Array of adjustable parameters.
        """
        self.Q_p = np.clip(np.diag(adjustable[:3]), np.diag([1e-4, 1e-4, 1e-4]), np.diag([1e3, 1e3, 1e3]))
        self.alpha_app_course = np.clip(adjustable[3:5], np.array([1e-6, 0.000001]), np.array([1e3, 1.0]))
        self.alpha_app_speed = np.clip(adjustable[5:7], np.array([1.0, 0.00001]), np.array([1e3, 1.0]))
        self.K_app_course = np.clip(adjustable[7], 0.001, 1e3)
        self.K_app_speed = np.clip(adjustable[8], 0.001, 1e3)
        self.alpha_cr = np.clip(adjustable[9:11], np.array([1e-6, 1e-6]), np.array([1.0, 1.0]))
        self.y_0_cr = np.clip(adjustable[11], -1e4, 1e4)
        self.alpha_ho = np.clip(adjustable[12:14], np.array([1e-6, 1e-6]), np.array([1.0, 1.0]))
        self.x_0_ho = np.clip(adjustable[14], -1e4, 1e4)
        self.alpha_ot = np.clip(adjustable[15:17], np.array([1e-6, 1e-6]), np.array([1.0, 1.0]))
        self.x_0_ot = np.clip(adjustable[17], -1e4, 1e4)
        self.y_0_ot = np.clip(adjustable[18], -1e4, 1e4)
        self.d_attenuation = np.clip(adjustable[19], 1.0, 1e4)
        self.w_colregs = np.clip(adjustable[20:23], np.array([1e-4, 1e-4, 1e-4]), np.array([1e4, 1e4, 1e4]))
        self.r_safe_do = np.clip(adjustable[23], 1.0, 1e4)
