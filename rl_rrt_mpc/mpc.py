"""
    mpc.py

    Summary:
        Contains a class for an MPC trajectory tracking controller.

    Author: Trym Tengesdal
"""
from dataclasses import dataclass
from typing import Optional

import casadi as csd
import numpy as np
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions


@dataclass
class MPCParams:
    T: float = 10.0
    dt: float = 0.5
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R: np.ndarray = np.diag([1.0, 1.0, 1.0])
    gamma: float = 0.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = MPCParams(T=config_dict["T"], dt=config_dict["dt"], Q=np.zeros(9), R=np.zeros(9), gamma=config_dict["gamma"])
        config.Q = np.diag(config_dict["Q"])
        config.R = np.diag(config_dict["R"])
        return config

    def to_dict(self):
        return {
            "T": self.T,
            "dt": self.dt,
            "Q": self.Q.diagonal(),
            "R": self.R.diagonal(),
            "gamma": self.gamma,
        }


class MPC:
    def __init__(self, config: Optional[MPCParams] = MPCParams()) -> None:
        self._params = config
        self._ocp_options: AcadosOcpOptions = AcadosOcpOptions()
        # self._model: AcadosModel = models.ShipModel().to_acados()

    def plan(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        pass
