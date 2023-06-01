"""
    rlmpc.py

    Summary:
        Contains a class for a Reinforcement Learning (RL) (N)MPC trajectory tracking/path following controller with collision avoidance functionality.


    Author: Trym Tengesdal
"""
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Type

import numpy as np
import rl_rrt_mpc.common.config_parsing as cp
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.mpc.casadi_mpc as casadi_mpc
import rl_rrt_mpc.mpc.models as models
import rl_rrt_mpc.mpc.parameters as parameters
import rl_rrt_mpc.mpc.set_generator as sg
import seacharts.enc as senc

uname_result = platform.uname()
if uname_result.machine == "arm64" and uname_result.system == "Darwin":
    ACADOS_COMPATIBLE = False  # ACADOS does not support arm64 and macOS yet
else:
    import rl_rrt_mpc.mpc.acados_mpc as acados_mpc

    ACADOS_COMPATIBLE = True


@dataclass
class SolverConfig:
    acados: dict = field(default_factory=dict)
    casadi: casadi_mpc.CasadiSolverOptions = casadi_mpc.CasadiSolverOptions()

    @classmethod
    def from_dict(self, config_dict: dict):
        config = SolverConfig(acados=config_dict["acados"], casadi=casadi_mpc.CasadiSolverOptions.from_dict(config_dict["casadi"]))
        return config


@dataclass
class Config:
    enable_acados: bool = False
    mpc: Type[parameters.IParams] = parameters.RLMPCParams()
    solver_options: SolverConfig = SolverConfig()

    @classmethod
    def from_dict(self, config_dict: dict):
        config = Config(
            enable_acados=config_dict["enable_acados"],
            mpc=parameters.RLMPCParams.from_dict(config_dict["params"]),
            solver_options=SolverConfig.from_dict(config_dict["solver_options"]),
        )
        return config


class MPC:
    def __init__(self, model: models.Telemetron, config: Optional[Config] = None, config_file: Optional[Path] = dp.rl_rrt_mpc_config) -> None:
        if config:
            self._params0 = config.mpc
            self._params = config.mpc
            self._solver_options: SolverConfig = config.solver_options
            self._acados_enabled: bool = config.enable_acados
        else:
            default_config = cp.extract(Config, config_file, dp.rl_rrt_mpc_schema)
            self._params0 = default_config.mpc
            self._params = default_config.mpc
            self._solver_options = default_config.solver_options
            self._acados_enabled = default_config.enable_acados

        self._casadi_mpc: casadi_mpc.CasadiMPC = casadi_mpc.CasadiMPC(model, self._params, self._solver_options.casadi)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc: acados_mpc.AcadosMPC = acados_mpc.AcadosMPC(model, self._params, self._solver_options.acados)

        self._initialized: bool = False
        self._s: float = 0.0
        self._set_generator: Optional[sg.SetGenerator] = None

    @property
    def params(self) -> parameters.IParams:
        return self._params

    def action_value(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Returns the Q(s, a) function value for the given state and action."""
        return self._casadi_mpc.action_value(state, action)

    def value(self, state: np.ndarray) -> np.ndarray:
        """Returns the V(s) value function value for the given state."""
        return self._casadi_mpc.value(state)

    def train(self, data) -> None:
        """Trains the RL-MPC using data (s, a, s+, a+, r+)"""

    def construct_ocp(self, nominal_trajectory: np.ndarray, xs: np.ndarray, do_list: list, so_list: list, enc: senc.ENC) -> None:
        """Constructs the Optimal Control Problem (OCP) for the RL-MPC COLAV algorithm.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow.
            - xs (np.ndarray): Current state of the ownship.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of static obstacle Polygon objects.
            - enc (senc.ENC): ENC object containing information about the ENC.
        """
        # self._casadi_mpc.construct_ocp(so_list, enc)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.construct_ocp(nominal_trajectory, xs, do_list, so_list, enc)

    def plan(
        self,
        t: float,
        nominal_trajectory: np.ndarray,
        nominal_inputs: Optional[np.ndarray],
        xs: np.ndarray,
        do_list: list,
        so_list: list,
        enc: Optional[senc.ENC],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track (position in NED and velocity in BODY) or path to follow.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of ALL static obstacle Polygon objects.
            - enc (Optional[senc.ENC]): Electronic Navigational Chart object.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - Tuple[np.ndarray, np.ndarray]: Optimal trajectory [eta, nu] x N and inputs for the ownship.
        """
        if self._acados_enabled:
            trajectory, inputs = self._acados_mpc.plan(t, nominal_trajectory, nominal_inputs, xs, do_list, so_list, **kwargs)
        else:
            trajectory, inputs, _ = self._casadi_mpc.plan(t, nominal_trajectory, nominal_inputs, xs, do_list, so_list, enc, **kwargs)
        return trajectory, inputs
