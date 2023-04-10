"""
    rlmpc.py

    Summary:
        Contains a class for a Reinforcement Learning (RL) (N)MPC trajectory tracking/path following controller with collision avoidance functionality.


    Author: Trym Tengesdal
"""
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rl_rrt_mpc.common.config_parsing as cp
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.rl_mpc.casadi_mpc as casadi_mpc
import rl_rrt_mpc.rl_mpc.models as models
import rl_rrt_mpc.rl_mpc.parameters as parameters
import seacharts.enc as senc

uname_result = platform.uname()
if uname_result.machine == "arm64" and uname_result.system == "Darwin":
    import rl_rrt_mpc.rl_mpc.acados_mpc as acados_mpc

    ACADOS_COMPATIBLE = False  # ACADOS does not support arm64 and macOS yet
else:
    ACADOS_COMPATIBLE = True

MAX_NUM_DO_CONSTRAINTS: int = 15
MAX_NUM_SO_CONSTRAINTS: int = 300


@dataclass
class SolverConfig:
    acados: dict = {}
    casadi: casadi_mpc.CasadiSolverOptions = casadi_mpc.CasadiSolverOptions()

    @classmethod
    def from_dict(self, config_dict: dict):
        config = SolverConfig(acados=config_dict["acados"], casadi=config_dict["casadi"])
        return config


@dataclass
class Config:
    enable_acados: bool = False
    rlmpc: parameters.RLMPCParams = parameters.RLMPCParams()
    solver_options: SolverConfig = SolverConfig()

    @classmethod
    def from_dict(self, config_dict: dict):
        config = Config(
            enable_acados=config_dict["enable_acados"],
            rlmpc=parameters.RLMPCParams.from_dict(config_dict["rl_mpc"]),
            solver_options=SolverConfig.from_dict(config_dict["solver_options"]),
        )
        return config


class RLMPC:
    def __init__(self, model: models.Telemetron, config: Optional[Config] = None, config_file: Optional[Path] = dp.rl_rrt_mpc_config) -> None:
        if config:
            self._params0: parameters.RLMPCParams = config.rlmpc
            self._params: parameters.RLMPCParams = config.rlmpc
            self._solver_options: SolverConfig = config.solver_options
        else:
            config = cp.extract(Config, config_file, dp.rl_rrt_mpc_schema)
            self._params0: parameters.RLMPCParams = config.rlmpc
            self._params: parameters.RLMPCParams = config.rlmpc
            self._solver_options = config.solver_options

        self._casadi_mpc: casadi_mpc.CasadiMPC(model, config.rlmpc, config.solver_options.casadi)
        self._acados_enabled = config.enable_acados
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc = acados_mpc.AcadosMPC(model, config.rlmpc, config.solver_options.acados)

        self._initialized = False
        self._s: float = 0.0

    @property
    def params(self) -> parameters.RLMPCParams:
        return self._params

    def train(self, data) -> None:
        """Trains the RL-MPC using data (s, a, s+, a+, r+)"""

    def plan(
        self, t: float, nominal_trajectory: np.ndarray | list, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for (x, y, psi, U).
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of static obstacle Polygon objects.

        Returns:
            - Tuple[np.ndarray, np.ndarray]: Optimal trajectory and inputs for the ownship.
        """
        N = int(self._params.T / self._params.dt)
        if self._acados_enabled:
            trajectory, inputs = self._acados_mpc.plan(t, nominal_trajectory, nominal_inputs, xs, do_list, so_list)
        else:
            trajectory, inputs = self._casadi_mpc.plan(t, nominal_trajectory, nominal_inputs, xs, do_list, so_list)
        return trajectory[:, :N], inputs[:, :N]
