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

    def _create_compatible_polygons(self, so_list: list, enc: senc.ENC) -> list:
        """Based on the chosen Static Obstacle constraint type, creates a list of compatible polygons.

        Args:
            so_list (list): List of all original static obstacle Polygon objects.
            enc (senc.ENC): ENC object.

        Returns:
            list: List of compatible static obstacle Polygon objects.
        """
        compatible_polygons: list = []
        if self._params.so_constraint_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
            compatible_polygons = []
        elif self._params.so_constraint_type == parameters.StaticObstacleConstraint.CIRCULAR:
            compatible_polygons = so_list
        elif self._params.so_constraint_type == parameters.StaticObstacleConstraint.ELLIPTICAL:
            compatible_polygons = so_list
        elif self._params.so_constraint_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            poly_points_x, poly_points_y = hf.create_point_list_from_polygons(so_list)
            self._set_generator = sg.SetGenerator(poly_points_x, poly_points_y)
        return compatible_polygons

    def construct_ocp(self, nominal_trajectory: np.ndarray, do_list: list, so_list: list, enc: senc.ENC) -> None:
        """Constructs the Optimal Control Problem (OCP) for the RL-MPC COLAV algorithm.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of ALL static obstacle Polygon objects.
            - enc (senc.ENC): ENC object containing information about the ENC.
        """
        compatible_polygons = self._create_compatible_polygons(so_list, enc)
        self._casadi_mpc.construct_ocp(compatible_polygons, enc)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.construct_ocp(nominal_trajectory, do_list, compatible_polygons, enc)

    def plan(self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track (in NED!) or path to follow.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of ALL static obstacle Polygon objects.

        Returns:
            - Tuple[np.ndarray, np.ndarray]: Optimal trajectory [eta, nu] x N and inputs for the ownship.
        """
        N = int(self._params.T / self._params.dt)
        # Convert velocity part of trajectory to body frame
        compatible_nominal_trajectory = nominal_trajectory.copy()
        if nominal_trajectory.shape[0] > 2:
            for k in range(nominal_trajectory.shape[1]):
                psi = compatible_nominal_trajectory[2, k]
                compatible_nominal_trajectory[3:6, k] = mf.Rpsi(psi).T @ compatible_nominal_trajectory[3:6, k]

        if self._acados_enabled:
            trajectory, inputs = self._acados_mpc.plan(compatible_nominal_trajectory, nominal_inputs, xs, do_list, so_list)
        else:
            trajectory, inputs, _ = self._casadi_mpc.plan(compatible_nominal_trajectory, nominal_inputs, xs, do_list, so_list)
        return trajectory[:, :N], inputs[:, :N]
