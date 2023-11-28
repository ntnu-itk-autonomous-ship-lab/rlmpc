"""
    risk_based_mpc.py

    Summary:
        Contains the class for a mid-level risk-aware MPC-based COLAV planner.

    Author: Trym Tengesdal
"""
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Type

import colav_simulator.core.models as cs_models
import numpy as np
import rl_rrt_mpc.common.config_parsing as cp
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.mpc.common as common
import rl_rrt_mpc.mpc.mid_level.casadi_mpc as casadi_mpc
import rl_rrt_mpc.mpc.models as models
import rl_rrt_mpc.mpc.parameters as mpc_parameters
import scipy.interpolate as interp
import seacharts.enc as senc

uname_result = platform.uname()
if uname_result.machine == "arm64" and uname_result.system == "Darwin":
    ACADOS_COMPATIBLE = False  # ACADOS does not support arm64 and macOS yet
else:
    import rl_rrt_mpc.mpc.mid_level.acados_mpc as acados_mpc

    ACADOS_COMPATIBLE = True


@dataclass
class Config:
    enable_acados: bool = False
    mpc: mpc_parameters.MidlevelMPCParams = mpc_parameters.MidlevelMPCParams()
    solver_options: common.SolverConfig = common.SolverConfig()
    model: Type[models.MPCModel] = models.KinematicCSOGWithAccelerationAndPathtimingParams()
    path_timing: models.DoubleIntegratorParams = models.DoubleIntegratorParams()

    @classmethod
    def from_dict(self, config_dict: dict):
        if "telemetron" in config_dict["model"]:
            model = models.Telemetron()
        else:
            model = models.KinematicCSOGWithAccelerationAndPathtiming(
                models.KinematicCSOGWithAccelerationAndPathtimingParams.from_dict(
                    config_dict["model"]["kinematic_csog_with_acceleration_and_path_timing"]
                )
            )

        config = Config(
            enable_acados=config_dict["enable_acados"],
            mpc=mpc_parameters.MidlevelMPCParams.from_dict(config_dict["params"]),
            solver_options=common.SolverConfig.from_dict(config_dict["solver_options"]),
            model=model,
        )
        return config


class MidlevelMPC:
    """Class for a mid-level COLAV planner with multiple economic goals. Nonlinear obstacle constraints."""

    def __init__(self, config: Optional[Config] = None, config_file: Optional[Path] = dp.rl_rrt_mpc_config) -> None:
        if config:
            self._params = config.mpc
            self._solver_options: common.SolverConfig = config.solver_options
            self._acados_enabled: bool = config.enable_acados
        else:
            default_config = cp.extract(Config, config_file, dp.rl_rrt_mpc_schema)
            self._params = default_config.mpc
            self._solver_options = default_config.solver_options
            self._acados_enabled = default_config.enable_acados

        self._casadi_mpc: casadi_mpc.CasadiMPC = casadi_mpc.CasadiMPC(
            config.model, self._params, self._solver_options.casadi
        )
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc: acados_mpc.AcadosMPC = acados_mpc.AcadosMPC(
                config.model, self._params, self._solver_options.acados
            )

    @property
    def params(self) -> mpc_parameters.MidlevelMPCParams:
        return self._params

    def action_value(self, state: np.ndarray, action: np.ndarray, parameters: np.ndarray) -> Tuple[float, dict]:
        """Returns the Q(s, a) action-value function value for the given state and action.

        Args:
            state (np.ndarray): Current state of the ownship.
            action (np.ndarray): Current action of the ownship.
            parameters (np.ndarray): Current adjustable RL-agent parameters.

        Returns:
            Tuple[float, dict]: The Q(s, a) action-value function value and the corresponding mpc solution dictionary.
        """
        return self._casadi_mpc.action_value(state, action, parameters)

    def value(self, state: np.ndarray) -> np.ndarray:
        """Returns the V(s) value function value for the given state."""
        return self._casadi_mpc.value(state)

    def train(self, data) -> None:
        """Trains the RL-MPC using data (s, a, s+, a+, r+)"""

    def construct_ocp(
        self,
        nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline],
        xs: np.ndarray,
        so_list: list,
        enc: senc.ENC,
        map_origin: np.ndarray = np.array([0.0, 0.0]),
        min_depth: int = 5,
    ) -> None:
        """Constructs the Optimal Control Problem (OCP) for the RL-MPC COLAV algorithm.

        Args:
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline]): Tuple containing the nominal path splines in x, y, heading and the speed.
            - xs (np.ndarray): Current state of the ownship on the form [x, y, chi, U]^T
            - so_list (list): List of static obstacle Polygon objects.
            - enc (senc.ENC): ENC object containing information about the ENC.
            - map_origin (np.ndarray, optional): Origin of the map. Defaults to np.array([0.0, 0.0]).
            - min_depth (int, optional): Minimum allowable depth for the vessel. Defaults to 5.
        """
        self._casadi_mpc.construct_ocp(nominal_path, so_list, enc, map_origin, min_depth)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.construct_ocp(nominal_path, xs, so_list, enc, map_origin, min_depth)

    def plan(
        self,
        t: float,
        xs: np.ndarray,
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        so_list: list,
        enc: senc.ENC,
        **kwargs
    ) -> dict:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - xs (np.ndarray): Current state on the form [x, y, chi, U]^T.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - so_list (list): List of ALL static obstacle Polygon objects.
            - enc (senc.ENC): Electronic Navigational Chart object.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - dict: Dictionary containing the optimal trajectory, inputs, slacks, course references (X[4, :]), speed references (U_d(s)), solver stats ++
        """
        if self._acados_enabled:
            mpc_soln = self._acados_mpc.plan(t, xs, do_cr_list, do_ho_list, do_ot_list, so_list, enc, **kwargs)
        else:
            mpc_soln = self._casadi_mpc.plan(t, xs, do_cr_list, do_ho_list, do_ot_list, so_list, enc, **kwargs)
        return mpc_soln
