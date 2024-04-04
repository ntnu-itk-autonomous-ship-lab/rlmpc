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

import colav_simulator.core.stochasticity as stoch
import numpy as np
import rlmpc.common.config_parsing as cp
import rlmpc.common.paths as dp
import rlmpc.mpc.common as common
import rlmpc.mpc.mid_level.casadi_mpc as casadi_mpc
import rlmpc.mpc.models as models
import rlmpc.mpc.parameters as mpc_parameters
import scipy.interpolate as interp
import seacharts.enc as senc

uname_result = platform.uname()
if uname_result.machine == "arm64" and uname_result.system == "Darwin":
    ACADOS_COMPATIBLE = False  # ACADOS does not support arm64 and macOS yet
else:
    import rlmpc.mpc.mid_level.acados_mpc as acados_mpc

    ACADOS_COMPATIBLE = True


@dataclass
class Config:
    enable_acados: bool = False
    mpc: mpc_parameters.MidlevelMPCParams = mpc_parameters.MidlevelMPCParams()
    solver_options: common.SolverConfig = common.SolverConfig()
    model: Type[models.MPCModel] = models.KinematicCSOGWithAccelerationAndPathtimingParams()

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

    def to_dict(self):
        return {
            "enable_acados": self.enable_acados,
            "params": self.mpc.to_dict(),
            "solver_options": self.solver_options.to_dict(),
            "model": self.model.params().to_dict(),
        }


class MidlevelMPC:
    """Class for a mid-level COLAV planner with multiple economic goals. Nonlinear obstacle constraints."""

    def __init__(self, config: Optional[Config] = None, config_file: Optional[Path] = dp.rlmpc_config) -> None:
        if config:
            self._params = config.mpc
            self._solver_options: common.SolverConfig = config.solver_options
            self._acados_enabled: bool = config.enable_acados
        else:
            default_config = cp.extract(Config, config_file, dp.rlmpc_config)
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
    def model_dims(self) -> Tuple[int, int]:
        return self._casadi_mpc.model.dims()

    @property
    def params(self) -> mpc_parameters.MidlevelMPCParams:
        return self._params

    @property
    def adjustable_params(self) -> np.ndarray:
        return self._casadi_mpc.get_adjustable_params()

    @property
    def fixed_params(self) -> np.ndarray:
        return self._casadi_mpc.get_fixed_params()

    def update_adjustable_params(self, delta: np.ndarray) -> None:
        self._casadi_mpc.update_adjustable_params(delta)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.update_adjustable_params(delta)

    def construct_ocp(
        self,
        nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline, float],
        xs: np.ndarray,
        so_list: list,
        enc: senc.ENC,
        map_origin: np.ndarray = np.array([0.0, 0.0]),
        min_depth: int = 5,
        tau: Optional[float] = None,
    ) -> None:
        """Constructs the Optimal Control Problem (OCP) for the RL-MPC COLAV algorithm.

        Args:
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline, float]): Tuple containing the nominal path splines in x, y, heading and the speed. The last element is the path length.
            - xs (np.ndarray): Current state of the ownship on the form [x, y, chi, U]^T
            - so_list (list): List of static obstacle Polygon objects.
            - enc (senc.ENC): ENC object containing information about the ENC.
            - map_origin (np.ndarray, optional): Origin of the map. Defaults to np.array([0.0, 0.0]).
            - min_depth (int, optional): Minimum allowable depth for the vessel. Defaults to 5.
            - tau (Optional[float], optional): Barrier parameter for the primal-dual interior point formulation used in the casadi nlp. Defaults to None.
        """
        self._casadi_mpc.construct_ocp(nominal_path, so_list, enc, map_origin, min_depth, tau)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.construct_ocp(nominal_path, xs, so_list, enc, map_origin, min_depth)

    def set_action_indices(self, action_indices: list):
        """Sets the indices of the action variables in the decision vector.

        Args:
            - action_indices (list): List of indices of the action variables in the decision vector.
        """
        self._casadi_mpc.set_action_indices(action_indices)

    def build_sensitivities(self, tau: Optional[float] = None) -> common.NLPSensitivities:
        """Builds the sensitivity of the KKT matrix function with respect to the decision variables and parameters.

        Args:
            - tau (Optional[float]): Barrier parameter for the primal-dual interior point formulation. Defaults to None.

        Returns:
            - common.NLPSensitivities: Class container of the sensitivity functions necessary for
                computing the score function  gradient in RL context.
        """
        return self._casadi_mpc.build_sensitivities(tau)

    def get_antigrounding_surface_functions(self) -> list:
        """Returns the anti-grounding surface functions.

        Returns:
            - list: List of anti-grounding surface functions.
        """
        return self._casadi_mpc.get_antigrounding_surface_functions()

    def plan(
        self,
        t: float,
        xs: np.ndarray,
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        so_list: list,
        enc: senc.ENC,
        perturb_nlp: bool = False,
        perturb_sigma: float = 0.001,
        prev_soln: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - xs (np.ndarray): Current state on the form [x, y, psi, u, v, r]^T.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - so_list (list): List of ALL static obstacle Polygon objects.
            - enc (senc.ENC): Electronic Navigational Chart object.
            - perturb_nlp (bool, optional): Perturb the NLP cost function or not. Used when using the MPC as a stochastic policy. Defaults to False.
            - perturb_sigma (float, optional): Standard deviation of the perturbation. Defaults to 0.001.
            - prev_soln (Optional[dict], optional): Previous solution to use as warm start. Defaults to None.
            - **kwargs: Additional keyword arguments such as an optional previous solution to use.

        Returns:
            - dict: Dictionary containing the optimal trajectory, inputs, slacks, course references (X[4, :]), speed references (U_d(s)), solver stats ++
        """
        if self._acados_enabled:
            mpc_soln = self._acados_mpc.plan(
                t, xs, do_cr_list, do_ho_list, do_ot_list, so_list, enc, prev_soln, **kwargs
            )
        else:
            mpc_soln = self._casadi_mpc.plan(
                t,
                xs,
                do_cr_list,
                do_ho_list,
                do_ot_list,
                so_list,
                enc,
                perturb_nlp=perturb_nlp,
                perturb_sigma=perturb_sigma,
                prev_soln=prev_soln,
                **kwargs
            )
        return mpc_soln
