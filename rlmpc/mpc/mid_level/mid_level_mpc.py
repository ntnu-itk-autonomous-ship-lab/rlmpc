"""
risk_based_mpc.py

Summary:
    Contains the class for a mid-level risk-aware MPC-based COLAV planner.

Author: Trym Tengesdal
"""

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import scipy.interpolate as interp
import seacharts.enc as senc

import rlmpc.common.config_parsing as cp
import rlmpc.common.paths as dp
import rlmpc.mpc.common as common
import rlmpc.mpc.mid_level.casadi_mpc as casadi_mpc
import rlmpc.mpc.models as models
import rlmpc.mpc.parameters as mpc_parameters

uname_result = platform.uname()
import rlmpc.mpc.mid_level.acados_mpc as acados_mpc

ACADOS_COMPATIBLE = True


@dataclass
class Config:
    enable_acados: bool = False
    mpc: mpc_parameters.MidlevelMPCParams = field(
        default_factory=lambda: mpc_parameters.MidlevelMPCParams()
    )
    solver_options: common.SolverConfig = field(
        default_factory=lambda: common.SolverConfig()
    )
    model: Type[models.MPCModel] = field(
        default_factory=lambda: models.KinematicCSOGWithAccelerationAndPathtimingParams()
    )

    @classmethod
    def from_dict(self, config_dict: dict):
        if "Viknes" in config_dict["model"]:
            model = models.Viknes()
        else:
            model = models.KinematicCSOGWithAccelerationAndPathtiming(
                models.KinematicCSOGWithAccelerationAndPathtimingParams.from_dict(
                    config_dict["model"][
                        "kinematic_csog_with_acceleration_and_path_timing"
                    ]
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
        if isinstance(self.model, models.Viknes):
            model_dict = {"viknes": ""}
        else:
            model_dict = {
                "kinematic_csog_with_acceleration_and_path_timing": self.model.params().to_dict(),
            }
        output_dict = {
            "enable_acados": self.enable_acados,
            "params": self.mpc.to_dict(),
            "solver_options": self.solver_options.to_dict(),
            "model": model_dict,
        }
        return output_dict


class MidlevelMPC:
    """Class for a mid-level COLAV planner with multiple economic goals. Nonlinear obstacle constraints."""

    def __init__(
        self,
        config: Optional[Config] = None,
        config_file: Optional[Path] = dp.rlmpc_config,
        identifier: str = "mpc",
        acados_code_gen_path: str = None,
    ) -> None:
        if config:
            self._solver_options: common.SolverConfig = config.solver_options
            self._acados_enabled: bool = config.enable_acados
        else:
            config = cp.extract(Config, config_file, dp.rlmpc_config)
            self._solver_options = config.solver_options
            self._acados_enabled = config.enable_acados

        self._casadi_mpc: casadi_mpc.CasadiMPC = casadi_mpc.CasadiMPC(
            config.model, config.mpc, self._solver_options.casadi, identifier
        )
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc: acados_mpc.AcadosMPC = acados_mpc.AcadosMPC(
                config.model,
                config.mpc,
                self._solver_options.acados,
                identifier=identifier,
                acados_code_gen_path=acados_code_gen_path,
            )
        self.sens: common.NLPSensitivities = None

    @property
    def model_dims(self) -> Tuple[int, int]:
        return self._casadi_mpc.model.dims()

    @property
    def params(self) -> mpc_parameters.MidlevelMPCParams:
        return self._casadi_mpc.params

    @property
    def adjustable_params(self) -> np.ndarray:
        return self._casadi_mpc.get_adjustable_params()

    @property
    def fixed_params(self) -> np.ndarray:
        return self._casadi_mpc.get_fixed_params()

    def set_adjustable_param_str_list(self, param_str_list: list[str]) -> None:
        self._casadi_mpc.set_adjustable_param_str_list(param_str_list)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.set_adjustable_param_str_list(param_str_list)

    def set_param_subset(self, subset: Dict[str, Any]) -> None:
        self._casadi_mpc.set_param_subset(subset)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.set_param_subset(subset)

    def reset(self) -> None:
        self._casadi_mpc.reset()
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.reset()

    def compute_path_variable_info(self, xs: np.ndarray) -> Tuple[float, float]:
        """Computes the path variable and its derivative from the current state.

        Args:
            xs (np.ndarray): State of the system on the form [x, y, psi, u, v, r]^T.

        Returns:
            Tuple[float, float]: Path variable and its derivative.
        """
        return self._casadi_mpc.compute_path_variable_info(xs)

    def dims(self) -> Tuple[int, int, int, int, int]:
        """Get the input, state and slack dimensions of the (casadi) MPC model.

        Returns:
            Tuple[int, int, int, int, int]: Input, state, slack dimension (for k != 0), total number of slacks, and g func dimensions.
        """
        return (
            *self._casadi_mpc.model.dims(),
            self._casadi_mpc.ns,
            self._casadi_mpc.ns_total,
            self._casadi_mpc.dim_g,
        )

    def construct_ocp(
        self,
        nominal_path: Tuple[
            interp.BSpline,
            interp.BSpline,
            interp.PchipInterpolator,
            interp.BSpline,
            float,
        ],
        so_list: list,
        enc: senc.ENC,
        map_origin: np.ndarray = np.array([0.0, 0.0]),
        min_depth: int = 5,
        tau: Optional[float] = None,
        debug: bool = False,
    ) -> None:
        """Constructs the Optimal Control Problem (OCP) for the RL-MPC COLAV algorithm.

        Args:
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline, float]): Tuple containing the nominal path splines in x, y, heading and the speed. The last element is the path length.
            - so_list (list): List of static obstacle Polygon objects.
            - enc (senc.ENC): ENC object containing information about the ENC.
            - map_origin (np.ndarray, optional): Origin of the map. Defaults to np.array([0.0, 0.0]).
            - min_depth (int, optional): Minimum allowable depth for the vessel. Defaults to 5.
            - tau (Optional[float], optional): Barrier parameter for the primal-dual interior point formulation used in the casadi nlp. Defaults to None.
            - debug (bool, optional): Debug flag. Defaults to False.
        """
        self._casadi_mpc.construct_ocp(
            nominal_path, so_list, enc, map_origin, min_depth, tau, debug
        )
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.construct_ocp(
                nominal_path, so_list, enc, map_origin, min_depth, debug
            )

    def model_prediction(
        self, xs: np.ndarray, U: np.ndarray, N: int, p: np.ndarray = np.array([])
    ) -> np.ndarray:
        """Predicts the state trajectory of the system using the model.

        Args:
            - xs (np.ndarray): Initial state of the system.
            - U (np.ndarray): Decision variables.
            - N (int): Prediction horizon.
            - p (np.ndarray, optional): Parameters of the model. Defaults to np.array([]).

        Returns:
            - np.ndarray: Predicted state trajectory of the system.
        """
        return self._casadi_mpc.model_prediction(xs, U, N, p)

    def set_action_indices(self, action_indices: list):
        """Sets the indices of the action variables in the decision vector.

        Args:
            - action_indices (list): List of indices of the action variables in the decision vector.
        """
        self._acados_mpc.set_action_indices(action_indices)
        self._casadi_mpc.set_action_indices(action_indices)

    def build_sensitivities(
        self, tau: Optional[float] = None
    ) -> common.NLPSensitivities:
        """Builds the sensitivity of the KKT matrix function with respect to the decision variables and parameters.

        Args:
            - tau (Optional[float]): Barrier parameter for the primal-dual interior point formulation. Defaults to None.

        Returns:
            - common.NLPSensitivities: Class container of the sensitivity functions necessary for
                computing the score function  gradient in RL context.
        """
        self.sens = self._casadi_mpc.build_sensitivities(tau)
        return self.sens

    def get_antigrounding_surface_functions(self) -> list:
        """Returns the anti-grounding surface functions.

        Returns:
            - list: List of anti-grounding surface functions.
        """
        return self._casadi_mpc.get_antigrounding_surface_functions()

    def set_params(self, params: mpc_parameters.MidlevelMPCParams) -> None:
        """Sets the parameters of the mid-level MPC.

        Args:
            - params (mpc_parameters.MidlevelMPCParams): Parameters of the mid-level MPC.
        """
        self._casadi_mpc.set_params(params)
        if self._acados_enabled and ACADOS_COMPATIBLE:
            self._acados_mpc.set_params(params)

    def decision_trajectories(
        self, solution: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, X, Sigma = self._casadi_mpc.extract_trajectories(solution)
        return U, X, Sigma

    def decision_variables(
        self, U: np.ndarray, X: np.ndarray, Sigma: np.ndarray
    ) -> np.ndarray:
        return self._casadi_mpc.decision_variables(U, X, Sigma)

    def plan(
        self,
        t: float,
        xs: np.ndarray,
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        warm_start: dict,
        perturb_nlp: bool = False,
        perturb_sigma: float = 0.001,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - xs (np.ndarray): Current state on the form [x, y, psi, u, v, r]^T.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - warm_start (Optional[dict]): Warm start solution to use before the next iteration.
            - perturb_nlp (bool, optional): Perturb the NLP cost function or not. Used when using the MPC as a stochastic policy. Defaults to False.
            - perturb_sigma (float, optional): Standard deviation of the perturbation. Defaults to 0.001.
            - **kwargs: Additional keyword arguments such as an optional previous solution to use.

        Returns:
            - dict: Dictionary containing the solution info (trajectory, decision variables, etc.)
        """
        mpc_soln_csd = None
        mpc_soln_ac = None
        if self._acados_enabled:
            mpc_soln_ac = self._acados_mpc.plan(
                t,
                xs,
                do_cr_list,
                do_ho_list,
                do_ot_list,
                warm_start,
                perturb_nlp=perturb_nlp,
                perturb_sigma=perturb_sigma,
                verbose=verbose,
                **kwargs,
            )
            mpc_soln_ac["soln"]["x"] = self._casadi_mpc.decision_variables(
                mpc_soln_ac["inputs"], mpc_soln_ac["trajectory"], mpc_soln_ac["slacks"]
            ).full()
        else:
            mpc_soln_csd = self._casadi_mpc.plan(
                t,
                xs,
                do_cr_list,
                do_ho_list,
                do_ot_list,
                warm_start,
                perturb_nlp=perturb_nlp,
                perturb_sigma=perturb_sigma,
                verbose=verbose,
                **kwargs,
            )
        mpc_soln = mpc_soln_ac if self._acados_enabled else mpc_soln_csd

        # if t < 2.0:
        #     self._check_optimality_conditions(mpc_soln_ac, mpc_soln_csd)

        return mpc_soln

    def _check_optimality_conditions(
        self, mpc_soln_ac: Optional[dict] = None, mpc_soln_csd: Optional[dict] = None
    ) -> None:
        """Check the optimality conditions for the acados and casadi solvers.

        Args:
            mpc_soln_ac (dict): Solution dictionary from the acados solver.
            mpc_soln_csd (dict): Solution dictionary from the casadi solver.
        """
        if mpc_soln_ac is None and mpc_soln_csd is None:
            return

        if mpc_soln_ac is not None:
            lam_g_ac = (
                mpc_soln_ac["soln"]["lam_g"].flatten()
                if self._acados_enabled
                else np.array([])
            )
            w_ac = (
                mpc_soln_ac["soln"]["x"].flatten()
                if self._acados_enabled
                else np.array([])
            )
            z_ac = np.concatenate(
                (
                    mpc_soln_ac["soln"]["x"].flatten(),
                    mpc_soln_ac["soln"]["lam_g"].flatten(),
                )
            )
            R_kkt_ac = self.sens.r_kkt(
                z_ac, mpc_soln_ac["p_fixed"], mpc_soln_ac["p"]
            ).full()
            dlag_dw_ac = self.sens.dlag_dw(
                mpc_soln_ac["soln"]["x"],
                lam_g_ac,
                mpc_soln_ac["p_fixed"],
                mpc_soln_ac["p"],
            ).full()

            eq_constr_ac = self.sens.G(
                w_ac, mpc_soln_ac["p_fixed"], mpc_soln_ac["p"]
            ).full()
            eq_jac_ac = self.sens.G_jac(
                w_ac, mpc_soln_ac["p_fixed"], mpc_soln_ac["p"]
            ).full()
            ineq_constr_ac = self.sens.H(
                w_ac, mpc_soln_ac["p_fixed"], mpc_soln_ac["p"]
            ).full()
            ineq_jac_ac = self.sens.H_jac(
                w_ac, mpc_soln_ac["p_fixed"], mpc_soln_ac["p"]
            ).full()
            n_eq_constr = eq_jac_ac.shape[0]
            active_constraints_ac = np.where(
                np.abs(np.concatenate((eq_constr_ac, ineq_constr_ac))) < 1e-4
            )[0]
            active_constr_jac_ac = np.concatenate((eq_jac_ac, ineq_jac_ac), axis=0)[
                active_constraints_ac, :
            ]
            rank_active_constr_jac_ac = np.linalg.matrix_rank(active_constr_jac_ac)
            max_rank_ac = np.min(active_constr_jac_ac.shape)
        else:
            dlag_dw_ac = np.array([0.0])
            eq_constr_ac = np.array([0.0])
            ineq_constr_ac = np.array([0.0])
            lam_g_ac = np.array([0.0])
            n_eq_constr = 0
            rank_active_constr_jac_ac = 0
            max_rank_ac = 0

        if mpc_soln_csd is not None:
            lam_g_csd = mpc_soln_csd["soln"]["lam_g"].flatten()
            lam_g_diff = lam_g_ac - lam_g_csd
            w_csd = mpc_soln_csd["soln"]["x"].flatten()
            w_diff = w_ac - w_csd
            z_csd = np.concatenate(
                (
                    mpc_soln_csd["soln"]["x"].flatten(),
                    mpc_soln_csd["soln"]["lam_g"].flatten(),
                )
            )
            R_kkt_csd = self.sens.r_kkt(
                z_csd, mpc_soln_csd["p_fixed"], mpc_soln_csd["p"]
            ).full()
            dlag_dw_csd = self.sens.dlag_dw(
                mpc_soln_csd["soln"]["x"],
                lam_g_csd,
                mpc_soln_csd["p_fixed"],
                mpc_soln_csd["p"],
            ).full()

            eq_constr_csd = self.sens.G(
                w_csd, mpc_soln_csd["p_fixed"], mpc_soln_csd["p"]
            ).full()
            eq_jac_csd = self.sens.G_jac(
                w_csd, mpc_soln_csd["p_fixed"], mpc_soln_csd["p"]
            ).full()
            ineq_constr_csd = self.sens.H(
                w_csd, mpc_soln_csd["p_fixed"], mpc_soln_csd["p"]
            ).full()
            ineq_jac_csd = self.sens.H_jac(
                w_csd, mpc_soln_csd["p_fixed"], mpc_soln_csd["p"]
            ).full()

            active_constraints_csd = np.where(
                np.abs(np.concatenate((eq_constr_csd, ineq_constr_csd))) < 1e-4
            )[0]
            active_constr_jac_csd = np.concatenate((eq_jac_csd, ineq_jac_csd), axis=0)[
                active_constraints_csd, :
            ]
            rank_active_constr_jac_csd = np.linalg.matrix_rank(active_constr_jac_csd)
            max_rank_csd = np.min(active_constr_jac_csd.shape)
        else:
            dlag_dw_csd = np.array([0.0])
            eq_constr_csd = np.array([0.0])
            ineq_constr_csd = np.array([0.0])
            lam_g_csd = np.array([0.0])
            rank_active_constr_jac_csd = 0
            max_rank_csd = 0

        # dep_rows_ac = cs_mf.find_dependent_rows(active_constr_jac_ac.copy())
        # dep_rows_csd = cs_mf.find_dependent_rows(active_constr_jac_csd.copy())
        # for dep in dep_rows_ac:
        #     print(self._casadi_mpc.g_str_list[active_constraints_ac[dep]])
        print("Checking first order necessary conditions for optimality...: ")
        print(
            "dlag_dw = 0 condition | acados: ",
            np.linalg.norm(dlag_dw_ac),
            " | casadi: ",
            np.linalg.norm(dlag_dw_csd),
        )
        print(
            f"c_i(x*) = 0 for all i in E | acados: {np.all(np.abs(eq_constr_ac) < 1e-4)} | casadi: {np.all(np.abs(eq_constr_csd) < 1e-4)}"
        )
        print(
            f"c_i(x*) <= 0 for all i in I | acados: {np.all(ineq_constr_ac < 1e-4)} | casadi: {np.all(ineq_constr_csd < 1e-4)}"
        )
        print(
            f"Lamda_i >= 0 for i in I | acados: {np.all(lam_g_ac[n_eq_constr:] >= 0)} | casadi: {np.all(lam_g_csd[n_eq_constr:] >= 0)}"
        )
        print(
            f"Lambda_i * c_i(x*) = 0 for all i in (I and E) | acados: {np.all(lam_g_ac * np.concatenate((eq_constr_ac, ineq_constr_ac)).flatten() < 1e-4)} | casadi: {np.all(lam_g_csd * np.concatenate((eq_constr_csd, ineq_constr_csd)).flatten() < 1e-4)}"
        )

        print(
            f"Active constraint jacobian rank/max_rank | acados: {rank_active_constr_jac_ac}/{max_rank_ac} | casadi: {rank_active_constr_jac_csd}/{max_rank_csd}"
        )

        # print("Checking second order necessary conditions for optimality: ")
        # hess_ac = self.sens.d2lag_d2w(
        #     w_ac, np.concatenate((mpc_soln_ac["p"], mpc_soln_ac["p_fixed"])), 1.0, lam_g_ac
        # ).full()
        # hess_csd = self.sens.d2lag_d2w(
        #     w_csd, np.concatenate((mpc_soln_csd["p"], mpc_soln_csd["p_fixed"])), 1.0, lam_g_csd
        # ).full()

        print("Hessian >= 0 on null space of equality constraints: ")
