"""
    common.py

    Summary:
        Contains common solver functions and configuration objects used by the various MPC-types.


    Author: Trym Tengesdal
"""

import pathlib
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import casadi as csd
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.common.math_functions as mf
import rlmpc.common.paths as dp
import rlmpc.mpc.models as models
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions
from acados_template.acados_ocp_solver import AcadosOcpSolver


@dataclass
class CasadiSolverOptions:
    solver_type: str = "ipopt"
    solver_tol: float = 1e-6
    print_level: int = 0
    print_time: int = 0
    mu_target: float = 1e-6
    mu_init: float = 1e-6
    acceptable_tol: float = 1e-6
    acceptable_obj_change_tol: float = 1e15
    max_iter: int = 1000
    warm_start_init_point: str = "no"
    verbose: bool = True
    jit: bool = True
    jit_flags: list = field(default_factory=lambda: ["-O0"])
    compiler: str = "clang"
    expand_mx_funcs_to_sx: bool = True
    derivative_test: str = "none"

    @classmethod
    def from_dict(cls, config_dict: dict):
        return CasadiSolverOptions(**config_dict)

    def to_dict(self):
        return asdict(self)

    def to_opt_settings(self):
        opts = {
            self.solver_type + ".tol": self.solver_tol,
            self.solver_type + ".print_level": self.print_level,
            "print_time": self.print_time,
            self.solver_type + ".mu_target": self.mu_target,
            self.solver_type + ".mu_init": self.mu_init,
            self.solver_type + ".acceptable_tol": self.acceptable_tol,
            self.solver_type + ".acceptable_obj_change_tol": self.acceptable_obj_change_tol,
            self.solver_type + ".max_iter": self.max_iter,
            self.solver_type + ".warm_start_init_point": self.warm_start_init_point,
            "verbose": self.verbose,
            "jit": self.jit,
            "jit_options": {"flags": self.jit_flags},
            "compiler": self.compiler,
            "expand": self.expand_mx_funcs_to_sx,
            self.solver_type + ".derivative_test": self.derivative_test,
        }
        return opts


class AcadosErrorCode(Enum):
    Success = 0
    Failure = 1
    MaxIter = 2
    MinStep = 3
    QPFailure = 4
    Ready = 5


def map_acados_error_code(error_code: int) -> str:
    return AcadosErrorCode(error_code).name


def parse_acados_solver_options(config_dict: dict):
    acados_solver_options = AcadosOcpOptions()
    acados_solver_options.nlp_solver_type = config_dict["nlp_solver_type"]
    acados_solver_options.nlp_solver_max_iter = config_dict["nlp_solver_max_iter"]
    acados_solver_options.nlp_solver_tol_eq = config_dict["nlp_solver_tol_eq"]
    acados_solver_options.nlp_solver_tol_ineq = config_dict["nlp_solver_tol_ineq"]
    acados_solver_options.nlp_solver_tol_comp = config_dict["nlp_solver_tol_comp"]
    acados_solver_options.nlp_solver_tol_stat = config_dict["nlp_solver_tol_stat"]
    acados_solver_options.nlp_solver_ext_qp_res = config_dict["nlp_solver_ext_qp_res"]
    acados_solver_options.qp_solver = config_dict["qp_solver_type"]
    acados_solver_options.qp_solver_iter_max = config_dict["qp_solver_iter_max"]
    acados_solver_options.qp_solver_warm_start = config_dict["qp_solver_warm_start"]
    acados_solver_options.qp_solver_tol_eq = config_dict["qp_solver_tol_eq"]
    acados_solver_options.qp_solver_tol_ineq = config_dict["qp_solver_tol_ineq"]
    acados_solver_options.qp_solver_tol_comp = config_dict["qp_solver_tol_comp"]
    acados_solver_options.qp_solver_tol_stat = config_dict["qp_solver_tol_stat"]
    acados_solver_options.hessian_approx = config_dict["hessian_approx_type"]
    acados_solver_options.globalization = config_dict["globalization"]
    if config_dict["regularize_method"] != "NONE":
        acados_solver_options.regularize_method = config_dict["regularize_method"]
    acados_solver_options.levenberg_marquardt = config_dict["levenberg_marquardt"]
    acados_solver_options.print_level = config_dict["print_level"]
    acados_solver_options.ext_fun_compile_flags = config_dict["ext_fun_compile_flags"]
    if "HPIPM" in config_dict["qp_solver_type"]:
        acados_solver_options.hpipm_mode = "BALANCE"
    return acados_solver_options


@dataclass
class SolverConfig:
    acados: dict = field(default_factory=dict)
    casadi: CasadiSolverOptions = CasadiSolverOptions()

    @classmethod
    def from_dict(self, config_dict: dict):
        config = SolverConfig(acados=config_dict["acados"], casadi=CasadiSolverOptions.from_dict(config_dict["casadi"]))
        return config

    def to_dict(self):
        return {"acados": self.acados, "casadi": self.casadi.to_dict()}


@dataclass
class NLPSensitivities:
    """Class for containing MPC problem sensitivity functions"""

    dlag_dw: csd.Function  # Partial derivative of the Lagrangian wrt the primal decision variables w = {U, X}
    dlag_dp_f: csd.Function  # Partial derivative of the Lagrangian wrt the fixed parameters
    dlag_dp: csd.Function  # Partial derivative of the Lagrangian wrt the adjustable parameters
    d2lag_d2w: (
        csd.Function
    )  # Second order partial derivative of the Lagrangian wrt the primal decision variables w = {U, X}, i.e. the Hessian

    da_dp: csd.Function  # Partial derivative of the RL action a wrt the adjustable parameters
    dz_dp: (
        csd.Function
    )  # Partial derivative of the NLP solution z = (decision variables, multipliers) wrt the adjustable parameters
    dr_dz: (
        csd.Function
    )  # Partial derivative of the KKT matrix wrt the NLP solution z = (decision variables, multipliers)
    dr_dp: csd.Function  # Partial derivative of the KKT matrix wrt the adjustable parameters
    dr_dp_f: csd.Function  # Partial derivative of the KKT matrix wrt the fixed parameters

    # See Gros and Zanon "Reinforcement Learning based on MPC and the Stochastic Policy Gradient Method" for info on the below sensitivity functions
    dr_dz_bar: Optional[
        (csd.Function)
    ]  # Partial derivative of the KKT matrix wrt the NLP solution (decision variables, multipliers) with first input replaced by stochastic perturbation vector d
    dr_dp_bar: Optional[
        csd.Function
    ]  # Partial derivative of the KKT matrix wrt the adjustable parameters with first input replaced by stochastic perturbation vector d
    dr_dp_f_bar: Optional[
        csd.Function
    ]  # Partial derivative of the KKT matrix wrt the fixed parameters with first fixed parameter replaced by first input vector
    d2r_dp_da: Optional[
        list[csd.Function]
    ]  # List of second order partial derivatives of the KKT matrix wrt the adjustable parameters and the action (first input vector). Length of list is equal to the number of adjustable parameters.
    d2r_dp_dz_bar: Optional[
        list[csd.Function]
    ]  # List of second order partial derivatives of the KKT matrix wrt the adjustable parameters and the NLP solution (decision variables, multipliers) with first input replaced by stochastic perturbation vector d. Length of list is equal to the number of adjustable parameters.
    d2r_dzdz_j_bar: Optional[
        list[csd.Function]
    ]  # List of second order partial derivatives of the KKT matrix wrt the NLP solution (decision variables, multipliers) with first element of (z_bar) by stochastic perturbation vector d. Length of list is equal to the number of decision variables
    d2r_dadz_j_bar: Optional[
        list[csd.Function]
    ]  # List of second order partial derivatives of the KKT matrix wrt the action (first input vector) and the NLP solution z_bar where the first input is replaced by stochastic perturbation vector d. Length of list is equal to the number of decision variables

    @classmethod
    def from_dict(cls, input_dict: dict):
        output = NLPSensitivities(
            dlag_dw=input_dict["dlag_dw"],
            dlag_dp_f=input_dict["dlag_dp_f"],
            dlag_dp=input_dict["dlag_dp"],
            d2lag_d2w=input_dict["d2lag_d2w"],
            da_dp=input_dict["da_dp"],
            dz_dp=input_dict["dz_dp"],
            dr_dz=input_dict["dr_dz"],
            dr_dp=input_dict["dr_dp"],
            dr_dp_f=input_dict["dr_dp_f"],
            dr_dz_bar=input_dict["dr_dz_bar"] if "dr_dz_bar" in input_dict else None,
            dr_dp_bar=input_dict["dr_dp_bar"] if "dr_dp_bar" in input_dict else None,
            dr_dp_f_bar=input_dict["dr_dp_f_bar"] if "dr_dp_f_bar" in input_dict else None,
            d2r_dp_da=input_dict["d2r_dp_da"] if "d2r_dp_da" in input_dict else None,
            d2r_dp_dz_bar=input_dict["d2r_dp_dz_bar"] if "d2r_dp_dz_bar" in input_dict else None,
            d2r_dzdz_j_bar=input_dict["d2r_dzdz_j_bar"] if "d2r_dzdz_j_bar" in input_dict else None,
            d2r_dadz_j_bar=input_dict["d2r_dadz_j_bar"] if "d2r_dadz_j_bar" in input_dict else None,
        )
        return output

    def generate_c_functions(self, only_da_dp: bool = True, path: Optional[pathlib.Path] = None) -> None:
        """Generates C code for the casadi sensitivity functions. SLOW Shit"""
        if path is None:
            path = dp.casadi_code_gen
        assert only_da_dp, "Only da_dp codegen is implemented for now"
        if only_da_dp:
            t_now = time.time()
            # path_str  = str(path / "da_dp.c")
            # self.da_dp.generate("da_dp.c", {"main": True})
            C = csd.Importer("da_dp.c", "shell")
            self.da_dp = csd.external("da_dp", C)
            print(f"Time to generate da_dp: {time.time() - t_now}")
            return

        for attr in self.__dict__.keys():
            if isinstance(getattr(self, attr), csd.Function):
                getattr(self, attr).generate(dp.casadi_code_gen / (attr + ".c"))
                C = csd.Importer(dp.casadi_code_gen / (attr + ".c"), "shell")
                setattr(self, attr, csd.external(dp.casadi_code_gen / (attr + ".c"), C))


def quadratic_cost(var: csd.MX, var_ref: csd.MX, W: csd.MX) -> csd.MX:
    """Forms the NMPC stage cost function used by the mid-level COLAV method.

    Args:
        var (csd.MX): Decision variable to weight quadratically (e.g. state, input, slack)
        var_ref (csd.MX): Reference (input or state fixed parameter) variable
        W (csd.MX): Weighting matrix for the decision variable error

    Returns:
        csd.MX: Cost function
    """
    return (var_ref - var).T @ W @ (var_ref - var)


def path_following_cost(x: csd.MX, p_ref: csd.MX, Q_p: csd.MX) -> Tuple[csd.MX, csd.MX, csd.MX]:
    """Computes the path following cost for an NMPFC COLAV. Assumes the ship model is augmented by the path timing dynamics.

    Args:
        - x (csd.MX): Current state.
        - path_ref (csd.MX): Path reference on the form [p_x_ref, p_y_ref, s_dot_ref]^T.
        - Q_p (csd.MX): Path following cost weight vector.

    Returns:
        Tuple[csd.MX, csd.MX, csd.MX, csd.MX]: Total cost, path deviation cost, path dot deviation cost.
    """
    # relevant states for the path following cost term is the position (x, y) and path timing derivative (s_dot)
    z = csd.vertcat(x[:2], x[5])  # [x, y, s_dot]
    assert z.shape[0] == p_ref.shape[0], "Path reference and output vector must have the same dimension."
    path_dev_cost = quadratic_cost(z[:2], p_ref[:2], csd.diag(Q_p[:2]))
    path_dot_dev_cost = quadratic_cost(z[2], p_ref[2], Q_p[2])
    return path_dev_cost + path_dot_dev_cost, path_dev_cost, path_dot_dev_cost


def path_following_cost_huber(x: csd.MX, p_ref: csd.MX, Q_p: csd.MX) -> Tuple[csd.MX, csd.MX, csd.MX]:
    """Computes the path following cost for an NMPFC COLAV using the Huber loss for position errors. Assumes the ship model is augmented by the path timing dynamics.

    Args:
        x (csd.MX): State vector.
        p_ref (csd.MX): Reference path [x, y, s_dot]
        Q_p (csd.MX): Path following cost weight vector.
        delta (csd.MX): Shape parameter for the Huber loss function.

    Returns:
        Tuple[csd.MX, csd.MX, csd.MX]: Total cost, path deviation cost, speed deviation cost.
    """
    z = csd.vertcat(x[:2], x[5])
    assert z.shape[0] == p_ref.shape[0], "Path reference and output vector must have the same dimension."
    path_dev_squared = (z[:2] - p_ref[:2]).T @ (z[:2] - p_ref[:2])
    path_dev_cost = Q_p[0] * huber_loss(path_dev_squared, delta=1.0)
    path_dot_dev_cost = Q_p[2] * (z[2] - p_ref[2]) ** 2
    return path_dev_cost + path_dot_dev_cost, path_dev_cost, path_dot_dev_cost


def huber_loss(x_squared: csd.MX, delta: csd.MX = 1.0) -> csd.MX:
    """Huber loss function.

    Args:
        x (csd.MX): distance input squared.
        delta (csd.MX): Shape parameter.

    Returns:
        csd.MX: Loss function value.
    """
    return (csd.sqrt(1.0 + x_squared / (delta**2)) - 1.0) * delta**2


def rate_cost(
    r: csd.MX, a: csd.MX, alpha_app: csd.MX, K_app: csd.MX, r_max: float, a_max: float
) -> Tuple[csd.MX, csd.MX, csd.MX]:
    """Computes the chattering cost associated with the rate of change of the course and speed references,
    and stimulates chosing apparent maneuvers.

    Args:
        r (csd.MX): Turn rate input.
        a (csd.MX): Acceleration input.
        alpha_app (csd.MX): Apparent maneuver cost parameters.
        K_app (csd.MX): Apparent maneuver cost weight.
        r_max (float): Maximum rate of change of the course reference.
        a_max (float): Maximum rate of change of the speed reference.

    Returns:
        Tuple[csd.MX, csd.MX, csd.MX]: Total cost, course cost, speed cost.
    """
    q_chi = alpha_app[0] * r**2 + (1.0 - csd.exp(-(r**2) / alpha_app[1]))
    q_chi_max = alpha_app[0] * r_max**2 + (1.0 - csd.exp(-(r_max**2) / alpha_app[1]))
    q_U = alpha_app[2] * a**2 + (1.0 - csd.exp(-(a**2) / alpha_app[3]))
    q_U_max = alpha_app[2] * a_max**2 + (1.0 - csd.exp(-(a_max**2) / alpha_app[3]))
    course_cost = K_app[0] * q_chi / q_chi_max
    speed_cost = K_app[1] * q_U / q_U_max
    return course_cost + speed_cost, course_cost, speed_cost


def colregs_cost(
    x: csd.MX,
    X_do_cr: csd.MX,
    X_do_ho: csd.MX,
    X_do_ot: csd.MX,
    nx_do: int,
    alpha_cr: csd.MX,
    y_0_cr: csd.MX,
    alpha_ho: csd.MX,
    x_0_ho: csd.MX,
    alpha_ot: csd.MX,
    x_0_ot: csd.MX,
    y_0_ot: csd.MX,
    d_attenuation: csd.MX,
    weights: csd.MX,
) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX]:
    """Computes the COLREGS cost for the COLAV MPC method using the potential function approach.

    Args:
        x (csd.MX): Current state.
        X_do_cr (csd.MX): Dynamic obstacle states in the crossing zone
        X_do_ho (csd.MX): Dynamic obstacle states in the head-on zone
        X_do_ot (csd.MX): Dynamic obstacle states in the overtaking zone
        nx_do (int): Number of states in the dynamic obstacle model.
        alpha_cr (csd.MX): Attenuation parameters for the give-way situation potential function.
        y_0_cr (csd.MX): Offset parameter for the give-way situation potential function.
        alpha_ho (csd.MX): Attenuation parameters for the head-on situation potential function.
        x_0_ho (csd.MX): Offset parameter for the head-on situation potential function.
        alpha_ot (csd.MX): Attenuation parameters for the overtaking situation potential function.
        x_0_ot (csd.MX): Offset parameter for the overtaking situation potential function.
        y_0_ot (csd.MX): Offset parameter for the overtaking situation potential function.
        d_attenuation (csd.MX): Attenuation parameter for the potential functions.
        weights (csd.MX): Weights for the COLREGS cost terms.

    Returns:
        Tuple[csd.MX, csd.MX, csd.MX, csd.MX]: Total weighted cost, crossing cost, head-on cost, overtaking cost.
    """
    n_do_per_zone = int(X_do_cr.shape[0] / nx_do)

    cr_term = 0.0
    ho_term = 0.0
    ot_term = 0.0
    for i in range(n_do_per_zone):
        x_aug_do_cr = X_do_cr[i * nx_do : (i + 1) * nx_do]
        R_chi_do_cr = mf.Rpsi2D_casadi(x_aug_do_cr[2])
        p_rel = R_chi_do_cr.T @ (x[:2] - x_aug_do_cr[:2])
        d_rel_squared = p_rel.T @ p_rel
        cr_term += cr_potential(p_rel, alpha_cr, y_0_cr) * csd.exp(-d_rel_squared / d_attenuation**2)

        x_aug_do_ho = X_do_ho[i * nx_do : (i + 1) * nx_do]
        R_chi_do_ho = mf.Rpsi2D_casadi(x_aug_do_ho[2])
        p_rel = R_chi_do_ho.T @ (x[:2] - x_aug_do_ho[:2])
        d_rel_squared = p_rel.T @ p_rel
        ho_term += ho_potential(p_rel, alpha_ho, x_0_ho) * csd.exp(-d_rel_squared / d_attenuation**2)

        x_aug_do_ot = X_do_ot[i * nx_do : (i + 1) * nx_do]
        R_chi_do_ot = mf.Rpsi2D_casadi(x_aug_do_ot[2])
        p_rel = R_chi_do_ot.T @ (x[:2] - x_aug_do_ot[:2])
        d_rel_squared = p_rel.T @ p_rel
        ot_term += ot_potential(p_rel, alpha_ot, x_0_ot, y_0_ot) * csd.exp(-d_rel_squared / d_attenuation**2)

    cost = weights[0] * cr_term + weights[1] * ho_term + weights[2] * ot_term
    return cost, weights[0] * cr_term, weights[1] * ho_term, weights[2] * ot_term


def potential_field_base_function(x: csd.MX) -> csd.MX:
    """Calculates the base nonlinear function for the potential field approach to COLREGS.

    Args:
        x (csd.MX): Input OS position relative to a target ship

    Returns:
        csd.MX: The base function value.
    """
    return x / csd.sqrt(x**2 + 1) + 1


def cr_potential(p: csd.MX, alpha: csd.MX, y_0: csd.MX) -> csd.MX:
    """Calculates the potential function for the crossing situation.

    Args:
        p (csd.MX): Position relative to the TS body-fixed frame.
        alpha (csd.MX): Parameter adjusting the steepness.
        y_0 (csd.MX): Parameter adjusting the attenuation on the port side of the ship.

    Returns:
        csd.MX: The potential function value.
    """
    return (
        0.25
        * potential_field_base_function(alpha[0] * p[0])
        * (potential_field_base_function(alpha[1] * (p[1] - y_0)) + 1)
    )


def ho_potential(p: csd.MX, alpha: csd.MX, x_0: csd.MX) -> csd.MX:
    """Calculates the potential function for the head-on situation.

    Args:
        p (csd.MX): Position relative to the TS body-fixed frame.
        alpha (csd.MX): Parameter adjusting the steepness.
        x_0 (csd.MX): Parameter adjusting the attenuation on the ship front.

    Returns:
        csd.MX: The potential function value.
    """
    return (
        0.25
        * (potential_field_base_function(alpha[0] * (x_0 - p[0])) + 1)
        * potential_field_base_function(alpha[1] * p[1])
    )


def ot_potential(p: csd.MX, alpha: csd.MX, x_0: csd.MX, y_0: csd.MX) -> csd.MX:
    """Calculates the potential function for the give-way situation.

    Args:
        p (csd.MX): Position relative to the TS body-fixed frame.
        alpha (csd.MX): Parameter adjusting the steepness.
        x_0 (csd.MX): Parameter adjusting the attenuation on the ship front.
        y_0 (csd.MX): Parameter adjusting the attenuation on the port side of the ship.


    Returns:
        csd.MX: The potential function value.
    """
    return (
        0.25
        * potential_field_base_function(-alpha[0] * (x_0 - p[0]))
        * potential_field_base_function(alpha[1] * csd.fabs(p[1] - y_0))
    )


def plot_casadi_solver_stats(stats: dict, show_plots: bool = True) -> None:
    """Plots solver statistics for the COLAV MPC method.

    Args:
        - stats (dict): Dictionary of solver statistics
        - show_plots (bool, optional): Whether to show plots or not. Defaults to True.
    """
    if show_plots:
        fig = plt.figure()
        axs = fig.subplot_mosaic(
            [
                ["cost"],
                ["inf"],
                ["barrier_parameter"],
                ["step_lengths"],
            ]
        )

        axs["cost"].plot(stats["iterations"]["obj"], "b")
        axs["cost"].set_xlabel("Iteration number")
        axs["cost"].set_ylabel("J")

        axs["inf"].plot(np.array(stats["iterations"]["inf_pr"]), "b--")
        axs["inf"].plot(np.array(stats["iterations"]["inf_du"]), "r--")
        axs["inf"].legend(["Primal Infeasibility", "Dual Infeasibility"])
        axs["inf"].set_xlabel("Iteration number")
        axs["inf"].set_ylabel("Inf norm")

        axs["barrier_parameter"].plot(np.array(stats["iterations"]["mu"]), "b--")
        axs["barrier_parameter"].set_xlabel("Iteration number")
        axs["barrier_parameter"].set_ylabel("tau")

        axs["step_lengths"].plot(np.array(stats["iterations"]["alpha_pr"]), "b--")
        axs["step_lengths"].plot(np.array(stats["iterations"]["alpha_du"]), "r--")
        axs["step_lengths"].legend(["Primal Step Length", "Dual Step Length"])
        axs["step_lengths"].set_xlabel("Iteration number")
        axs["step_lengths"].set_ylabel("Step Length")
        plt.show(block=False)
    return None


def construct_acados_ocp(
    model: models.MPCModel,
    cost_function: csd.MX,
    l1_slack_penalty: float,
    l2_slack_penalty: float,
    do_constraints: csd.MX,
    so_constraints: csd.MX,
    parameters: csd.MX,
    parameter_values: np.ndarray,
    horizon: float,
    time_step: float,
    solver_options: AcadosOcpOptions,
) -> AcadosOcpSolver:
    """
    Constructs the OCP for the NMPC COLAV problem using ACADOS.

    Args:
        model (models.MPCModel): Model of the system.
        cost_function (csd.MX): Cost function of the OCP.
        l1_slack_penalty (float): L1 penalty on the slack variables.
        l2_slack_penalty (float): L2 penalty on the slack variables.
        do_constraints (csd.MX): Dynamic obstacle constraints.
        so_constraints (csd.MX): Static obstacle constraints.
        parameters (csd.MX): Parameters of the OCP.
        parameter_values (np.ndarray): Values of the parameters.
        horizon (float): Planning horizon.
        time_step (float): Time step of the OCP.
        solver_options (AcadosOcpOptions): AcadosOCPSolver options.

    Returns:
        AcadosOcpSolver: ACADOS OCP solver.
    """
    acados_ocp: AcadosOcp = AcadosOcp()
    acados_ocp.solver_options = solver_options
    acados_ocp.model = model.as_acados()
    acados_ocp.dims.N = int(horizon / time_step)
    acados_ocp.solver_options.qp_solver_cond_N = acados_ocp.dims.N
    acados_ocp.solver_options.tf = horizon

    nx = acados_ocp.model.x.size()[0]
    nu = acados_ocp.model.u.size()[0]
    acados_ocp.dims.nx = nx
    acados_ocp.dims.nu = nu

    acados_ocp.cost.cost_type = "EXTERNAL"
    acados_ocp.cost.cost_type_e = "EXTERNAL"
    acados_ocp.model.cost_expr_ext_cost = cost_function
    acados_ocp.model.cost_expr_ext_cost_e = cost_function

    approx_inf = 1e6
    lbu, ubu, lbx, ubx = model.get_input_state_bounds()

    # Input constraints lbu <= u <= ubu ∀ u
    acados_ocp.constraints.idxbu = np.array(range(nu))
    acados_ocp.constraints.lbu = lbu
    acados_ocp.constraints.ubu = ubu

    # State constraints lbx <= x <= ubx ∀ x
    acados_ocp.constraints.x0 = lbx
    acados_ocp.constraints.idxbx = np.array(range(nx))
    acados_ocp.constraints.lbx = lbx[acados_ocp.constraints.idxbx]
    acados_ocp.constraints.ubx = ubx[acados_ocp.constraints.idxbx]

    acados_ocp.constraints.idxbx_e = np.array(range(nx))
    acados_ocp.constraints.lbx_e = lbx[acados_ocp.constraints.idxbx_e]
    acados_ocp.constraints.ubx_e = ubx[acados_ocp.constraints.idxbx_e]

    # Dynamic and static obstacle constraints together on the form -inf <= h(x, u, p) <= 0 + s_upper
    n_path_constr = len(so_constraints) + len(do_constraints)
    if n_path_constr:
        acados_ocp.constraints.lh = -approx_inf * np.ones(n_path_constr)
        acados_ocp.constraints.lh_e = acados_ocp.constraints.lh
        acados_ocp.constraints.uh = np.zeros(n_path_constr)
        acados_ocp.constraints.uh_e = acados_ocp.constraints.uh

        # Slacks on dynamic obstacle and static obstacle constraints
        acados_ocp.constraints.idxsh = np.array(range(n_path_constr))
        acados_ocp.constraints.idxsh_e = np.array(range(n_path_constr))

        acados_ocp.cost.Zl = 0 * l2_slack_penalty * np.ones(n_path_constr)
        acados_ocp.cost.Zl_e = 0 * l2_slack_penalty * np.ones(n_path_constr)
        acados_ocp.cost.Zu = l2_slack_penalty * np.ones(n_path_constr)
        acados_ocp.cost.Zu_e = l2_slack_penalty * np.ones(n_path_constr)
        acados_ocp.cost.zl = 0 * l1_slack_penalty * np.ones(n_path_constr)
        acados_ocp.cost.zl_e = 0 * l1_slack_penalty * np.ones(n_path_constr)
        acados_ocp.cost.zu = l1_slack_penalty * np.ones(n_path_constr)
        acados_ocp.cost.zu_e = l1_slack_penalty * np.ones(n_path_constr)

        con_h_expr = []
        con_h_expr.extend(so_constraints)
        con_h_expr.extend(do_constraints)

        acados_ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
        acados_ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

    acados_ocp.model.p = parameters
    acados_ocp.dims.np = acados_ocp.model.p.size()[0]
    acados_ocp.parameter_values = parameter_values

    solver_json = "acados_ocp_" + acados_ocp.model.name + ".json"
    acados_ocp.code_export_directory = dp.acados_code_gen.as_posix()
    return AcadosOcpSolver(acados_ocp, json_file=solver_json)
