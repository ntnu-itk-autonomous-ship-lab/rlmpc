"""
    common.py

    Summary:
        Contains common solver functions and configuration objects used by the various MPC-types.


    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass, field

import casadi as csd
import matplotlib.pyplot as plt
import numpy as np
from acados_template.acados_ocp import AcadosOcpOptions


@dataclass
class CasadiSolverOptions:
    solver_type: str = "ipopt"
    solver_tol: float = 1e-6
    print_level: int = 0
    print_time: int = 0
    mu_target: float = 1e-6
    mu_init: float = 1e-6
    acceptable_tol: float = 1e-6
    acceptable_obj_change_tol: float = 1e-6
    max_iter: int = 1000
    warm_start_init_point: str = "no"
    verbose: bool = True
    jit: bool = True
    jit_flags: list = field(default_factory=lambda: ["-O0"])
    compiler: str = "clang"
    expand_mx_funcs_to_sx: bool = True

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
        }
        return opts


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

        axs["step_lengths"].plot(np.array(stats["iterations"]["alpha_pr"]), "b--")
        axs["step_lengths"].plot(np.array(stats["iterations"]["alpha_du"]), "r--")
        axs["step_lengths"].legend(["Primal Step Length", "Dual Step Length"])
        axs["step_lengths"].set_xlabel("Iteration number")
        axs["step_lengths"].set_ylabel("Step Length")
        plt.show(block=False)
    return None
