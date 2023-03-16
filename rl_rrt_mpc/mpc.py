"""
    mpc.py

    Summary:
        Contains a class for an MPC trajectory tracking controller.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import Optional

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.config_parsing as cp
import scipy
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions, AcadosOcpSolver, AcadosSimSolver


@dataclass
class OCPSolverOptions:
    nlp_solver_type: str = "SQP"
    qp_solver_type: str = "FULL_CONDENSING_QPOASES"
    hessian_approx_type: str = "GAUSS_NEWTON"
    globalization: str = "MERIT_BACKTRACKING"
    nlp_solver_max_iter: int = 100
    nlp_solver_tol_stat: float = 1e-6
    qp_solver_iter_max: int = 100
    qp_solver_warm_start: int = 0
    levenberg_marquardt: float = 0.0
    print_level: int = 3

    @classmethod
    def from_dict(cls, config_dict: dict):
        return OCPSolverOptions(**config_dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class MPCParams:
    T: float = 10.0
    dt: float = 0.5
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R: np.ndarray = np.diag([1.0, 1.0, 1.0])
    gamma: float = 0.0
    solver_options: AcadosOcpOptions = AcadosOcpOptions()

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = MPCParams(
            T=config_dict["T"],
            dt=config_dict["dt"],
            Q=np.diag(config_dict["Q"]),
            R=np.diag(config_dict["R"]),
            gamma=config_dict["gamma"],
            solver_options=AcadosOcpOptions(),
        )
        config.solver_options.nlp_solver_type = config_dict["solver_options"]["nlp_solver_type"]
        config.solver_options.qp_solver_type = config_dict["solver_options"]["qp_solver_type"]
        config.solver_options.hessian_approx_type = config_dict["solver_options"]["hessian_approx_type"]
        config.solver_options.globalization = config_dict["solver_options"]["globalization"]
        config.solver_options.nlp_solver_max_iter = config_dict["solver_options"]["nlp_solver_max_iter"]
        config.solver_options.nlp_solver_tol_stat = config_dict["solver_options"]["nlp_solver_tol_stat"]
        config.solver_options.qp_solver_iter_max = config_dict["solver_options"]["qp_solver_iter_max"]
        config.solver_options.qp_solver_warm_start = config_dict["solver_options"]["qp_solver_warm_start"]
        config.solver_options.levenberg_marquardt = config_dict["solver_options"]["levenberg_marquardt"]
        config.solver_options.print_level = config_dict["solver_options"]["print_level"]
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
    def __init__(self, params: Optional[MPCParams] = MPCParams()) -> None:
        self._params = params
        self._ocp_options: AcadosOcpOptions = AcadosOcpOptions()
        self._ocp: AcadosOcp = AcadosOcp()
        self._setup_ocp_solver_options()
        # self._model: AcadosModel = models.ShipModel().to_acados()

    def _setup_ocp_solver_options(self) -> None:
        self._ocp.solver_options.qp_solver = self._params.solver_options.qp_solver_type
        self._ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self._ocp.solver_options.integrator_type = "ERK"
        self._ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
        self._ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"  # PARTIAL_CONDENSING_HPIPM
        self._ocp.solver_options.print_level = 3
        self._ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self._ocp.solver_options.nlp_solver_max_iter = 5000
        self._ocp.solver_options.nlp_solver_tol_stat = 1e-6
        self._ocp.solver_options.levenberg_marquardt = 0.1
        self._ocp.solver_options.sim_method_num_steps = 15
        self._ocp.solver_options.qp_solver_iter_max = 100
        self._ocp.solver_options.hessian_approx = "EXACT"
        self._ocp.solver_options.exact_hess_constr = 0
        self._ocp.solver_options.exact_hess_dyn = 0

    def _setup_ocp(self) -> AcadosOcp:

        # set dimensions

        self.ocp.dims.N = N_horizon

        # set cost module
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
        R_mat = 2 * np.diag([1e-2])

        self.ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

        self.ocp.cost.W_e = Q_mat

        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[4, 0] = 1.0
        self.ocp.cost.Vu = Vu

        self.ocp.cost.Vx_e = np.eye(nx)

        self.ocp.cost.yref = np.zeros((ny,))
        self.ocp.cost.yref_e = np.zeros((ny_e,))

        # set constraints
        Fmax = 80
        x0 = np.array([0.0, np.pi, 0.0, 0.0])
        self.ocp.constraints.constr_type = "BGH"
        self.ocp.constraints.lbu = np.array([-Fmax])
        self.ocp.constraints.ubu = np.array([+Fmax])
        self.ocp.constraints.x0 = x0
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.qp_solver_cond_N = N_horizon

        # set prediction horizon
        self.ocp.solver_options.tf = Tf

        solver_json = "acados_ocp_" + model.name + ".json"
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file=solver_json)

    def plan(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        pass


def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_pendulum_ode_model()
    ocp.model = model

    Tf = 1.0
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    N_horizon = 20

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
    R_mat = 2 * np.diag([1e-2])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.W_e = Q_mat

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[4, 0] = 1.0
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    Fmax = 80
    x0 = np.array([0.0, np.pi, 0.0, 0.0])
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = "acados_ocp_" + model.name + ".json"
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file=solver_json)

    # N - maximum number of bangs
    N = 7
    Tf = N
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    # set dimensions
    ocp.dims.N = N

    # set cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost = dt
    ocp.model.cost_expr_ext_cost_e = 0

    ocp.constraints.lbu = np.array([-a_max, 0.0])
    ocp.constraints.ubu = np.array([+a_max, dt_max])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.x0 = x0
    ocp.constraints.lbx_e = xf
    ocp.constraints.ubx_e = xf
    ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

    # set prediction horizon
    ocp.solver_options.tf = Tf

    # set options

    Nsim = 100
    simX = np.ndarray((Nsim + 1, nx))
    simU = np.ndarray((Nsim, nu))

    simX[0, :] = x0

    # closed loop
    for i in range(Nsim):

        # solve ocp and get next control input
        simU[i, :] = acados_ocp_solver.solve_for_x0(x0_bar=simX[i, :])

        # simulate system
        simX[i + 1, :] = acados_integrator.simulate(x=simX[i, :], u=simU[i, :])

    # plot results
    plot_pendulum(np.linspace(0, Tf / N_horizon * Nsim, Nsim + 1), Fmax, simU, simX)


if __name__ == "__main__":
    main()
