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
import scipy
from acados_template.acados_ocp import AcadosModel, AcadosOcp, AcadosOcpOptions, AcadosOcpSolver
import seacharts.enc as senc

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


class MPC:
    def __init__(self, model: AcadosModel, params: Optional[MPCParams] = MPCParams()) -> None:
        self._ocp: AcadosOcp = AcadosOcp()
        if params:
            self._params = params
        self._setup_ocp(model)

    def _construct_cost_function(self) -> None:
        nx = self._ocp.model.x.size()[0]
        nu = self._ocp.model.u.size()[0]
        ny = nx + nu
        ny_e = nx
        cost = csd.mtimes([y.T, self._params.Q, y]) + csd.mtimes([u.T, self._params.R, u])
        self._ocp.cost.cost_type = "EXTERNAL"
        self._ocp.cost.cost_type_e = "EXTERNAL"
        self._ocp.cost.W = scipy.linalg.block_diag(self._params.Q, self._params.R)
        self._ocp.cost.W_e = self._params.Q

        self._ocp.cost.Vx = np.zeros((ny, nx))
        self._ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[4, 0] = 1.0
        self._ocp.cost.Vu = Vu

        self._ocp.cost.Vx_e = np.eye(nx)

        self._ocp.cost.yref = np.zeros((ny,))
        self._ocp.cost.yref_e = np.zeros((ny_e,))

    def _construct_constraints(self, do_list: list, enc: senc.ENC) -> None:

    def _setup_ocp(self, model: AcadosModel) -> None:

        self._ocp.solver_options = self._params.solver_options
        self._ocp.model = model
        nx = self._ocp.model.x.size()[0]
        nu = self._ocp.model.u.size()[0]
        ny = nx + nu
        ny_e = nx
        Tf = self._params.T

        self._ocp.dims.N = self._params.T / self._params.dt
        self._ocp.solver_options.qp_solver_cond_N = self._ocp.dims.N



        # set constraints
        Fmax = 80
        x0 = np.array([0.0, np.pi, 0.0, 0.0])
        self._ocp.constraints.constr_type = "BGH"
        self._ocp.constraints.lbu = np.array([-Fmax])
        self._ocp.constraints.ubu = np.array([+Fmax])
        self._ocp.constraints.x0 = x0
        self._ocp.constraints.idxbu = np.array([0])
        self._ocp.solver_options.tf = Tf
        solver_json = "acados_ocp_" + model.name + ".json"
        self._solver = AcadosOcpSolver(self._ocp, json_file=solver_json)
        self._integrator = AcadosSimSolver(self._ocp, json_file=solver_json)

    def plan(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        pass
