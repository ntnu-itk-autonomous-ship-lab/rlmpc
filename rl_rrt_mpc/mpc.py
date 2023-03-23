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
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.models as models
import seacharts.enc as senc
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions
from acados_template.acados_ocp_solver import AcadosOcpSolver


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
    reference_traj_bbox_buffer: float = 500.0
    T: float = 10.0
    dt: float = 0.5
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R: np.ndarray = np.diag([1.0, 1.0, 1.0])
    gamma: float = 0.0
    solver_options: AcadosOcpOptions = AcadosOcpOptions()
    max_num_so_constraints: int = 1000
    max_num_do_constraints: int = 1000

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = MPCParams(
            reference_traj_bbox_buffer=config_dict["reference_traj_bbox_buffer"],
            T=config_dict["T"],
            dt=config_dict["dt"],
            Q=np.diag(config_dict["Q"]),
            R=np.diag(config_dict["R"]),
            gamma=config_dict["gamma"],
            solver_options=AcadosOcpOptions(),
        )
        config.solver_options.nlp_solver_type = config_dict["solver_options"]["nlp_solver_type"]
        config.solver_options.qp_solver_type = config_dict["solver_options"]["qp_solver_type"]
        config.solver_options.hessian_approx = config_dict["solver_options"]["hessian_approx_type"]
        config.solver_options.globalization = config_dict["solver_options"]["globalization"]
        config.solver_options.nlp_solver_max_iter = config_dict["solver_options"]["nlp_solver_max_iter"]
        config.solver_options.nlp_solver_tol_stat = config_dict["solver_options"]["nlp_solver_tol_stat"]
        config.solver_options.qp_solver_iter_max = config_dict["solver_options"]["qp_solver_iter_max"]
        config.solver_options.qp_solver_warm_start = config_dict["solver_options"]["qp_solver_warm_start"]
        config.solver_options.levenberg_marquardt = config_dict["solver_options"]["levenberg_marquardt"]
        config.solver_options.print_level = config_dict["solver_options"]["print_level"]
        return config


class MPC:
    def __init__(self, model: models.TelemetronAcados, params: Optional[MPCParams] = MPCParams()) -> None:
        self._ocp: AcadosOcp = AcadosOcp()
        self._model = model
        if params:
            self._params = params

    def plan(self, t: float, nominal_trajectory: np.ndarray, nominal_inputs: np.ndarray, xs: np.ndarray, do_list: list, so_list: list, **kwargs) -> np.ndarray:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            t (float): Current time.
            nominal_trajectory (np.ndarray): Nominal reference trajectory to track.
            xs (np.ndarray): Current state.
            do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            so_list (list): List of static obstacle Polygon objects

        Returns:
            np.ndarray: Optimal trajectory for the ownship.
        """
        references = np.empty(9)

        self._ocp.set("constraints.x0", xs)
        self._ocp.set("init_x", nominal_trajectory)
        self._ocp.set("init_u", nominal_inputs)

        status = self._solver.solve()
        return references

    def construct_ocp(self, do_list: list, so_list: list, enc: Optional[senc.ENC] = None) -> None:
        """Constructs the OCP for the NMPC problem.

        Args:
            do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            so_list (list): List of static obstacle Polygon objects
            enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

        """
        self._ocp.model = self._model.as_acados()
        self._ocp.solver_options = self._params.solver_options
        self._ocp.dims.N = int(self._params.T / self._params.dt)
        self._ocp.solver_options.qp_solver_cond_N = self._ocp.dims.N
        self._ocp.solver_options.tf = self._params.T

        nx = self._ocp.model.x.size()[0]
        nu = self._ocp.model.u.size()[0]
        self._ocp.dims.nx = nx
        self._ocp.dims.nu = nu

        x = self._ocp.model.x
        u = self._ocp.model.u

        Qvec = csd.SX.sym("Q", 36)
        kappa = csd.SX.sym("kappa", 1)
        t = csd.SX.sym("t", 1)

        adjustable_params = csd.vertcat(Qvec, kappa, t)
        x_ref = csd.SX.sym("x_ref", 6)
        fixed_params = x_ref

        p = csd.vertcat(adjustable_params, fixed_params)
        self._ocp.model.p = p

        self._ocp.cost.cost_type = "EXTERNAL"
        self._ocp.cost.cost_type_e = "EXTERNAL"
        Qmtrx = hf.casadi_matrix_from_vector(Qvec)
        self._ocp.model.cost_expr_ext_cost = (x_ref - x).T @ Qmtrx @ (x_ref - x) + kappa * t
        self._ocp.model.cost_expr_ext_cost_e = (x_ref - x).T @ Qmtrx @ (x_ref - x)

        # # soften
        # ocp.constraints.idxsh = np.array([0])
        # ocp.cost.zl = 1e5 * np.array([1])
        # ocp.cost.zu = 1e5 * np.array([1])
        # ocp.cost.Zl = 1e5 * np.array([1])
        # ocp.cost.Zu = 1e5 * np.array([1])

        self._ocp.constraints.constr_type = "BGH"
        min_Fx = self._model.params.Fx_limits[0]
        max_Fx = self._model.params.Fx_limits[1]
        min_Fy = self._model.params.Fy_limits[0]
        max_Fy = self._model.params.Fy_limits[1]
        lever_arm = self._model.params.l_r
        max_turn_rate = self._model.params.r_max
        max_speed = self._model.params.U_max

        # Input constraints
        self._ocp.constraints.idxbu = np.array(range(nu))
        self._ocp.constraints.lbu = np.array(
            [
                min_Fx,
                min_Fy,
                lever_arm * min_Fy,
            ]
        )
        self._ocp.constraints.ubu = np.array([max_Fx, max_Fy, lever_arm * max_Fy])

        # State constraints
        self._ocp.constraints.idxbx = np.array(range(nx))
        self._ocp.constraints.lbx = np.array([-np.inf, -np.inf, -np.inf, 0.0, -0.6 * max_speed, -max_turn_rate])
        self._ocp.constraints.ubx = np.array([np.inf, np.inf, np.inf, max_speed, 0.6 * max_speed, max_turn_rate])

        self._ocp.constraints.idxbx_e = np.array(range(nx))
        self._ocp.constraints.lbx_e = np.array([-np.inf, -np.inf, -np.inf, 0.0, -0.6 * max_speed, -max_turn_rate])
        self._ocp.constraints.ubx_e = np.array([np.inf, np.inf, np.inf, max_speed, 0.6 * max_speed, max_turn_rate])

        # Dynamic and static obstacle constraints
        d_safe_so = csd.SX.sym("d_safe_so", 1)
        d_safe_do = csd.SX.sym("d_safe_do", 1)

        p = self._ocp.model.p
        p = csd.vertcat(p, d_safe_so, d_safe_do)
        self._ocp.model.p = p

        self._ocp.constraints.lh = np.zeros(self._params.max_num_so_constraints + self._params.max_num_so_constraints)
        self._ocp.constraints.lh_e = self._ocp.constraints.lh
        self._ocp.constraints.uh = np.inf * np.ones(self._params.max_num_so_constraints + self._params.max_num_so_constraints)
        self._ocp.constraints.uh_e = self._ocp.constraints.uh

        con_h_expr = []
        so_splines_xy, so_splines_der_xy = hf.compute_splines_from_polygons(so_list, enc)
        n_so = len(so_splines_xy)
        for j in range(self._params.max_num_so_constraints):
            if j < n_so:

            # generate spline of static obstacle
            y_poly, x_poly = so_list[j].exterior.xy
            so_spline_j = csd.interpolant("so_spline_" + str(j), "bspline", [x_poly], y_poly)
            con_h_expr.append((x - so_spline_j(x)).T @ (x - so_spline_j(x)) - d_safe_so**2)

        for i in range(self._params.max_num_so_constraints, self._params.max_num_so_constraints + self._params.max_num_do_constraints):
            (_, _, _, length, width) = do_list[i - self._params.max_num_so_constraints]
            x_do_i = csd.SX.sym("x_do_" + str(i), 4)
            chi_do_i = csd.atan2(x_do_i[3], x_do_i[2])
            p = csd.vertcat(p, x_do_i)
            con_h_expr.append(((x[0] - x_do_i[0]) ** 2 * csd.cos(chi_do_i) / length**2) + ((x[0] - x_do_i[0]) ** 2 * csd.cos(chi_do_i) / width**2))

        self._ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
        self._ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

        solver_json = "acados_ocp_" + self._ocp.model.name + ".json"
        self._solver = AcadosOcpSolver(self._ocp, json_file=solver_json)

        self._solver.set()
