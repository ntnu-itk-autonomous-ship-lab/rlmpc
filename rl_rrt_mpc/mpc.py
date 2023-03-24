"""
    mpc.py

    Summary:
        Contains a class for an MPC trajectory tracking controller.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.models as models
import seacharts.enc as senc
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions
from acados_template.acados_ocp_solver import AcadosOcpSolver

MAX_NUM_DO_CONSTRAINTS: int = 15
MAX_NUM_SO_CONSTRAINTS: int = 1000
P_ADJUSTABLE_IDX_START: int = 0  # Index of first adjustable parameter in the parameter vector
P_ADJUSTABLE_IDX_END: int = 39  # Index of last adjustable parameter + 1 in the parameter vector
P_XREF_IDX_START: int = 39  # Index of first reference state element in the parameter vector
P_XREF_IDX_END: int = 46  # Index of last reference state element + 1 in the parameter vector
P_SO_IDX_START: int = -1  # Index of first static obstacle constraint element in the parameter vector
P_SO_IDX_END: int = -1 * MAX_NUM_SO_CONSTRAINTS  # Index of last static obstacle constraint element parameter + 1 in the parameter vector.
P_DO_IDX_START: int = 46  # Index of first dynamic obstacle constraint element in the parameter vector
P_DO_IDX_END: int = 46 + (4 + 1 + 1) * MAX_NUM_DO_CONSTRAINTS  # Index of last dynamic obstacle constraint element parameter + 1 in the parameter vector.


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
    d_safe_so: float = 5.0
    d_safe_do: float = 5.0
    solver_options: AcadosOcpOptions = AcadosOcpOptions()

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = MPCParams(
            reference_traj_bbox_buffer=config_dict["reference_traj_bbox_buffer"],
            T=config_dict["T"],
            dt=config_dict["dt"],
            Q=np.diag(config_dict["Q"]),
            R=np.diag(config_dict["R"]),
            gamma=config_dict["gamma"],
            d_safe_so=config_dict["d_safe_so"],
            d_safe_do=config_dict["d_safe_do"],
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
            self._params0: MPCParams = params
            self._params: MPCParams = params

    def update_adjustable_params(self, params: list) -> list:
        """Updates the RL-tuneable parameters in the NMPC.

        Args:
            params (list): List of parameters to update. The order of the parameters are:
                - Q
                - gamma
                - d_safe_so
                - d_safe_do

        Returns:
            list: List of newly updated parameters.
        """
        self._params.Q = np.reshape(params[0:36], (6, 6))
        self._params.gamma = params[36]
        self._params.d_safe_so = params[37]
        self._params.d_safe_do = params[38]

    def get_adjustable_params(self) -> list:
        """Returns the RL-tuneable parameters in the NMPC.

        Returns:
            list: List of parameters. The order of the parameters are:
                - Q
                - gamma
                - d_safe_so
                - d_safe_do
        """
        return [self._params.Q.reshape(), self._params.gamma, self._params.d_safe_so, self._params.d_safe_do]

    def plan(self, t: float, nominal_trajectory: np.ndarray, nominal_inputs: np.ndarray, xs: np.ndarray, do_list: list, so_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            t (float): Current time.
            nominal_trajectory (np.ndarray): Nominal reference trajectory to track.
            nominal_inputs (np.ndarray): Nominal reference inputs to track.
            xs (np.ndarray): Current state.
            do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            so_list (list): List of static obstacle Polygon objects

        Returns:
            Tuple[np.ndarray, np.ndarray]: Optimal trajectory and inputs for the ownship.
        """
        self._update_ocp(t, nominal_trajectory, nominal_inputs, xs, do_list, so_list, self._params)
        status = self._ocp_solver.solve()
        t_solve = self._ocp_solver.get_stats("time_tot")
        cost = self._ocp_solver.get_stats("cost")

        trajectory = np.array((self._ocp.dims.nx, self._ocp.dims.N + 1))
        inputs = np.array((self._ocp.dims.nu, self._ocp.dims.N))
        for i in range(self._ocp.dims.N):
            trajectory[:, i] = self._ocp_solver.get(i, "x").T
            inputs[:, i] = self._ocp_solver.get(i, "u").T
        print(f"MPC: | Runtime: {t_solve} | Cost: {cost}")
        return trajectory, inputs

    def _update_ocp(
        self, t: float, nominal_trajectory: np.ndarray, nominal_inputs: np.ndarray, xs: np.ndarray, do_list: list, so_list: list, adjustable_params: list
    ) -> None:
        """Updates the OCP (cost and constraints) with the current info available

        Args:
            t (float): Current time.
            nominal_trajectory (np.ndarray): Nominal reference trajectory to track.
            nominal_inputs (np.ndarray): Nominal reference inputs to track.
            xs (np.ndarray): Current state.
            do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            so_list (list): List of static obstacle Polygon objects
            adjustable_params (list): List of adjustable parameters to be passed to the cost function
        """
        self._ocp_solver.constraints_set(0, "lbx", xs)
        self._ocp_solver.constraints_set(0, "ubx", xs)

        for i in range(self._ocp.dims.N + 1):
            p_i = self.create_parameter_values(adjustable_params, nominal_trajectory, do_list, so_list, i)
            self._ocp_solver.set(i, "p", p_i)

    def construct_ocp(self, nominal_trajectory: np.ndarray, do_list: list, so_list: list, enc: Optional[senc.ENC] = None) -> None:
        """Constructs the OCP for the NMPC problem.

        Args:
            nominal_trajectory (np.ndarray): Nominal reference trajectory to track.
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
        gamma = csd.SX.sym("gamma", 1)
        x_ref = csd.SX.sym("x_ref", 6)
        fixed_params = x_ref

        self._ocp.cost.cost_type = "EXTERNAL"
        self._ocp.cost.cost_type_e = "EXTERNAL"
        Qmtrx = hf.casadi_matrix_from_vector(Qvec)
        self._ocp.model.cost_expr_ext_cost = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)
        self._ocp.model.cost_expr_ext_cost_e = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)

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

        approx_inf = 1e15
        # State constraints
        self._ocp.constraints.idxbx = np.array(range(nx))
        self._ocp.constraints.lbx = np.array([-approx_inf, -approx_inf, -approx_inf, 0.0, -0.6 * max_speed, -max_turn_rate])
        self._ocp.constraints.ubx = np.array([approx_inf, approx_inf, approx_inf, max_speed, 0.6 * max_speed, max_turn_rate])

        self._ocp.constraints.idxbx_e = np.array(range(nx))
        self._ocp.constraints.lbx_e = np.array([-approx_inf, -approx_inf, -approx_inf, 0.0, -0.6 * max_speed, -max_turn_rate])
        self._ocp.constraints.ubx_e = np.array([approx_inf, approx_inf, approx_inf, max_speed, 0.6 * max_speed, max_turn_rate])

        # Dynamic and static obstacle constraints
        d_safe_so = csd.SX.sym("d_safe_so", 1)
        d_safe_do = csd.SX.sym("d_safe_do", 1)

        self._ocp.constraints.lh = np.zeros(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._ocp.constraints.lh_e = self._ocp.constraints.lh
        self._ocp.constraints.uh = approx_inf * np.ones(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._ocp.constraints.uh_e = self._ocp.constraints.uh

        con_h_expr = []
        so_splines_xy, so_splines_der_xy = hf.compute_splines_from_polygons(so_list, enc)
        n_so = len(so_splines_xy)
        for j in range(MAX_NUM_SO_CONSTRAINTS):
            # if j < n_so:
            # # generate spline of static obstacle
            # y_poly, x_poly = so_list[j].exterior.xy
            # so_spline_j = csd.interpolant("so_spline_" + str(j), "bspline", [x_poly], y_poly)
            # con_h_expr.append((x - so_spline_j(x)).T @ (x - so_spline_j(x)) - d_safe_so**2)
            con_h_expr.append(x[0])

        for i in range(MAX_NUM_DO_CONSTRAINTS):
            x_do_i = csd.SX.sym("x_do_" + str(i), 4)
            l_do_i = csd.SX.sym("l_do_" + str(i), 1)
            w_do_i = csd.SX.sym("w_do_" + str(i), 1)
            chi_do_i = csd.atan2(x_do_i[3], x_do_i[2])
            fixed_params = csd.vertcat(fixed_params, x_do_i, l_do_i, w_do_i)
            con_h_expr.append(((x[0] - x_do_i[0]) ** 2 * csd.cos(chi_do_i) / l_do_i**2) + ((x[1] - x_do_i[1]) ** 2 * csd.cos(chi_do_i) / w_do_i**2) - d_safe_do**2)

        # Parameters consist of RL adjustable parameters, and fixed parameters (either nominal trajectory or dynamic obstacle related). The model parameters are considered fixed.
        adjustable_params = csd.vertcat(Qvec, gamma, d_safe_so, d_safe_do)
        self._ocp.model.p = csd.vertcat(adjustable_params, fixed_params)
        self._ocp.dims.np = self._ocp.model.p.size()[0]

        self._ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
        self._ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

        initial_adjustable_params = [*self._params.Q.reshape(nx**2).tolist(), self._params.gamma, self._params.d_safe_so, self._params.d_safe_do]
        self._ocp.parameter_values = self.create_parameter_values(initial_adjustable_params, nominal_trajectory, do_list, so_list, 0)

        solver_json = "acados_ocp_" + self._ocp.model.name + ".json"
        self._ocp_solver = AcadosOcpSolver(self._ocp, json_file=solver_json)

    def create_parameter_values(self, adjustable_params: list, nominal_trajectory: np.ndarray, do_list: list, so_list: list, stage_idx: int) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            adjustable_params (list): List of adjustable parameter values
            nominal_trajectory (np.ndarray): Nominal trajectory to be followed
            do_list (list): List of dynamic obstacles
            so_list (list): List of static obstacles
            stage_idx (int): Stage index for the shooting node to consider

        Returns:
            np.ndarray: Parameter vector to be used as input to solver
        """
        parameter_values = np.concatenate((np.array(adjustable_params), np.array(nominal_trajectory[:, stage_idx])))
        n_do = len(do_list)
        dt = self._params.dt

        for j in range(MAX_NUM_SO_CONSTRAINTS):
            continue

        for i in range(MAX_NUM_DO_CONSTRAINTS):
            t = i * dt
            if i < n_do:
                (ID, state, cov, length, width) = do_list[i]
                parameter_values = np.concatenate((parameter_values, np.array([state[0] + t * state[2], state[1] + t * state[3], state[2], state[3], length, width])))
            else:
                parameter_values = np.concatenate((parameter_values, np.array([0.0, 0.0, 0.0, 0.0, 5.0, 2.0])))
        return parameter_values
