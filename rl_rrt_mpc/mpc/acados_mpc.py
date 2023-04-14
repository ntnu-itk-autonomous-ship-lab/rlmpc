"""
    acados_mpc.py

    Summary:
        Contains a class (impl in Acados) for an NMPC trajectory tracking/path following controller with incorporated collision avoidance.

    Author: Trym Tengesdal
"""
from typing import Optional, Tuple, Type, TypeVar

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.mpc.models as models
import rl_rrt_mpc.mpc.parameters as parameters
import seacharts.enc as senc
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions
from acados_template.acados_ocp_solver import AcadosOcpSolver

MAX_NUM_DO_CONSTRAINTS: int = 15
MAX_NUM_SO_CONSTRAINTS: int = 200

ParamClass = TypeVar("ParamClass", bound=parameters.IParams)


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
    acados_solver_options.hessian_approx = config_dict["hessian_approx_type"]
    acados_solver_options.globalization = config_dict["globalization"]
    acados_solver_options.levenberg_marquardt = config_dict["levenberg_marquardt"]
    acados_solver_options.print_level = config_dict["print_level"]
    return acados_solver_options


class AcadosMPC:
    def __init__(self, model: models.Telemetron, params: ParamClass, solver_options: dict) -> None:
        self._acados_ocp: AcadosOcp = AcadosOcp()
        self._acados_ocp.solver_options = parse_acados_solver_options(solver_options)
        self._model = model

        self._params0: ParamClass = params
        self._params: ParamClass = params

        nx, nu = self._model.dims()
        self._x_warm_start: np.ndarray = np.zeros(nx)
        self._u_warm_start: np.ndarray = np.zeros(nu)
        self._initialized = False
        self._map_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # In east-north coordinates
        self._s: float = 0.0

    @property
    def params(self):
        return self._params

    def update_adjustable_params(self, params: list) -> None:
        """Updates the RL-tuneable parameters in the NMPC.

        Args:
            params (list): List of parameters to update. The order of the parameters are:
                - Q
                - d_safe_so
                - d_safe_do

        Returns:
            - list: List of newly updated parameters.
        """
        nx = self._acados_ocp.model.x.size()[0]
        if self._params.path_following:
            dim_Q = 2
        else:
            dim_Q = nx
        self._params.Q = np.reshape(params[0 : dim_Q * dim_Q], (dim_Q, dim_Q))
        self._params.d_safe_so = params[dim_Q * dim_Q]
        self._params.d_safe_do = params[dim_Q * dim_Q + 1]

    def get_adjustable_params(self) -> np.ndarray:
        """Returns the RL-tuneable parameters in the NMPC.

        Returns:
            np.ndarray: Array of parameters. The order of the parameters are:
                - Q
                - d_safe_so
                - d_safe_do
        """
        return self._params.adjustable

    def _set_initial_warm_start(self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray]) -> None:
        """Sets the initial warm start state (and input) trajectory for the NMPC.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
        """
        self._x_warm_start = nominal_trajectory[:6, :]

        if nominal_inputs is not None and nominal_inputs.size > 0:
            self._u_warm_start = nominal_inputs

    def plan(self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of static obstacle Polygon objects.

        Returns:
            - Tuple[np.ndarray, np.ndarray]: Optimal trajectory and inputs for the ownship.
        """
        if not self._initialized:
            self._set_initial_warm_start(nominal_trajectory, nominal_inputs)
            self._initialized = True

        self._update_ocp(nominal_trajectory, nominal_inputs, xs, do_list, so_list)
        status = self._acados_ocp_solver.solve()
        self._acados_ocp_solver.print_statistics()
        t_solve = self._acados_ocp_solver.get_stats("time_tot")
        cost_val = self._acados_ocp_solver.get_cost()

        trajectory = np.zeros((self._acados_ocp.dims.nx, self._acados_ocp.dims.N + 1))
        inputs = np.zeros((self._acados_ocp.dims.nu, self._acados_ocp.dims.N))
        for i in range(self._acados_ocp.dims.N + 1):
            trajectory[:, i] = self._acados_ocp_solver.get(i, "x")
            if i < self._acados_ocp.dims.N:
                inputs[:, i] = self._acados_ocp_solver.get(i, "u").T
        print(f"NMPC: | Runtime: {t_solve} | Cost: {cost_val}")
        self._x_warm_start = trajectory.copy()
        self._u_warm_start = inputs.copy()
        return trajectory[:, : self._acados_ocp.dims.N], inputs[:, : self._acados_ocp.dims.N]

    def _update_ocp(self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list) -> None:
        """Updates the OCP (cost and constraints) with the current info available

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
        """
        self._acados_ocp_solver.constraints_set(0, "lbx", xs)
        self._acados_ocp_solver.constraints_set(0, "ubx", xs)
        for i in range(self._acados_ocp.dims.N + 1):
            self._acados_ocp_solver.set(i, "x", self._x_warm_start[:, i])
            if i < self._acados_ocp.dims.N and nominal_inputs is not None and nominal_inputs.size > 0:
                self._acados_ocp_solver.set(i, "u", self._u_warm_start[:, i])
            p_i = self.create_parameter_values(nominal_trajectory, do_list, so_list, i)
            self._acados_ocp_solver.set(i, "p", p_i)
        print("OCP updated")

    def construct_ocp(self, nominal_trajectory: np.ndarray, do_list: list, so_list: list, enc: senc.ENC) -> None:
        """Constructs the OCP for the NMPC problem using ACADOS.

         Class constructs an ACADOS tailored OCP on the form:
            min     ∫ Lc(x, u, p) dt + Tc_theta(xf)  (from 0 to Tf)
            s.t.    xdot = f_expl(x, u)
                    lbx <= x <= ubx ∀ x
                    lbu <= u <= ubu ∀ u
                    lbh <= h(x, u, p) <= ubh

            where x, u and p are the state, input and parameter vector, respectively.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.

        """
        self._map_bbox = enc.bbox
        self._acados_ocp.model = self._model.as_acados()
        self._acados_ocp.dims.N = int(self._params.T / self._params.dt)
        self._acados_ocp.solver_options.qp_solver_cond_N = self._acados_ocp.dims.N
        self._acados_ocp.solver_options.tf = self._params.T

        nx = self._acados_ocp.model.x.size()[0]
        nu = self._acados_ocp.model.u.size()[0]
        self._acados_ocp.dims.nx = nx
        self._acados_ocp.dims.nu = nu

        x = self._acados_ocp.model.x
        u = self._acados_ocp.model.u

        self._acados_ocp.cost.cost_type = "EXTERNAL"
        self._acados_ocp.cost.cost_type_e = "EXTERNAL"

        if self._params.path_following:
            dim_Q = 2
            Qscaling = np.eye(2)
        else:  # trajectory tracking
            dim_Q = nx
            Qscaling = np.eye(dim_Q)
            # Qscaling = np.diag(
            #     [
            #         1.0 / (self._map_bbox[3] - self._map_bbox[1]) ** 2,
            #         1.0 / (self._map_bbox[2] - self._map_bbox[0]) ** 2,
            #         1.0 / (2.0 * np.pi) ** 2,
            #         1.0 / max_speed**2,
            #         1.0 / (0.6 * max_speed) ** 2,
            #         1.0 / (2.0 * max_turn_rate) ** 2,
            #     ]
            # )
        x_ref = csd.MX.sym("x_ref", dim_Q)
        Q_vec = csd.MX.sym("Q_vec", dim_Q * dim_Q, 1)
        Qmtrx = (
            hf.casadi_matrix_from_vector(
                Q_vec,
                dim_Q,
                dim_Q,
            )
            @ Qscaling
        )
        gamma = csd.MX.sym("gamma", 1)
        self._acados_ocp.model.cost_expr_ext_cost = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)
        self._acados_ocp.model.cost_expr_ext_cost_e = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)
        fixed_params = csd.vertcat(x_ref, gamma)

        approx_inf = 1e10
        lbu, ubu, lbx, ubx = self._model.get_input_state_bounds()

        # Input constraints
        self._acados_ocp.constraints.idxbu = np.array(range(nu))
        self._acados_ocp.constraints.lbu = lbu
        self._acados_ocp.constraints.ubu = ubu

        # State constraints
        self._acados_ocp.constraints.idxbx_0 = np.array(range(nx))
        self._acados_ocp.constraints.lbx_0 = lbx
        self._acados_ocp.constraints.ubx_0 = ubx

        self._acados_ocp.constraints.idxbx = np.array([0, 1, 3, 4, 5])
        self._acados_ocp.constraints.lbx = lbx[self._acados_ocp.constraints.idxbx]
        self._acados_ocp.constraints.ubx = ubx[self._acados_ocp.constraints.idxbx]

        self._acados_ocp.constraints.idxbx_e = np.array([0, 1, 3, 4, 5])
        self._acados_ocp.constraints.lbx_e = lbx[self._acados_ocp.constraints.idxbx_e]
        self._acados_ocp.constraints.ubx_e = ubx[self._acados_ocp.constraints.idxbx_e]

        # Dynamic and static obstacle constraints
        d_safe_so = csd.MX.sym("d_safe_so", 1)
        d_safe_do = csd.MX.sym("d_safe_do", 1)

        self._acados_ocp.constraints.lh = np.zeros(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.constraints.lh_e = self._acados_ocp.constraints.lh
        self._acados_ocp.constraints.uh = approx_inf * np.ones(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.constraints.uh_e = self._acados_ocp.constraints.uh

        # Slacks on dynamic obstacle and static obstacle constraints
        self._acados_ocp.constraints.idxsh = np.array(range(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS))
        self._acados_ocp.constraints.idxsh_e = np.array(range(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS))

        self._acados_ocp.cost.Zl = np.zeros(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.cost.Zl_e = np.zeros(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.cost.Zu = np.zeros(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.cost.Zu_e = np.zeros(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.cost.zl = 1e5 * np.ones(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.cost.zl_e = 1e5 * np.ones(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.cost.zu = 1e5 * np.ones(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._acados_ocp.cost.zu_e = 1e5 * np.ones(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)

        con_h_expr = []

        # Static obstacle polygon constraints
        # so_surfaces = hf.compute_surface_approximations_from_polygons(so_list, enc)
        n_so = 0  # len(so_surfaces)
        for j in range(MAX_NUM_SO_CONSTRAINTS):
            if j < n_so:
                con_h_expr.append(0.0)  # so_surfaces[j](x[:2]))
            else:
                con_h_expr.append(0.0)

        # Ellipsoidal DO constraints
        epsilon_do = 0.0001
        for i in range(MAX_NUM_DO_CONSTRAINTS):
            x_do_i = csd.MX.sym("x_do_" + str(i), 4)
            l_do_i = csd.MX.sym("l_do_" + str(i), 1)
            w_do_i = csd.MX.sym("w_do_" + str(i), 1)
            chi_do_i = csd.atan2(x_do_i[3], x_do_i[2])
            Rchi_do_i = mf.Rpsi2D_casadi(chi_do_i)
            p_diff_do_frame = Rchi_do_i @ (x[0:2] - x_do_i[0:2])
            weights = hf.casadi_matrix_from_nested_list([[1.0 / (l_do_i + d_safe_do) ** 2, 0.0], [0.0, 1.0 / (w_do_i + d_safe_do) ** 2]])
            fixed_params = csd.vertcat(fixed_params, x_do_i, l_do_i, w_do_i)
            con_h_expr.append(csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon_do) - csd.log(1 + epsilon_do))

        # Parameters consist of RL adjustable parameters, and fixed parameters
        # (either nominal trajectory or dynamic obstacle related).
        # The model parameters are considered fixed.
        adjustable_params = csd.vertcat(Q_vec, d_safe_so, d_safe_do)
        self._acados_ocp.model.p = csd.vertcat(adjustable_params, fixed_params)
        self._acados_ocp.dims.np = self._acados_ocp.model.p.size()[0]

        self._acados_ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
        self._acados_ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

        self._acados_ocp.parameter_values = self.create_parameter_values(nominal_trajectory, do_list, so_list, 0)

        solver_json = "acados_ocp_" + self._acados_ocp.model.name + ".json"
        # self._acados_ocp.code_export_directory = "../generated_ocp_" + self._acados_ocp.model.name
        self._acados_ocp_solver: AcadosOcpSolver = AcadosOcpSolver(self._acados_ocp, json_file=solver_json)

    def create_parameter_values(self, nominal_trajectory: np.ndarray, do_list: list, so_list: list, stage_idx: int) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for (x, y, psi, U).
            - do_list (list): List of dynamic obstacles.
            - so_list (list): List of static obstacles.
            - stage_idx (int): Stage index for the shooting node to consider

        Returns:
            - np.ndarray: Parameter vector to be used as input to solver
        """
        adjustable_params = self.get_adjustable_params()

        fixed_parameter_values = []
        if self._params.path_following:
            x_ref_stage = nominal_trajectory[0:2, stage_idx]
        else:
            x_ref_stage = nominal_trajectory[:6, stage_idx]
        fixed_parameter_values.extend(x_ref_stage.tolist())
        fixed_parameter_values.append(self._params.gamma)
        n_do = len(do_list)
        dt = self._params.dt

        for j in range(MAX_NUM_SO_CONSTRAINTS):
            continue

        for i in range(MAX_NUM_DO_CONSTRAINTS):
            t = stage_idx * dt
            if i < n_do:
                (ID, state, cov, length, width) = do_list[i]
                fixed_parameter_values.extend([state[0] + t * state[2], state[1] + t * state[3], state[2], state[3], length, width])
            else:
                fixed_parameter_values.extend([self._map_bbox[1], self._map_bbox[0], 0.0, 0.0, 5.0, 2.0])
        return np.concatenate((adjustable_params, np.array(fixed_parameter_values)))
