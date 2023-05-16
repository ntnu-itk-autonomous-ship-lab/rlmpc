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
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.mpc.models as models
import rl_rrt_mpc.mpc.parameters as parameters
import seacharts.enc as senc
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions
from acados_template.acados_ocp_solver import AcadosOcpSolver

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

        self._p_fixed: csd.MX = csd.MX.sym("p_fixed", 0)
        self._p_adjustable: csd.MX = csd.MX.sym("p_adjustable", 0)
        self._p: csd.MX = csd.vertcat(self._p_fixed, self._p_adjustable)

        self._p_fixed_so_values: np.ndarray = np.zeros(0)
        self._p_adjustable_values: np.ndarray = np.zeros(0)

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

    def plan(
        self, xs: np.ndarray, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], do_list: list, so_list: list, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of static obstacle Polygon objects.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - Tuple[np.ndarray, np.ndarray]: Optimal trajectory and inputs for the ownship.
        """
        if not self._initialized:
            self._set_initial_warm_start(nominal_trajectory, nominal_inputs)
            self._initialized = True

        self._update_ocp(xs, nominal_trajectory, nominal_inputs, do_list, so_list)
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

    def _update_ocp(self, xs: np.ndarray, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], do_list: list, so_list: list, **kwargs) -> None:
        """Updates the OCP (cost and constraints) with the current info available

        Args:
            - xs (np.ndarray): Current state.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        """
        self._acados_ocp_solver.constraints_set(0, "lbx", xs)
        self._acados_ocp_solver.constraints_set(0, "ubx", xs)
        for i in range(self._acados_ocp.dims.N + 1):
            self._acados_ocp_solver.set(i, "x", self._x_warm_start[:, i])
            if i < self._acados_ocp.dims.N and nominal_inputs is not None and nominal_inputs.size > 0:
                self._acados_ocp_solver.set(i, "u", self._u_warm_start[:, i])
            p_i = self.create_parameter_values(xs, nominal_trajectory, do_list, so_list, i)
            self._acados_ocp_solver.set(i, "p", p_i)
        print("OCP updated")

    def construct_ocp(self, xs: np.ndarray, nominal_trajectory: np.ndarray, do_list: list, so_list: list, enc: senc.ENC) -> None:
        """Constructs the OCP for the NMPC problem using ACADOS.

         Class constructs an ACADOS tailored OCP on the form:
            min     ∫ Lc(x, u, p) dt + Tc_theta(xf)  (from 0 to Tf)
            s.t.    xdot = f_expl(x, u)
                    lbx <= x <= ubx ∀ x
                    lbu <= u <= ubu ∀ u
                    lbh <= h(x, u, p) <= ubh

            where x, u and p are the state, input and parameter vector, respectively.

        Args:
            - xs (np.ndarray): Current state.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.

        """
        self._map_bbox = enc.bbox
        self._acados_ocp.model = self._model.as_acados()
        ship_vertices = self._model.params().ship_vertices
        self._acados_ocp.dims.N = int(self._params.T / self._params.dt)
        self._acados_ocp.solver_options.qp_solver_cond_N = self._acados_ocp.dims.N
        self._acados_ocp.solver_options.tf = self._params.T

        nx = self._acados_ocp.model.x.size()[0]
        nu = self._acados_ocp.model.u.size()[0]
        self._acados_ocp.dims.nx = nx
        self._acados_ocp.dims.nu = nu

        x = self._acados_ocp.model.x
        u = self._acados_ocp.model.u
        num_fixed_ocp_params = 0
        num_adjustable_ocp_params = 0

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
        Q_vec = csd.MX.sym("Q_vec", dim_Q * dim_Q, 1)
        Qmtrx = (
            hf.casadi_matrix_from_vector(
                Q_vec,
                dim_Q,
                dim_Q,
            )
            @ Qscaling
        )
        self._p_adjustable = csd.vertcat(Q_vec)
        num_adjustable_ocp_params += dim_Q * dim_Q

        x_ref = csd.MX.sym("x_ref", dim_Q)
        gamma = csd.MX.sym("gamma", 1)
        self._acados_ocp.model.cost_expr_ext_cost = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)
        self._acados_ocp.model.cost_expr_ext_cost_e = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)
        self._p_fixed = csd.vertcat(x_ref, gamma)
        num_fixed_ocp_params += dim_Q + 1

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
        self._p_adjustable = csd.vertcat(self._p_adjustable, d_safe_so, d_safe_do)
        num_adjustable_ocp_params += 2

        self._acados_ocp.constraints.lh = np.zeros(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.constraints.lh_e = self._acados_ocp.constraints.lh
        self._acados_ocp.constraints.uh = approx_inf * np.ones(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.constraints.uh_e = self._acados_ocp.constraints.uh

        # Slacks on dynamic obstacle and static obstacle constraints
        self._acados_ocp.constraints.idxsh = np.array(range(self._params.max_num_so_constr + self._params.max_num_do_constr))
        self._acados_ocp.constraints.idxsh_e = np.array(range(self._params.max_num_so_constr + self._params.max_num_do_constr))

        self._acados_ocp.cost.Zl = np.zeros(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.cost.Zl_e = np.zeros(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.cost.Zu = np.zeros(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.cost.Zu_e = np.zeros(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.cost.zl = 1e5 * np.ones(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.cost.zl_e = 1e5 * np.ones(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.cost.zu = 1e5 * np.ones(self._params.max_num_so_constr + self._params.max_num_do_constr)
        self._acados_ocp.cost.zu_e = 1e5 * np.ones(self._params.max_num_so_constr + self._params.max_num_do_constr)

        # Static obstacle constraint parameters
        so_pars = csd.MX.sym("so_pars", 0)
        A_so_constr = csd.MX.sym("A_so_constr", 0)
        b_so_constr = csd.MX.sym("b_so_constr", 0)
        so_surfaces = []
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
            so_surfaces = mapf.compute_surface_approximations_from_polygons(so_list, enc)
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
            so_pars = csd.MX.sym("so_pars", 3, self._params.max_num_so_constr)  # (x_c, y_c, r) x self._params.max_num_so_constr
            self._p_fixed = csd.vertcat(self._p_fixed, csd.reshape(so_pars, -1, 1))
            num_fixed_ocp_params += 3 * self._params.max_num_so_constr  # so_pars
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
            so_pars = csd.MX.sym("so_pars", 2 + 2 * 2, self._params.max_num_so_constr)  # (x_c, y_c, A_c.flatten().tolist()) x self._params.max_num_so_constr
            self._p_fixed = csd.vertcat(self._p_fixed, csd.reshape(so_pars, -1, 1))
            num_fixed_ocp_params += 4 * self._params.max_num_so_constr  # so_pars
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            A_so_constr = csd.MX.sym("A_so_constr", self._params.max_num_so_constr, 2)
            b_so_constr = csd.MX.sym("b_so_constr", self._params.max_num_so_constr, 1)
            self._p_fixed = csd.vertcat(self._p_fixed, csd.reshape(A_so_constr, -1, 1))
            self._p_fixed = csd.vertcat(self._p_fixed, b_so_constr)
            num_fixed_ocp_params += self._params.max_num_so_constr * 3  # A_so_constr and b_so_constr
        else:
            raise ValueError("Unknown static obstacle constraint type.")

        con_h_expr = []
        so_constr_list = self._create_static_obstacle_constraint(x, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, d_safe_so)
        con_h_expr.extend(so_constr_list)

        do_constr_list = self._create_dynamic_obstacle_constraint(x, [], 6, d_safe_do)
        con_h_expr.extend(do_constr_list)

        # Parameters consist of RL adjustable parameters, and fixed parameters
        # (either nominal trajectory or dynamic obstacle related).
        # The model parameters are considered fixed.
        self._acados_ocp.model.p = csd.vertcat(self._p_adjustable, self._p_fixed)
        self._acados_ocp.dims.np = self._acados_ocp.model.p.size()[0]

        self._acados_ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
        self._acados_ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

        self._p_fixed_so_values = self._create_fixed_so_parameter_values(so_list, xs, nominal_trajectory, enc)
        self._acados_ocp.parameter_values = self.create_parameter_values(xs, nominal_trajectory, do_list, so_list, 0)

        solver_json = "acados_ocp_" + self._acados_ocp.model.name + ".json"
        # self._acados_ocp.code_export_directory = "../generated_ocp_" + self._acados_ocp.model.name
        self._acados_ocp_solver: AcadosOcpSolver = AcadosOcpSolver(self._acados_ocp, json_file=solver_json)

    def _create_static_obstacle_constraint(
        self,
        x_k: csd.MX,
        so_pars: csd.MX,
        A_so_constr: Optional[csd.MX],
        b_so_constr: Optional[csd.MX],
        so_surfaces: Optional[list],
        ship_vertices: np.ndarray,
        d_safe_so: csd.MX,
    ) -> list:
        """Creates the static obstacle constraints for the NLP at the current stage, based on the chosen static obstacle constraint type.

        Args:
            - x_k (csd.MX): State vector at the current stage in the OCP.
            - so_pars (csd.MX): Parameters of the static obstacles, used for circular, ellipsoidal constraints
            - A_so_constr (Optional[csd.MX]): Convex safe set constraint matrix if convex safe set constraints are used.
            - b_so_constr (Optional[csd.MX]): Convex safe set constraint vector if convex safe set constraints are used.
            - so_surfaces (Optional[list]): Parametric surface approximations for the static obstacles, if parametric surface constraints are used.
            - ship_vertices (np.ndarray): Vertices of the ship model.
            - d_safe_so (csd.MX): Safety distance to static obstacles.

        Returns:
            list: List of static obstacle constraints at the current stage in the OCP.
        """
        epsilon = 1e-6
        so_constr_list = []
        if self._params.max_num_so_constr == 0:
            return so_constr_list

        if self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            assert A_so_constr is not None and b_so_constr is not None, "Convex safe set constraints must be provided for this constraint type."
            so_constr_list.append(
                # A_so_constr @ x_k[0:2]
                # - b_so_constr
                csd.vec(A_so_constr @ (mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * d_safe_so + x_k[0:2]) - b_so_constr)
            )
        else:
            if self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
                assert so_pars.shape[0] == 3, "Static obstacle parameters with dim 3 in first axis must be provided for this constraint type."
                for j in range(self._params.max_num_so_constr):
                    x_c, y_c, r_c = so_pars[0, j], so_pars[1, j], so_pars[2, j]
                    so_constr_list.append(csd.log(r_c**2 + epsilon) - csd.log(((x_k[0] - x_c) ** 2) + (x_k[1] - y_c) ** 2 + epsilon))
            elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
                assert so_pars.shape[0] == 4, "Static obstacle parameters with dim 4 in first axis must be provided for this constraint type."
                for j in range(self._params.max_num_so_constr):
                    x_e, y_e, A_e = so_pars[0, j], so_pars[1, j], so_pars[2:, j]
                    A_e = csd.reshape(A_e, 2, 2)
                    p_diff_do_frame = x_k[0:2] - csd.vertcat(x_e, y_e)
                    weights = A_e / d_safe_so**2
                    so_constr_list.append(csd.log(1 + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon))
            elif self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
                assert so_surfaces is not None, "Parametric surfaces must be provided for this constraint type."
                n_so = len(so_surfaces)
                for j in range(self._params.max_num_so_constr):
                    if j < n_so:
                        so_constr_list.append(csd.vec(so_surfaces[j](mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * d_safe_so + x_k[0:2])))
                    else:
                        so_constr_list.append(0.0)
        return so_constr_list

    def _create_dynamic_obstacle_constraint(self, x_k: csd.MX, X_do_k: csd.MX, nx_do: int, d_safe_do: csd.MX) -> list:
        """Creates the dynamic obstacle constraints for the NLP at the current stage.

        Args:
            x_k (csd.MX): State vector at the current stage in the OCP.
            sigma_k (csd.MX): Sigma vector at the current stage in the OCP.
            X_do_k (csd.MX): Decision variables of the dynamic obstacles at the current stage in the OCP.
            nx_do (int): Dimension of fixed parameter vector for a dynamic obstacle.
            d_safe_do (csd.MX): Safety distance to dynamic obstacles.

        Returns:
            list: List of dynamic obstacle constraints at the current stage in the OCP.
        """
        do_constr_list = []
        epsilon = 1e-6
        for i in range(self._params.max_num_do_constr):
            x_aug_do_i = X_do_k[nx_do * i : nx_do * (i + 1)]
            x_do_i = x_aug_do_i[0:4]
            l_do_i = x_aug_do_i[4]
            w_do_i = x_aug_do_i[5]
            Rchi_do_i = mf.Rpsi2D_casadi(x_do_i[2])
            p_diff_do_frame = Rchi_do_i @ (x_k[0:2] - x_do_i[0:2])
            weights = hf.casadi_matrix_from_nested_list([[1.0 / (l_do_i + d_safe_do) ** 2, 0.0], [0.0, 1.0 / (w_do_i + d_safe_do) ** 2]])
            do_constr_list.append(csd.log(1 + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon))
        return do_constr_list

    def create_parameter_values(self, state: np.ndarray, nominal_trajectory: np.ndarray, do_list: list, so_list: list, stage_idx: int) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - state (np.ndarray): Current state of the ship.
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

        fixed_so_parameter_values = self._update_so_parameter_values(so_list, state, nominal_trajectory, dt)
        fixed_parameter_values.extend(fixed_so_parameter_values)

        for i in range(self._params.max_num_do_constr):
            t = stage_idx * dt
            if i < n_do:
                (ID, state, cov, length, width) = do_list[i]
                fixed_parameter_values.extend([state[0] + t * state[2], state[1] + t * state[3], state[2], state[3], length, width])
            else:
                fixed_parameter_values.extend([self._map_bbox[1], self._map_bbox[0], 0.0, 0.0, 5.0, 2.0])
        return np.concatenate((adjustable_params, np.array(fixed_parameter_values)))

    def _create_fixed_so_parameter_values(self, so_list: list, state: np.ndarray, nominal_trajectory: np.ndarray, enc: Optional[senc.ENC] = None, **kwargs) -> np.ndarray:
        """Creates the fixed parameter values for the static obstacle constraints.

        Args:
            - so_list (list): List of static obstacles.
            - state (np.ndarray): Current state of the system.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow.
            - enc (senc.ENC): Electronic Navigation Chart (ENC) object.

        Returns:
            np.ndarray: Fixed parameter vector for static obstacles to be used as input to solver
        """
        fixed_so_parameter_values = []
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
            for (c, r) in so_list:
                fixed_so_parameter_values.extend([c[0], c[1], r])
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
            for (c, A) in so_list:
                fixed_so_parameter_values.extend([c[0], c[1], *A.flatten().tolist()])
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.TRIANGULARBOUNDARY:
            for triangle in so_list:
                fixed_so_parameter_values.extend(*triangle)
        return np.array(fixed_so_parameter_values)

    def _update_so_parameter_values(self, so_list: list, state: np.ndarray, nominal_trajectory: np.ndarray, enc: senc.ENC, **kwargs) -> list:
        """Updates the parameter values for the static obstacle constraints in case of changing constraints.

        Args:
            - so_list (list): List of static obstacles.
            - state (np.ndarray): Current state of the system.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow.
            - enc (senc.ENC): Electronic Navigation Chart (ENC) object.

        Returns:
            np.ndarray: Fixed parameter vector for static obstacles to be used as input to solver
        """
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            assert len(so_list) == 2, "Approximate convex safe set constraint requires constraint variables A and b"
            A, b = so_list[0], so_list[1]
            self._p_fixed_so_values = np.concatenate((A.flatten(), b.flatten()), axis=0)
        return self._p_fixed_so_values.tolist()
