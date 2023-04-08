"""
    mpc.py

    Summary:
        Contains a class for an NMPC trajectory tracking/path following controller with collision avoidance functionality.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.rl_mpc.integrators as integrators
import rl_rrt_mpc.rl_mpc.models as models
import seacharts.enc as senc
import rl_rrt_mpc.rl_mpc.nmpc_casadi as nmpc_casadi

if ACADOS_COMPATIBLE:
    import rl_rrt_mpc.rl_mpc.nmpc_acados as nmpc_acados


MAX_NUM_DO_CONSTRAINTS: int = 15
MAX_NUM_SO_CONSTRAINTS: int = 300


@dataclass
class NMPCParams:
    acados_nmpc_params: nmpc_acados.AcadosNMPCParams
    casadi_nmpc_params: nmpc_casadi.CasadiNMPCParams


    @classmethod
    def from_dict(cls, config_dict: dict):
        config = NMPCParams(
            reference_traj_bbox_buffer=config_dict["reference_traj_bbox_buffer"],
            T=config_dict["T"],
            dt=config_dict["dt"],
            Q=np.diag(config_dict["Q"]),
            R=np.diag(config_dict["R"]),
            gamma=config_dict["gamma"],
            d_safe_so=config_dict["d_safe_so"],
            d_safe_do=config_dict["d_safe_do"],
            spline_reference=config_dict["spline_reference"],
            path_following=config_dict["path_following"],
            acados=config_dict["acados"],
            casadi_solver_options=CasadiSolverOptions(),
            acados_solver_options=AcadosOcpOptions(),
        )

        if config.path_following and config.Q.shape[0] != 2:
            raise ValueError("Q must be a 2x2 matrix when path_following is True.")

        if not config.path_following and config.Q.shape[0] != 6:
            raise ValueError("Q must be a 6x6 matrix when path_following is False (trajectory tracking).")

        config.casadi_solver_options = CasadiSolverOptions.from_dict(config_dict["casadi_solver_options"])
        config.acados_solver_options = config.parse_acados_solver_options(config_dict["acados_solver_options"])
        return config

    def parse_acados_solver_options(self, config_dict: dict):
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


class NMPC:
    def __init__(self, model: models.TelemetronAcados, params: Optional[NMPCParams] = NMPCParams()) -> None:
        self._acados_ocp: AcadosOcp = AcadosOcp()
        self._model = model
        if params:
            self._params0: NMPCParams = params
            self._params: NMPCParams = params

        nx, nu = self._model.dims
        self._x_warm_start: np.ndarray = np.zeros(nx)
        self._u_warm_start: np.ndarray = np.zeros(nu)
        self._initialized = False
        self._map_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # In east-north coordinates
        self._s: float = 0.0

    @property
    def params(self) -> NMPCParams:
        return self._params

    def update_adjustable_params(self, params: list) -> None:
        """Updates the RL-tuneable parameters in the NMPC.

        Args:
            params (list): List of parameters to update. The order of the parameters are:
                - Q
                - gamma
                - d_safe_so
                - d_safe_do

        Returns:
            - list: List of newly updated parameters.
        """
        nx = self._acados_ocp.model.x.size()[0]
        if self._params.path_following:
            self._params.Q = np.reshape(params[0 : 2 * 2], (2, 2))
        else:
            self._params.Q = np.reshape(params[0 : nx * nx], (nx, nx))
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
        nx = self._acados_ocp.model.x.size()[0]
        return [*self._params.Q.reshape((nx * nx)).tolist(), self._params.gamma, self._params.d_safe_so, self._params.d_safe_do]

    def _compute_path_variable_derivative(self, s: float, nominal_trajectory: list, xs: Optional[np.ndarray]) -> float:
        """Computes the path variable dynamics, i.e. the derivative of the path variable s.

        Args:
            - s (float): Path variable.
            - nominal_trajectory (list): Nominal reference trajectory to track. As list of splines for (x, y, psi, v).
            - xs (Optional[np.ndarray]): Own-ship state.

        Returns:
            float: Derivative of the path variable s.
        """
        x_spline = nominal_trajectory[0]
        y_spline = nominal_trajectory[1]
        speed_spline = nominal_trajectory[3]

        # Use speed spline to compute the path variable derivative, i.e. use a speed profile
        s_dot = speed_spline(s) / np.sqrt(0.0001 + np.power(x_spline(s, 1), 2.0) + np.power(y_spline(s, 1), 2.0))
        # Reference vehicle propagation slows down when the actual vehicle is far away from the reference vehicle
        # s_dot = speed_spline(s) * (1 - 0.1 * np.tanh(np.sqrt((x_spline(s) - xs[0]) ** 2 + (y_spline(s) - xs[1]) ** 2)))
        return s_dot

    def _set_initial_warm_start(self, nominal_trajectory: np.ndarray | list, nominal_inputs: Optional[np.ndarray]) -> None:
        """Sets the initial warm start state (and input) trajectory for the NMPC.

        Args:
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for (x, y, psi, U).
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
        """
        if isinstance(nominal_trajectory, list) and self._params.spline_reference:
            N = int(self._params.T / self._params.dt)
            self._x_warm_start = np.zeros((6, N))
            for i in range(N):
                t = (i * self._params.dt) / (self._params.T - self._params.dt)
                x_d = nominal_trajectory[0](t)
                y_d = nominal_trajectory[1](t)
                x_dot_d = nominal_trajectory[0](t, 1)
                y_dot_d = nominal_trajectory[1](t, 1)
                psi_d = nominal_trajectory[2](t)
                r_d = nominal_trajectory[2](t, 1)
                Rpsi = mf.Rpsi2D(psi_d)
                v_ne = np.array([x_dot_d, y_dot_d])
                v_body = Rpsi.T @ v_ne

                self._x_warm_start[:, i] = np.array(
                    [
                        x_d,
                        y_d,
                        psi_d,
                        v_body[0],
                        v_body[1],
                        r_d,
                    ]
                )
        else:
            self._x_warm_start = nominal_trajectory

        if nominal_inputs is not None:
            self._u_warm_start = nominal_inputs

    def plan(
        self, t: float, nominal_trajectory: np.ndarray | list, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for (x, y, psi, U).
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

        self._update_ocp(t, nominal_trajectory, nominal_inputs, xs, do_list, so_list)
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

    def _update_ocp(self, t: float, nominal_trajectory: np.ndarray | list, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list) -> None:
        """Updates the OCP (cost and constraints) with the current info available

        Args:
            - t (float): Current time.
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for (x, y, psi, U).
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
        """
        adjustable_params = self.get_adjustable_params()
        self._acados_ocp_solver.constraints_set(0, "lbx", xs)
        self._acados_ocp_solver.constraints_set(0, "ubx", xs)
        for i in range(self._acados_ocp.dims.N + 1):
            self._acados_ocp_solver.set(i, "x", self._x_warm_start[:, i])
            if i < self._acados_ocp.dims.N:
                self._acados_ocp_solver.set(i, "u", self._u_warm_start[:, i])
            p_i = self.create_parameter_values(adjustable_params, nominal_trajectory, do_list, so_list, i)
            self._acados_ocp_solver.set(i, "p", p_i)
        print("OCP updated")

    def construct_ocp(self, nominal_trajectory: np.ndarray | list, do_list: list, so_list: list, enc: senc.ENC) -> None:
        """Constructs the OCP for the NMPC problem. Depending on whether or not ACADOS is used, the OCP is constructed using the acados_template or the casadi framework.

        Args:
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for  x, y, psi, U.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.

        """
        self._map_bbox = enc.bbox
        if self._params.acados:
            self._construct_acados_ocp(nominal_trajectory, do_list, so_list, enc)
        else:
            self._construct_casadi_ocp(nominal_trajectory, do_list, so_list, enc)

    def _construct_acados_ocp(self, nominal_trajectory: np.ndarray | list, do_list: list, so_list: list, enc: senc.ENC) -> None:
        """Constructs the OCP for the NMPC problem. Depending on whether or not ACADOS is used, the OCP is constructed using the acados_template or the casadi framework.

         Class constructs an ACADOS tailored OCP on the form:
            min     ∫ Lc(x, u, p) dt + Tc_theta(xf)  (from 0 to Tf)
            s.t.    xdot = f_expl(x, u)
                    xlb <= x <= xub ∀ x
                    ulb <= u <= uub ∀ u
                    hlb <= h(x, u, p) <= hub

            where x, u and p are the state, input and parameter vector, respectively.

        Args:
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for  x, y, psi, U.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.

        """
        min_Fx = self._model.params.Fx_limits[0]
        max_Fx = self._model.params.Fx_limits[1]
        min_Fy = self._model.params.Fy_limits[0]
        max_Fy = self._model.params.Fy_limits[1]
        lever_arm = self._model.params.l_r
        max_turn_rate = self._model.params.r_max
        max_speed = self._model.params.U_max

        self._acados_ocp.model = self._model.as_acados()
        self._acados_ocp.solver_options = self._params.solver_options
        self._acados_ocp.dims.N = int(self._params.T / self._params.dt)
        self._acados_ocp.solver_options.qp_solver_cond_N = self._acados_ocp.dims.N
        self._acados_ocp.solver_options.tf = self._params.T

        nx = self._acados_ocp.model.x.size()[0]
        nu = self._acados_ocp.model.u.size()[0]
        self._acados_ocp.dims.nx = nx
        self._acados_ocp.dims.nu = nu

        x = self._acados_ocp.model.x
        u = self._acados_ocp.model.u

        gamma = csd.MX.sym("gamma", 1)

        self._acados_ocp.cost.cost_type = "EXTERNAL"
        self._acados_ocp.cost.cost_type_e = "EXTERNAL"

        if self._params.path_following:
            x_ref = csd.MX.sym("x_ref", 2)
            Qvec = csd.MX.sym("Q", 2 * 2)
            Qscaling = np.eye(2)
        else:  # trajectory tracking
            x_ref = csd.MX.sym("x_ref", nx)
            Qvec = csd.MX.sym("Q", nx * nx)
            Qscaling = np.eye(nx)
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

        Qmtrx = hf.casadi_matrix_from_vector(Qvec) @ Qscaling
        self._acados_ocp.model.cost_expr_ext_cost = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)
        self._acados_ocp.model.cost_expr_ext_cost_e = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)
        fixed_params = x_ref

        # # soften
        # ocp.constraints.idxsh = np.array([0])
        # ocp.cost.zl = 1e5 * np.array([1])
        # ocp.cost.zu = 1e5 * np.array([1])
        # ocp.cost.Zl = 1e5 * np.array([1])
        # ocp.cost.Zu = 1e5 * np.array([1])

        approx_inf = 1e10

        # Input constraints
        self._acados_ocp.constraints.idxbu = np.array(range(nu))
        self._acados_ocp.constraints.lbu = np.array(
            [
                min_Fx,
                min_Fy,
                lever_arm * min_Fy,
            ]
        )
        self._acados_ocp.constraints.ubu = np.array([max_Fx, max_Fy, lever_arm * max_Fy])

        # State constraints
        lbx = np.array([self._map_bbox[1], self._map_bbox[0], -np.pi, 0.0, -0.6 * max_speed, -max_turn_rate])
        ubx = np.array([self._map_bbox[3], self._map_bbox[2], np.pi, max_speed, 0.6 * max_speed, max_turn_rate])
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
        adjustable_params = csd.vertcat(Qvec, gamma, d_safe_so, d_safe_do)
        self._acados_ocp.model.p = csd.vertcat(adjustable_params, fixed_params)
        self._acados_ocp.dims.np = self._acados_ocp.model.p.size()[0]

        self._acados_ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
        self._acados_ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

        initial_adjustable_params = [*self._params.Q.reshape(nx**2).tolist(), self._params.gamma, self._params.d_safe_so, self._params.d_safe_do]
        self._acados_ocp.parameter_values = self.create_parameter_values(initial_adjustable_params, nominal_trajectory, do_list, so_list, 0)

        solver_json = "acados_ocp_" + self._acados_ocp.model.name + ".json"
        # self._acados_ocp.code_export_directory = "../generated_ocp_" + self._acados_ocp.model.name
        self._acados_ocp_solver: AcadosOcpSolver = AcadosOcpSolver(self._acados_ocp, json_file=solver_json)

    def _construct_casadi_ocp(self, nominal_trajectory: np.ndarray | list, do_list: list, so_list: list, enc: senc.ENC) -> None:
        """
        Class constructs a CASADI OCP on the form:
            min     ∫ Lc(x, u) dt + Tc(xf)  (from 0 to T)
            s.t.    dx/dt = xdot(x, u)
                    xlb <= x <= xub ∀ x
                    ulb <= u <= uub ∀ u
                    clb <= c(x, u, p) <= cub ∀ x, u

        Args:
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for  x, y, psi, U.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.
        """
        N = int(self._params.T / self._params.dt)

        xdot, x, u = self._model.as_casadi()



        # Create symbolic constraint
        self.c = csd.Function("C", [x, u, p], [c], ["x", "u", "p"], ["c"])

        # Create symbolic integrator
        erk4 = integrators.ERK4(x, u, xdot, None, self._params.dt)

        # For plotting x and u given w
        x_plot, u_plot, s_plot = [], [], []

        # Start with an empty NLP (no states w and no constraints g)
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []
        cost = 0

        # "Lift" initial conditions
        x_0 = csd.MX.sym("X_0", x.shape[0])
        x_k = x_0
        w.append(x_k)
        lbw.append(x_0)
        ubw.append(x_0)
        x_plot.append(x_k)

        # Formulate the NLP by iterating over shoting intervals
        for k in range(N):
            # New NLP variable for the control
            u_k = csd.MX.sym("U_" + str(k), u.shape[0])
            w.append(u_k)
            lbw.append(lbu)
            ubw.append(ubu)
            u_plot.append(u_k)

            # Integrate from t_k -> t_k+1 and add variables
            x_k_end, _, _, _, _, _, _, _ = erk4(x_k, u_k)

            # New NLP variable for state at end of interval
            x_k = csd.MX.sym("X_" + str(k + 1), x.shape[0])
            w.append(x_k)
            x_plot.append(x_k)

            # Add constraint on shooting gap
            g.append(x_k_end - x_k)
            lbg.append([0] * x.shape[0])
            ubg.append([0] * x.shape[0])

            # Add nonlinear constraints
            if slack is not None:
                Sk = csd.MX.sym("S_" + str(k + 1), len(clb))
                self.w.append(Sk)
                self.lbw.append([0] * len(clb))
                self.ubw.append([np.inf] * len(clb))
                s_plot.append(Sk)
                g.append(self.c(x_k, u_k, self.p) - Sk)
                cost += slack * csd.sum1(Sk)
            else:
                g.append(self.c(x_k, u_k, self.p))
            lbg.append(clb)
            ubg.append(cub)

            # Add cost contribution
            cost = cost + (x_k) *

        # Add terminal cost
        terminal_cost = csd.Function("Tc", [x, p], [Tc], ["x", "p"], ["terminal_cost"])
        cost = cost + terminal_cost(x_k, self.p)

        # Vectorize and finalize the NLP
        self._casadi_w = csd.vertcat(*w)
        self._casadi_g = csd.vertcat(*g)
        self._casadi_lbw = csd.vertcat(*lbw)
        self._casadi_ubw = csd.vertcat(*ubw)
        self._casadi_lbg = csd.vertcat(*lbg)
        self._casadi_ubg = csd.vertcat(*ubg)
        self._casadi_cost = cost

        # Useful function for extracting x and u trajectories from w vector
        self._casadi_slack = csd.Function("slack", [self._casadi_w], [csd.horzcat(*s_plot)], ["w"], ["s"])
        self._casadi_decision_trajectories = csd.Function(
            "decision_trajectories", [self.w], [csd.horzcat(*x_plot), csd.horzcat(*u_plot), csd.horzcat(*s_plot)], ["w"], ["x", "u", "s"]
        )
        self._casadi_decision_variables = csd.Function(
            "decision_variables", [csd.horzcat(*x_plot), csd.horzcat(*u_plot), csd.horzcat(*s_plot)], [self._casadi_w], ["x", "u", "s"], ["w"]
        )

        # Useful function for generating bounds
        self._casadi_bounds = csd.Function("bounds", [x_0, p], [self._casadi_lbw, self._casadi_ubw, self._casadi_lbg, self._casadi_ubg], ["x_0"], ["lbx", "ubx", "lbg", "ubg"])

        problem = {"f": self._casadi_cost, "x": self._casadi_w, "g": self._casadi_g, "p": self._p}
        self.solver = csd.nlpsol("solver", self._params.solver, problem, self._params.casadi_solver_options)

    def create_parameter_values(self, adjustable_params: list, nominal_trajectory: np.ndarray | list, do_list: list, so_list: list, stage_idx: int) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - adjustable_params (list): List of adjustable parameter values
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for (x, y, psi, U).
            - do_list (list): List of dynamic obstacles.
            - so_list (list): List of static obstacles.
            - stage_idx (int): Stage index for the shooting node to consider

        Returns:
            - np.ndarray: Parameter vector to be used as input to solver
        """
        parameter_values = np.array(adjustable_params)

        if self._params.spline_reference:
            assert isinstance(nominal_trajectory, list)
            s_stage = self._s
            s_dot = self._compute_path_variable_derivative(s_stage, nominal_trajectory, None)
            for i in range(stage_idx):
                s_dot = self._compute_path_variable_derivative(s_stage, nominal_trajectory, None)
                s_stage += self._params.dt * s_dot
            x_d = nominal_trajectory[0](s_stage)
            y_d = nominal_trajectory[1](s_stage)
            x_dot_d = nominal_trajectory[0](s_stage, 1) * s_dot
            y_dot_d = nominal_trajectory[1](s_stage, 1) * s_dot
            psi_d = nominal_trajectory[2](s_stage)
            r_d = nominal_trajectory[2](s_stage, 1)
            Rpsi = mf.Rpsi2D(psi_d)
            v_ne = np.array([x_dot_d, y_dot_d])
            v_body = Rpsi.T @ v_ne
            x_ref_stage = np.array(
                [
                    x_d,
                    y_d,
                    psi_d,
                    v_body[0],
                    v_body[1],
                    r_d,
                ]
            )
        else:
            assert isinstance(nominal_trajectory, np.ndarray)
            x_ref_stage = nominal_trajectory[:, stage_idx]

        if self._params.path_following:
            x_ref_stage = x_ref_stage[0:2]

        parameter_values = np.concatenate((parameter_values, x_ref_stage))
        n_do = len(do_list)
        dt = self._params.dt

        for j in range(MAX_NUM_SO_CONSTRAINTS):
            continue

        for i in range(MAX_NUM_DO_CONSTRAINTS):
            t = stage_idx * dt
            if i < n_do:
                (ID, state, cov, length, width) = do_list[i]
                parameter_values = np.concatenate((parameter_values, np.array([state[0] + t * state[2], state[1] + t * state[3], state[2], state[3], length, width])))
            else:
                parameter_values = np.concatenate((parameter_values, np.array([self._map_bbox[1], self._map_bbox[0], 0.0, 0.0, 5.0, 2.0])))
        return parameter_values

    def build_sensitivity(self, cost, eq_constr, Hu, Hx, Hs):
        # Sensitivity
        lamb = csd.MX.sym("lambda", eq_constr.shape[0])
        mu_u = csd.MX.sym("muu", Hu.shape[0])
        mu_x = csd.MX.sym("mux", Hx.shape[0])
        mu_s = csd.MX.sym("mux", Hs.shape[0])
        mult = csd.vertcat(lamb, mu_u, mu_x, mu_s)

        lagrangian = (
            cost
            + csd.transpose(lamb) @ eq_constr
            + csd.transpose(mu_u) @ Hu
            + csd.transpose(mu_x) @ Hx
            + csd.transpose(mu_s) @ Hs
        )
        lagrangian_function = csd.Function("Lag", [self.Opt_Vars, mult, self.Pf, self.P], [lagrangian])
        lagrangian_function_derivative = lagrangian_function.factory(
            "dLagfunc",
            ["i0", "i1", "i2", "i3"],
            ["jac:o0:i0", "jac:o0:i2", "jac:o0:i3"],
        )
        dLdw, dLdPf, dLdP = lagrangian_function_derivative(self._casadi_w, mult, self.Pf, self.P)
