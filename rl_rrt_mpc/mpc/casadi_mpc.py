"""
    casadi_mpc.py

    Summary:
        Contains a class for an (RL) NMPC (impl in casadi) trajectory tracking/path following controller with incorporated collision avoidance.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple, TypeVar

import casadi as csd
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.mpc.integrators as integrators
import rl_rrt_mpc.mpc.models as models
import rl_rrt_mpc.mpc.parameters as parameters
import rl_rrt_mpc.mpc.set_generator as sg
import seacharts.enc as senc

ParamClass = TypeVar("ParamClass", bound=parameters.IParams)


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


class CasadiMPC:
    def __init__(
        self, model: models.Telemetron, params: Optional[ParamClass] = parameters.RLMPCParams(), solver_options: CasadiSolverOptions = CasadiSolverOptions()
    ) -> None:
        self._model = model
        if params:
            self._params0: ParamClass = params
            self._params: ParamClass = params

        self._solver_options: CasadiSolverOptions = solver_options
        self._initialized = False
        self._map_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # In east-north coordinates

        self._opt_vars: csd.MX = csd.MX.sym("opt_vars", 0)  # Optimization variables w
        self._lbw: np.ndarray = np.array([])
        self._ubw: np.ndarray = np.array([])
        self._vsolver: csd.Function = csd.Function("vsolver", [], [])
        self._prev_vsoln: dict = {"x": [], "lam_x0": [], "lam_g": []}
        self._lbg_v: np.ndarray = np.array([])
        self._ubg_v: np.ndarray = np.array([])
        self._qsolver: csd.Function = csd.Function("qsolver", [], [])
        self._prev_qsoln: dict = {"x": [], "lam_x0": [], "lam_g": []}
        self._lbg_q: np.ndarray = np.array([])
        self._ubg_q: np.ndarray = np.array([])

        self._dlag_v: csd.Function = csd.Function("dlag_v", [], [])
        self._dlag_q: csd.Function = csd.Function("dlag_q", [], [])

        self._num_ocp_params: int = 0
        self._num_fixed_ocp_params: int = 0
        self._num_adjustable_ocp_params: int = 0
        self._p_fixed: csd.MX = csd.MX.sym("p_fixed", 0)
        self._p_adjustable: csd.MX = csd.MX.sym("p_adjustable", 0)
        self._p: csd.MX = csd.vertcat(self._p_fixed, self._p_adjustable)

        self._set_generator: Optional[sg.SetGenerator] = None
        self._p_fixed_so_values: np.ndarray = np.array([])
        self._p_fixed_values: np.ndarray = np.array([])
        self._p_adjustable_values: np.ndarray = np.array([])

        self._decision_trajectories: csd.Function = csd.Function("decision_trajectories", [], [])
        self._decision_variables: csd.Function = csd.Function("decision_variables", [], [])
        self._equality_constraints: csd.Function = csd.Function("equality_constraints", [], [])
        self._inequality_constraints: csd.Function = csd.Function("inequality_constraints", [], [])

    @property
    def params(self):
        return self._params

    def get_adjustable_params(self) -> np.ndarray:
        """Returns the RL-tuneable parameters in the NMPC.

        Returns:
            np.ndarray: Array of parameters. The order of the parameters are:
                - Q (flattened)
                - d_safe_so
                - d_safe_do
        """
        return self._params.adjustable

    def _set_initial_warm_start(self, xs: np.ndarray, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray]) -> None:
        """Sets the initial warm start decision trajectory [U, X, Sigma] flattened for the NMPC.

        Args:
            - xs (np.ndarray): Initial state of the system.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow. Used to set the initial warm start state trajectory.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected. Used to set the initial warm start input trajectory, if provided.
        """
        nx, nu = self._model.dims()
        N = int(self._params.T / self._params.dt)
        if nominal_inputs is not None:
            w = nominal_inputs.T.flatten()
        else:
            w = np.zeros(N * nu)
        nominal_trajectory_cpy = nominal_trajectory.copy()
        psi = nominal_trajectory_cpy[2, :]
        psi_unwrapped = np.unwrap(np.concatenate(([xs[2]], psi)))[1:]
        nominal_trajectory_cpy[2, :] = psi_unwrapped
        w = np.concatenate((w, nominal_trajectory_cpy[0:nx, 0 : N + 1].T.flatten()))
        w = np.concatenate((w, np.zeros((self._params.max_num_so_constr + self._params.max_num_do_constr) * (N + 1))))
        self._prev_vsoln["x"] = w.tolist()
        self._prev_vsoln["lam_x"] = np.zeros(w.shape[0]).tolist()
        self._prev_vsoln["lam_g"] = np.zeros(self._lbg_v.shape[0]).tolist()

        # self._prev_vsoln = {"x": [], "lam_x": [], "lam_g": []}

    def plan(
        self,
        nominal_trajectory: np.ndarray,
        nominal_inputs: Optional[np.ndarray],
        xs: np.ndarray,
        do_list: list,
        so_list: list,
        enc: Optional[senc.ENC],
        show_plots: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track or path to follow.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List ofrelevant static obstacle Polygon objects.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - Tuple[np.ndarray, np.ndarray, dict]: Optimal trajectory and inputs for the ownship. Also, the NLP solution dictionary is returned.
        """
        if not self._initialized:
            self._set_initial_warm_start(xs, nominal_trajectory, nominal_inputs)
            self._p_fixed_so_values = self._create_fixed_so_parameter_values(so_list, xs, nominal_trajectory, enc, **kwargs)
            self._initialized = True
        N = int(self._params.T / self._params.dt)
        action = None
        if nominal_inputs is not None:
            action = nominal_inputs[:, 0]

        parameter_values = self.create_parameter_values(xs, action, nominal_trajectory, do_list, so_list, enc, **kwargs)
        soln = self._vsolver(
            x0=self._prev_vsoln["x"],
            lam_x0=self._prev_vsoln["lam_x"],
            lam_g0=self._prev_vsoln["lam_g"],
            p=parameter_values,
            lbx=self._lbw,
            ubx=self._ubw,
            lbg=self._lbg_v,
            ubg=self._ubg_v,
        )
        stats = self._vsolver.stats()
        if not stats["success"]:
            RuntimeError("Problem is Infeasible")

        g_eq_vals = self._equality_constraints(soln["x"], parameter_values).full()
        g_ineq_vals = self._inequality_constraints(soln["x"], parameter_values).full()
        X, U, Sigma = self._decision_trajectories(soln["x"])
        X = X.full()
        U = U.full()
        Sigma = Sigma.full()
        psi = X[2, :]
        psi = np.unwrap(np.concatenate(([xs[2]], psi)))[1:]
        X[2, :] = psi
        self._prev_vsoln["x"] = self._decision_variables(X, U, Sigma)
        self._prev_vsoln["lam_x"] = soln["lam_x"].full()
        self._prev_vsoln["lam_g"] = soln["lam_g"].full()
        # plot_solver_stats(stats, show_plots)
        print(f"Inf norm opt slack variables: {np.max(Sigma)} | Max g_ineq: {np.max(g_ineq_vals)}")
        return X[:, :N], U, soln

    def construct_ocp(self, so_list: list, enc: senc.ENC) -> None:
        """Constructs the OCP for the NMPC problem using pure Casadi.

        Class constructs a "CASADI" (ipopt) tailored OCP on the form (same as for the ACADOS MPC):
            min     ∫ Lc(x, u, p) dt + Tc_theta(xf)  (from 0 to Tf)
            s.t.    xdot = f_expl(x, u)
                    lbx <= x <= ubx ∀ x
                    lbu <= u <= ubu ∀ u
                    0 <= sigma ∀ sigma
                    lbh <= h(x, u, p) <= ubh

            where x, u and p are the state, input and parameter vector, respectively.

            Since this is Casadi, this OCP must be converted to an NLP on the form

            min_w     J(w, p)
            s.t.    lbw <= w <= ubw ∀ w
                    lbg <= g(w, p) <= ubg

        Args:
            - so_list (list): List of compatible static obstacle Polygon objects with the static obstacle constraint type.
            - enc (senc.ENC): ENC object.
        """
        self._map_bbox = enc.bbox
        N = int(self._params.T / self._params.dt)
        dt = self._params.dt

        nx, nu = self._model.dims()
        xdot, x, u = self._model.as_casadi()

        # Box constraints on NLP decision variables
        lbu_k, ubu_k, lbx_k, ubx_k = self._model.get_input_state_bounds()
        lbx_k[0] = self._map_bbox[1]
        lbx_k[1] = self._map_bbox[0]
        ubx_k[0] = self._map_bbox[3]
        ubx_k[1] = self._map_bbox[2]
        lbu = np.tile(lbu_k, N)
        ubu = np.tile(ubu_k, N)
        lbx = np.tile(lbx_k, N + 1)
        ubx = np.tile(ubx_k, N + 1)
        lbsigma = np.array([0] * (N + 1) * (self._params.max_num_so_constr + self._params.max_num_do_constr))
        ubsigma = np.array([np.inf] * (N + 1) * (self._params.max_num_so_constr + self._params.max_num_do_constr))
        self._lbw = np.concatenate((lbu, lbx, lbsigma))
        self._ubw = np.concatenate((ubu, ubx, ubsigma))

        g_eq_list = []  # NLP equality constraints
        g_ineq_list = []  # NLP inequality constraints
        p_fixed, p_adjustable = [], []  # NLP parameters
        num_fixed_ocp_params, num_adjustable_ocp_params = 0, 0

        # NLP decision variables
        U = []
        X = []
        Sigma = []

        # Initial state constraint
        x_0 = csd.MX.sym("x_0_constr", nx, 1)
        x_k = csd.MX.sym("x_0", nx, 1)
        X.append(x_k)
        g_eq_list.append(x_0 - x_k)
        p_fixed.append(x_0)
        num_fixed_ocp_params += nx  # x_0

        # Initial slack
        sigma_k = csd.MX.sym("sigma_0", self._params.max_num_so_constr + self._params.max_num_do_constr, 1)
        Sigma.append(sigma_k)

        # Add the initial action u_0 as parameter, relevant for the Q-function approximator
        u_0 = csd.MX.sym("u_0_constr", nu, 1)
        p_fixed.append(u_0)
        num_fixed_ocp_params += nu  # u_0

        if self._params.path_following:
            dim_Q = 2
        else:
            dim_Q = nx
        X_ref = csd.MX.sym("X_ref", dim_Q, N + 1)
        Q_vec = csd.MX.sym("Q_vec", dim_Q * dim_Q, 1)
        Qmtrx = hf.casadi_matrix_from_vector(Q_vec, dim_Q, dim_Q)
        p_fixed.append(csd.reshape(X_ref, dim_Q * (N + 1), 1))
        num_fixed_ocp_params += dim_Q * (N + 1)  # X_ref

        p_adjustable.append(Q_vec)
        num_adjustable_ocp_params += dim_Q * dim_Q  # Q_vec

        gamma = csd.MX.sym("gamma", 1)
        p_fixed.append(gamma)
        num_fixed_ocp_params += 1  # gamma

        # Slack weighting matrix W (dim = 1 x (self._params.max_num_so_constr + self._params.max_num_do_constr))
        W = csd.MX.sym("W", self._params.max_num_so_constr + self._params.max_num_do_constr, 1)
        p_fixed.append(W)
        num_fixed_ocp_params += self._params.max_num_so_constr + self._params.max_num_do_constr  # W

        # Safety zone parameters
        d_safe_so = csd.MX.sym("d_safe_so", 1)
        d_safe_do = csd.MX.sym("d_safe_do", 1)
        p_adjustable.append(d_safe_so)
        p_adjustable.append(d_safe_do)
        num_adjustable_ocp_params += 2  # d_safe_so and d_safe_do

        ship_vertices = self._model.params().ship_vertices

        # Dynamic obstacle augmented state parameters (x, y, Vx, Vy, length, width) * N + 1
        nx_do = 6
        X_do = csd.MX.sym("X_do", nx_do * self._params.max_num_do_constr, N + 1)
        p_fixed.append(csd.reshape(X_do, -1, 1))

        # Static obstacle constraint parameters
        so_pars = csd.MX.sym("so_pars", 0)
        A_so_constr = csd.MX.sym("A_so_constr", 0)
        b_so_constr = csd.MX.sym("b_so_constr", 0)
        so_surfaces = []
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
            so_surfaces = mapf.compute_surface_approximations_from_polygons(so_list, enc)
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
            so_pars = csd.MX.sym("so_pars", 3, self._params.max_num_so_constr)  # (x_c, y_c, r) x self._params.max_num_so_constr
            p_fixed.append(csd.reshape(so_pars, -1, 1))
            num_fixed_ocp_params += 3 * self._params.max_num_so_constr  # so_pars
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
            so_pars = csd.MX.sym("so_pars", 2 + 2 * 2, self._params.max_num_so_constr)  # (x_c, y_c, A_c.flatten().tolist()) x self._params.max_num_so_constr
            p_fixed.append(csd.reshape(so_pars, -1, 1))
            num_fixed_ocp_params += 4 * self._params.max_num_so_constr  # so_pars
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            A_so_constr = csd.MX.sym("A_so_constr", self._params.max_num_so_constr, 2)
            b_so_constr = csd.MX.sym("b_so_constr", self._params.max_num_so_constr, 1)
            p_fixed.append(csd.reshape(A_so_constr, -1, 1))
            p_fixed.append(csd.reshape(b_so_constr, -1, 1))
            num_fixed_ocp_params += self._params.max_num_so_constr * 3  # A_so_constr and b_so_constr
        else:
            raise ValueError("Unknown static obstacle constraint type.")

        # Cost function
        J = 0.0

        # Create symbolic integrator for the shooting gap constraints and discretized cost function
        stage_cost = quadratic_cost(x_k[0:dim_Q], X_ref[:, 0], Qmtrx)
        erk4 = integrators.ERK4(x=x, p=u, ode=xdot, quad=stage_cost, h=dt)
        for k in range(N):
            u_k = csd.MX.sym("u_" + str(k), nu, 1)
            U.append(u_k)

            # Sum stage costs
            J += gamma**k * (quadratic_cost(x_k[0:dim_Q], X_ref[:, k], Qmtrx) + W.T @ sigma_k)

            so_constr_k = self._create_static_obstacle_constraint(x_k, sigma_k, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, d_safe_so)
            g_ineq_list.extend(so_constr_k)

            do_constr_k = self._create_dynamic_obstacle_constraint(x_k, sigma_k, X_do, nx_do, d_safe_do)
            g_ineq_list.extend(do_constr_k)

            # Shooting gap constraints
            x_k_end, _, _, _, _, _, _, _ = erk4(x_k, u_k)
            x_k = csd.MX.sym("x_" + str(k + 1), nx, 1)
            X.append(x_k)
            g_eq_list.append(x_k_end - x_k)

            sigma_k = csd.MX.sym("sigma_" + str(k + 1), self._params.max_num_so_constr + self._params.max_num_do_constr, 1)
            Sigma.append(sigma_k)

        # Terminal costs and constraints
        J += gamma**N * (quadratic_cost(x_k[:dim_Q], X_ref[:, N], Qmtrx) + W.T @ sigma_k)

        so_constr_N = self._create_static_obstacle_constraint(x_k, sigma_k, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, d_safe_so)
        g_ineq_list.extend(so_constr_N)

        do_constr_N = self._create_dynamic_obstacle_constraint(x_k, sigma_k, X_do[:, N], nx_do, d_safe_do)
        g_ineq_list.extend(do_constr_N)

        # Vectorize and finalize the NLP
        g_eq = csd.vertcat(*g_eq_list)
        g_ineq = csd.vertcat(*g_ineq_list)

        lbg_eq = [0.0] * g_eq.shape[0]
        ubg_eq = [0.0] * g_eq.shape[0]
        lbg_ineq = [-np.inf] * g_ineq.shape[0]
        ubg_ineq = [0.0] * g_ineq.shape[0]
        self._lbg_v = np.concatenate((lbg_eq, lbg_ineq), axis=0)
        self._ubg_v = np.concatenate((ubg_eq, ubg_ineq), axis=0)

        self._p_fixed = csd.vertcat(*p_fixed)
        self._p_adjustable = csd.vertcat(*p_adjustable)
        self._p = csd.vertcat(*p_adjustable, *p_fixed)

        self._num_fixed_ocp_params = num_fixed_ocp_params
        self._num_adjustable_ocp_params = num_adjustable_ocp_params
        self._num_ocp_params = num_fixed_ocp_params + num_adjustable_ocp_params
        self._p_fixed_values = np.zeros((self._num_fixed_ocp_params, 1))
        self._p_adjustable_values = self.get_adjustable_params()

        self._opt_vars = csd.vertcat(*U, *X, *Sigma)

        # Create value (v) function approximation MPC solver
        vnlp_prob = {
            "f": J,
            "x": self._opt_vars,
            "p": self._p,
            "g": csd.vertcat(g_eq, g_ineq),
        }
        self._vsolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob, self._solver_options.to_opt_settings())

        self._equality_constraints = csd.Function("equality_constraints", [self._opt_vars, self._p], [g_eq], ["w", "p"], ["g_eq"])
        self._inequality_constraints = csd.Function("inequality_constraints", [self._opt_vars, self._p], [g_ineq], ["w", "p"], ["g_ineq"])
        if self._params.max_num_so_constr + self._params.max_num_do_constr == 0.0:
            self._decision_trajectories = csd.Function(
                "decision_trajectories",
                [self._opt_vars],
                [
                    csd.reshape(csd.vertcat(*X), nx, -1),
                    csd.reshape(csd.vertcat(*U), nu, -1),
                    csd.vertcat(*Sigma),
                ],
                ["w"],
                ["X", "U", "Sigma"],
            )
            self._decision_variables = csd.Function(
                "decision_variables",
                [
                    csd.reshape(csd.vertcat(*X), nx, -1),
                    csd.reshape(csd.vertcat(*U), nu, -1),
                    csd.vertcat(*Sigma),
                ],
                [self._opt_vars],
                ["X", "U", "Sigma"],
                ["w"],
            )
        else:
            self._decision_trajectories = csd.Function(
                "decision_trajectories",
                [self._opt_vars],
                [
                    csd.reshape(csd.vertcat(*X), nx, -1),
                    csd.reshape(csd.vertcat(*U), nu, -1),
                    csd.reshape(csd.vertcat(*Sigma), self._params.max_num_so_constr + self._params.max_num_do_constr, -1),
                ],
                ["w"],
                ["X", "U", "Sigma"],
            )
            self._decision_variables = csd.Function(
                "decision_variables",
                [
                    csd.reshape(csd.vertcat(*X), nx, -1),
                    csd.reshape(csd.vertcat(*U), nu, -1),
                    csd.reshape(csd.vertcat(*Sigma), self._params.max_num_so_constr + self._params.max_num_do_constr, -1),
                ],
                [self._opt_vars],
                ["X", "U", "Sigma"],
                ["w"],
            )
        # self._dlag_v = self.build_sensitivity(J, g_eq, g_ineq)

        # Create action-value (q or Q(s, a)) function approximation
        g_eq = csd.vertcat(g_eq, u_0 - U[0])
        lbg_eq = [0.0] * g_eq.shape[0]
        ubg_eq = [0.0] * g_eq.shape[0]
        self._lbg_q = np.concatenate((lbg_eq, lbg_ineq), axis=0)
        self._ubg_q = np.concatenate((ubg_eq, ubg_ineq), axis=0)

        qnlp_prob = {"f": J, "x": self._opt_vars, "p": self._p, "g": csd.vertcat(g_eq, g_ineq)}
        self._qsolver = csd.nlpsol("qsolver", "ipopt", qnlp_prob, self._solver_options.to_opt_settings())
        # self._dlag_q = self.build_sensitivity(J, g_eq, g_ineq)

    def _create_static_obstacle_constraint(
        self,
        x_k: csd.MX,
        sigma_k: csd.MX,
        so_pars: list,
        A_so_constr: Optional[csd.MX],
        b_so_constr: Optional[csd.MX],
        so_surfaces: Optional[list],
        ship_vertices: np.ndarray,
        d_safe_so: csd.MX,
    ) -> list:
        """Creates the static obstacle constraints for the NLP at the current stage, based on the chosen static obstacle constraint type.

        Args:
            - x_k (csd.MX): State vector at the current stage in the OCP.
            - sigma_k (csd.MX): Sigma vector at the current stage in the OCP.
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
                # - sigma_k[: self._params.max_num_so_constr]
                csd.vec(A_so_constr @ (mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * d_safe_so + x_k[0:2]) - b_so_constr - sigma_k[: self._params.max_num_so_constr])
            )
        else:
            if self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
                assert so_pars.shape[0] == 3, "Static obstacle parameters with dim 3 in first axis must be provided for this constraint type."
                for j in range(self._params.max_num_so_constr):
                    x_c, y_c, r_c = so_pars[0, j], so_pars[1, j], so_pars[2, j]
                    so_constr_list.append(csd.log(r_c**2 - sigma_k[j] + epsilon) - csd.log(((x_k[0] - x_c) ** 2) + (x_k[1] - y_c) ** 2 + epsilon))
            elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
                assert so_pars.shape[0] == 4, "Static obstacle parameters with dim 4 in first axis must be provided for this constraint type."
                for j in range(self._params.max_num_so_constr):
                    x_e, y_e, A_e = so_pars[0, j], so_pars[1, j], so_pars[2:, j]
                    A_e = csd.reshape(A_e, 2, 2)
                    p_diff_do_frame = x_k[0:2] - csd.vertcat(x_e, y_e)
                    weights = A_e / d_safe_so**2
                    so_constr_list.append(csd.log(1 - sigma_k[j] + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon))
            elif self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
                assert so_surfaces is not None, "Parametric surfaces must be provided for this constraint type."
                n_so = len(so_surfaces)
                for j in range(self._params.max_num_so_constr):
                    if j < n_so:
                        # so_constr_list.append(so_surfaces[j](x_k[0:2]) - sigma_k[j])
                        so_constr_list.append(csd.vec(so_surfaces[j](mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * d_safe_so + x_k[0:2]) - sigma_k[j]))
                    else:
                        so_constr_list.append(-sigma_k[j])
        return so_constr_list

    def _create_dynamic_obstacle_constraint(self, x_k: csd.MX, sigma_k: csd.MX, X_do_k: csd.MX, nx_do: int, d_safe_do: csd.MX) -> list:
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
            do_constr_list.append(csd.log(1 - sigma_k[self._params.max_num_so_constr + i] + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon))
        return do_constr_list

    def create_parameter_values(
        self, state: np.ndarray, action: Optional[np.ndarray], nominal_trajectory: np.ndarray, do_list: list, so_list: list, enc: Optional[senc.ENC] = None, **kwargs
    ) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - state (np.ndarray): Current state of the system.
            - action (np.ndarray): Current action of the system.
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track or path to follow.
            - do_list (list): List of dynamic obstacles.
            - so_list (list): List of static obstacles.
            - enc (Optional[senc.ENC]): Electronic Navigation Chart (ENC) object.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.


        Returns:
            - np.ndarray: Parameter vector to be used as input to solver
        """
        nx, nu = self._model.dims()
        N = int(self._params.T / self._params.dt)
        adjustable_parameter_values = self.get_adjustable_params()
        fixed_parameter_values: list = []
        fixed_parameter_values.extend(state.tolist())
        if action is not None:
            fixed_parameter_values.extend(action.tolist())
        else:
            fixed_parameter_values.extend([0.0] * nu)

        dim_Q = nx
        if self._params.path_following:
            dim_Q = 2

        fixed_parameter_values.extend(nominal_trajectory[0:dim_Q, : N + 1].T.flatten().tolist())
        fixed_parameter_values.append(self._params.gamma)
        W = 1e3 * np.ones(self._params.max_num_so_constr + self._params.max_num_do_constr)
        fixed_parameter_values.extend(W.tolist())
        n_do = len(do_list)
        for k in range(N + 1):
            t = k * self._params.dt
            for i in range(self._params.max_num_do_constr):
                if i < n_do:
                    (ID, state, cov, length, width) = do_list[i]
                    chi = np.atan2(state[3], state[2])
                    U = np.sqrt(state[2] ** 2 + state[3] ** 2)
                    fixed_parameter_values.extend([state[0] + t * U * np.cos(chi), state[1] + t * U * np.sin(chi), chi, U, length, width])
                else:
                    fixed_parameter_values.extend([self._map_bbox[1], self._map_bbox[0], 0.0, 0.0, 5.0, 2.0])

        so_parameter_values = self._update_so_parameter_values(so_list, state, nominal_trajectory, enc, **kwargs)
        fixed_parameter_values.extend(so_parameter_values)
        return np.concatenate((adjustable_parameter_values, np.array(fixed_parameter_values)), axis=0)

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

    def build_sensitivity(self, cost: csd.MX, g_eq: csd.MX, g_ineq: csd.MX) -> dict:
        """Builds the sensitivity of the Lagrangian (lag) defined by the inputs.

        L = cost + lamb.T @ g_eq + mu.T @ g_ineq

        Args:
            cost (_type_): Cost function
            g_eq (_type_): Equality constraints
            g_ineq (_type_): Inequality constraints

        Returns:
            dict: Dictionary containing the lagrangian function and its derivative funcition + sensitivities wrt decision variables and parameters.
        """
        lamb = csd.MX.sym("lambda", g_eq.shape[0])
        mu = csd.MX.sym("mu", g_ineq.shape[0])
        mult = csd.vertcat(lamb, mu)

        lag = cost + csd.transpose(lamb) @ g_eq + csd.transpose(mu) @ g_ineq
        lag_func = csd.Function("lagrangian", [self._opt_vars, mult, self._p_fixed, self._p_adjustable], [lag])
        dlag_func = lag_func.factory(
            "lagrangian_derivative_func",
            ["i0", "i1", "i2", "i3"],
            ["jac:o0:i0", "jac:o0:i2", "jac:o0:i3"],
        )

        # Compute the lagrangian sensitivities wrt decision variables and parameters
        dlag_dw, dlag_dp_fixed, dlag_dp_adjustable = dlag_func(self._opt_vars, mult, self._p_fixed, self._p_adjustable)

        # # # Build KKT matrix
        # R_kkt = csd.vertcat(
        #     csd.transpose(dlag_dw),
        #     g_eq,
        #     mu * g_ineq + self._etau,
        # )

        # # z contains all variables of the lagrangian
        # z = csd.vertcat(self._opt_vars, lamb, mu)

        # # Generate sensitivity of the KKT matrix
        # R_func = csd.Function("kkt_matrix_func", [z, self._p_fixed, self._p_adjustable], [R_kkt])
        # dR_kkt_func = R_func.factory("kkt_matrix_derivative_func", ["i0", "i1", "i2"], ["jac:o0:i0", "jac:o0:i2"])
        # [dR_kkt_dz, dR_kkt_dp] = dR_kkt_func(z, self._p_fixed, self._p_adjustable)

        # # Generate sensitivity of the optimal solution
        # dz_dp = -csd.inv(dR_kkt_dz) @ dR_kkt_dp
        # dz_dp_func = csd.Function("dz_dp_func", [z, self._p_fixed, self._p_adjustable], [dz_dp])
        # dR_kkt_dz_func = csd.Function("dR_kkt_dz_func", [z, self._p_fixed, self._p_adjustable], [dR_kkt_dz])
        # dR_kkt_dp_func = csd.Function("dR_kkt_dp_func", [z, self._p_fixed, self._p_adjustable], [dR_kkt_dp])

        output_dict = {
            "lag": lag_func,
            "dlag_func": dlag_func,
            "dlag_dw": dlag_dw,
            "dlag_dp_fixed": dlag_dp_fixed,
            "dlag_dp_adjustable": dlag_dp_adjustable,
        }
        return output_dict

    def policy_gradient_wrt_parameters(self, state: np.ndarray, soln: dict, parameter_values: np.ndarray) -> np.ndarray:
        """Computes the sensitivity of the policy output with respect to the learnable parameters.

        This is basically the Jacobian/gradient of the policy output with respect to the learnable parameters.

        Args:
            state (np.ndarray): State vector
            soln (dict): Solution dictionary
            parameter_values (np.ndarray): Parameter vector

        Returns:
            np.ndarray: Sensitivity of the policy output with respect to the learnable parameters
        """
        nx, nu = self._model.dims()
        w = soln["x"].full()
        lamb_g = soln["lamb_g"].full()
        z = np.concatenate((w, lamb_g), axis=0)

        parameter_values[:nx] = state
        jacob_act = self.dPi(z, self._p_fixed_values, self._p_adjustable_values).full()
        return jacob_act[:nu, :]

    def action_value(self, state: np.ndarray, action: np.ndarray, parameter_values: np.ndarray):
        """Computes the action value function Q(s, a) for a given state and action.

        Args:
            state (np.ndarray): State vector
            action (np.ndarray): Action vector
            parameter_values (np.ndarray, optional): Parameter vector for the MPC NLP problem.

        Returns:
            _type_: _description_
        """
        nx, nu = self._model.dims()
        self._p_fixed_values[:nx, :] = state
        self._p_fixed_values[nx : nx + nu, :] = action

        qsoln = self._qsolver(
            x0=state,
            p=np.concatenate([self._p_fixed_values, self._p_adjustable_values]),
            lbg=self._lbg_q,
            ubg=self._ubg_q,
        )
        fl = self._qsolver.stats()
        if not fl["success"]:
            RuntimeError("Problem is Infeasible")

        q = qsoln["f"].full()[0, 0]
        return q, qsoln

    def dQdP(self, state: np.ndarray, action: np.ndarray, soln: dict, parameter_values):
        # Gradient of action-value fn Q wrt lernable param
        # state, action, act_wt need to be from qsoln (garbage in garbage out)
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()

        nx, nu = self._model.dims()
        self._p_fixed_values[:nx, :] = state
        self._p_fixed_values[nx : nx + nu, :] = action

        _, _, dlag_dp_adjustable = self._dlag_q(x, lam_g, self._p_fixed_values, parameter_values)
        return dlag_dp_adjustable.full()

    def parameter_update(self, lr, dJ, parameter_values):
        # Param update scheme
        if self._constrained_updates:
            self._p_adjustable_values = self.constrained_param_update(lr, dJ, parameter_values)
        else:
            self._p_adjustable_values -= lr * dJ

    def constrained_parameter_update(self, learning_rate: float, dJ: np.ndarray, parameter_values: np.ndarray) -> np.ndarray:
        """Constrained parameter update scheme to ensure stable MPC formulation

        Args:
            learning_rate (float): Learning rate for the parameter update
            dJ (np.ndarray): Gradient of the cost function
            parameter_values (np.ndarray): Adjustable parameter vector before the update

        Returns:
            np.ndarray: Updated parameter vector
        """
        # SDP for param update to ensure stable MPC formulation
        nx, _ = self._model.dims()
        dp = cvx.Variable((self._num_adjustable_ocp_params, 1))
        J_up = 0.5 * cvx.sum_squares(dp) + learning_rate * dJ.T @ dp
        p_next = parameter_values + dp
        constraint = []
        constraint += [
            cvx.reshape(
                p_next[0 : nx * nx],
                (nx, nx),
            )
            >> 0.0
        ]
        prob = cvx.Problem(cvx.Minimize(J_up), constraint)
        prob.solve(solver="CVXOPT")
        P_up = parameter_values + dp.value
        return P_up


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


def plot_solver_stats(stats: dict, show_plots: bool = True) -> None:
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
