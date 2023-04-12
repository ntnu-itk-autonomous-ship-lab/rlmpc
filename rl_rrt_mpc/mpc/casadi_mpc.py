"""
    casadi_mpc.py

    Summary:
        Contains a class for an (RL) NMPC (impl in casadi) trajectory tracking/path following controller with incorporated collision avoidance.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.mpc.integrators as integrators
import rl_rrt_mpc.mpc.models as models
import rl_rrt_mpc.mpc.parameters as parameters
import seacharts.enc as senc

MAX_NUM_DO_CONSTRAINTS: int = 15
MAX_NUM_SO_CONSTRAINTS: int = 300


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
        }
        return opts


class CasadiMPC:
    def __init__(
        self, model: models.Telemetron, params: Optional[parameters.RLMPCParams] = parameters.RLMPCParams(), solver_options: CasadiSolverOptions = CasadiSolverOptions()
    ) -> None:
        self._model = model
        if params:
            self._params0: parameters.RLMPCParams = params
            self._params: parameters.RLMPCParams = params

        self._solver_options: CasadiSolverOptions = solver_options

        nx, nu = self._model.dims()
        self._x_warm_start: np.ndarray = np.zeros(nx)
        self._u_warm_start: np.ndarray = np.zeros(nu)
        self._initialized = False
        self._map_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # In east-north coordinates

        self._opt_vars: csd.MX = csd.MX("opt_vars", 0)  # Optimization variables w
        self._lbw: np.ndarray = np.array([])
        self._ubw: np.ndarray = np.array([])
        self._vsolver: csd.Function = csd.Function("vsolver", [], [])
        self._lbg_v: np.ndarray = np.array([])
        self._ubg_v: np.ndarray = np.array([])
        self._qsolver: csd.Function = csd.Function("qsolver", [], [])
        self._lbg_q: np.ndarray = np.array([])
        self._ubg_q: np.ndarray = np.array([])

        self._dlag_v: csd.Function = csd.Function("dlag_v", [], [])
        self._dlag_q: csd.Function = csd.Function("dlag_q", [], [])
        self._decision_trajectories_func: csd.Function = csd.Function("decision_trajectories_func", [], [])

        self._num_ocp_params: int = 0
        self._num_fixed_ocp_params: int = 0
        self._num_adjustable_ocp_params: int = 0
        self._p_fixed: csd.MX = csd.MX("p_fixed", 0)
        self._p_adjustable: csd.MX = csd.MX("p_adjustable", 0)
        self._p: csd.MX = csd.vertcat(self._p_fixed, self._p_adjustable)

        self._p_fixed_values: np.ndarray = np.array([])
        self._p_adjustable_values: np.ndarray = np.array([])

    @property
    def params(self) -> parameters.RLMPCParams:
        return self._params

    def get_adjustable_params(self) -> list:
        """Returns the RL-tuneable parameters in the NMPC.

        Returns:
            list: List of parameters. The order of the parameters are:
                - Q (flattened)
                - d_safe_so
                - d_safe_do
        """
        return self._params.adjustable

    def _set_initial_warm_start(self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray]) -> None:
        """Sets the initial warm start state (and input) trajectory for the NMPC.

        Args:
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track or path to follow.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
        """
        self._x_warm_start = nominal_trajectory

        if nominal_inputs is not None:
            self._u_warm_start = nominal_inputs

    def plan(self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track or path to follow.
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

        nx, nu = self._model.dims()
        N = int(self._params.T / self._params.dt)
        action = None
        if nominal_inputs is not None:
            action = nominal_inputs[:, 0]

        parameter_values = self.create_parameter_values(xs, action, nominal_trajectory, do_list, so_list)
        soln = self._vsolver(
            x0=xs,
            p=parameter_values,
            lbg=self._lbg_v,
            ubg=self._ubg_v,
        )
        stats = self._vsolver.stats()
        if not stats["success"]:
            RuntimeError("Problem is Infeasible")

        opt_vars = soln["x"].full()
        act0 = np.array(opt_vars[:nu])[:, 0]

        print("Soln")
        print(opt_vars[: nu * N, :].T)
        print(
            opt_vars[
                nu * N : nu * N + nx * N,
                :,
            ].T
        )
        print(
            opt_vars[
                nu * N + nx * N :,
                :,
            ].T
        )
        return act0, soln

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

            min     J(w, p)
            s.t.    lbw <= x <= ubw ∀ w
                    lbg <= g(w, p) <= ubg

        Args:
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.
        """
        self._map_bbox = enc.bbox
        N = int(self._params.T / self._params.dt)
        dt = self._params.dt

        nx, nu = self._model.dims()
        xdot, x, u = self._model.as_casadi()

        # NLP decision variables
        U = csd.MX.sym("U", nu, N)
        X = csd.MX.sym("X", nx, N + 1)
        Sigma = csd.MX.sym("Sigma", MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS, N + 1)
        w = csd.vertcat(csd.reshape(U, -1, 1), csd.reshape(X, -1, 1), csd.reshape(Sigma, -1, 1))

        # Box constraints on NLP decision variables
        lbu_k, ubu_k, lbx_k, ubx_k = self._model.get_input_state_bounds()
        lbu = [lbu_k] * N
        ubu = [ubu_k] * N
        lbx = [lbx_k] * (N + 1)
        ubx = [ubx_k] * (N + 1)
        lbsigma = [0] * (N + 1) * (MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        ubsigma = [np.inf] * (N + 1) * (MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._lbw = np.array([*lbu, *lbx, *lbsigma])
        self._ubw = np.array([*ubu, *ubx, *ubsigma])

        g_eq_list = []  # NLP equality constraints
        g_ineq_list = []  # NLP inequality constraints
        p, p_fixed, p_adjustable = [], [], []  # NLP parameters
        num_fixed_ocp_params, num_adjustable_ocp_params = 0, 0

        # Initial state constraint
        x_0 = csd.MX.sym("x_0", nx, 1)
        g_eq_list.append(x_0 - X[:, 0])
        p_fixed.append(x_0)
        num_fixed_ocp_params += nx  # x_0

        # Also add the initial action u_0 as parameter, relevant for the Q-function approximator
        u_0 = csd.MX.sym("u_0", nu, 1)
        p_fixed.append(u_0)
        num_fixed_ocp_params += nu  # u_0

        if self._params.path_following:
            dim_Q = nx * nx
            X_ref = csd.MX.sym("X_ref", nx, N + 1)
        else:
            dim_Q = 2
            X_ref = csd.MX.sym("X_ref", 2, N + 1)
        Q_vec = csd.MX.sym("Q_vec", dim_Q * dim_Q, 1)
        Qmtrx = hf.casadi_matrix_from_vector(Q_vec, dim_Q, dim_Q)
        p_fixed.append(csd.reshape(X_ref, dim_Q * (N + 1), 1))
        num_fixed_ocp_params += dim_Q * N + 1  # X_ref

        p_adjustable.append(Q_vec)
        num_adjustable_ocp_params += dim_Q * dim_Q  # Q_vec

        gamma = csd.MX.sym("gamma", 1)
        p_fixed.append(gamma)
        num_fixed_ocp_params += 1  # gamma

        # Slack weighting matrix W (dim = 1 x (MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS))
        W = csd.MX.sym("W", 1, MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        p_fixed.append(csd.reshape(W, -1, 1))
        num_fixed_ocp_params += MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS  # W

        # Dynamic obstacle augmented states (x, y, Vx, Vy, length, width) * N + 1
        epsilon_do = 0.00001
        nx_do = 6
        X_do = csd.MX.sym("X_do", nx_do * MAX_NUM_DO_CONSTRAINTS, N + 1)
        p_fixed.append(csd.reshape(X_do, -1, 1))

        # Safety zone parameters
        d_safe_so = csd.MX.sym("d_safe_so", 1)
        d_safe_do = csd.MX.sym("d_safe_do", 1)
        p_adjustable.append(d_safe_so)
        p_adjustable.append(d_safe_do)
        num_adjustable_ocp_params += 2  # d_safe_so and d_safe_do

        # Cost function
        J = 0

        so_surfaces = hf.compute_surface_approximations_from_polygons(so_list, enc)

        # Create symbolic integrator for the shooting gap constraints
        erk4 = integrators.ERK4(x, u, xdot, None, dt)
        for k in range(N):
            # Sum stage costs
            J += gamma**k * quadratic_cost(X[:, k], X_ref[:, k], Qmtrx) + W @ Sigma[:, k]

            # Shooting gap constraints
            x_k_next = erk4(X[:, k], U[:, k])
            g_eq_list.append(x_k_next - X[:, k + 1])

            # Static obstacle constraints
            for j in range(MAX_NUM_SO_CONSTRAINTS):
                g_ineq_list.append(0.0)

            # Dynamic obstacle constraints
            for i in range(MAX_NUM_DO_CONSTRAINTS):
                x_aug_do_i = X_do[nx_do * i : nx_do * (i + 1), k]
                x_do_i = x_aug_do_i[0:4]
                l_do_i = x_aug_do_i[4]
                w_do_i = x_aug_do_i[5]
                chi_do_i = csd.atan2(x_do_i[3], x_do_i[2])
                Rchi_do_i = mf.Rpsi2D_casadi(chi_do_i)
                p_diff_do_frame = Rchi_do_i @ (x[0:2] - x_do_i[0:2])
                weights = hf.casadi_matrix_from_nested_list([[1.0 / (l_do_i + d_safe_do) ** 2, 0.0], [0.0, 1.0 / (w_do_i + d_safe_do) ** 2]])
                g_ineq_list.append(csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon_do) - csd.log(1 + epsilon_do))

        # Add terminal cost
        J += gamma**N * quadratic_cost(X[:, N], X_ref[:, N], Qmtrx) + W @ Sigma[:, N]

        # Vectorize and finalize the NLP
        g_eq = csd.vertcat(*g_eq_list)
        g_ineq = csd.vertcat(*g_ineq_list)

        lbg_eq = [0.0] * g_eq.shape[0]
        ubg_eq = [0.0] * g_eq.shape[0]
        lbg_ineq = [0.0] * g_ineq.shape[0]
        ubg_ineq = [np.inf] * g_ineq.shape[0]
        self._lbg_v = np.concatenate((lbg_eq, lbg_ineq), axis=0)
        self._ubg_v = np.concatenate((ubg_eq, ubg_ineq), axis=0)

        self._p_fixed = csd.vertcat(*p_fixed)
        self._p_adjustable = csd.vertcat(*p_adjustable)
        self._p = csd.vertcat(*p_fixed, *p_adjustable)

        self._num_fixed_ocp_params = num_fixed_ocp_params
        self._num_adjustable_ocp_params = num_adjustable_ocp_params
        self._num_ocp_params = num_fixed_ocp_params + num_adjustable_ocp_params
        self._p_fixed_values = np.zeros((self._num_fixed_ocp_params, 1))
        self._p_adjustable_values = np.zeros((self._num_adjustable_ocp_params, 1))

        self._opt_vars = w

        # Create value (v) function approximation MPC solver
        vnlp_prob = {
            "f": J,
            "x": w,
            "p": self._p,
            "g": csd.vertcat(g_eq, g_ineq),
        }
        self._vsolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob, self._solver_options.to_opt_settings())
        self._dlag_v = self.build_sensitivity(J, g_eq, g_ineq)

        # Create action-value (q or Q(s, a)) function approximation
        u_0 = csd.MX.sym("u_0", nu, 1)
        g_eq = csd.vertcat(g_eq, u_0 - U[:, 0])
        lbg_eq = [0.0] * g_eq.shape[0]
        ubg_eq = [0.0] * g_eq.shape[0]
        self._lbg_q = np.concatenate((lbg_eq, lbg_ineq), axis=0)
        self._ubg_q = np.concatenate((ubg_eq, ubg_ineq), axis=0)

        qnlp_prob = {"f": J, "x": w, "p": p, "g": csd.vertcat(g_eq, g_ineq)}
        self._qsolver = csd.nlpsol("qsolver", "ipopt", qnlp_prob, self._solver_options.to_opt_settings())
        self._dlag_q = self.build_sensitivity(J, g_eq, g_ineq)

        # Useful functions
        self._decision_trajectories_func = csd.Function("decision_trajectories", [w], [U, X, Sigma], ["w"], ["U", "X", "Sigma"])

    def create_parameter_values(self, state: np.ndarray, action: Optional[np.ndarray], nominal_trajectory: np.ndarray, do_list: list, so_list: list) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - state (np.ndarray): Current state of the system.
            - action (np.ndarray): Current action of the system.
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track or path to follow.
            - do_list (list): List of dynamic obstacles.
            - so_list (list): List of static obstacles.

        Returns:
            - np.ndarray: Parameter vector to be used as input to solver
        """
        adjustable_parameter_values = self.get_adjustable_params()

        nx, nu = self._model.dims()
        N = int(self._params.T / self._params.dt)
        fixed_parameter_values = state
        if action is not None:
            fixed_parameter_values = np.concatenate((fixed_parameter_values, action), axis=0)

        if self._params.path_following:
            fixed_parameter_values = np.concatenate((fixed_parameter_values, nominal_trajectory[0:2, :].T.flatten()), axis=0)
        else:
            fixed_parameter_values = np.concatenate((fixed_parameter_values, nominal_trajectory.T.flatten()), axis=0)

        n_do = len(do_list)
        for k in range(N):
            t = k * self._params.dt
            for j in range(MAX_NUM_SO_CONSTRAINTS):
                continue

            for i in range(MAX_NUM_DO_CONSTRAINTS):
                if i < n_do:
                    (ID, state, cov, length, width) = do_list[i]
                    fixed_parameter_values = np.concatenate(
                        (fixed_parameter_values, np.array([state[0] + t * state[2], state[1] + t * state[3], state[2], state[3], length, width]))
                    )
                else:
                    fixed_parameter_values = np.concatenate((fixed_parameter_values, np.array([self._map_bbox[1], self._map_bbox[0], 0.0, 0.0, 5.0, 2.0])))
        return np.concatenate((fixed_parameter_values, adjustable_parameter_values), axis=0)

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
