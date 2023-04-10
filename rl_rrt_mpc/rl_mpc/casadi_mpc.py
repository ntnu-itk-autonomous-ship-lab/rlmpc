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
import rl_rrt_mpc.rl_mpc.integrators as integrators
import rl_rrt_mpc.rl_mpc.models as models
import rl_rrt_mpc.rl_mpc.parameters as parameters
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


class CasadiMPC:
    def __init__(
        self, model: models.Telemetron, params: Optional[parameters.RLMPCParams] = parameters.RLMPCParams(), solver_options: CasadiSolverOptions = CasadiSolverOptions()
    ) -> None:
        self._model = model
        if params:
            self._params0: parameters.RLMPCParams = params
            self._params: parameters.RLMPCParams = params

        self._solver_options: CasadiSolverOptions = solver_options

        nx, nu = self._model.dims
        self._x_warm_start: np.ndarray = np.zeros(nx)
        self._u_warm_start: np.ndarray = np.zeros(nu)
        self._initialized = False
        self._map_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # In east-north coordinates
        self._s: float = 0.0

    @property
    def params(self) -> parameters.RLMPCParams:
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
        nx, nu = self._model.dims()
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
        nx, nu = self._model.dims()
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

        nx, nu = self._model.dims()
        parameter_values = self.create_parameter_values(t, nominal_trajectory, nominal_inputs, xs, do_list, so_list)
        soln = self.vsolver(
            x0=xs,
            p=parameter_values,
            lbg=self.lbg_vcsd,
            ubg=self.ubg_vcsd,
        )
        stats = self._solver.stats()
        if not fl["success"]:
            RuntimeError("Problem is Infeasible")

        opt_vars = soln["x"].full()
        act0 = np.array(opt_vars[:nu])[:, 0]

        print("Soln")
        print(opt_vars[: nu * self.N, :].T)
        print(
            opt_vars[
                nu * self.N : nu * self.N + nx * self.N,
                :,
            ].T
        )
        print(
            opt_vars[
                nu * self.N + nx * self.N :,
                :,
            ].T
        )
        return act0, soln

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
        self._solver.constraints_set(0, "lbx", xs)
        self._solver.constraints_set(0, "ubx", xs)
        for i in range(N + 1):
            self._solver.set(i, "x", self._x_warm_start[:, i])
            if i < N:
                self._solver.set(i, "u", self._u_warm_start[:, i])
            p_i = self.create_parameter_values(adjustable_params, nominal_trajectory, do_list, so_list, i)
            self._solver.set(i, "p", p_i)
        print("OCP updated")

    def construct_ocp(self, nominal_trajectory: np.ndarray | list, do_list: list, so_list: list, enc: senc.ENC) -> None:
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
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Either as np.ndarray or as list of splines for  x, y, psi, U.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.
        """
        self._map_bbox = enc.bbox
        N = int(self._params.T / self._params.dt)
        dt = self._params.dt

        nx, nu = self._model.dims()
        xdot, x, u = self._model.as_casadi()

        # NLP decision variables
        X = csd.MX.sym("X", nx, N + 1)
        U = csd.MX.sym("U", nu, N)
        Sigma = csd.MX.sym("Sigma", 1, N + 1)
        w = csd.vertcat(X, U, Sigma)

        # Box constraints on NLP decision variables
        lbu_k, ubu_k, lbx_k, ubx_k = self._model.get_input_state_bounds()
        lbu = [lbu_k] * N
        ubu = [ubu_k] * N
        lbx = [lbx_k] * (N + 1)
        ubx = [ubx_k] * (N + 1)
        lbsigma = [0] * (N + 1)
        ubsigma = [np.inf] * (N + 1)
        lbw = np.concatenate((lbx, lbu, lbsigma), axis=0)
        ubw = np.concatenate((ubx, ubu, ubsigma), axis=0)

        g, lbg, ubg = [], [], []  # NLP inequality constraints
        p, p_fixed, p_adjustable = [], [], []  # NLP parameters

        gamma = csd.MX.sym("gamma", 1)
        if self._params.path_following:
            Q_vec = csd.MX.sym("Qmtrx", nx * nx, 1)
            X_ref = csd.MX.sym("X_ref", nx, N + 1)
        else:
            Q_vec = csd.MX.sym("Qmtrx", 2 * 2, 1)
            X_ref = csd.MX.sym("X_ref", 2, N + 1)
        Qmtrx = hf.casadi_matrix_from_vector(Q_vec, nx, nx)
        p_fixed.append(X_ref)

        p_adjustable.append(gamma)
        p_adjustable.append(Q_vec)

        # Create symbolic constraint
        g_func = csd.Function("G", [x, u, p], [c], ["x", "u", "p"], ["c"])

        erk4 = integrators.ERK4(x, u, xdot, None, dt)

        x_0 = csd.MX.sym("x_0", nx, 1)
        lbw.append(x_0)
        ubw.append(x_0)

        J = 0
        for k in range(N):
            # Sum stage costs
            J += gamma**k * quadratic_cost(X[:, k], X_ref[:, k], Qmtrx)

            # Shooting gap constraints
            x_k_next = erk4(X[:, k], U[:, k])
            g.append(x_k_next - X[:, k + 1])
            lbg.append(np.zeros(nx))
            ubg.append(np.zeros(nx))

            # Static obstacle constraints

            # Dynamic obstacle constraints

        # Add terminal cost
        J += gamma**N * (self.terminal_cost(X[:, self.N], s_ref, tQ, tq) + W @ self.Sigma[:, self.N])

        # Vectorize and finalize the NLP
        self.g = ca.vertcat(*self.g)
        self.lbg = ca.vertcat(*self.lbg)
        self.ubg = ca.vertcat(*self.ubg)

        u_0 = csd.MX.sym("u_0", nu, 1)
        qsolver_extra_constraint = U[:, 0] - u_0

        # Usefull function for extracting x and u trajectories from w vector
        self.trajectories = ca.Function("trajectories", [self.w], [ca.horzcat(*x_plot), ca.horzcat(*u_plot)], ["w"], ["x", "u"])
        self.collocation = ca.Function("collocation_points", [self.w], [ca.horzcat(*x_collocation)], ["w"], ["x"])
        self.slack = ca.Function("slack", [self.w], [ca.horzcat(*s_plot)], ["w"], ["s"])
        self.decision_trajectories = ca.Function(
            "decision_trajectories", [self.w], [ca.horzcat(*x_collocation, Xk), ca.horzcat(*u_plot), ca.horzcat(*s_plot)], ["w"], ["x", "u", "s"]
        )
        self.decision_variables = ca.Function(
            "decision_variables", [ca.horzcat(*x_collocation, Xk), ca.horzcat(*u_plot), ca.horzcat(*s_plot)], [self.w], ["x", "u", "s"], ["w"]
        )

        # Usefull function for generating bounds
        self.bounds = ca.Function("bounds", [X0], [self.lbw, self.ubw, self.lbg, self.ubg], ["x0"], ["lbx", "ubx", "lbg", "ubg"])

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

        lagrangian = cost + csd.transpose(lamb) @ eq_constr + csd.transpose(mu_u) @ Hu + csd.transpose(mu_x) @ Hx + csd.transpose(mu_s) @ Hs
        lagrangian_function = csd.Function("Lag", [self.Opt_Vars, mult, self.Pf, self.P], [lagrangian])
        lagrangian_function_derivative = lagrangian_function.factory(
            "dLagfunc",
            ["i0", "i1", "i2", "i3"],
            ["jac:o0:i0", "jac:o0:i2", "jac:o0:i3"],
        )
        dLdw, dLdPf, dLdP = lagrangian_function_derivative(self._casadi_w, mult, self.Pf, self.P)

    def dPidP(self, state, soln, param_val=None):
        # Sensitivity of policy output with respect to learnable param
        # i.e. gradient of action wrt to param_val
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()
        z = np.concatenate((x, lam_g), axis=0)

        self.pf_val[:nx, :] = state[:, None]
        param_val = param_val if param_val is not None else self.param_val
        jacob_act = self.dPi(z, self.pf_val[:, 0], param_val[:, 0]).full()
        return jacob_act[:nu, :]

    def Q_value(self, state, action, param_val=None):
        # Action-value function evaluation
        nx, nu = self._model.dims()
        self.pf_val[:nx, :] = state[:, None]
        self.pf_val[nx : nx + nu, :] = action[:, None]
        param_val = param_val if param_val is not None else self.param_val
        X0 = self.env.get_initial_guess(self.N)

        qsoln = self.qsolver(
            x0=X0,
            p=np.concatenate([self.pf_val, param_val])[:, 0],
            lbg=self.lbg_qcsd,
            ubg=self.ubg_qcsd,
        )
        fl = self.qsolver.stats()
        if not fl["success"]:
            RuntimeError("Problem is Infeasible")

        q = qsoln["f"].full()[0, 0]
        return q, qsoln

    def dQdP(self, state, action, soln, param_val=None):
        # Gradient of action-value fn Q wrt lernable param
        # state, action, act_wt need to be from qsoln (garbage in garbage out)
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()

        self.pf_val[:nx, :] = state[:, None]
        self.pf_val[nx : nx + nu, :] = action[:, None]
        param_val = param_val if param_val is not None else self.param_val

        _, _, dLdP = self.dLagQ(x, lam_g, self.pf_val[:, 0], param_val[:, 0])
        return dLdP.full()

    def param_update(self, lr, dJ, param_val=None):
        # Param update scheme
        param_val = param_val if param_val is not None else self.param_val
        if self.constrained_updates:
            self.param_val = self.constraint_param_update(lr, dJ, param_val)
        else:
            self.param_val -= lr * dJ

    def constraint_param_update(self, lr, dJ, param_val):
        # SDP for param update to ensure stable MPC formulation
        dP = cvx.Variable((self.n_P, 1))
        J_up = 0.5 * cvx.sum_squares(dP) + lr * dJ.T @ dP
        P_next = param_val + dP
        constraint = []
        for cost_type in self.cost_defn:
            if cost_type == "diagQ":
                constraint += [cvx.diag(P_next[self.indP_q[0] : self.indP_q[1], 0]) >> 0.0]
            elif cost_type == "diagR":
                constraint += [cvx.diag(P_next[self.indP_r[0] : self.indP_r[1], 0]) >> 0.0]
            elif cost_type == "fullQ":
                constraint += [
                    cvx.reshape(
                        P_next[self.indP_fq[0] : self.indP_fq[1], 0],
                        (nx, nx),
                    )
                    >> 0.0
                ]
            elif cost_type == "fullR":
                constraint += [
                    cvx.reshape(
                        P_next[self.indP_fr[0] : self.indP_fr[1], 0],
                        (nu, nu),
                    )
                    >> 0.0
                ]
            elif cost_type == "fullQu" or cost_type == "fullRu":
                raise BaseException("Constrained update for cost function type not implemented")
        prob = cvx.Problem(cvx.Minimize(J_up), constraint)
        prob.solve(solver="CVXOPT")
        P_up = param_val + dP.value
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
