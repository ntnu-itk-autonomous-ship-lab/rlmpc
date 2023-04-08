"""
    mpc.py

    Summary:
        Contains a class for an NMPC trajectory tracking/path following controller.

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


@dataclass
class CasadiNMPCParams:
    reference_traj_bbox_buffer: float = 500.0
    T: float = 10.0
    dt: float = 0.5
    Q: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R: np.ndarray = np.diag([1.0, 1.0, 1.0])
    gamma: float = 0.0
    d_safe_so: float = 5.0
    d_safe_do: float = 5.0
    spline_reference: bool = False
    path_following: bool = False
    acados: bool = False
    casadi_solver_options: CasadiSolverOptions = CasadiSolverOptions()

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = CasadiNMPCParams(
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
        )

        if config.path_following and config.Q.shape[0] != 2:
            raise ValueError("Q must be a 2x2 matrix when path_following is True.")

        if not config.path_following and config.Q.shape[0] != 6:
            raise ValueError("Q must be a 6x6 matrix when path_following is False (trajectory tracking).")

        config.casadi_solver_options = CasadiSolverOptions.from_dict(config_dict["casadi_solver_options"])
        return config


class CasadiNMPC:
    def __init__(self, model: models.Telemetron, params: Optional[CasadiNMPCParams] = CasadiNMPCParams()) -> None:
        self._model = model
        if params:
            self._params0: CasadiNMPCParams = params
            self._params: CasadiNMPCParams = params

        nx, nu = self._model.dims
        self._x_warm_start: np.ndarray = np.zeros(nx)
        self._u_warm_start: np.ndarray = np.zeros(nu)
        self._initialized = False
        self._map_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # In east-north coordinates
        self._s: float = 0.0

    @property
    def params(self) -> CasadiNMPCParams:
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
        """Constructs the OCP for the NMPC problem using pure Casadi.

        Class constructs a CASADI OCP on the form (same form as for the ACADOS OCP):
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
        self._map_bbox = enc.bbox
        N = int(self._params.T / self._params.dt)

        xdot, x, u = self._model.as_casadi()

        lbu, ubu, lbx, ubx = self._model.get_input_state_bounds()

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
