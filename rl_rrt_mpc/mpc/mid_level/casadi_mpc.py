"""
    casadi_mpc.py

    Summary:
        Casadi MPC class for the mid-level COLAV planner.

    Author: Trym Tengesdal
"""
import time
from typing import Optional, Tuple

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.integrators as integrators
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.common.set_generator as sg
import rl_rrt_mpc.mpc.common as mpc_common
import rl_rrt_mpc.mpc.models as models
import rl_rrt_mpc.mpc.parameters as parameters
import scipy.interpolate as interp
import seacharts.enc as senc


class CasadiMPC:
    def __init__(
        self,
        model: models.MPCModel,
        path_timing: models.MPCModel,
        params: parameters.MidlevelMPCParams,
        solver_options: mpc_common.CasadiSolverOptions,
    ) -> None:
        self._model = model
        self._path_timing = path_timing
        self._params0: parameters.MidlevelMPCParams = params
        self._params: parameters.MidlevelMPCParams = params

        self._solver_options: mpc_common.CasadiSolverOptions = solver_options
        self._initialized_v: bool = False
        self._initialized_q: bool = False
        self._map_origin: np.ndarray = np.array([])

        self._opt_vars: csd.MX = csd.MX.sym("opt_vars", 0)
        self._lbw: np.ndarray = np.array([])
        self._ubw: np.ndarray = np.array([])
        self._vsolver: csd.Function = csd.Function("vsolver", [], [])
        self._current_warmstart_v: dict = {"x": [], "lam_x0": [], "lam_g": []}
        self._lbg_v: np.ndarray = np.array([])
        self._ubg_v: np.ndarray = np.array([])
        self._qsolver: csd.Function = csd.Function("qsolver", [], [])
        self._current_warmstart_q: dict = {"x": [], "lam_x0": [], "lam_g": []}
        self._lbg_q: np.ndarray = np.array([])
        self._ubg_q: np.ndarray = np.array([])

        self._dlag_v: csd.Function = csd.Function("dlag_v", [], [])
        self._dlag_q: csd.Function = csd.Function("dlag_q", [], [])

        self._num_ocp_params: int = 0
        self._num_fixed_ocp_params: int = 0
        self._num_adjustable_ocp_params: int = 0
        self._p_mdl = csd.MX.sym("p_mdl", 0)
        self._p_fixed: csd.MX = csd.MX.sym("p_fixed", 0)
        self._p_adjustable: csd.MX = csd.MX.sym("p_adjustable", 0)
        self._p: csd.MX = csd.vertcat(self._p_fixed, self._p_adjustable)

        self._set_generator: Optional[sg.SetGenerator] = None
        self._p_ship_mdl_values: np.ndarray = np.array([self._model.params().T_chi, self._model.params().T_U])
        self._p_fixed_so_values: np.ndarray = np.array([])
        self._p_fixed_values: np.ndarray = np.array([])
        self._p_adjustable_values: np.ndarray = np.array([])

        self._decision_trajectories: csd.Function = csd.Function("decision_trajectories", [], [])
        self._decision_variables: csd.Function = csd.Function("decision_variables", [], [])
        self._static_obstacle_constraints: csd.Function = csd.Function("static_obstacle_constraints", [], [])
        self._dynamic_obstacle_constraints: csd.Function = csd.Function("dynamic_obstacle_constraints", [], [])
        self._equality_constraints: csd.Function = csd.Function("equality_constraints", [], [])
        self._inequality_constraints: csd.Function = csd.Function("inequality_constraints", [], [])
        self._t_prev: float = 0.0
        self._xs_prev: np.ndarray = np.array([])
        self._min_depth: int = 5

        self._x_path: csd.Function = csd.Function("x_path", [], [])
        self._x_path_coeffs: np.ndarray = np.array([])
        self._y_path: csd.Function = csd.Function("y_path", [], [])
        self._y_path_coeffs: np.ndarray = np.array([])
        self._U_ref: float = 0.0
        self._s: float = 0.0
        self._s_dot: float = 0.0
        self._s_final_value: float = 0.0

    @property
    def params(self):
        return self._params

    def get_adjustable_params(self) -> np.ndarray:
        """Returns the RL-tuneable parameters in the MPC.

        Returns:
            np.ndarray: Array of parameters.
        """
        mdl_adjustable_params = np.array([])
        if isinstance(self._model, models.AugmentedKinematicCSOG):
            mdl_params = self._model.params()
            mdl_adjustable_params = np.array([mdl_params.T_chi, mdl_params.T_U])
        mpc_adjustable_params = self.params.adjustable()
        return np.concatenate((mdl_adjustable_params, mpc_adjustable_params))

    def _set_path_information(self, nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, float]) -> None:
        """Sets the path information for the MPC.

        Args:
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, inter.PchipInterpolator, float]): Tuple containing the nominal path splines in x, y, heading and the nominal speed reference.
        """
        x_spline, y_spline, _, U_ref = nominal_path
        s = csd.MX.sym("s", 1)
        x_path_coeffs = csd.MX.sym("x_path_coeffs", x_spline.c.shape[0])
        x_path = csd.bspline(s, x_path_coeffs, [[*x_spline.t]], [x_spline.k], 1, {})
        self._x_path = csd.Function("x_path", [s, x_path_coeffs], [x_path])
        self._x_path_coeffs_values = x_spline.c
        self._x_path_coeffs = x_path_coeffs

        y_path_coeffs = csd.MX.sym("y_path_coeffs", y_spline.c.shape[0])
        y_path = csd.bspline(s, y_path_coeffs, [[*y_spline.t]], [y_spline.k], 1, {})
        self._y_path = csd.Function("y_path", [s, y_path_coeffs], [y_path])
        self._y_path_coeffs_values = y_spline.c
        self._y_path_coeffs = y_path_coeffs
        self._s_final_value = x_spline.t[-1]
        self._U_ref = U_ref
        self._path_timing.set_min_path_variable(0.0)
        self._path_timing.set_max_path_variable(self._s_final_value)

    def _model_prediction(self, xs: np.ndarray, u: np.ndarray, N: int) -> np.ndarray:
        """Euler prediction of the ship model and path timing model, concatenated.

        Args:
            - xs (np.ndarray): Starting state of the system (x, y, chi, U).
            - u (np.ndarray): Input to apply.
            - dt (float): Time step.
            - N (int): Number of steps to predict.

        Returns:
            np.ndarray: Predicted states.
        """
        nx, nu = self._model.dims()
        nx_p, nu_p = self._path_timing.dims()
        u_ship = u[:nu]
        u_path = u[nu:]
        X_ship = self._model.euler_n_step(xs[:nx], u_ship, self._p_ship_mdl_values, self._params.dt, N)
        X_path = self._path_timing.euler_n_step(xs[nx:], u_path, np.array([]), self._params.dt, N)
        return np.concatenate((X_ship, X_path))

    def _update_path_variables(self, xs: np.ndarray, dt: float) -> None:
        """Updates the path variables s and s_dot.

        Args:
            xs (np.ndarray): Current state of the system (x, y, chi, U).
            dt (float): Time step between the previous MPC iteration and the current.
        """
        nx, nu = self._model.dims()
        dt_mpc = self._params.dt
        shift = int(dt / dt_mpc)
        _, X, _ = self._decision_trajectories(self._current_warmstart_v["x"])
        self._s = X[nx, shift]
        self._s_dot = X[nx + 1, shift]

    def _create_initial_warm_start(self, xs: np.ndarray, dim_g: int, enc: Optional[senc.ENC] = None) -> dict:
        """Sets the initial warm start decision trajectory [U, X, Sigma] flattened for the NMPC.

        Args:
            - xs (np.ndarray): Initial state of the system (x, y, chi, U)
            - dim_g (int): Dimension/length of the constraints.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
        """
        n_colregs_zones = 3
        nx, nu = self._model.dims()
        nx_p, nu_p = self._path_timing.dims()
        N = int(self._params.T / self._params.dt)
        w = np.zeros((nu + nu_p, N))

        path_timing_input = 1.0
        w[2, :] = path_timing_input * np.ones(N)
        w = w.flatten()

        xs_k = np.zeros(nx + nx_p)
        xs_k[:4] = xs[:4]
        xs_k[4] = xs[2]
        xs_k[5] = 0.7 * self._U_ref

        self._s = 0.0
        self._s_dot = 0.0
        xs_k[nx:] = np.array([self._s, self._s_dot])

        n_attempts = 3
        success = False
        path_timing_input = 1.0
        u_attempts = [
            np.array([0.0, 0.0, path_timing_input]),
            np.array([0.0, -0.05, path_timing_input]),
            np.array([0.0, 0.05, path_timing_input]),
            np.zeros(nu + nu_p),
        ]
        for i in range(n_attempts):
            warm_start_traj = self._model_prediction(xs_k, u_attempts[i], N + 1)
            positions = np.array([warm_start_traj[1, :] + self._map_origin[1], warm_start_traj[0, :] + self._map_origin[0]])
            min_dist, _, _ = mapf.compute_closest_grounding_dist(positions, self._min_depth, enc)
            if enc is not None:
                t_ws = warm_start_traj + np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(nx + nx_p, 1)
                hf.plot_trajectory(t_ws, enc, "orange")
            if min_dist > self._params.r_safe_so:
                success = True
                break

        assert success, "Could not create initial warm start solution"
        chi = warm_start_traj[2, :]
        chi_unwrapped = np.unwrap(np.concatenate(([xs_k[2]], chi)))[1:]
        warm_start_traj[2, :] = chi_unwrapped
        w = np.concatenate((w, warm_start_traj.T.flatten()))
        w = np.concatenate((w, np.zeros((self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone) * (N + 1))))
        warm_start = {"x": w.tolist(), "lam_x": np.zeros(w.shape[0]).tolist(), "lam_g": np.zeros(dim_g).tolist()}

        return warm_start

    def _try_to_create_warm_start_solution(
        self,
        X_prev: np.ndarray,
        U_prev: np.ndarray,
        Sigma_prev: np.ndarray,
        u_mod: np.ndarray,
        offset: int,
        n_shifts: int,
        enc: senc.ENC,
    ) -> Tuple[np.ndarray, bool]:
        """Creates a shifted warm start trajectory from the previous trajectory and the new control input.
        Args:
            - X_prev (np.ndarray): The previous trajectory.
            - U_prev (np.ndarray): The previous control inputs.
            - Sigma_prev (np.ndarray): The previous slack variables.
            - u_mod (np.ndarray): The new control input to apply at the end of the previous trajectory.
            - offset (int): The offset from the last sample in the previous trajectory, to apply the modified input vector.
            - dt (float): Time step
            - n_shifts (int): Number of shifts to perform on the previous trajectory.
            - enc (senc.ENC): The ENC object containing the map.

        Returns:
            Tuple[np.ndarray, bool]: The new warm start and a boolean indicating if the warm start was successful.
        """
        # Simulate the system from t_N to t_N+n_shifts with the last input
        Sigma_warm_start = np.concatenate((Sigma_prev[:, n_shifts:], np.tile(Sigma_prev[:, -1], (n_shifts, 1)).T), axis=1)

        if offset == 0:
            inputs_past_N = np.tile(u_mod, (n_shifts, 1)).T
            states_past_N = self._model_prediction(X_prev[:, -1], u_mod, n_shifts)
        else:
            inputs_past_N = np.tile(u_mod, (n_shifts + offset, 1)).T
            states_past_N = self._model_prediction(X_prev[:, -offset], u_mod, n_shifts + offset)

        if offset == 0:
            U_warm_start = np.concatenate((U_prev[:, n_shifts:], inputs_past_N), axis=1)
            X_warm_start = np.concatenate((X_prev[:, n_shifts:], states_past_N), axis=1)
        else:
            U_warm_start = np.concatenate((U_prev[:, n_shifts:-offset], inputs_past_N), axis=1)
            X_warm_start = np.concatenate((X_prev[:, n_shifts:-offset], states_past_N), axis=1)
        psi = X_warm_start[2, :].tolist()
        X_warm_start[2, :] = np.unwrap(np.concatenate(([psi[0]], psi)))[1:]
        w_warm_start = np.concatenate((U_warm_start.T.flatten(), X_warm_start.T.flatten(), Sigma_warm_start.T.flatten()))

        pos_past_N = states_past_N[:2, :] + self._map_origin.reshape(2, 1)
        pos_past_N[0, :] = states_past_N[1, :] + self._map_origin[1]
        pos_past_N[1, :] = states_past_N[0, :] + self._map_origin[0]
        min_dist, _, _ = mapf.compute_closest_grounding_dist(pos_past_N, self._min_depth, enc)
        if min_dist <= self._params.r_safe_so:
            return w_warm_start, False

        return w_warm_start, True

    def _shift_warm_start(self, prev_warm_start: dict, xs: np.ndarray, dt: float, enc: senc.ENC) -> dict:
        """Shifts the warm start decision trajectory [U, X, Sigma] dt units ahead.

        Args:
            - prev_warm_start (dict): Warm start decision trajectory to shift.
            - xs (np.ndarray): Current state of the system on the form (x, y, chi, U)^T.
            - dt (float): Time to shift the warm start decision trajectory.
            - enc (senc.ENC): Electronic Navigational Chart object.
        """
        nx, nu = self._model.dims()
        nx_p, nu_p = self._path_timing.dims()
        lbu_ship, ubu_ship, _, _ = self._model.get_input_state_bounds()
        lbu_p, ubu_p, _, _ = self._path_timing.get_input_state_bounds()
        lbu = np.concatenate((lbu_ship, lbu_p))
        ubu = np.concatenate((ubu_ship, ubu_p))
        n_attempts = 3
        n_shifts = int(dt / self._params.dt)
        w_prev = np.array(prev_warm_start["x"])
        X_prev, U_prev, Sigma_prev = self._decision_trajectories(w_prev)
        X_prev = X_prev.full()
        U_prev = U_prev.full()
        Sigma_prev = Sigma_prev.full()
        offsets = [0, int(2.5 * n_shifts), int(2.5 * n_shifts), int(2.5 * n_shifts)]
        path_timing_input = 1.0
        u_attempts = [
            U_prev[:, -1],
            np.array([U_prev[0, -offsets[1]], lbu[1], path_timing_input]),
            np.array([U_prev[0, -offsets[2]], ubu[1], path_timing_input]),
            np.zeros(nu + nu_p),
        ]
        success = False
        for i in range(n_attempts):
            w_warm_start, success = self._try_to_create_warm_start_solution(X_prev, U_prev, Sigma_prev, u_attempts[i], offsets[i], n_shifts, enc)
            if success:
                break
        assert success, "Could not create warm start solution"
        new_warm_start = prev_warm_start
        new_warm_start["x"] = w_warm_start.tolist()
        return new_warm_start

    def plan(
        self,
        t: float,
        xs: np.ndarray,
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        so_list: list,
        enc: Optional[senc.ENC],
        show_plots: bool = False,
        **kwargs,
    ) -> dict:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - xs (np.ndarray): Current state.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - so_list (list): List ofrelevant static obstacle Polygon objects.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - dict: Dictionary containing the optimal trajectory, inputs, slacks and solver stats.
        """
        if not self._initialized_v:
            self._current_warmstart_v = self._create_initial_warm_start(xs, self._lbg_v.shape[0], enc)
            self._p_fixed_so_values = self._create_fixed_so_parameter_values(so_list, xs, enc, **kwargs)
            self._xs_prev = xs
            self._initialized_v = True

        action = None

        psi = xs[2]
        xs_unwrapped = xs.copy()
        xs_unwrapped[2] = np.unwrap(np.array([self._xs_prev[2], psi]))[1]
        self._xs_prev = xs_unwrapped
        dt = t - self._t_prev
        if dt > 0.0:
            self._update_path_variables(xs_unwrapped, dt)
            self._current_warmstart_v = self._shift_warm_start(self._current_warmstart_v, xs_unwrapped, dt, enc)

        parameter_values = self.create_parameter_values(xs_unwrapped, action, do_cr_list, do_ho_list, do_ot_list, so_list, enc, **kwargs)
        t_start = time.time()
        soln = self._vsolver(
            x0=self._current_warmstart_v["x"],
            lam_x0=self._current_warmstart_v["lam_x"],
            lam_g0=self._current_warmstart_v["lam_g"],
            p=parameter_values,
            lbx=self._lbw,
            ubx=self._ubw,
            lbg=self._lbg_v,
            ubg=self._ubg_v,
        )
        t_solve = time.time() - t_start
        stats = self._vsolver.stats()
        if stats["return_status"] == "Maximum_Iterations_Exceeded":
            # Use solution unless it is infeasible, then use previous solution.
            g_eq_vals = self._equality_constraints(soln["x"], parameter_values).full()
            g_ineq_vals = self._inequality_constraints(soln["x"], parameter_values).full()
            if np.any(g_eq_vals > 1e-6) or np.any(g_ineq_vals > 1e-6):
                soln = self._current_warmstart_v
        elif stats["return_status"] == "Infeasible_Problem_Detected":
            raise RuntimeError("Infeasible solution found.")

        so_constr_vals = self._static_obstacle_constraints(soln["x"], parameter_values).full()
        do_constr_vals = self._dynamic_obstacle_constraints(soln["x"], parameter_values).full()
        g_eq_vals = self._equality_constraints(soln["x"], parameter_values).full()
        g_ineq_vals = self._inequality_constraints(soln["x"], parameter_values).full()
        X, U, Sigma = self._extract_trajectories(soln)
        self._current_warmstart_v["x"] = self._decision_variables(X, U, Sigma)
        self._current_warmstart_v["lam_x"] = soln["lam_x"].full()
        self._current_warmstart_v["lam_g"] = soln["lam_g"].full()
        cost_val = soln["f"].full()[0][0]
        final_residuals = [stats["iterations"]["inf_du"][-1], stats["iterations"]["inf_pr"][-1]]
        mpc_common.plot_casadi_solver_stats(stats, show_plots)
        print(
            f"NMPC: | Runtime: {t_solve} | Cost: {cost_val} | sl (max, argmax): ({np.max(Sigma)}, {np.argmax(Sigma)}) | so_constr (max, argmax): ({np.max(so_constr_vals)}, {np.argmax(so_constr_vals)})"
        )
        self._t_prev = t
        output = {
            "trajectory": X,
            "inputs": U,
            "lower_slacks": [],
            "upper_slacks": Sigma,
            "so_constr_vals": so_constr_vals,
            "do_constr_vals": [],
            "t_solve": t_solve,
            "cost_val": cost_val,
            "n_iter": stats["iter_count"],
            "final_residuals": final_residuals,
        }
        return output

    def _extract_trajectories(self, soln: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the optimal trajectory, inputs and slacks from the solution dictionary.

        Args:
            soln (dict): Solution dictionary.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Optimal trajectory, inputs and slacks.
        """
        X, U, Sigma = self._decision_trajectories(soln["x"])
        X = X.full()
        U = U.full()
        Sigma = Sigma.full()
        psi = X[2, :]
        psi = np.unwrap(np.concatenate(([psi[0]], psi)))[1:]
        X[2, :] = psi
        return X, U, Sigma

    def construct_ocp(
        self,
        nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, float],
        so_list: list,
        enc: senc.ENC,
        map_origin: np.ndarray = np.array([0.0, 0.0]),
        min_depth: int = 5,
    ) -> None:
        """Constructs the OCP for the NMPC problem using pure Casadi.

        Class constructs a "CASADI" (ipopt) tailored OCP on the form (same as for the ACADOS MPC):
            min     ∫ Lc(x, u, p) dt + Tc(xf)  (from 0 to Tf)
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
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, float]): Tuple containing the nominal path splines in x, y, heading and the nominal speed reference.
            - so_list (list): List of compatible static obstacle Polygon objects with the static obstacle constraint type.
            - enc (senc.ENC): ENC object.
            - map_origin (np.ndarray, optional): Origin of the map. Defaults to np.array([0.0, 0.0]).
            - min_depth (int, optional): Minimum allowable depth for the vessel. Defaults to 5.
        """
        self._set_path_information(nominal_path)
        self._min_depth = min_depth
        self._map_origin = map_origin
        N = int(self._params.T / self._params.dt)
        dt = self._params.dt
        n_colregs_zones = 3

        # Ship model and path timing dynamics
        nx_ship, nu_ship = self._model.dims()
        xdot_ship, x_ship, u_ship, p_ship = self._model.as_casadi()
        lbu_ship_k, ubu_ship_k, lbx_ship_k, ubx_ship_k = self._model.get_input_state_bounds()

        nx_p, nu_p = self._path_timing.dims()
        xdot_p, x_p, u_p, p_mdl_p = self._path_timing.as_casadi()
        lbu_p_k, ubu_p_k, lbx_p_k, ubx_p_k = self._path_timing.get_input_state_bounds()

        # Concatenate the ship model and path timing dynamics
        nx = nx_ship + nx_p
        nu = nu_ship + nu_p
        xdot = csd.vertcat(xdot_ship, xdot_p)
        x = csd.vertcat(x_ship, x_p)
        u = csd.vertcat(u_ship, u_p)
        self._p_mdl = csd.vertcat(p_ship, p_mdl_p)
        lbu_k = np.concatenate((lbu_ship_k, lbu_p_k))
        ubu_k = np.concatenate((ubu_ship_k, ubu_p_k))
        lbx_k = np.concatenate((lbx_ship_k, lbx_p_k))
        ubx_k = np.concatenate((ubx_ship_k, ubx_p_k))

        # Box constraints on NLP decision variables
        lbu = np.tile(lbu_k, N)
        ubu = np.tile(ubu_k, N)
        lbx = np.tile(lbx_k, N + 1)
        ubx = np.tile(ubx_k, N + 1)
        lbsigma = np.array([0] * (N + 1) * (self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone))
        ubsigma = np.array([np.inf] * (N + 1) * (self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone))
        self._lbw = np.concatenate((lbu, lbx, lbsigma))
        self._ubw = np.concatenate((ubu, ubx, ubsigma))

        g_eq_list = []  # NLP equality constraints
        g_ineq_list = []  # NLP inequality constraints
        p_fixed, p_adjustable = [], []  # NLP parameters

        # NLP decision variables
        U = []
        X = []
        Sigma = []

        p_adjustable.append(self._p_mdl)

        # Initial state constraint
        x_0 = csd.MX.sym("x_0_constr", nx, 1)
        x_k = csd.MX.sym("x_0", nx, 1)
        X.append(x_k)
        g_eq_list.append(x_0 - x_k)
        p_fixed.append(x_0)

        # Initial slack
        sigma_k = csd.MX.sym("sigma_0", self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone, 1)
        Sigma.append(sigma_k)

        # Add the initial action u_0 as parameter, relevant for the Q-function approximator
        u_0 = csd.MX.sym("u_0_constr", nu, 1)
        p_fixed.append(u_0)

        # Path following, speed deviation, chattering and fuel cost parameters
        dim_Q_p = self._params.Q_p.shape[0]
        s_final = csd.MX.sym("s_final", 1, 1)
        Q_p_vec = csd.MX.sym("Q_vec", dim_Q_p, 1)  # diagonal elements of Q_p
        Q_p = hf.casadi_diagonal_matrix_from_vector(Q_p_vec)
        alpha_app_course = csd.MX.sym("alpha_app_course", 2, 1)
        alpha_app_speed = csd.MX.sym("alpha_app_speed", 2, 1)
        K_app_course = csd.MX.sym("K_app_course", 1, 1)
        K_app_speed = csd.MX.sym("K_app_speed", 1, 1)

        U_ref = csd.MX.sym("U_ref", 1, 1)
        K_speed = csd.MX.sym("K_speed", 1, 1)
        K_fuel = csd.MX.sym("K_fuel", 1, 1)

        p_fixed.append(self._x_path_coeffs)
        p_fixed.append(self._y_path_coeffs)
        p_fixed.append(s_final)
        p_fixed.append(U_ref)

        gamma = csd.MX.sym("gamma", 1)
        p_fixed.append(gamma)

        p_adjustable.append(Q_p_vec)
        p_adjustable.append(alpha_app_course)
        p_adjustable.append(alpha_app_speed)
        p_adjustable.append(K_app_course)
        p_adjustable.append(K_app_speed)
        p_adjustable.append(K_speed)
        p_adjustable.append(K_fuel)

        # COLREGS cost parameters
        alpha_gw = csd.MX.sym("alpha_gw", 2, 1)
        y_0_gw = csd.MX.sym("y_0_gw", 1, 1)
        alpha_ho = csd.MX.sym("alpha_ho", 2, 1)
        x_0_ho = csd.MX.sym("x_0_ho", 1, 1)
        alpha_ot = csd.MX.sym("alpha_ot", 2, 1)
        x_0_ot = csd.MX.sym("x_0_ot", 1, 1)
        y_0_ot = csd.MX.sym("y_0_ot", 1, 1)
        colregs_weights = csd.MX.sym("w", 3, 1)

        p_adjustable.append(alpha_gw)
        p_adjustable.append(y_0_gw)
        p_adjustable.append(alpha_ho)
        p_adjustable.append(x_0_ho)
        p_adjustable.append(alpha_ot)
        p_adjustable.append(x_0_ot)
        p_adjustable.append(y_0_ot)
        p_adjustable.append(colregs_weights)

        # Slack weighting matrix W (dim = 1 x (self._params.max_num_so_constr + 3 * self._params.max_num_do_constr_per_zone))
        W = csd.MX.sym("W", self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone, 1)
        p_fixed.append(W)

        # Safety zone parameters
        r_safe_so = csd.MX.sym("r_safe_so", 1)
        r_safe_do = csd.MX.sym("r_safe_do", 1)
        p_fixed.append(r_safe_so)
        p_adjustable.append(r_safe_do)

        ship_vertices = self._model.params().ship_vertices

        # Dynamic obstacle augmented state parameters (x, y, chi, U, length, width) * N + 1 for colregs situations GW, HO and OT
        nx_do = 6
        X_do_gw = csd.MX.sym("X_do", nx_do * self._params.max_num_do_constr_per_zone, N + 1)
        X_do_ho = csd.MX.sym("X_do", nx_do * self._params.max_num_do_constr_per_zone, N + 1)
        X_do_ot = csd.MX.sym("X_do", nx_do * self._params.max_num_do_constr_per_zone, N + 1)

        p_fixed.append(csd.reshape(X_do_gw, -1, 1))
        p_fixed.append(csd.reshape(X_do_ho, -1, 1))
        p_fixed.append(csd.reshape(X_do_ot, -1, 1))

        # Static obstacle constraint parameters
        so_pars = csd.MX.sym("so_pars", 0)
        A_so_constr = csd.MX.sym("A_so_constr", 0)
        b_so_constr = csd.MX.sym("b_so_constr", 0)
        so_surfaces = []
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
            so_surfaces = mapf.compute_surface_approximations_from_polygons(so_list, enc, safety_margins=[self._params.r_safe_so], map_origin=self._map_origin)[0]
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
            so_pars = csd.MX.sym("so_pars", 3, self._params.max_num_so_constr)  # (x_c, y_c, r) x self._params.max_num_so_constr
            p_fixed.append(csd.reshape(so_pars, -1, 1))
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
            so_pars = csd.MX.sym("so_pars", 2 + 2 * 2, self._params.max_num_so_constr)  # (x_c, y_c, A_c.flatten().tolist()) x self._params.max_num_so_constr
            p_fixed.append(csd.reshape(so_pars, -1, 1))
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            A_so_constr = csd.MX.sym("A_so_constr", self._params.max_num_so_constr, 2)
            b_so_constr = csd.MX.sym("b_so_constr", self._params.max_num_so_constr, 1)
            p_fixed.append(csd.reshape(A_so_constr, -1, 1))
            p_fixed.append(csd.reshape(b_so_constr, -1, 1))
        else:
            raise ValueError("Unknown static obstacle constraint type.")

        # Cost function
        J = 0.0

        so_constr_list = []
        do_constr_list = []

        # Create symbolic integrator for the shooting gap constraints and discretized cost function
        stage_cost = csd.MX.sym("stage_cost", 1)
        erk4 = integrators.ERK4(x=x, p=csd.vertcat(u, self._p_mdl), ode=xdot, quad=stage_cost, h=dt)
        for k in range(N):
            u_k = csd.MX.sym("u_" + str(k), nu, 1)
            U.append(u_k)

            s_k = x_k[nx_ship]
            p_ref_k = csd.vertcat(self._x_path(s_k, self._x_path_coeffs), self._y_path(s_k, self._y_path_coeffs), s_final)

            # Sum stage costs
            J += gamma**k * (
                mpc_common.path_following_cost(x_k, p_ref_k, U_ref, Q_p, K_speed, nx_ship)
                + mpc_common.rate_cost(u_k, csd.vertcat(alpha_app_course, alpha_app_speed), csd.vertcat(K_app_course, K_app_speed), r_max=ubu[0], U_dot_max=ubu[1])
                + mpc_common.colregs_cost(
                    x_k, X_do_gw[:, k], X_do_ho[:, k], X_do_ot[:, k], nx_do, alpha_gw, y_0_gw, alpha_ho, x_0_ho, alpha_ot, x_0_ot, y_0_ot, colregs_weights
                )
                + W.T @ sigma_k
            )

            so_constr_k = self._create_static_obstacle_constraint(x_k, sigma_k, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, r_safe_so)
            so_constr_list.extend(so_constr_k)
            g_ineq_list.extend(so_constr_k)

            X_do_k = csd.vertcat(X_do_gw[:, k], X_do_ho[:, k], X_do_ot[:, k])
            do_constr_k = self._create_dynamic_obstacle_constraint(x_k, sigma_k, X_do_k, nx_do, r_safe_do)
            do_constr_list.extend(do_constr_k)
            g_ineq_list.extend(do_constr_k)

            # Shooting gap constraints
            x_k_end, _, _, _, _, _, _, _ = erk4(x_k, csd.vertcat(u_k, self._p_mdl))
            x_k = csd.MX.sym("x_" + str(k + 1), nx, 1)
            X.append(x_k)
            g_eq_list.append(x_k_end - x_k)

            sigma_k = csd.MX.sym("sigma_" + str(k + 1), self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone, 1)
            Sigma.append(sigma_k)

        # Terminal costs and constraints
        p_ref_N = csd.vertcat(self._x_path(x_k[nx_ship], self._x_path_coeffs), self._y_path(x_k[nx_ship], self._y_path_coeffs), s_final)
        J += gamma**N * (mpc_common.path_following_cost(x_k, p_ref_N, U_ref, Q_p, K_speed, nx_ship) + W.T @ sigma_k)

        so_constr_N = self._create_static_obstacle_constraint(x_k, sigma_k, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, r_safe_so)
        so_constr_list.extend(so_constr_N)
        g_ineq_list.extend(so_constr_N)

        X_do_N = csd.vertcat(X_do_gw[:, N], X_do_ho[:, N], X_do_ot[:, N])
        do_constr_N = self._create_dynamic_obstacle_constraint(x_k, sigma_k, X_do_N, nx_do, r_safe_do)
        do_constr_list.extend(do_constr_N)
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

        self._p_fixed_values = np.zeros((self._p_fixed.shape[0], 1))
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

        self._static_obstacle_constraints = csd.Function(
            "static_obstacle_constraints", [self._opt_vars, self._p], [csd.vertcat(*so_constr_list)], ["w", "p"], ["so_constr"]
        )
        self._dynamic_obstacle_constraints = csd.Function(
            "dynamic_obstacle_constraints",
            [self._opt_vars, self._p],
            [csd.vertcat(*do_constr_list)],
            ["w", "p"],
            ["do_constr"],
        )
        self._equality_constraints = csd.Function("equality_constraints", [self._opt_vars, self._p], [g_eq], ["w", "p"], ["g_eq"])
        self._inequality_constraints = csd.Function("inequality_constraints", [self._opt_vars, self._p], [g_ineq], ["w", "p"], ["g_ineq"])
        if self._params.max_num_so_constr + self._params.max_num_do_constr_per_zone == 0.0:
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
                    csd.reshape(csd.vertcat(*Sigma), self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone, -1),
                ],
                ["w"],
                ["X", "U", "Sigma"],
            )
            self._decision_variables = csd.Function(
                "decision_variables",
                [
                    csd.reshape(csd.vertcat(*X), nx, -1),
                    csd.reshape(csd.vertcat(*U), nu, -1),
                    csd.reshape(csd.vertcat(*Sigma), self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone, -1),
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
        r_safe_so: csd.MX,
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
            - r_safe_so (csd.MX): Safety distance to static obstacles.

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
                csd.vec(A_so_constr @ (mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * r_safe_so + x_k[0:2]) - b_so_constr - sigma_k[: self._params.max_num_so_constr])
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
                    weights = A_e / r_safe_so**2
                    so_constr_list.append(csd.log(1 - sigma_k[j] + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon))
            elif self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
                assert so_surfaces is not None, "Parametric surfaces must be provided for this constraint type."
                n_so = len(so_surfaces)
                for j in range(self._params.max_num_so_constr):
                    if j < n_so:
                        so_constr_list.append(so_surfaces[j](x_k[0:2].reshape((1, 2))) - sigma_k[j])
                        # vertices = mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * r_safe_so + x_k[0:2]
                        # vertices = vertices.reshape((-1, 2))
                        # for i in range(vertices.shape[0]):
                        #     so_constr_list.append(csd.vec(so_surfaces[j](vertices[i, :]) - sigma_k[j]))
                    else:
                        so_constr_list.append(-sigma_k[j])
        return so_constr_list

    def _create_dynamic_obstacle_constraint(self, x_k: csd.MX, sigma_k: csd.MX, X_do_k: csd.MX, nx_do: int, r_safe_do: csd.MX) -> list:
        """Creates the dynamic obstacle constraints for the NLP at the current stage.

        Args:
            x_k (csd.MX): State vector at the current stage in the OCP.
            sigma_k (csd.MX): Sigma vector at the current stage in the OCP.
            X_do_k (csd.MX): Decision variables of the dynamic obstacles (in all colregs zones) at the current stage in the OCP.
            nx_do (int): Dimension of fixed parameter vector for a dynamic obstacle.
            r_safe_do (csd.MX): Safety distance to dynamic obstacles.

        Returns:
            list: List of dynamic obstacle constraints at the current stage in the OCP.
        """
        do_constr_list = []
        epsilon = 1e-6
        n_do = int(X_do_k.shape[0] / nx_do)
        for i in range(n_do):
            x_aug_do_i = X_do_k[nx_do * i : nx_do * (i + 1)]
            x_do_i = x_aug_do_i[0:4]
            l_do_i = x_aug_do_i[4]
            w_do_i = x_aug_do_i[5]
            Rchi_do_i = mf.Rpsi2D_casadi(x_do_i[2])
            p_diff_do_frame = Rchi_do_i @ (x_k[0:2] - x_do_i[0:2])
            weights = hf.casadi_matrix_from_nested_list([[1.0 / (l_do_i + r_safe_do) ** 2, 0.0], [0.0, 1.0 / (w_do_i + r_safe_do) ** 2]])
            do_constr_list.append(csd.log(1 - sigma_k[self._params.max_num_so_constr + i] + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon))
        return do_constr_list

    def create_parameter_values(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray],
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        so_list: list,
        enc: Optional[senc.ENC] = None,
        **kwargs,
    ) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - state (np.ndarray): Current state of the system on the form (x, y, chi, U)^T.
            - action (np.ndarray, optional): Current action of the system on the form (chi_d, U_d)^T.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - so_list (list): List of static obstacles.
            - enc (Optional[senc.ENC]): Electronic Navigation Chart (ENC) object.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - np.ndarray: Parameter vector to be used as input to solver
        """
        n_colregs_zones = 3
        nx, nu = self._model.dims()
        nx_p, nu_p = self._path_timing.dims()
        N = int(self._params.T / self._params.dt)

        adjustable_parameter_values = self.get_adjustable_params()
        fixed_parameter_values: list = []

        state_aug = np.zeros((nx + nx_p, 1))
        state_aug[0:4] = state.reshape((4, 1))

        if action is None:
            action = np.array([state[2], state[3]])

        state_aug[4:nx] = action.reshape((2, 1))  # state is augmented with action when using the AugmentedKinematicCSOG model.
        state_aug[nx:] = np.array([self._s, self._s_dot]).reshape((2, 1))  # and path dynamics due to the integrated path timing model.
        fixed_parameter_values.extend(state_aug.flatten().tolist())  # x0
        fixed_parameter_values.extend([0.0] * (nu + nu_p))  # u0
        fixed_parameter_values.extend(self._x_path_coeffs_values.tolist())
        fixed_parameter_values.extend(self._y_path_coeffs_values.tolist())
        fixed_parameter_values.append(self._s_final_value)
        fixed_parameter_values.append(self._U_ref)
        fixed_parameter_values.append(self._params.gamma)

        W = self._params.w_L1 * np.ones(self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone)
        fixed_parameter_values.extend(W.tolist())
        fixed_parameter_values.append(self._params.r_safe_so)

        do_parameter_values_cr = self._create_do_parameter_values(state, do_cr_list, **kwargs)
        do_parameter_values_ho = self._create_do_parameter_values(state, do_ho_list, **kwargs)
        do_parameter_values_ot = self._create_do_parameter_values(state, do_ot_list, **kwargs)
        fixed_parameter_values.extend(do_parameter_values_cr)
        fixed_parameter_values.extend(do_parameter_values_ho)
        fixed_parameter_values.extend(do_parameter_values_ot)

        so_parameter_values = self._update_so_parameter_values(so_list, state, enc, **kwargs)
        fixed_parameter_values.extend(so_parameter_values)
        return np.concatenate((adjustable_parameter_values, np.array(fixed_parameter_values)), axis=0)

    def _create_do_parameter_values(self, state: np.ndarray, do_list: list, **kwargs) -> list:
        """Creates the parameter values for the dynamic obstacle constraints.

        Args:
            state (np.ndarray): Current state of the system on the form (x, y, chi, U).
            do_list (list): List of dynamic obstacles in a colregs zone.

        Returns:
            list: List of dynamic obstacle parameters to be used as input to solver
        """
        N = int(self._params.T / self._params.dt)
        do_parameter_values = []
        n_do = len(do_list)
        for k in range(N + 1):
            t = k * self._params.dt
            for i in range(self._params.max_num_do_constr_per_zone):
                if i < n_do:
                    (ID, do_state, cov, length, width) = do_list[i]
                    chi = np.atan2(do_state[3], do_state[2])
                    U = np.sqrt(do_state[2] ** 2 + do_state[3] ** 2)
                    do_parameter_values.extend([do_state[0] + t * U * np.cos(chi), do_state[1] + t * U * np.sin(chi), chi, U, length, width])
                else:
                    do_parameter_values.extend([state[0] - 10000.0, state[1] - 10000.0, 0.0, 0.0, 10.0, 3.0])
        return do_parameter_values

    def _create_fixed_so_parameter_values(self, so_list: list, state: np.ndarray, enc: Optional[senc.ENC] = None, **kwargs) -> np.ndarray:
        """Creates the fixed parameter values for the static obstacle constraints.

        Args:
            - so_list (list): List of static obstacles.
            - state (np.ndarray): Current state of the system on the form (x, y, chi, U).
            - enc (senc.ENC): Electronic Navigation Chart (ENC) object.

        Returns:
            np.ndarray: Fixed parameter vector for static obstacles to be used as input to solver
        """
        if len(so_list) == 0:
            return []
        fixed_so_parameter_values = []
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
            for (c, r) in so_list:
                fixed_so_parameter_values.extend([c[0], c[1], r])
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
            for (c, A) in so_list:
                fixed_so_parameter_values.extend([c[0], c[1], *A.flatten().tolist()])
        return fixed_so_parameter_values

    def _update_so_parameter_values(self, so_list: list, state: np.ndarray, enc: senc.ENC, **kwargs) -> list:
        """Updates the parameter values for the static obstacle constraints in case of changing constraints.

        Args:
            - so_list (list): List of static obstacles.
            - state (np.ndarray): Current state of the system on the form (x, y, chi, U).
            - enc (senc.ENC): Electronic Navigation Chart (ENC) object.

        Returns:
            np.ndarray: Fixed parameter vector for static obstacles to be used as input to solver
        """
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            assert len(so_list) == 2, "Approximate convex safe set constraint requires constraint variables A and b"
            A, b = so_list[0], so_list[1]
            self._p_fixed_so_values = np.concatenate((A.flatten(), b.flatten()), axis=0).tolist()
        return self._p_fixed_so_values

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

    def action_value(self, state: np.ndarray, action: np.ndarray, parameter_values: np.ndarray, show_plots: bool = False) -> Tuple[float, dict]:
        """Computes the action value function Q(s, a) for a given state and action.

        Args:
            - state (np.ndarray): State vector
            - action (np.ndarray): Action vector
            - parameter_values (np.ndarray, optional): Adjustable parameter vector for the MPC NLP problem.
            - show_plots (bool, optional): Whether to show plots or not. Defaults to False.

        Returns:
            Tuple[float, dict]: Action value function Q(s, a) and corresponding solution dictionary
        """
        if not self._initialized_q:
            self._current_warmstart_q = self._create_initial_warm_start(xs, nominal_trajectory, nominal_inputs, self._lbg_q.shape[0])
            self._p_fixed_so_values = self._create_fixed_so_parameter_values(so_list, xs, nominal_trajectory, enc, **kwargs)
            self._xs_prev = xs
            self._initialized_q = True

        psi = xs[2]
        xs_unwrapped = xs.copy()
        xs_unwrapped[2] = np.unwrap(np.array([self._xs_prev[2], psi]))[1]
        self._xs_prev = xs_unwrapped
        dt = t - self._t_prev
        if dt > 0.0:
            self._current_warmstart_q = self._shift_warm_start(self._current_warmstart_q, xs_unwrapped, dt, enc)

        parameter_values = self.create_parameter_values(xs_unwrapped, action, nominal_trajectory, nominal_inputs, do_list, so_list, enc, **kwargs)
        t_start = time.time()

        soln = self._qsolver(
            x0=self._current_warmstart_q["x"],
            lam_x0=self._current_warmstart_q["lam_x"],
            lam_g0=self._current_warmstart_q["lam_g"],
            p=parameter_values,
            lbx=self._lbw,
            ubx=self._ubw,
            lbg=self._lbg_q,
            ubg=self._ubg_q,
        )
        t_solve = time.time() - t_start
        stats = self._qsolver.stats()
        if stats["return_status"] == "Maximum_Iterations_Exceeded":
            # Use solution unless it is infeasible, then use previous solution.
            g_eq_vals = self._equality_constraints(soln["x"], parameter_values).full()
            g_ineq_vals = self._inequality_constraints(soln["x"], parameter_values).full()
            if np.any(g_eq_vals > 1e-6) or np.any(g_ineq_vals > 1e-6):
                soln = self._current_warmstart_q
        elif stats["return_status"] == "Infeasible_Problem_Detected":
            raise RuntimeError("Infeasible solution found.")

        so_constr_vals = self._static_obstacle_constraints(soln["x"], parameter_values).full()
        g_eq_vals = self._equality_constraints(soln["x"], parameter_values).full()
        g_ineq_vals = self._inequality_constraints(soln["x"], parameter_values).full()
        X, U, Sigma = self._extract_trajectories(soln)
        self._current_warmstart_q["x"] = self._decision_variables(X, U, Sigma)
        self._current_warmstart_q["lam_x"] = soln["lam_x"].full()
        self._current_warmstart_q["lam_g"] = soln["lam_g"].full()
        cost_val = soln["f"].full()[0][0]
        final_residuals = [stats["iterations"]["inf_du"][-1], stats["iterations"]["inf_pr"][-1]]
        hf.plot_solver_stats(stats, show_plots)
        print(
            f"NMPC: | Runtime: {t_solve} | Cost: {cost_val} | sl (max, argmax): ({np.max(Sigma)}, {np.argmax(Sigma)}) | so_constr (max, argmax): ({np.max(so_constr_vals)}, {np.argmax(so_constr_vals)})"
        )
        self._t_prev = t
        output = {
            "trajectory": X,
            "inputs": U,
            "lower_slacks": [],
            "upper_slacks": Sigma,
            "so_constr_vals": so_constr_vals,
            "do_constr_vals": [],
            "t_solve": t_solve,
            "cost_val": cost_val,
            "n_iter": stats["iter_count"],
            "final_residuals": final_residuals,
        }
        return output

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
