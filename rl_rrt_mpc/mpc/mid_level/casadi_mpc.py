"""
    casadi_mpc.py

    Summary:
        Casadi MPC class for the mid-level COLAV planner.

    Author: Trym Tengesdal
"""
import time
from typing import Optional, Tuple

import casadi as csd
import colav_simulator.common.map_functions as cs_mapf
import matplotlib.pyplot as plt
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
        params: parameters.MidlevelMPCParams,
        solver_options: mpc_common.CasadiSolverOptions,
    ) -> None:
        self._model = model
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
        self._x_path_coeffs: csd.MX = csd.MX.sym("x_path_coeffs", 0)
        self._x_path_coeffs_values: np.ndarray = np.array([])
        self._y_path: csd.Function = csd.Function("y_path", [], [])
        self._y_path_coeffs: csd.MX = csd.MX.sym("y_path_coeffs", 0)
        self._y_path_coeffs_values: np.ndarray = np.array([])
        self._speed_spline: csd.Function = csd.Function("speed_spline", [], [])
        self._speed_spl_coeffs: csd.MX = csd.MX.sym("speed_spl_coeffs", 0)
        self._speed_spl_coeffs_values: np.ndarray = np.array([])
        self._s: float = 0.0
        self._s_dot: float = 0.0
        self._s_final_value: float = 0.0

        # debugging functions
        self._p_path = csd.MX.sym("p_path", 0)

        self._p_rate = csd.MX.sym("p_rate", 0)
        self._p_colregs = csd.MX.sym("p_colregs", 0)
        self._path_dev_cost = csd.Function("path_dev_cost", [], [])
        self._speed_dev_cost = csd.Function("speed_ref_cost", [], [])
        self._course_rate_cost = csd.Function("course_rate_cost", [], [])
        self._speed_rate_cost = csd.Function("speed_rate_cost", [], [])
        self._crossing_cost = csd.Function("crossing_cost", [], [])
        self._head_on_cost = csd.Function("head_on_cost", [], [])
        self._overtaking_cost = csd.Function("overtaking_cost", [], [])

    @property
    def params(self):
        return self._params

    def get_adjustable_params(self) -> np.ndarray:
        """Returns the RL-tuneable parameters in the MPC.

        Returns:
            np.ndarray: Array of parameters.
        """
        mdl_params = self._model.params()
        mdl_adjustable_params = np.array([mdl_params.T_chi, mdl_params.T_U])
        mpc_adjustable_params = self.params.adjustable()
        return np.concatenate((mdl_adjustable_params, mpc_adjustable_params))

    def _set_path_information(
        self, nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, float]
    ) -> None:
        """Sets the path information for the MPC.

        Args:
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, inter.PchipInterpolator, float]): Tuple containing the nominal path splines in x, y, heading and the nominal speed reference.
        """
        x_spline, y_spline, _, speed_spline = nominal_path
        s = csd.MX.sym("s", 1)
        x_path_coeffs = csd.MX.sym("x_path_coeffs", x_spline.c.shape[0])
        x_path = csd.bspline(s, x_path_coeffs, [[*x_spline.t]], [x_spline.k], 1, {})
        self._x_path = csd.Function("x_path", [s, x_path_coeffs], [x_path])
        self._x_path_coeffs_values = x_spline.c
        self._x_path_coeffs = x_path_coeffs

        x_spline_der = x_spline.derivative()
        x_dot_path_coeffs = csd.MX.sym("x_dot_path_coeffs", x_spline_der.c.shape[0])
        x_dot_path = csd.bspline(s, x_dot_path_coeffs, [[*x_spline_der.t]], [x_spline_der.k], 1, {})
        self._x_dot_path = csd.Function("x_dot_path", [s, x_dot_path_coeffs], [x_dot_path])
        self._x_dot_path_coeffs_values = x_spline_der.c
        self._x_dot_path_coeffs = x_dot_path_coeffs

        y_path_coeffs = csd.MX.sym("y_path_coeffs", y_spline.c.shape[0])
        y_path = csd.bspline(s, y_path_coeffs, [[*y_spline.t]], [y_spline.k], 1, {})
        self._y_path = csd.Function("y_path", [s, y_path_coeffs], [y_path])
        self._y_path_coeffs_values = y_spline.c
        self._y_path_coeffs = y_path_coeffs

        y_spline_der = y_spline.derivative()
        y_dot_path_coeffs = csd.MX.sym("y_dot_path_coeffs", y_spline_der.c.shape[0])
        y_dot_path = csd.bspline(s, y_dot_path_coeffs, [[*y_spline_der.t]], [y_spline_der.k], 1, {})
        self._y_dot_path = csd.Function("y_dot_path", [s, y_dot_path_coeffs], [y_dot_path])
        self._y_dot_path_coeffs_values = y_spline_der.c
        self._y_dot_path_coeffs = y_dot_path_coeffs

        self._speed_spl_coeffs_values = speed_spline.c
        speed_spl_coeffs = csd.MX.sym("speed_spl_coeffs", speed_spline.c.shape[0])
        speed_spl = csd.bspline(s, speed_spl_coeffs, [[*speed_spline.t]], [speed_spline.k], 1, {})
        self._speed_spline = csd.Function("speed_spline", [s, speed_spl_coeffs], [speed_spl])
        self._speed_spl_coeffs = speed_spl_coeffs

        self._s_final_value = x_spline.t[-1]
        self._model.set_min_path_variable(0.0)
        self._model.set_max_path_variable(self._s_final_value)
        self._model.set_path_derivative_splines(
            self._x_dot_path, self._x_dot_path_coeffs, self._y_dot_path, self._y_dot_path_coeffs
        )
        self._model.setup_equations_of_motion()

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
        p = np.concatenate((self._p_ship_mdl_values, self._x_dot_path_coeffs_values, self._y_dot_path_coeffs_values))
        X = self._model.erk4_n_step(xs, u, p, self._params.dt, N)
        return X

    def _update_path_variables(self, dt: float) -> None:
        """Updates the path variables s and s_dot.

        Args:
            dt (float): Time step between the previous MPC iteration and the current.
        """
        self._s = self._s + self._s_dot * dt
        self._s_dot = self.compute_path_variable_derivative(self._s)

    def _create_initial_warm_start(self, xs: np.ndarray, dim_g: int, enc: senc.ENC) -> dict:
        """Sets the initial warm start decision trajectory [U, X, Sigma] flattened for the NMPC.

        Args:
            - xs (np.ndarray): Initial state of the system (x, y, chi, U)
            - dim_g (int): Dimension/length of the constraints.
            - enc (senc.ENC): ENC object containing the map info.
        """
        self._s = 0.0
        self._s_dot = self.compute_path_variable_derivative(self._s)

        n_colregs_zones = 3
        nx, nu = self._model.dims()
        N = int(self._params.T / self._params.dt)
        w = np.zeros((nu, N))

        path_timing_input = 1.0
        w[1, :] = path_timing_input * np.ones(N)
        w = w.flatten()

        xs_k = np.zeros(nx)
        xs_k[:4] = xs[:4]
        xs_k[4] = xs[2]
        xs_k[5:] = np.array([self._s, self._s_dot])

        n_attempts = 7
        success = False
        path_timing_input = 0.05
        u_attempts = [
            np.array([0.0, 0.0]),
            np.array([0.01, 0.0]),
            np.array([-0.01, 0.0]),
            np.array([0.01, -0.05]),
            np.array([-0.01, -0.05]),
            np.array([0.0, -0.05]),
            np.zeros(nu),
        ]
        for i in range(n_attempts):
            warm_start_traj = self._model_prediction(xs_k, u_attempts[i], N + 1)
            positions = np.array(
                [warm_start_traj[1, :] + self._map_origin[1], warm_start_traj[0, :] + self._map_origin[0]]
            )
            distance_vectors = cs_mapf.compute_distance_vectors_to_grounding(positions, self._min_depth, enc)
            dv_norms = np.linalg.norm(distance_vectors, axis=0)
            min_dist = np.min(dv_norms)
            if min_dist > self._params.r_safe_so:
                success = True
                break

        assert success, "Could not create initial warm start solution"
        chi = warm_start_traj[2, :]
        chi_unwrapped = np.unwrap(np.concatenate(([xs_k[2]], chi)))[1:]
        chi_d = warm_start_traj[4, :]
        chi_d = np.unwrap(np.concatenate(([xs_k[2]], chi_d)))[1:]
        warm_start_traj[2, :] = chi_unwrapped
        warm_start_traj[4, :] = chi_d
        w = np.concatenate((w, warm_start_traj.T.flatten()))
        w = np.concatenate(
            (
                w,
                np.zeros(
                    (self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone)
                    * (N + 1)
                ),
            )
        )
        warm_start = {"x": w.tolist(), "lam_x": np.zeros(w.shape[0]).tolist(), "lam_g": np.zeros(dim_g).tolist()}
        shifted_ws_traj = warm_start_traj + np.array(
            [self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0, 0.0]
        ).reshape(nx, 1)
        cs_mapf.plot_trajectory(shifted_ws_traj, enc, "orange")
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
        Sigma_warm_start = np.concatenate(
            (Sigma_prev[:, n_shifts:], np.tile(Sigma_prev[:, -1], (n_shifts, 1)).T), axis=1
        )

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
        w_warm_start = np.concatenate(
            (U_warm_start.T.flatten(), X_warm_start.T.flatten(), Sigma_warm_start.T.flatten())
        )

        pos_past_N = states_past_N[:2, :] + self._map_origin.reshape(2, 1)
        pos_past_N[0, :] = states_past_N[1, :] + self._map_origin[1]
        pos_past_N[1, :] = states_past_N[0, :] + self._map_origin[0]
        distance_vectors = cs_mapf.compute_distance_vectors_to_grounding(pos_past_N, self._min_depth, enc)
        dv_norms = np.linalg.norm(distance_vectors, axis=0)
        min_dist = np.min(dv_norms)
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
        self._s = self._s + self._s_dot * dt
        nx, nu = self._model.dims()
        lbu, ubu, _, _ = self._model.get_input_state_bounds()
        n_attempts = 3
        n_shifts = int(dt / self._params.dt)
        w_prev = np.array(prev_warm_start["x"])
        X_prev, U_prev, Sigma_prev = self._decision_trajectories(w_prev)
        X_prev = X_prev.full()
        U_prev = U_prev.full()
        Sigma_prev = Sigma_prev.full()
        offsets = [0, int(2.5 * n_shifts), int(2.5 * n_shifts), int(2.5 * n_shifts)]
        path_timing_input = 0.5
        u_attempts = [
            U_prev[:, -1],
            np.array([U_prev[0, -offsets[1]], path_timing_input]),
            np.array([U_prev[0, -offsets[2]], path_timing_input]),
            np.zeros(nu),
        ]
        success = False
        for i in range(n_attempts):
            w_warm_start, success = self._try_to_create_warm_start_solution(
                X_prev, U_prev, Sigma_prev, u_attempts[i], offsets[i], n_shifts, enc
            )
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

        parameter_values, do_cr_params, do_ho_params, do_ot_params = self.create_parameter_values(
            xs_unwrapped, action, do_cr_list, do_ho_list, do_ot_list, so_list, enc, **kwargs
        )
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
            if dt > 0.0 and np.any(g_eq_vals > 1e-6) or np.any(g_ineq_vals > 1e-6):
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
        self.plot_cost_function_values(X, U, Sigma, do_cr_params, do_ho_params, do_ot_params, show_plots)
        self.plot_solution_trajectory(X, U, Sigma)
        arg_max_sigma, max_sigma = -1, -1
        if Sigma.size > 0:
            arg_max_sigma, max_sigma = np.argmax(Sigma), np.max(Sigma)
        arg_max_do_constr, max_do_constr = -1, -1
        if do_constr_vals.size > 0:
            arg_max_do_constr, max_do_constr = np.argmax(do_constr_vals), np.max(do_constr_vals)

        arg_max_so_constr, max_so_constr = -1, -1
        if so_constr_vals.size > 0:
            arg_max_so_constr, max_so_constr = np.argmax(so_constr_vals), np.max(so_constr_vals)

        print(
            f"Mid-level COLAV: \n\t- Runtime: {t_solve} \n\t- Cost: {cost_val} \n\t- Upper slacks (max, argmax): ({max_sigma}, {arg_max_sigma}) \n\t- Static obstacle constraints (max, argmax): ({max_so_constr}, {arg_max_so_constr}) \n\t- Dynamic obstacle constraints (max, argmax): ({max_do_constr}, {arg_max_do_constr})"
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
        chi = X[2, :]
        chi = np.unwrap(np.concatenate(([chi[0]], chi)))[1:]
        X[2, :] = chi
        chi_d = X[4, :]
        chi_d = np.unwrap(np.concatenate(([chi_d[0]], chi_d)))[1:]
        X[4, :] = chi_d
        return X, U, Sigma

    def compute_path_variable_derivative(self, s: float | csd.MX) -> float | csd.MX:
        """Computes the path variable derivative.

        Args:
            s (float | csd.MX): Path variable.

        Returns:
            float | csd.MX: Path variable derivative.
        """
        epsilon = 1e-9
        if isinstance(s, float):
            s_dot = self._speed_spline(s, self._speed_spl_coeffs_values) / np.sqrt(
                epsilon
                + self._x_dot_path(s, self._x_dot_path_coeffs_values) ** 2
                + self._y_dot_path(s, self._y_dot_path_coeffs_values) ** 2
            )
            return s_dot.full()[0][0]
        else:
            return self._speed_spline(s, self._speed_spl_coeffs) / csd.sqrt(
                epsilon
                + self._x_dot_path(s, self._x_dot_path_coeffs) ** 2
                + self._y_dot_path(s, self._y_dot_path_coeffs) ** 2
            )

    def construct_ocp(
        self,
        nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline],
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
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline]): Tuple containing the nominal path splines in x, y, heading and speed.
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
        nx, nu = self._model.dims()
        xdot, x, u, p = self._model.as_casadi()
        lbu_k, ubu_k, lbx_k, ubx_k = self._model.get_input_state_bounds()

        self._p_mdl = p

        # Box constraints on NLP decision variables
        lbu = np.tile(lbu_k, N)
        ubu = np.tile(ubu_k, N)
        lbx = np.tile(lbx_k, N + 1)
        ubx = np.tile(ubx_k, N + 1)
        lbsigma = np.array(
            [0] * (N + 1) * (self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone)
        )
        ubsigma = np.array(
            [np.inf]
            * (N + 1)
            * (self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone)
        )
        self._lbw = np.concatenate((lbu, lbx, lbsigma))
        self._ubw = np.concatenate((ubu, ubx, ubsigma))

        g_eq_list = []  # NLP equality constraints
        g_ineq_list = []  # NLP inequality constraints
        p_fixed, p_adjustable = [], []  # NLP parameters

        # NLP decision variables
        U = []
        X = []
        Sigma = []

        p_adjustable.append(self._p_mdl[0])  # T_chi, T_U
        p_adjustable.append(self._p_mdl[1])

        # Initial state constraint
        x_0 = csd.MX.sym("x_0_constr", nx, 1)
        x_k = csd.MX.sym("x_0", nx, 1)
        X.append(x_k)
        g_eq_list.append(x_0 - x_k)
        p_fixed.append(x_0)

        # Initial slack
        sigma_k = csd.MX.sym(
            "sigma_0", self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone, 1
        )
        Sigma.append(sigma_k)

        # Add the initial action u_0 as parameter, relevant for the Q-function approximator
        u_0 = csd.MX.sym("u_0_constr", nu, 1)
        p_fixed.append(u_0)

        # Path following, speed deviation, chattering and fuel cost parameters
        dim_Q_p = self._params.Q_p.shape[0]
        Q_p_vec = csd.MX.sym("Q_vec", dim_Q_p, 1)  # diagonal elements of Q_p
        Q_p = hf.casadi_diagonal_matrix_from_vector(Q_p_vec)
        alpha_app_course = csd.MX.sym("alpha_app_course", 2, 1)
        alpha_app_speed = csd.MX.sym("alpha_app_speed", 2, 1)
        K_app_course = csd.MX.sym("K_app_course", 1, 1)
        K_app_speed = csd.MX.sym("K_app_speed", 1, 1)

        K_fuel = csd.MX.sym("K_fuel", 1, 1)

        p_fixed.append(self._x_path_coeffs)
        p_fixed.append(self._y_path_coeffs)
        p_fixed.append(self._x_dot_path_coeffs)
        p_fixed.append(self._y_dot_path_coeffs)
        p_fixed.append(self._speed_spl_coeffs)

        gamma = csd.MX.sym("gamma", 1)
        p_fixed.append(gamma)

        p_adjustable.append(Q_p_vec)
        p_adjustable.append(alpha_app_course)
        p_adjustable.append(alpha_app_speed)
        p_adjustable.append(K_app_course)
        p_adjustable.append(K_app_speed)
        p_adjustable.append(K_fuel)

        # COLREGS cost parameters
        alpha_cr = csd.MX.sym("alpha_cr", 2, 1)
        y_0_cr = csd.MX.sym("y_0_cr", 1, 1)
        alpha_ho = csd.MX.sym("alpha_ho", 2, 1)
        x_0_ho = csd.MX.sym("x_0_ho", 1, 1)
        alpha_ot = csd.MX.sym("alpha_ot", 2, 1)
        x_0_ot = csd.MX.sym("x_0_ot", 1, 1)
        y_0_ot = csd.MX.sym("y_0_ot", 1, 1)
        colregs_weights = csd.MX.sym("w", 3, 1)

        p_adjustable.append(alpha_cr)
        p_adjustable.append(y_0_cr)
        p_adjustable.append(alpha_ho)
        p_adjustable.append(x_0_ho)
        p_adjustable.append(alpha_ot)
        p_adjustable.append(x_0_ot)
        p_adjustable.append(y_0_ot)
        p_adjustable.append(colregs_weights)

        # Slack weighting matrix W (dim = 1 x (self._params.max_num_so_constr + 3 * self._params.max_num_do_constr_per_zone))
        W = csd.MX.sym(
            "W", self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone, 1
        )
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

        self._p_path = csd.vertcat(
            self._x_path_coeffs,
            self._y_path_coeffs,
            self._x_dot_path_coeffs,
            self._y_dot_path_coeffs,
            self._speed_spl_coeffs,
            Q_p_vec,
        )
        self._p_rate = csd.vertcat(alpha_app_course, alpha_app_speed, K_app_course, K_app_speed)
        self._p_colregs = csd.vertcat(alpha_cr, y_0_cr, alpha_ho, x_0_ho, alpha_ot, x_0_ot, y_0_ot, colregs_weights)

        # Static obstacle constraint parameters
        so_pars = csd.MX.sym("so_pars", 0)
        A_so_constr = csd.MX.sym("A_so_constr", 0)
        b_so_constr = csd.MX.sym("b_so_constr", 0)
        so_surfaces = []
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
            so_surfaces = mapf.compute_surface_approximations_from_polygons(
                so_list, enc, safety_margins=[self._params.r_safe_so], map_origin=self._map_origin
            )[0]
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
            so_pars = csd.MX.sym(
                "so_pars", 3, self._params.max_num_so_constr
            )  # (x_c, y_c, r) x self._params.max_num_so_constr
            p_fixed.append(csd.reshape(so_pars, -1, 1))
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
            so_pars = csd.MX.sym(
                "so_pars", 2 + 2 * 2, self._params.max_num_so_constr
            )  # (x_c, y_c, A_c.flatten().tolist()) x self._params.max_num_so_constr
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
        # stage_cost = csd.MX.sym("stage_cost", 1)
        erk4 = integrators.ERK4(x=x, p=csd.vertcat(u, self._p_mdl), ode=xdot, quad=csd.vertcat([]), h=dt)
        s_k_ref = self._s
        U_dot_max = 0.5  # m/s^2
        eps = 1e-9
        for k in range(N):
            u_k = csd.MX.sym("u_" + str(k), nu, 1)
            U.append(u_k)

            p_ref_k = csd.vertcat(
                self._x_path(s_k_ref, self._x_path_coeffs), self._y_path(s_k_ref, self._y_path_coeffs)
            )
            s_dot_ref_k = self.compute_path_variable_derivative(s_k_ref)
            path_ref_k = csd.vertcat(p_ref_k, s_dot_ref_k)
            s_k_ref = s_dot_ref_k * dt + s_k_ref
            # Sum stage costs
            U_d_k = x_k[6] * csd.sqrt(
                + self._x_dot_path(x_k[5], self._x_dot_path_coeffs) ** 2
                + self._y_dot_path(x_k[5], self._y_dot_path_coeffs) ** 2
            )
            path_following_cost, path_dev_cost, speed_dev_cost = mpc_common.path_following_cost(x_k, path_ref_k, Q_p)
            rate_cost, course_rate_cost, speed_rate_cost = mpc_common.rate_cost(
                u_k,
                csd.vertcat(alpha_app_course, alpha_app_speed),
                csd.vertcat(K_app_course, K_app_speed),
                r_max=ubu[0],
                U_dot_max=U_dot_max,
            )
            colregs_cost, crossing_cost, head_on_cost, overtaking_cost = mpc_common.colregs_cost(
                x_k,
                X_do_gw[:, k],
                X_do_ho[:, k],
                X_do_ot[:, k],
                nx_do,
                alpha_cr,
                y_0_cr,
                alpha_ho,
                x_0_ho,
                alpha_ot,
                x_0_ot,
                y_0_ot,
                colregs_weights,
            )
            slack_penalty_cost = W.T @ sigma_k
            J += gamma**k * (path_following_cost + rate_cost + colregs_cost + slack_penalty_cost)
            if k == 0:
                self._path_dev_cost = csd.Function("path_dev_cost", [x_k, self._p_path], [path_dev_cost])
                self._speed_dev_cost = csd.Function("speed_ref_cost", [x_k, self._p_path], [speed_dev_cost])
                self._course_rate_cost = csd.Function("course_rate_cost", [u_k, self._p_rate], [course_rate_cost])
                self._speed_rate_cost = csd.Function("speed_rate_cost", [u_k, self._p_rate], [speed_rate_cost])
                self._crossing_cost = csd.Function(
                    "crossing_cost", [x_k, X_do_gw[:, k], self._p_colregs], [crossing_cost]
                )
                self._head_on_cost = csd.Function("head_on_cost", [x_k, X_do_ho[:, k], self._p_colregs], [head_on_cost])
                self._overtaking_cost = csd.Function(
                    "overtaking_cost", [x_k, X_do_ot[:, k], self._p_colregs], [overtaking_cost]
                )
                self._slack_penalty_cost = csd.Function("slack_penalty_cost", [sigma_k], [slack_penalty_cost])

            so_constr_k = self._create_static_obstacle_constraint(
                x_k, sigma_k, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, r_safe_so
            )
            so_constr_list.extend(so_constr_k)
            g_ineq_list.extend(so_constr_k)

            X_do_k = csd.vertcat(X_do_gw[:, k], X_do_ho[:, k], X_do_ot[:, k])
            do_constr_k = self._create_dynamic_obstacle_constraint(x_k, sigma_k, X_do_k, nx_do, r_safe_do)
            do_constr_list.extend(do_constr_k)
            g_ineq_list.extend(do_constr_k)

            # Shooting gap constraints
            # x_0_test_1 = np.array([0.0, 0.0, -0.78, 5.0, -0.78, 3.0, 0.0, 0.0])
            # x_0_test_2 = x_0_test_1.copy()
            # u_0_test = np.array([0.0, 0.0, 1.0])
            # for l in range(10):
            #     x_1, _, _, _, _, _, _, _ = erk4(x=x_0_test_1, p=csd.vertcat(u_0_test, self._p_ship_mdl_values))
            #     x_1_alt = self._model_prediction(x_0_test_2, u_0_test, 2)[:, -1]
            #     print(f"diff: {x_1 - x_1_alt}")
            #     x_0_test_1 = x_1
            #     x_0_test_2 = x_1_alt
            x_k_end, _, _, _, _, _, _, _ = erk4(x_k, csd.vertcat(u_k, self._p_mdl))
            x_k = csd.MX.sym("x_" + str(k + 1), nx, 1)
            X.append(x_k)
            g_eq_list.append(x_k_end - x_k)

            sigma_k = csd.MX.sym(
                "sigma_" + str(k + 1),
                self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone,
                1,
            )
            Sigma.append(sigma_k)

        # Terminal costs and constraints
        p_ref_N = csd.vertcat(self._x_path(s_k_ref, self._x_path_coeffs), self._y_path(s_k_ref, self._y_path_coeffs))
        s_dot_ref_N = self.compute_path_variable_derivative(s_k_ref)
        path_ref_N = csd.vertcat(p_ref_N, s_dot_ref_N)
        path_following_cost, _, _ = mpc_common.path_following_cost(x_k, path_ref_N, Q_p)
        J += gamma**N * (path_following_cost + W.T @ sigma_k)

        so_constr_N = self._create_static_obstacle_constraint(
            x_k, sigma_k, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, r_safe_so
        )
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
            "static_obstacle_constraints",
            [self._opt_vars, self._p],
            [csd.vertcat(*so_constr_list)],
            ["w", "p"],
            ["so_constr"],
        )
        self._dynamic_obstacle_constraints = csd.Function(
            "dynamic_obstacle_constraints",
            [self._opt_vars, self._p],
            [csd.vertcat(*do_constr_list)],
            ["w", "p"],
            ["do_constr"],
        )
        self._equality_constraints = csd.Function(
            "equality_constraints", [self._opt_vars, self._p], [g_eq], ["w", "p"], ["g_eq"]
        )
        self._inequality_constraints = csd.Function(
            "inequality_constraints", [self._opt_vars, self._p], [g_ineq], ["w", "p"], ["g_ineq"]
        )
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
                    csd.reshape(
                        csd.vertcat(*Sigma),
                        self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone,
                        -1,
                    ),
                ],
                ["w"],
                ["X", "U", "Sigma"],
            )
            self._decision_variables = csd.Function(
                "decision_variables",
                [
                    csd.reshape(csd.vertcat(*X), nx, -1),
                    csd.reshape(csd.vertcat(*U), nu, -1),
                    csd.reshape(
                        csd.vertcat(*Sigma),
                        self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone,
                        -1,
                    ),
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
        epsilon = 1e-9
        so_constr_list = []
        if self._params.max_num_so_constr == 0:
            return so_constr_list

        if self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            assert (
                A_so_constr is not None and b_so_constr is not None
            ), "Convex safe set constraints must be provided for this constraint type."
            so_constr_list.append(
                # A_so_constr @ x_k[0:2]
                # - b_so_constr
                # - sigma_k[: self._params.max_num_so_constr]
                csd.vec(
                    A_so_constr @ (mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * r_safe_so + x_k[0:2])
                    - b_so_constr
                    - sigma_k[: self._params.max_num_so_constr]
                )
            )
        else:
            if self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
                assert (
                    so_pars.shape[0] == 3
                ), "Static obstacle parameters with dim 3 in first axis must be provided for this constraint type."
                for j in range(self._params.max_num_so_constr):
                    x_c, y_c, r_c = so_pars[0, j], so_pars[1, j], so_pars[2, j]
                    so_constr_list.append(
                        csd.log(r_c**2 - sigma_k[j] + epsilon)
                        - csd.log(((x_k[0] - x_c) ** 2) + (x_k[1] - y_c) ** 2 + epsilon)
                    )
            elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
                assert (
                    so_pars.shape[0] == 4
                ), "Static obstacle parameters with dim 4 in first axis must be provided for this constraint type."
                for j in range(self._params.max_num_so_constr):
                    x_e, y_e, A_e = so_pars[0, j], so_pars[1, j], so_pars[2:, j]
                    A_e = csd.reshape(A_e, 2, 2)
                    p_diff_do_frame = x_k[0:2] - csd.vertcat(x_e, y_e)
                    weights = A_e / r_safe_so**2
                    so_constr_list.append(
                        csd.log(1 - sigma_k[j] + epsilon)
                        - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon)
                    )
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

    def _create_dynamic_obstacle_constraint(
        self, x_k: csd.MX, sigma_k: csd.MX, X_do_k: csd.MX, nx_do: int, r_safe_do: csd.MX
    ) -> list:
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
        epsilon = 1e-9
        n_do = int(X_do_k.shape[0] / nx_do)
        for i in range(n_do):
            x_aug_do_i = X_do_k[nx_do * i : nx_do * (i + 1)]
            x_do_i = x_aug_do_i[0:4]
            l_do_i = x_aug_do_i[4]
            w_do_i = x_aug_do_i[5]
            Rchi_do_i = mf.Rpsi2D_casadi(x_do_i[2])
            p_diff_do_frame = Rchi_do_i @ (x_k[0:2] - x_do_i[0:2])
            weights = hf.casadi_matrix_from_nested_list(
                [[1.0 / (l_do_i + r_safe_do) ** 2, 0.0], [0.0, 1.0 / (w_do_i + r_safe_do) ** 2]]
            )
            do_constr_list.append(
                csd.log(1 - sigma_k[self._params.max_num_so_constr + i] + epsilon)
                - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon)
            )
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - state (np.ndarray): Current state of the system on the form (x, y, chi, U)^T.
            - action (np.ndarray, optional): Current action of the system on the form (chi_d, u_p)^T.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - so_list (list): List of static obstacles.
            - enc (Optional[senc.ENC]): Electronic Navigation Chart (ENC) object.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of parameter vector values, and arrays of dynamic obstacle parameter values for each colregs zone.
        """
        assert len(state) == 4, "State must be of length 4."
        n_colregs_zones = 3
        nx, nu = self._model.dims()

        adjustable_parameter_values = self.get_adjustable_params()
        fixed_parameter_values: list = []

        state_aug = np.zeros((nx, 1))
        state_aug[0:4] = state.reshape((4, 1))

        if action is None:
            action = np.array([state[2], 0.0])

        # state is augmented with the course action when using the HalfAugmentedKinematicCSOG model.
        state_aug[4] = action[0]
        # and path dynamics due to the integrated path timing model with velocity assignment
        state_aug[5:] = np.array([self._s, self._s_dot]).reshape((2, 1))
        fixed_parameter_values.extend(state_aug.flatten().tolist())  # x0
        fixed_parameter_values.extend([0.0] * nu)  # u0
        fixed_parameter_values.extend(self._x_path_coeffs_values.tolist())
        fixed_parameter_values.extend(self._y_path_coeffs_values.tolist())
        fixed_parameter_values.extend(self._x_dot_path_coeffs_values.tolist())
        fixed_parameter_values.extend(self._y_dot_path_coeffs_values.tolist())
        fixed_parameter_values.extend(self._speed_spl_coeffs_values.tolist())

        # fixed_parameter_values.append(self._s_final_value)
        fixed_parameter_values.append(self._params.gamma)

        W = self._params.w_L1 * np.ones(
            self._params.max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone
        )
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
        return (
            np.concatenate((adjustable_parameter_values, np.array(fixed_parameter_values)), axis=0),
            np.array(do_parameter_values_cr),
            np.array(do_parameter_values_ho),
            np.array(do_parameter_values_ot),
        )

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
                    do_parameter_values.extend(
                        [do_state[0] + t * U * np.cos(chi), do_state[1] + t * U * np.sin(chi), chi, U, length, width]
                    )
                else:
                    do_parameter_values.extend([state[0] - 10000.0, state[1] - 10000.0, 0.0, 0.0, 10.0, 3.0])
        return do_parameter_values

    def _create_fixed_so_parameter_values(
        self, so_list: list, state: np.ndarray, enc: Optional[senc.ENC] = None, **kwargs
    ) -> np.ndarray:
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
            for c, r in so_list:
                fixed_so_parameter_values.extend([c[0], c[1], r])
        elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
            for c, A in so_list:
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

    def action_value(
        self, state: np.ndarray, action: np.ndarray, parameter_values: np.ndarray, show_plots: bool = False
    ) -> Tuple[float, dict]:
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
            self._current_warmstart_q = self._create_initial_warm_start(
                xs, nominal_trajectory, nominal_inputs, self._lbg_q.shape[0]
            )
            self._p_fixed_so_values = self._create_fixed_so_parameter_values(
                so_list, xs, nominal_trajectory, enc, **kwargs
            )
            self._xs_prev = xs
            self._initialized_q = True

        psi = xs[2]
        xs_unwrapped = xs.copy()
        xs_unwrapped[2] = np.unwrap(np.array([self._xs_prev[2], psi]))[1]
        self._xs_prev = xs_unwrapped
        dt = t - self._t_prev
        if dt > 0.0:
            self._current_warmstart_q = self._shift_warm_start(self._current_warmstart_q, xs_unwrapped, dt, enc)

        parameter_values = self.create_parameter_values(
            xs_unwrapped, action, nominal_trajectory, nominal_inputs, do_list, so_list, enc, **kwargs
        )
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

    def constrained_parameter_update(
        self, learning_rate: float, dJ: np.ndarray, parameter_values: np.ndarray
    ) -> np.ndarray:
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

    def plot_cost_function_values(
        self,
        X: np.ndarray,
        U: np.ndarray,
        Sigma: np.ndarray,
        do_cr_parameters: np.ndarray,
        do_ho_parameters: np.ndarray,
        do_ot_parameters: np.ndarray,
        show_plots: bool = False,
    ) -> None:
        """Plots the cost function values for the solution.

        Args:
            - X (np.ndarray): State trajectory.
            - U (np.ndarray): Input trajectory.
            - Sigma (np.ndarray): Slack trajectory.
            - do_cr_parameters (np.ndarray): Crossing dynamic obstacle parameters.
            - do_ho_parameters (np.ndarray): Head-on dynamic obstacle parameters.
            - do_ot_parameters (np.ndarray): Overtaking dynamic obstacle parameters.
            - show_plots (bool, optional): Whether to show the plots or not. Defaults to False.
        """
        path_dev_cost = np.zeros(X.shape[1])
        speed_dev_cost = np.zeros(X.shape[1])
        course_rate_cost = np.zeros(X.shape[1])
        speed_rate_cost = np.zeros(X.shape[1])
        crossing_cost = np.zeros(X.shape[1])
        head_on_cost = np.zeros(X.shape[1])
        overtaking_cost = np.zeros(X.shape[1])
        w_colregs = self._params.w_colregs
        colregs_cost = np.zeros(X.shape[1])
        X_do_cr = do_cr_parameters.reshape(6, -1)
        X_do_ho = do_ho_parameters.reshape(6, -1)
        X_do_ot = do_ot_parameters.reshape(6, -1)
        p_path_values = np.array(
            [
                *self._x_path_coeffs_values.tolist(),
                *self._y_path_coeffs_values.tolist(),
                *self._x_dot_path_coeffs_values.tolist(),
                *self._y_dot_path_coeffs_values.tolist(),
                *self._speed_spl_coeffs_values.tolist(),
                *self._params.Q_p.diagonal().flatten().tolist(),
            ]
        )
        p_rate_values = np.array(
            [
                *self._params.alpha_app_course.tolist(),
                *self._params.alpha_app_speed.tolist(),
                self._params.K_app_course,
                self._params.K_app_speed,
            ]
        )
        p_colregs_values = np.array(
            [
                *self._params.alpha_cr.tolist(),
                self._params.y_0_cr,
                *self._params.alpha_ho.tolist(),
                self._params.x_0_ho,
                *self._params.alpha_ot.tolist(),
                self._params.x_0_ot,
                self._params.y_0_ot,
                *self._params.w_colregs.tolist(),
            ]
        )
        for k in range(X.shape[1]):
            x_do_cr_k = np.zeros((0, 1))
            if X_do_cr.size > 6:
                x_do_cr_k = X_do_cr[:, k]
            x_do_ho_k = np.zeros((0, 1))
            if X_do_ho.size > 6:
                x_do_ho_k = X_do_ho[:, k]
            x_do_ot_k = np.zeros((0, 1))
            if X_do_ot.size > 6:
                x_do_ot_k = X_do_ot[:, k]
            path_dev_cost[k] = self._path_dev_cost(X[:, k], p_path_values)
            speed_dev_cost[k] = self._speed_dev_cost(X[:, k], p_path_values)

            crossing_cost[k] = self._crossing_cost(X[:, k], x_do_cr_k, p_colregs_values)
            head_on_cost[k] = self._head_on_cost(X[:, k], x_do_ho_k, p_colregs_values)
            overtaking_cost[k] = self._overtaking_cost(X[:, k], x_do_ot_k, p_colregs_values)
            colregs_cost[k] = (
                w_colregs[0] * crossing_cost[k] + w_colregs[1] * head_on_cost[k] + w_colregs[2] * overtaking_cost[k]
            )

            if k < U.shape[1]:
                course_rate_cost[k] = self._course_rate_cost(U[:, k], p_rate_values)
                speed_rate_cost[k] = self._speed_rate_cost(U[:, k], p_rate_values)

        fig = plt.figure(figsize=(12, 8))
        axes = fig.subplot_mosaic(
            [
                ["p(s_k) - p_k"],
                ["s_cost"],
                ["chi_rate_cost"],
                ["U_rate_cost"],
                ["colregs_cost"],
            ]
        )
        axes["p(s_k) - p_k"].plot(path_dev_cost, label=r"$p(s_k) - p_k$ cost")
        axes["p(s_k) - p_k"].set_ylabel("cost")
        axes["p(s_k) - p_k"].set_xlabel("k")
        axes["p(s_k) - p_k"].legend()

        axes["s_cost"].plot(speed_dev_cost, label=r"$\dot{s}_{ref} - \dot{s}$ cost")
        axes["s_cost"].set_ylabel("cost")
        axes["s_cost"].set_xlabel("k")
        axes["s_cost"].legend()

        axes["chi_rate_cost"].plot(course_rate_cost, label=r"$\dot{\chi}_d$ cost")
        axes["chi_rate_cost"].set_ylabel("cost")
        axes["chi_rate_cost"].set_xlabel("k")
        axes["chi_rate_cost"].legend()

        axes["U_rate_cost"].plot(speed_rate_cost, label=r"$\dot{U}_d$ cost")
        axes["U_rate_cost"].set_ylabel("cost")
        axes["U_rate_cost"].set_xlabel("k")
        axes["U_rate_cost"].legend()

        axes["colregs_cost"].plot(colregs_cost, color="b", label="colregs cost")
        axes["colregs_cost"].plot(crossing_cost, color="r", label="crossing cost")
        axes["colregs_cost"].plot(head_on_cost, color="g", label="head on cost")
        axes["colregs_cost"].plot(overtaking_cost, color="y", label="overtaking cost")
        axes["colregs_cost"].set_ylabel("cost")
        axes["colregs_cost"].set_xlabel("k")
        axes["colregs_cost"].legend()
        plt.show(block=False)
        print("Plotted cost function values")

    def plot_solution_trajectory(self, X: np.ndarray, U: np.ndarray, Sigma: np.ndarray) -> None:
        """Plots the solution trajectory.

        Args:
            X (np.ndarray): State trajectory.
            U (np.ndarray): Input trajectory.
            Sigma (np.ndarray): Slack trajectory.
        """
        s_dot_ref = np.zeros(X.shape[1])
        s_k_ref = self._s
        epsilon = 1e-9
        speed_refs = np.zeros(X.shape[1])
        mpc_speed_refs = np.zeros(X.shape[1])
        for k in range(X.shape[1]):
            x_dot_path = self._x_dot_path(s_k_ref, self._x_dot_path_coeffs_values)
            y_dot_path = self._y_dot_path(s_k_ref, self._y_dot_path_coeffs_values)
            s_dot_ref[k] = self._speed_spline(s_k_ref, self._speed_spl_coeffs_values) / (
                np.sqrt(x_dot_path**2 + y_dot_path**2) + epsilon
            )
            speed_refs[k] = self._speed_spline(s_k_ref, self._speed_spl_coeffs_values)
            mpc_speed_refs[k] = X[6, k] * np.sqrt(x_dot_path**2 + y_dot_path**2 + epsilon)
            s_k_ref += s_dot_ref[k] * self._params.dt

        fig = plt.figure(figsize=(12, 8))
        axes = fig.subplot_mosaic(
            [
                ["course"],
                ["course_rate_input"],
                ["speed"],
                ["speed_rate_input"],
                ["path_variable"],
                ["path_dot_variable"],
                ["path_input"],
            ]
        )
        axes["course"].plot(np.rad2deg(X[2, :]), color="b", label=r"$\chi$")
        axes["course"].plot(np.rad2deg(X[4, :]), color="r", label=r"$\chi_d$")
        axes["course"].set_ylabel("[deg]")
        axes["course"].set_xlabel("k")
        axes["course"].legend()

        axes["course_rate_input"].plot(np.rad2deg(U[0, :]), color="b", label=r"$\dot{\chi}_d$")
        axes["course_rate_input"].set_ylabel("[deg/s]")
        axes["course_rate_input"].set_xlabel("k")
        axes["course_rate_input"].legend()

        axes["speed"].plot(X[3, :], color="b", label=r"$U$")
        axes["speed"].plot(mpc_speed_refs, color="r", label=r"$U_{d}(\omega)$")
        axes["speed"].plot(speed_refs, color="g", linestyle="--", label=r"$U_{ref}$")
        axes["speed"].set_ylabel("[m/s]")
        axes["speed"].set_xlabel("k")
        axes["speed"].legend()

        axes["speed_rate_input"].plot(U[1, :], color="b", label=r"$\dot{U}_d$")
        axes["speed_rate_input"].set_ylabel("[m/s2]")
        axes["speed_rate_input"].set_xlabel("k")
        axes["speed_rate_input"].legend()

        axes["path_variable"].plot(X[5, :], color="b", label=r"$s$")
        axes["path_variable"].set_ylabel("[m]")
        axes["path_variable"].set_xlabel("k")
        axes["path_variable"].legend()

        axes["path_dot_variable"].plot(X[6, :], color="b", label=r"$\dot{s}$")
        axes["path_dot_variable"].plot(s_dot_ref, color="r", label=r"$\dot{s}_{ref}$")
        axes["path_dot_variable"].set_ylabel("[m/s]")
        axes["path_dot_variable"].set_xlabel("k")
        axes["path_dot_variable"].legend()

        axes["path_input"].plot(U[1, :], color="b", label="path input")
        axes["path_input"].set_ylabel("[m/s2]")
        axes["path_input"].set_xlabel("k")
        axes["path_input"].legend()

        plt.show(block=False)
        print("Plotted solution trajectory")
