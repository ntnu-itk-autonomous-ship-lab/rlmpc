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
        self._initialized: bool = False
        self._map_origin: np.ndarray = np.array([])
        self._prev_cost: float = np.inf
        self._so_surfaces: list = []

        self._nlp_perturbation: csd.MX = csd.MX.sym("nlp_perturbation", 0)

        self._cost_function: csd.MX = csd.MX.sym("cost_function", 0)
        self._g_eq: csd.MX = csd.MX.sym("g_eq", 0)
        self._g_ineq: csd.MX = csd.MX.sym("g_ineq", 0)
        self._nlp_eq: csd.MX = csd.MX.sym("nlp_eq", 0)
        self._nlp_ineq: csd.MX = csd.MX.sym("nlp_ineq", 0)

        self._opt_vars: csd.MX = csd.MX.sym("opt_vars", 0)
        self._opt_vars_list: list[csd.MX] = []
        self._solver: csd.Function = csd.Function("vsolver", [], [])
        self._current_warmstart: dict = {"x": [], "lam_x0": [], "lam_g": []}
        self._lbg: np.ndarray = np.array([])
        self._ubg: np.ndarray = np.array([])

        self._p_mdl = csd.MX.sym("p_mdl", 0)
        self._p_fixed: csd.MX = csd.MX.sym("p_fixed", 0)
        self._p_fixed_list: list[csd.MX] = []
        self._p_adjustable: csd.MX = csd.MX.sym("p_adjustable", 0)
        self._p_adjustable_list: list[csd.MX] = []
        self._p: csd.MX = csd.MX.sym("p", 0)
        self._p_list: list[csd.MX] = []

        self._set_generator: Optional[sg.SetGenerator] = None
        self._p_ship_mdl_values: np.ndarray = np.array([])
        self._p_fixed_so_values: np.ndarray = np.array([])
        self._p_fixed_values: np.ndarray = np.array([])
        self._p_adjustable_values: np.ndarray = np.array([])

        self._t_prev: float = 0.0
        self._xs_prev: np.ndarray = np.array([])
        self._min_depth: int = 5

        self._x_path: csd.Function = csd.Function("x_path", [], [])
        self._x_path_coeffs: csd.MX = csd.MX.sym("x_path_coeffs", 0)
        self._x_path_coeffs_values: np.ndarray = np.array([])
        self._x_dot_path: csd.Function = csd.Function("x_dot_path", [], [])
        self._x_dot_path_coeffs: csd.MX = csd.MX.sym("x_dot_path_coeffs", 0)
        self._x_dot_path_coeffs_values: np.ndarray = np.array([])

        self._y_path: csd.Function = csd.Function("y_path", [], [])
        self._y_path_coeffs: csd.MX = csd.MX.sym("y_path_coeffs", 0)
        self._y_path_coeffs_values: np.ndarray = np.array([])
        self._y_dot_path: csd.Function = csd.Function("y_dot_path", [], [])
        self._y_dot_path_coeffs: csd.MX = csd.MX.sym("y_dot_path_coeffs", 0)
        self._y_dot_path_coeffs_values: np.ndarray = np.array([])

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
        self._path_dot_dev_cost = csd.Function("path_dot_dev_cost", [], [])
        self._speed_dev_cost = csd.Function("speed_dev_cost", [], [])
        self._course_rate_cost = csd.Function("course_rate_cost", [], [])
        self._speed_rate_cost = csd.Function("speed_rate_cost", [], [])
        self._crossing_cost = csd.Function("crossing_cost", [], [])
        self._head_on_cost = csd.Function("head_on_cost", [], [])
        self._overtaking_cost = csd.Function("overtaking_cost", [], [])
        self._decision_trajectories: csd.Function = csd.Function("decision_trajectories", [], [])
        self._decision_variables: csd.Function = csd.Function("decision_variables", [], [])
        self._static_obstacle_constraints: csd.Function = csd.Function("static_obstacle_constraints", [], [])
        self._dynamic_obstacle_constraints: csd.Function = csd.Function("dynamic_obstacle_constraints", [], [])
        self._equality_constraints: csd.Function = csd.Function("equality_constraints", [], [])
        self._equality_constraints_jacobian: csd.Function = csd.Function("equality_constraints_jacobian", [], [])
        self._inequality_constraints: csd.Function = csd.Function("inequality_constraints", [], [])
        self._inequality_constraints_jacobian: csd.Function = csd.Function("inequality_constraints_jacobian", [], [])
        self._box_inequality_constraints: csd.Function = csd.Function("box_inequality_constraints", [], [])

    @property
    def params(self):
        return self._params

    def get_adjustable_params(self) -> np.ndarray:
        """Returns the RL-tuneable parameters in the MPC.

        Returns:
            np.ndarray: Array of parameters.
        """
        mdl_adjustable_params = np.array([])
        mpc_adjustable_params = self.params.adjustable()
        return np.concatenate((mdl_adjustable_params, mpc_adjustable_params))

    def _set_path_information(
        self, nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline]
    ) -> None:
        """Sets the path information for the MPC.

        Args:
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, inter.PchipInterpolator, interp.BSpline]): Tuple containing the nominal path splines in x, y, heading and the nominal speed reference.
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
        self._model.set_min_path_variable(self._s)
        self._model.set_max_path_variable(self._s_final_value)
        print("Path information set. | s_final: ", self._s_final_value)

    def _model_prediction(self, xs: np.ndarray, u: np.ndarray, N: int) -> np.ndarray:
        """Euler prediction of the ship model and path timing model, concatenated.

        Args:
            - xs (np.ndarray): Starting state of the system (x, y, chi, U).
            - u (np.ndarray): Input to apply, either nu x N or nu x 1.
            - dt (float): Time step.
            - N (int): Number of steps to predict.

        Returns:
            np.ndarray: Predicted states.
        """
        p = self._p_ship_mdl_values
        X = self._model.erk4_n_step(xs, u, p, self._params.dt, N)
        return X

    def _create_initial_warm_start(self, xs: np.ndarray, dim_g: int, do_list, enc: senc.ENC) -> dict:
        """Sets the initial warm start decision trajectory [U, X, Sigma] flattened for the NMPC.

        Args:
            - xs (np.ndarray): Initial state of the system (x, y, psi, u, v, r)^T.
            - dim_g (int): Dimension/length of the constraints.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - enc (senc.ENC): ENC object containing the map info.
        """
        self._s = 0.0
        self._s_dot = self.compute_path_variable_derivative(self._s)

        n_colregs_zones = 3
        nx, nu = self._model.dims()
        N = int(self._params.T / self._params.dt)
        inputs = np.zeros((nu, N))

        xs_k = np.zeros(nx)
        xs_k[:3] = xs[:3]
        xs_k[3] = np.sqrt(xs[3] ** 2 + xs[4] ** 2)
        xs_k[4:] = np.array([self._s, self._s_dot])

        n_attempts = 7
        success = False
        u_attempts = [
            np.zeros(nu),
            np.array([-0.02, 0.0, 0.0]),
            np.array([0.02, 0.0, 0.0]),
            np.array([-0.02, -0.05, 0.0]),
            np.array([0.02, -0.05, 0.0]),
            np.array([0.0, -0.05, 0.0]),
        ]
        do_list_shifted = []
        for do in do_list:
            do_state = do[1]
            do_state_shifted = do_state.copy()
            do_state_shifted[0] = do_state[1] + self._map_origin[1]
            do_state_shifted[1] = do_state[0] + self._map_origin[0]
            do_state_shifted[2] = do_state[3]
            do_state_shifted[3] = do_state[2]
            do_shifted = (do[0], do_state_shifted, do[2], do[3], do[4])
            do_list_shifted.append(do_shifted)

        for i in range(n_attempts):
            inputs = np.tile(u_attempts[i], (N, 1)).T
            warm_start_traj = self._model_prediction(xs_k, inputs, N + 1)
            positions = np.array(
                [warm_start_traj[1, :] + self._map_origin[1], warm_start_traj[0, :] + self._map_origin[0]]
            )
            min_dist_do, min_dist_so, _, _ = cs_mapf.compute_minimum_distance_to_collision_and_grounding(
                positions,
                do_list_shifted,
                enc,
                self._params.T + self._params.dt,
                self._params.dt,
                self._min_depth,
                disable_bbox_check=True,
            )
            if min_dist_so > self._params.r_safe_so and min_dist_do > self._params.r_safe_do:
                success = True
                break

        assert success, "Could not create initial warm start solution"
        chi = warm_start_traj[2, :]
        chi_unwrapped = np.unwrap(np.concatenate(([xs_k[2]], chi)))[1:]
        warm_start_traj[2, :] = chi_unwrapped
        max_num_so_constr = self._params.max_num_so_constr
        if self._so_surfaces:
            max_num_so_constr = len(self._so_surfaces) if self._params.max_num_so_constr > 0 else 0
        w = np.concatenate(
            (
                inputs.T.flatten(),
                warm_start_traj.T.flatten(),
                np.zeros(
                    (nx + max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone) * (N + 1)
                ),
            )
        )

        warm_start = {"x": w.tolist(), "lam_x": np.zeros(w.shape[0]).tolist(), "lam_g": np.zeros(dim_g).tolist()}
        # shifted_ws_traj = warm_start_traj + np.array(
        #     [self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]
        # ).reshape(nx, 1)
        # cs_mapf.plot_trajectory(shifted_ws_traj, enc, "orange")
        return warm_start

    def _try_to_create_warm_start_solution(
        self,
        xs: np.ndarray,
        X_prev: np.ndarray,
        U_prev: np.ndarray,
        Sigma_prev: np.ndarray,
        u_mod: np.ndarray,
        offset: int,
        n_shifts: int,
        do_list: list,
        enc: senc.ENC,
    ) -> Tuple[np.ndarray, bool]:
        """Creates a shifted warm start trajectory from the previous trajectory and the new control input,
        possibly with an offset start back in time.
        Args:
            - xs (np.ndarray): Current state of the ownship.
            - X_prev (np.ndarray): The previous trajectory.
            - U_prev (np.ndarray): The previous control inputs.
            - Sigma_prev (np.ndarray): The previous slack variables.
            - u_mod (np.ndarray): The new control input to apply at the end of the previous trajectory.
            - offset (int): The offset from the last sample in the previous trajectory, to apply the modified input vector.
            - dt (float): Time step
            - n_shifts (int): Number of shifts to perform on the previous trajectory.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) with EN coordinates.
            - enc (senc.ENC): The ENC object containing the map.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, bool]: The new warm start, corresponding state and input trajectories, and a boolean indicating if the warm start was successful.
        """
        N = int(self._params.T / self._params.dt)
        nx, nu = self._model.dims()
        Sigma_warm_start = np.concatenate(
            (Sigma_prev[:, n_shifts:], np.tile(Sigma_prev[:, -1], (n_shifts, 1)).T), axis=1
        )

        if offset == 0:
            inputs_past_N = np.tile(u_mod, (n_shifts, 1)).T
            U_warm_start = np.concatenate((U_prev[:, n_shifts:], inputs_past_N), axis=1)
        else:
            inputs_past_N = np.tile(u_mod, (n_shifts + offset, 1)).T
            U_warm_start = np.concatenate((U_prev[:, n_shifts:-offset], inputs_past_N), axis=1)

        xs_init_mpc = np.zeros(nx)
        xs_init_mpc[:3] = xs[:3]
        xs_init_mpc[3] = np.sqrt(xs[3] ** 2 + xs[4] ** 2)
        s_shifted = X_prev[4, n_shifts]
        s_dot_shifted = X_prev[5, n_shifts]
        xs_init_mpc[4:] = np.array([s_shifted, s_dot_shifted])
        X_warm_start = self._model_prediction(xs_init_mpc, U_warm_start, N + 1)
        chi = X_warm_start[2, :].tolist()
        X_warm_start[2, :] = np.unwrap(np.concatenate(([chi[0]], chi)))[1:]
        w_warm_start = np.concatenate(
            (U_warm_start.T.flatten(), X_warm_start.T.flatten(), Sigma_warm_start.T.flatten())
        )

        positions = X_warm_start[:2, :] + self._map_origin.reshape(2, 1)
        positions[0, :] = X_warm_start[1, :] + self._map_origin[1]
        positions[1, :] = X_warm_start[0, :] + self._map_origin[0]
        min_dist_do, min_dist_so, _, _ = cs_mapf.compute_minimum_distance_to_collision_and_grounding(
            positions,
            do_list,
            enc,
            self._params.T + self._params.dt,
            self._params.dt,
            self._min_depth,
            disable_bbox_check=True,
        )
        shifted_pos_traj = X_warm_start[:2, :] + np.array([self._map_origin[0], self._map_origin[1]]).reshape(2, 1)
        # if min_dist_so <= self._params.r_safe_so or min_dist_do <= self._params.r_safe_do:
        # cs_mapf.plot_trajectory(shifted_pos_traj, enc, "black")
        # return w_warm_start, X_warm_start, U_warm_start, False
        cs_mapf.plot_trajectory(shifted_pos_traj, enc, "pink")
        return w_warm_start, X_warm_start, U_warm_start, True

    def _shift_warm_start(self, xs: np.ndarray, prev_warm_start: dict, dt: float, do_list: list, enc: senc.ENC) -> dict:
        """Shifts the warm start decision trajectory [U, X, Sigma] dt units ahead.

        Args:
            - xs (np.ndarray): Current state of the ownship.
            - prev_warm_start (dict): Warm start decision trajectory to shift.
            - dt (float): Time to shift the warm start decision trajectory.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - enc (senc.ENC): Electronic Navigational Chart object.
        """
        _, nu = self._model.dims()
        n_attempts = 6
        n_shifts = int(dt / self._params.dt)
        N = int(self._params.T / self._params.dt)
        U_prev, X_prev, Sigma_prev = self._decision_trajectories(prev_warm_start["x"])
        X_prev = X_prev.full()
        U_prev = U_prev.full()
        Sigma_prev = Sigma_prev.full()
        offsets = [
            0,
            int(0.4 * N),
            int(0.4 * N),
            int(0.7 * N),
            int(0.7 * N),
            int(0.9 * N),
        ]
        u_attempts = [
            U_prev[:, -1],
            np.array([0.05, -0.05, 0.0]),
            np.array([-0.05, -0.05, 0.0]),
            np.array([0.05, -0.1, 0.0]),
            np.array([-0.05, -0.1, 0.0]),
            np.array([0.0, -0.1, 0.0]),
        ]

        do_list_shifted = []
        for do in do_list:
            do_state = do[1]
            do_state_shifted = do_state.copy()
            do_state_shifted[0] = do_state[1] + self._map_origin[1]
            do_state_shifted[1] = do_state[0] + self._map_origin[0]
            do_state_shifted[2] = do_state[3]
            do_state_shifted[3] = do_state[2]
            do_shifted = (do[0], do_state_shifted, do[2], do[3], do[4])
            do_list_shifted.append(do_shifted)

        success = False
        for i in range(n_attempts):
            w_warm_start, X_warm_start, U_warm_start, success = self._try_to_create_warm_start_solution(
                xs, X_prev, U_prev, Sigma_prev, u_attempts[i], offsets[i], n_shifts, do_list_shifted, enc
            )
            if success:
                break
        assert success, "Could not create warm start solution"
        new_warm_start = prev_warm_start
        new_warm_start["x"] = w_warm_start.tolist()
        self._s = X_warm_start[4, 0]
        self._s_dot = X_warm_start[5, 0]
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
        perturb_nlp: bool = False,
        perturb_sigma: float = 0.001,
        show_plots: bool = False,
        **kwargs,
    ) -> dict:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - xs (np.ndarray): Current state [x, y, psi, u, v, r]^T of the ownship.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - so_list (list): List ofrelevant static obstacle Polygon objects.
            - enc (Optional[senc.ENC]): ENC object containing the map info.
            - perturb_nlp (bool, optional): Whether to perturb the NLP. Defaults to False.
            - perturb_sigma (float, optional): What standard deviation to use for generating the perturbation. Defaults to 0.001.
            - show_plots (bool, optional): Whether to show plots. Defaults to False.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - dict: Dictionary containing the optimal trajectory, inputs, slacks and solver stats.
        """
        if not self._initialized:
            self._current_warmstart = self._create_initial_warm_start(
                xs, self._lbg.shape[0], do_list=do_cr_list + do_ho_list + do_ot_list, enc=enc
            )
            self._p_fixed_so_values = self._create_fixed_so_parameter_values(so_list, xs, enc, **kwargs)
            self._xs_prev = xs
            self._initialized = True
            self._prev_cost = np.inf

        chi = xs[2]
        xs_unwrapped = xs.copy()
        xs_unwrapped[2] = np.unwrap(np.array([self._xs_prev[2], chi]))[1]
        self._xs_prev = xs_unwrapped
        dt = t - self._t_prev
        if dt > 0.0:
            self._current_warmstart = self._shift_warm_start(
                xs_unwrapped, self._current_warmstart, dt, do_list=do_cr_list + do_ho_list + do_ot_list, enc=enc
            )

        parameter_values, do_cr_params, do_ho_params, do_ot_params = self.create_parameter_values(
            self._current_warmstart,
            xs_unwrapped,
            None,
            do_cr_list,
            do_ho_list,
            do_ot_list,
            so_list,
            enc,
            perturb_nlp=perturb_nlp,
            perturb_sigma=perturb_sigma,
            **kwargs,
        )

        # Check initial start feasibility wrt equality and inequality constraints:
        U_ws, X_ws, Sigma_ws = self._decision_trajectories(self._current_warmstart["x"])
        w_sub_ws = np.concatenate((U_ws.full().T.flatten(), X_ws.full().T.flatten()))
        g_eq_vals = self._equality_constraints(w_sub_ws, parameter_values).full().flatten()
        g_ineq_vals = self._inequality_constraints(self._current_warmstart["x"], parameter_values).full().flatten()
        if np.any(np.abs(g_eq_vals) > 1e-6):
            print(
                f"Warm start is infeasible wrt equality constraints at rows: {np.argwhere(np.abs(g_eq_vals) > 1e-6)}!"
            )
        if np.any(g_ineq_vals > 1e-6):
            print(f"Warm start is infeasible wrt inequality constraints at row: {np.argwhere(g_ineq_vals > 1e-6)}!")

        t_start = time.time()
        soln = self._solver(
            x0=self._current_warmstart["x"],
            lam_x0=self._current_warmstart["lam_x"],
            lam_g0=self._current_warmstart["lam_g"],
            p=parameter_values,
            lbg=self._lbg,
            ubg=self._ubg,
        )
        t_solve = time.time() - t_start
        stats = self._solver.stats()
        cost_val = soln["f"].full()[0][0]
        lam_x = soln["lam_x"].full()
        lam_g = soln["lam_g"].full()
        U, X, Sigma = self._extract_trajectories(soln)
        w_sub = np.concatenate((U.T.flatten(), X.T.flatten()))
        self.print_solution_info(soln, parameter_values, stats, t_solve)
        self.plot_solution_trajectory(X, U, Sigma)
        self.plot_cost_function_values(X, U, Sigma, do_cr_params, do_ho_params, do_ot_params, show_plots)

        if not stats["success"]:
            mpc_common.plot_casadi_solver_stats(stats, True)
            self.plot_cost_function_values(X, U, Sigma, do_cr_params, do_ho_params, do_ot_params, show_plots)
            self.plot_solution_trajectory(X, U, Sigma)
            if stats["return_status"] == "Maximum_Iterations_Exceeded":
                # Use solution unless it is infeasible, then use previous solution.
                g_eq_vals = self._equality_constraints(w_sub, parameter_values).full().flatten()
                g_ineq_vals = self._inequality_constraints(soln["x"], parameter_values).full().flatten()
                if dt > 0.0 and np.any(np.abs(g_eq_vals) > 1e-6) or np.any(g_ineq_vals > 1e-6):
                    soln = self._current_warmstart
                    soln["f"] = self._prev_cost
                    cost_val = self._prev_cost
                    lam_x = self._current_warmstart["lam_x"]
                    lam_g = self._current_warmstart["lam_g"]
            elif stats["return_status"] == "Infeasible_Problem_Detected":
                # Should not happen with slacks
                raise RuntimeError("Infeasible solution found.")

        so_constr_vals = self._static_obstacle_constraints(soln["x"], parameter_values).full()
        do_constr_vals = self._dynamic_obstacle_constraints(soln["x"], parameter_values).full()
        g_eq_vals = self._equality_constraints(w_sub, parameter_values).full()
        g_ineq_vals = self._inequality_constraints(soln["x"], parameter_values).full()

        # self.print_solution_info(soln, parameter_values, stats, t_solve)
        # self.plot_cost_function_values(X, U, Sigma, do_cr_params, do_ho_params, do_ot_params, show_plots)
        # self.plot_solution_trajectory(X, U, Sigma)
        # mpc_common.plot_casadi_solver_stats(stats, show_plots)

        self._current_warmstart["x"] = self._decision_variables(U, X, Sigma)
        self._current_warmstart["lam_x"] = lam_x
        self._current_warmstart["lam_g"] = lam_g
        self._t_prev = t
        self._prev_inputs = U[:, 0]
        final_residuals = [stats["iterations"]["inf_du"][-1], stats["iterations"]["inf_pr"][-1]]
        output = {
            "soln": soln,
            "trajectory": X,
            "inputs": U,
            "slacks": Sigma,
            "so_constr_vals": so_constr_vals,
            "do_constr_vals": do_constr_vals,
            "g_eq_vals": g_eq_vals,
            "g_ineq_vals": g_ineq_vals,
            "t_solve": t_solve,
            "cost_val": cost_val,
            "n_iter": stats["iter_count"],
            "final_residuals": final_residuals,
        }
        return output

    def _extract_trajectories(self, soln: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the optimal inputs, trajectory and slacks from the solution dictionary.

        Args:
            soln (dict): Solution dictionary.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Optimal inputs, trajectory and slacks.
        """
        U, X, Sigma = self._decision_trajectories(soln["x"])
        X = X.full()
        U = U.full()
        Sigma = Sigma.full()
        chi = X[2, :]
        chi = np.unwrap(np.concatenate(([chi[0]], chi)))[1:]
        X[2, :] = chi
        return U, X, Sigma

    def compute_path_variable_derivative(self, s: float | csd.MX) -> float | csd.MX:
        """Computes the path variable derivative.

        Args:
            s (float | csd.MX): Path variable.

        Returns:
            float | csd.MX: Path variable derivative.
        """
        epsilon = 1e-8
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
        approx_inf = 1e10

        # Ship model and path timing dynamics
        nx, nu = self._model.dims()
        xdot, x, u, p = self._model.as_casadi()
        lbu_k, ubu_k, lbx_k, ubx_k = self._model.get_input_state_bounds()

        self._p_mdl = p

        hu = []  # Box constraints on inputs
        hx = []  # Box constraints on states
        hs, hs_bx, hs_so, hs_do = [], [], [], []  # Box constraints on sigma

        g_eq_list = []  # NLP equality constraints
        g_ineq_list = []  # NLP inequality constraints
        p_fixed, p_adjustable = [], []  # NLP parameters

        # NLP decision variables
        U = []
        X = []
        Sigma, Sigma_bx, Sigma_do, Sigma_so = [], [], [], []

        if self._p_mdl.shape[0] > 0:
            p_adjustable.append(self._p_mdl)

        # NLP perturbation (may be zero or randomly generated if the MPC is used as a stochastic policy)
        self._nlp_perturbation = csd.MX.sym("nlp_perturbation", nu, 1)
        p_fixed.append(self._nlp_perturbation)

        # Initial state constraint
        x_0 = csd.MX.sym("x_0_constr", nx, 1)
        x_k = csd.MX.sym("x_0", nx, 1)
        X.append(x_k)
        g_eq_list.append(x_0 - x_k)
        p_fixed.append(x_0)

        sigma_bx_k = csd.MX.sym(
            "sigma_bx_0",
            ubx_k.shape[0],
            1,
        )
        Sigma_bx.append(sigma_bx_k)

        # Add the initial action u_0 as parameter, relevant for the Q-function approximator
        u_0 = csd.MX.sym("u_0_constr", nu, 1)
        p_fixed.append(u_0)

        # Path following, speed deviation, chattering and fuel cost parameters
        dim_Q_p = self._params.Q_p.shape[0]
        Q_p_vec = csd.MX.sym("Q_vec", dim_Q_p, 1)
        alpha_app_course = csd.MX.sym("alpha_app_course", 2, 1)
        alpha_app_speed = csd.MX.sym("alpha_app_speed", 2, 1)
        K_app_course = csd.MX.sym("K_app_course", 1, 1)
        K_app_speed = csd.MX.sym("K_app_speed", 1, 1)
        K_fuel = csd.MX.sym("K_fuel", 1, 1)

        path_derivative_refs_list = []
        for k in range(N + 1):
            s_dot_ref_k = csd.MX.sym("s_ref_" + str(k), 1, 1)
            path_derivative_refs_list.append(s_dot_ref_k)
        p_fixed.append(self._x_path_coeffs)
        p_fixed.append(self._y_path_coeffs)
        p_fixed.append(self._speed_spl_coeffs)
        p_fixed.append(csd.vertcat(*path_derivative_refs_list))

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
        d_attenuation = csd.MX.sym("d_attenuation", 1, 1)
        colregs_weights = csd.MX.sym("colregs_weights", 3, 1)

        p_adjustable.append(alpha_cr)
        p_adjustable.append(y_0_cr)
        p_adjustable.append(alpha_ho)
        p_adjustable.append(x_0_ho)
        p_adjustable.append(alpha_ot)
        p_adjustable.append(x_0_ot)
        p_adjustable.append(y_0_ot)
        p_adjustable.append(d_attenuation)
        p_adjustable.append(colregs_weights)

        max_num_so_constr = self._params.max_num_so_constr

        # Static obstacle constraint parameters
        so_pars = csd.MX.sym("so_pars", 0)
        A_so_constr = csd.MX.sym("A_so_constr", 0)
        b_so_constr = csd.MX.sym("b_so_constr", 0)
        so_surfaces = []
        max_num_so_constr = self._params.max_num_so_constr
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
            so_surfaces = mapf.compute_surface_approximations_from_polygons(
                so_list, enc, safety_margins=[self._params.r_safe_so], map_origin=self._map_origin
            )[0]
            max_num_so_constr = min(len(so_surfaces), self._params.max_num_so_constr)
            self._so_surfaces = so_surfaces
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

        # Slack weighting matrix W (dim = 1 x (max_num_so_constr + 3 * self._params.max_num_do_constr_per_zone))
        W_bx = csd.MX.sym("W_bx", nx, 1)
        W_so = csd.MX.sym("W_so", max_num_so_constr, 1)
        W_do = csd.MX.sym("W_do", n_colregs_zones * self._params.max_num_do_constr_per_zone, 1)
        p_fixed.append(W_bx)
        p_fixed.append(W_so)
        p_fixed.append(W_do)

        # Safety zone parameters
        r_safe_so = csd.MX.sym("r_safe_so", 1)
        r_safe_do = csd.MX.sym("r_safe_do", 1)
        p_fixed.append(r_safe_so)
        p_adjustable.append(r_safe_do)

        ship_vertices = self._model.params().ship_vertices

        # Dynamic obstacle augmented state parameters (x, y, chi, U, length, width) * N + 1 for colregs situations GW, HO and OT
        nx_do = 6
        X_do_cr = csd.MX.sym("X_do_cr", nx_do * self._params.max_num_do_constr_per_zone, N + 1)
        X_do_ho = csd.MX.sym("X_do_ho", nx_do * self._params.max_num_do_constr_per_zone, N + 1)
        X_do_ot = csd.MX.sym("X_do_ot", nx_do * self._params.max_num_do_constr_per_zone, N + 1)

        p_fixed.append(csd.reshape(X_do_cr, -1, 1))
        p_fixed.append(csd.reshape(X_do_ho, -1, 1))
        p_fixed.append(csd.reshape(X_do_ot, -1, 1))

        self._p_path = csd.vertcat(
            self._x_path_coeffs,
            self._y_path_coeffs,
            self._speed_spl_coeffs,
            csd.vertcat(*path_derivative_refs_list),
            Q_p_vec,
        )
        self._p_rate = csd.vertcat(
            alpha_app_course,
            alpha_app_speed,
            K_app_course,
            K_app_speed,
        )
        self._p_colregs = csd.vertcat(
            alpha_cr, y_0_cr, alpha_ho, x_0_ho, alpha_ot, x_0_ot, y_0_ot, d_attenuation, colregs_weights
        )

        # Cost function
        J = self._nlp_perturbation.T @ u_0

        so_constr_list = []
        do_constr_list = []

        # Create symbolic integrator for the shooting gap constraints and discretized cost function
        # stage_cost = csd.MX.sym("stage_cost", 1)
        erk4 = integrators.ERK4(x=x, p=csd.vertcat(u, self._p_mdl), ode=xdot, quad=csd.vertcat([]), h=dt)
        for k in range(N):
            u_k = csd.MX.sym("u_" + str(k), nu, 1)
            U.append(u_k)
            hu.append(lbu_k - u_k)  # lbu <= u_k
            hu.append(u_k - ubu_k)  # u_k <= ubu
            hs_bx.append(-sigma_bx_k)  # 0 <= sigma_bx_k
            hs_bx.append(sigma_bx_k - approx_inf)  # sigma_bx_k <= inf
            hx.append(lbx_k - x_k - sigma_bx_k)  # lbx - sigma_bx <= x_k
            hx.append(x_k - sigma_bx_k - ubx_k)  # x_k <= ubx + sigma_bx

            slack_penalty_cost = W_bx.T @ sigma_bx_k

            sigma_so_k = csd.MX.sym(
                "sigma_so_" + str(k),
                max_num_so_constr,
                1,
            )
            sigma_do_k = csd.MX.sym(
                "sigma_do_" + str(k),
                n_colregs_zones * self._params.max_num_do_constr_per_zone,
                1,
            )

            if max_num_so_constr > 0:
                Sigma_so.append(sigma_so_k)
                hs_so.append(-sigma_so_k)  # 0 <= sigma_so_k
                hs_so.append(sigma_so_k - approx_inf)  # sigma_so_k <= 1e12 = inf
                slack_penalty_cost += W_so.T @ sigma_so_k
            if self._params.max_num_do_constr_per_zone > 0:
                Sigma_do.append(sigma_do_k)
                hs_do.append(-sigma_do_k)  # 0 <= sigma_do_k
                hs_do.append(sigma_do_k - approx_inf)  # sigma_do_k <= 1e12 = inf
                slack_penalty_cost += W_do.T @ sigma_do_k

            x_path_k = self._x_path(x_k[4], self._x_path_coeffs)
            y_path_k = self._y_path(x_k[4], self._y_path_coeffs)
            s_dot_ref_k = path_derivative_refs_list[k]
            path_ref_k = csd.vertcat(x_path_k, y_path_k, s_dot_ref_k)

            # Sum stage cost
            path_following_cost, path_dev_cost, path_dot_dev_cost = mpc_common.path_following_cost_huber(
                x_k, path_ref_k, Q_p_vec
            )
            speed_dev_cost = Q_p_vec[2] * (x_k[3] - self._speed_spline(x_k[4], self._speed_spl_coeffs)) ** 2

            rate_cost, course_rate_cost, speed_rate_cost = mpc_common.rate_cost(
                u_k[0],
                u_k[1],
                csd.vertcat(alpha_app_course, alpha_app_speed),
                csd.vertcat(K_app_course, K_app_speed),
                r_max=ubu_k[0],
                a_max=ubu_k[1],
            )
            colregs_cost, crossing_cost, head_on_cost, overtaking_cost = mpc_common.colregs_cost(
                x_k,
                X_do_cr[:, k],
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
                d_attenuation,
                colregs_weights,
            )

            J += gamma**k * (
                path_following_cost
                + speed_dev_cost
                + rate_cost
                + slack_penalty_cost
                + u_k.T @ np.diag([0.0001, 0.0001, 0.0001]) @ u_k
            )
            if k == 0:
                self._path_dev_cost = csd.Function("path_dev_cost", [x_k, self._p_path], [path_dev_cost])
                self._path_dot_dev_cost = csd.Function("path_dot_dev_cost", [x_k, self._p_path], [path_dot_dev_cost])
                self._speed_dev_cost = csd.Function("speed_dev_cost", [x_k, self._p_path], [speed_dev_cost])
                self._course_rate_cost = csd.Function("course_rate_cost", [u_k, self._p_rate], [course_rate_cost])
                self._speed_rate_cost = csd.Function("speed_rate_cost", [u_k, self._p_rate], [speed_rate_cost])
                self._crossing_cost = csd.Function("crossing_cost", [x_k, X_do_cr, self._p_colregs], [crossing_cost])
                self._head_on_cost = csd.Function("head_on_cost", [x_k, X_do_ho, self._p_colregs], [head_on_cost])
                self._overtaking_cost = csd.Function(
                    "overtaking_cost", [x_k, X_do_ot, self._p_colregs], [overtaking_cost]
                )
                self._slack_penalty_cost = csd.Function(
                    "slack_penalty_cost", [sigma_bx_k, sigma_so_k, sigma_do_k], [slack_penalty_cost]
                )

            so_constr_k = self._create_static_obstacle_constraint(
                x_k, sigma_so_k, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, r_safe_so
            )
            so_constr_list.extend(so_constr_k)
            g_ineq_list.extend(so_constr_k)

            X_do_k = csd.vertcat(X_do_cr[:, k], X_do_ho[:, k], X_do_ot[:, k])
            do_constr_k = self._create_dynamic_obstacle_constraint(x_k, sigma_do_k, X_do_k, nx_do, r_safe_do)
            do_constr_list.extend(do_constr_k)
            g_ineq_list.extend(do_constr_k)

            x_k_end, _, _, _, _, _, _, _ = erk4(x_k, csd.vertcat(u_k, self._p_mdl))
            x_k = csd.MX.sym("x_" + str(k + 1), nx, 1)
            sigma_bx_k = csd.MX.sym(
                "sigma_bx_" + str(k + 1),
                ubx_k.shape[0],
                1,
            )
            Sigma_bx.append(sigma_bx_k)
            X.append(x_k)
            g_eq_list.append(x_k_end - x_k)

        # Terminal costs and constraints
        hs_bx.append(-sigma_bx_k)  # 0 <= sigma_bx_k
        hs_bx.append(sigma_bx_k - approx_inf)  # sigma_bx_k <= 1e12 = inf
        hx.append(lbx_k - x_k - sigma_bx_k)  # lbx - sigma_bx <= x_k
        hx.append(x_k - sigma_bx_k - ubx_k)  # x_k <= ubx + sigma_bx
        slack_penalty_cost = W_bx.T @ sigma_bx_k
        if max_num_so_constr > 0:
            sigma_so_k = csd.MX.sym(
                "sigma_so_" + str(N),
                max_num_so_constr,
                1,
            )
            Sigma_so.append(sigma_so_k)
            hs_so.append(-sigma_so_k)  # 0 <= sigma_so_k
            hs_so.append(sigma_so_k - approx_inf)  # sigma_so_k <= inf
            slack_penalty_cost += W_so.T @ sigma_so_k
        if self._params.max_num_do_constr_per_zone > 0:
            sigma_do_k = csd.MX.sym(
                "sigma_do_" + str(N),
                n_colregs_zones * self._params.max_num_do_constr_per_zone,
                1,
            )
            Sigma_do.append(sigma_do_k)
            hs_do.append(-sigma_do_k)  # 0 <= sigma_do_k
            hs_do.append(sigma_do_k - approx_inf)  # sigma_do_k <= inf
            slack_penalty_cost += W_do.T @ sigma_do_k

        x_path_k = self._x_path(x_k[4], self._x_path_coeffs)
        y_path_k = self._y_path(x_k[4], self._y_path_coeffs)
        s_dot_ref_k = path_derivative_refs_list[-1]
        path_ref_k = csd.vertcat(x_path_k, y_path_k, s_dot_ref_k)
        path_following_cost, _, _ = mpc_common.path_following_cost_huber(x_k, path_ref_k, Q_p_vec)
        speed_dev_cost = Q_p_vec[2] * (x_k[3] - self._speed_spline(x_k[4], self._speed_spl_coeffs)) ** 2

        J += gamma**N * (path_following_cost + speed_dev_cost + slack_penalty_cost)

        so_constr_N = self._create_static_obstacle_constraint(
            x_k, sigma_so_k, so_pars, A_so_constr, b_so_constr, so_surfaces, ship_vertices, r_safe_so
        )
        so_constr_list.extend(so_constr_N)
        g_ineq_list.extend(so_constr_N)

        X_do_N = csd.vertcat(X_do_cr[:, N], X_do_ho[:, N], X_do_ot[:, N])
        do_constr_N = self._create_dynamic_obstacle_constraint(x_k, sigma_do_k, X_do_N, nx_do, r_safe_do)
        do_constr_list.extend(do_constr_N)
        g_ineq_list.extend(do_constr_N)

        hs = hs_bx + hs_so + hs_do

        g_ineq_list = [*hu, *hx, *hs, *g_ineq_list]

        # Vectorize and finalize the NLP
        g_eq = csd.vertcat(*g_eq_list)
        g_ineq = csd.vertcat(*g_ineq_list)
        Sigma = Sigma_bx + Sigma_so + Sigma_do

        lbg_eq = [-1e-09] * g_eq.shape[0]
        ubg_eq = [1e-09] * g_eq.shape[0]
        lbg_ineq = [-np.inf] * g_ineq.shape[0]
        ubg_ineq = [1e-09] * g_ineq.shape[0]
        self._lbg = np.concatenate((lbg_eq, lbg_ineq), axis=0)
        self._ubg = np.concatenate((ubg_eq, ubg_ineq), axis=0)

        self._p_fixed = csd.vertcat(*p_fixed)
        self._p_adjustable = csd.vertcat(*p_adjustable)
        self._p = csd.vertcat(*p_adjustable, *p_fixed)
        self._p_fixed_list = p_fixed
        self._p_adjustable_list = p_adjustable
        self._p_list = p_adjustable + p_fixed

        self._p_fixed_values = np.zeros((self._p_fixed.shape[0], 1))
        self._p_adjustable_values = self.get_adjustable_params()

        self._opt_vars = csd.vertcat(*U, *X, *Sigma)
        self._opt_vars_list = U + X + Sigma
        ns = nx + max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone

        self._cost_function = J
        self._g_eq = g_eq
        self._g_ineq = g_ineq
        g = csd.vertcat(g_eq, g_ineq)

        # Create IPOPT solver object
        nlp_prob = {
            "f": J,
            "x": self._opt_vars,
            "p": self._p,
            "g": g,
        }
        self._solver = csd.nlpsol("solver", "ipopt", nlp_prob, self._solver_options.to_opt_settings())

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
        w_sub = csd.vertcat(*U, *X)
        self._equality_constraints = csd.Function(
            "equality_constraints", [w_sub, self._p], [g_eq], ["w_sub", "p"], ["g_eq"]
        )
        self._equality_constraints_jacobian = csd.Function(
            "equality_constraints_jacobian",
            [w_sub, self._p],
            [csd.jacobian(g_eq, w_sub)],
            ["w_sub", "p"],
            ["dg_dw"],
        )
        self._inequality_constraints = csd.Function(
            "inequality_constraints", [self._opt_vars, self._p], [g_ineq], ["w", "p"], ["g_ineq"]
        )
        self._inequality_constraints_jacobian = csd.Function(
            "inequality_constraints_jacobian",
            [self._opt_vars, self._p],
            [csd.jacobian(g_ineq, self._opt_vars)],
            ["w", "p"],
            ["dg_dw"],
        )
        self._box_inequality_constraints = csd.Function(
            "box_inequality_constraints", [self._opt_vars, self._p], [csd.vertcat(*hu, *hx, *hs)], ["w", "p"], ["g"]
        )
        self._decision_trajectories = csd.Function(
            "decision_trajectories",
            [self._opt_vars],
            [
                csd.reshape(csd.vertcat(*U), nu, -1),
                csd.reshape(csd.vertcat(*X), nx, -1),
                csd.reshape(csd.vertcat(*Sigma), ns, -1),
            ],
            ["w"],
            ["U", "X", "Sigma"],
        )
        self._decision_variables = csd.Function(
            "decision_variables",
            [
                csd.reshape(csd.vertcat(*U), nu, -1),
                csd.reshape(csd.vertcat(*X), nx, -1),
                csd.reshape(csd.vertcat(*Sigma), ns, -1),
            ],
            [self._opt_vars],
            ["U", "X", "Sigma"],
            ["w"],
        )

        Hu = csd.vertcat(*hu)
        Hx = csd.vertcat(*hx)
        Hs = csd.vertcat(*hs)
        G = g_eq
        H = csd.vertcat(Hu, Hx, Hs, g_ineq)
        self._nlp_eq = G
        self._nlp_ineq = H
        self._nlp_hess_lag = self._solver.get_function("nlp_hess_l")
        # self.build_sensitivities(tau=self._solver_options.mu_target)

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
                for j in range(n_so):
                    so_constr_list.append(so_surfaces[j](x_k[0:2].reshape((1, 2))) - sigma_k[j])
                    # vertices = mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * r_safe_so + x_k[0:2]
                    # vertices = vertices.reshape((-1, 2))
                    # for i in range(vertices.shape[0]):
                    #     so_constr_list.append(csd.vec(so_surfaces[j](vertices[i, :]) - sigma_k[j]))
        return so_constr_list

    def _create_dynamic_obstacle_constraint(
        self, x_k: csd.MX, sigma_k: csd.MX, X_do_k: csd.MX, nx_do: int, r_safe_do: csd.MX
    ) -> list:
        """Creates the dynamic obstacle constraints for the NLP at the current stage.

        Args:
            x_k (csd.MX): State vector at the current stage in the OCP.
            sigma_k (csd.MX): Sigma vector at the current stage in the OCP for dynamic obstacle constraints.
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
            p_diff_do_frame = Rchi_do_i.T @ (x_k[0:2] - x_do_i[0:2])
            weights = hf.casadi_matrix_from_nested_list(
                [[1.0 / (0.5 * l_do_i + r_safe_do) ** 2, 0.0], [0.0, 1.0 / (0.5 * w_do_i + r_safe_do) ** 2]]
            )
            # do_constr_list.append(
            #     1.0 - sigma_k[i] - p_diff_do_frame.T @ weights @ p_diff_do_frame
            # )
            do_constr_list.append(
                csd.log(1 - sigma_k[i] + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon)
            )
        return do_constr_list

    def create_parameter_values(
        self,
        warm_start: dict,
        state: np.ndarray,
        action: Optional[np.ndarray],
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        so_list: list,
        enc: Optional[senc.ENC] = None,
        perturb_nlp: bool = False,
        perturb_sigma: float = 0.001,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - warm_start (dict): Dictionary containing the warm start values for the decision variables.
            - state (np.ndarray): Current state of the system on the form (x, y, psi, u, v, r)^T.
            - action (np.ndarray, optional): Current action of the system on the form (r, a)^T.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - so_list (list): List of static obstacles.
            - enc (Optional[senc.ENC]): Electronic Navigation Chart (ENC) object.
            - perturb_nlp (bool, optional): Whether to perturb the NLP problem. Defaults to False.
            - perturb_sigma (float, optional): Standard deviation of the perturbation. Defaults to 0.001.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of parameter vector values, and arrays of dynamic obstacle parameter values for each colregs zone.
        """
        assert len(state) == 6, "State must be of length 6."
        n_colregs_zones = 3
        nx, nu = self._model.dims()

        adjustable_parameter_values = self.get_adjustable_params()
        fixed_parameter_values: list = []

        d = np.zeros((nu, 1))
        if perturb_nlp:
            d = np.random.normal(0.0, perturb_sigma, size=(nu, 1))
        fixed_parameter_values.extend(d.flatten().tolist())  # d

        state_aug = np.zeros((nx, 1))
        U = np.sqrt(state[3] ** 2 + state[4] ** 2)
        state_aug[0:2] = state[0:2].reshape((2, 1))
        state_aug[2] = state[2]
        state_aug[3] = U

        if action is None:
            action = np.zeros((nu, 1))

        # augmented path dynamics due to the integrated path timing model with velocity assignment
        state_aug[4:] = np.array([self._s, self._s_dot]).reshape((2, 1))
        fixed_parameter_values.extend(state_aug.flatten().tolist())  # x0
        fixed_parameter_values.extend(action.flatten().tolist())  # u0

        path_parameter_values, _, _ = self._create_path_parameter_values()
        fixed_parameter_values.extend(path_parameter_values)

        fixed_parameter_values.append(self._params.gamma)

        so_parameter_values = self._update_so_parameter_values(so_list, state, enc, **kwargs)
        fixed_parameter_values.extend(so_parameter_values)

        max_num_so_constr = min(len(self._so_surfaces), self._params.max_num_so_constr)
        slack_size = nx + max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone
        W = self._params.w_L1 * np.ones(slack_size)
        fixed_parameter_values.extend(W.tolist())
        fixed_parameter_values.append(self._params.r_safe_so)

        do_parameter_values_cr = self._create_do_parameter_values(state, do_cr_list, **kwargs)
        do_parameter_values_ho = self._create_do_parameter_values(state, do_ho_list, **kwargs)
        do_parameter_values_ot = self._create_do_parameter_values(state, do_ot_list, **kwargs)
        fixed_parameter_values.extend(do_parameter_values_cr)
        fixed_parameter_values.extend(do_parameter_values_ho)
        fixed_parameter_values.extend(do_parameter_values_ot)

        return (
            np.concatenate((adjustable_parameter_values, np.array(fixed_parameter_values)), axis=0),
            np.array(do_parameter_values_cr),
            np.array(do_parameter_values_ho),
            np.array(do_parameter_values_ot),
        )

    def _create_path_parameter_values(self) -> Tuple[list, np.ndarray, np.ndarray]:
        """Creates the parameter values for the path constraints.

        Returns:
            Tuple[list, np.ndarray, np.ndarray]: List of path parameters to be used as input to solver, and the path der var and path var references
        """
        path_parameter_values = []
        path_parameter_values.extend(self._x_path_coeffs_values.tolist())
        path_parameter_values.extend(self._y_path_coeffs_values.tolist())
        path_parameter_values.extend(self._speed_spl_coeffs_values.tolist())

        N = int(self._params.T / self._params.dt)
        path_derivative_refs = np.zeros(N + 1)
        path_variable_refs = np.zeros(N + 1)
        s_ref_k = self._s
        dt = self._params.dt
        for k in range(N + 1):
            s_dot_ref_k = self.compute_path_variable_derivative(s_ref_k)
            # s_dot_ref_k = np.clip(s_dot_ref_k, 0.0, self._model.params().s_dot_max)
            path_derivative_refs[k] = s_dot_ref_k
            path_variable_refs[k] = s_ref_k
            k1 = self.compute_path_variable_derivative(s_ref_k)
            k2 = self.compute_path_variable_derivative(s_ref_k + 0.5 * dt * k1)
            k3 = self.compute_path_variable_derivative(s_ref_k + 0.5 * dt * k2)
            k4 = self.compute_path_variable_derivative(s_ref_k + dt * k3)
            s_ref_k = s_ref_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # s_ref_k = np.clip(s_ref_k, 0.0, self._s_final_value)

        path_parameter_values.extend(path_derivative_refs.tolist())
        return path_parameter_values, path_derivative_refs, path_variable_refs

    def _create_do_parameter_values(self, state: np.ndarray, do_list: list) -> list:
        """Creates the parameter values for the dynamic obstacle constraints.

        Args:
            state (np.ndarray): Current state of the system on the form (x, y, psi, u, v, r)^T.
            do_list (list): List of dynamic obstacles in a colregs zone.

        Returns:
            list: List of dynamic obstacle parameters to be used as input to solver
        """
        csog_state = np.array([state[0], state[1], state[2], np.sqrt(state[3] ** 2 + state[4] ** 2)])
        N = int(self._params.T / self._params.dt)
        do_parameter_values = []
        n_do = len(do_list)
        for k in range(N + 1):
            t = k * self._params.dt
            for i in range(self._params.max_num_do_constr_per_zone):
                if i < n_do:
                    (ID, do_state, cov, length, width) = do_list[i]
                    chi = np.arctan2(do_state[3], do_state[2])
                    U = np.sqrt(do_state[2] ** 2 + do_state[3] ** 2)
                    do_parameter_values.extend(
                        [
                            do_state[0] + t * U * np.cos(chi),
                            do_state[1] + t * U * np.sin(chi),
                            chi,
                            U,
                            length,
                            width,
                        ]
                    )
                else:
                    do_parameter_values.extend([csog_state[0] - 1e10, csog_state[1] - 1e10, 0.0, 0.0, 10.0, 2.0])
        return do_parameter_values

    def _create_fixed_so_parameter_values(
        self, so_list: list, state: np.ndarray, enc: Optional[senc.ENC] = None
    ) -> np.ndarray:
        """Creates the fixed parameter values for the static obstacle constraints.

        Args:
            - so_list (list): List of static obstacles.
            - state (np.ndarray): Current state of the system on the form (x, y, psi, u, v, r)^T.
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
            - state (np.ndarray): Current state of the system on the form (x, y, psi, u, v, r)^T.
            - enc (senc.ENC): Electronic Navigation Chart (ENC) object.

        Returns:
            np.ndarray: Fixed parameter vector for static obstacles to be used as input to solver
        """
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            assert len(so_list) == 2, "Approximate convex safe set constraint requires constraint variables A and b"
            A, b = so_list[0], so_list[1]
            self._p_fixed_so_values = np.concatenate((A.flatten(), b.flatten()), axis=0).tolist()
        return self._p_fixed_so_values

    def build_sensitivities(self, tau: float = 0.01) -> mpc_common.NLPSensitivities:
        """Builds the sensitivity of the KKT matrix function with respect to the decision variables and parameters.

        Args:
            - tau (float): Barrier parameter for the primal-dual interior point method formulation.

        Returns:
            - mpc_common.NLPSensitivities: Class containing the sensitivity functions necessary for
                computing the score function  gradient in RL context, and more.
        """
        output_dict = {}
        G = self._nlp_eq
        H = self._nlp_ineq
        lamb = csd.MX.sym("lambda", G.shape[0])
        mu = csd.MX.sym("mu", H.shape[0])
        multipliers = csd.vertcat(lamb, mu)

        lag = self._cost_function + csd.transpose(lamb) @ G + csd.transpose(mu) @ H
        lag_func = csd.Function(
            "lagrangian",
            [self._opt_vars, multipliers, self._p_fixed, self._p_adjustable],
            [lag],
        )
        dlag_func = lag_func.factory(
            "lagrangian_derivative_func",
            ["i0", "i1", "i2", "i3"],
            ["jac:o0:i0", "jac:o0:i2", "jac:o0:i3"],
        )

        # z contains all variables contained in a solution, i.e. decision variables and multipliers
        _, nu = self._model.dims()
        z = csd.vertcat(self._opt_vars, lamb, mu)

        d2lag_d2w_func = self._solver.get_function("nlp_hess_l")

        # Compute the lagrangian sensitivities wrt decision variables and parameters
        dlag_dw, dlag_dp_f, dlag_dp = dlag_func(self._opt_vars, multipliers, self._p_fixed, self._p_adjustable)
        dlag_dw_func = csd.Function(
            "dlag_dw_func",
            [self._opt_vars, multipliers, self._p_fixed, self._p_adjustable],
            [dlag_dw],
            ["w", "multipliers", "p_fixed", "p_adjustable"],
            ["dlag_dw"],
        )
        dlag_dp_f_func = csd.Function(
            "dlag_dp_f_func",
            [self._opt_vars, multipliers, self._p_fixed, self._p_adjustable],
            [dlag_dp_f],
            ["w", "multipliers", "p_fixed", "p_adjustable"],
            ["dlag_dp_f"],
        )
        dlag_dp_func = csd.Function(
            "dlag_dp_func",
            [self._opt_vars, multipliers, self._p_fixed, self._p_adjustable],
            [dlag_dp],
            ["w", "multipliers", "p_fixed", "p_adjustable"],
            ["dlag_dp"],
        )

        output_dict.update(
            {"dlag_dw": dlag_dw_func, "dlag_dp_f": dlag_dp_f_func, "dlag_dp": dlag_dp_func, "d2lag_d2w": d2lag_d2w_func}
        )

        # Build KKT matrix
        R_kkt = csd.vertcat(
            csd.transpose(dlag_dw),
            G,
            csd.diag(mu) * H + tau,
        )

        # # Generate sensitivity of the KKT matrix
        r_func = csd.Function("kkt_matrix_func", [z, self._p_fixed, self._p_adjustable], [R_kkt])
        dr_func = r_func.factory(
            "kkt_matrix_derivative_func", ["i0", "i1", "i2"], ["jac:o0:i0", "jac:o0:i1", "jac:o0:i2"]
        )
        [dr_dz, dr_dp_f, dr_dp] = dr_func(z, self._p_fixed, self._p_adjustable)
        dR_dz_func = csd.Function("dR_dz_func", [z, self._p_fixed, self._p_adjustable], [dr_dz])
        dr_dp_func = csd.Function("dr_dp_func", [z, self._p_fixed, self._p_adjustable], [dr_dp])
        dr_dp_f_func = csd.Function("dr_dp_f_func", [z, self._p_fixed, self._p_adjustable], [dr_dp_f])
        output_dict.update({"dr_dz": dR_dz_func, "dr_dp": dr_dp_func, "dr_dp_f": dr_dp_f_func})

        # Generate sensitivity of the optimal solution using the implicit function theorem, with respect to the parameters and
        # the perturbation (if using stochastic policies)
        # dz_dp = -csd.inv(dr_dz) @ dr_dp
        # dz_dp_func = csd.Function("dz_dp_func", [z, self._p_fixed, self._p_adjustable], [dz_dp])
        # dz_dp_f = -csd.inv(dr_dz) @ dr_dp_f
        # dz_dp_f_func = csd.Function("dz_dp_f_func", [z, self._p_fixed, self._p_adjustable], [dz_dp_f])

        # Generate sensitivity of z_bar (contains perturbation as first element, with u0 as a fixed parameter)
        # with respect to the parameters and the perturbation (if using stochastic policies)

        z_bar_list = [self._nlp_perturbation] + self._opt_vars_list[1:] + [lamb, mu]
        z_bar = csd.vertcat(*z_bar_list)
        p_fixed_bar_list = [self._opt_vars_list[0]] + self._p_fixed_list[1:]
        p_fixed_bar = csd.vertcat(*p_fixed_bar_list)
        r_bar_func = csd.Function("kkt_matrix_bar_func", [z_bar, p_fixed_bar, self._p_adjustable], [R_kkt])
        dr_bar_func: csd.Function = r_bar_func.factory(
            "kkt_matrix_bar_derivative_func", ["i0", "i1", "i2"], ["jac:o0:i0", "jac:o0:i1", "jac:o0:i2"]
        )
        [dr_dz_bar, dr_dp_f, dr_dp] = dr_bar_func(z_bar, p_fixed_bar, self._p_adjustable)
        dr_dz_bar_func = csd.Function(
            "dr_dz_bar_func",
            [z_bar, p_fixed_bar, self._p_adjustable],
            [dr_dz_bar],
            ["z_bar", "p_fixed_bar", "p_adjustable"],
            ["dr_dz_bar"],
        )
        dr_dp_bar_func = csd.Function(
            "dr_dp_func",
            [z_bar, p_fixed_bar, self._p_adjustable],
            [dr_dp],
            ["z_bar", "p_fixed_bar", "p_adjustable"],
            ["dr_dp_bar"],
        )
        dr_dp_f_bar_func = csd.Function(
            "dr_dp_f_func",
            [z_bar, p_fixed_bar, self._p_adjustable],
            [dr_dp_f],
            ["z_bar", "p_fixed_bar", "p_adjustable"],
            ["dr_dp_f_bar"],
        )

        output_dict.update({"dr_dz_bar": dr_dz_bar_func, "dr_dp_bar": dr_dp_bar_func, "dr_dp_f_bar": dr_dp_f_bar_func})

        # avoid usage of casadi inv
        # dr_dz_bar_inv = csd.inv(dr_dz_bar)
        # dz_bar_dp = -dr_dz_bar_inv @ dr_dp_bar
        # dz_bar_dp_f = -dr_dz_bar_inv @ dr_dp_f_bar

        # dz_bar_da = dz_bar_dp_f[0:nu]
        # dz_bar_da_func = csd.Function("dz_bar_da_func", [z_bar, p_fixed_bar, self._p_adjustable], [dz_bar_da])

        # Create second order sensitivity functions necessary for the constrained (MPC-based) stochastic policy
        n_p = len(self._p_adjustable_list)
        d2r_dp_da_list = []
        d2r_dp_dz_bar_list = []
        dr_da = dr_dp_f[0:nu]  # a is action
        for idx_p in range(n_p):
            d2r_dpi_dz_bar = csd.jacobian(dr_dz_bar, self._p_adjustable_list[idx_p])
            d2r_dpi_dz_bar_func = csd.Function(
                "d2R_dp" + str(idx_p) + "_dz_bar_func",
                [z_bar, p_fixed_bar, self._p_adjustable],
                [d2r_dpi_dz_bar],
                ["z_bar", "p_fixed_bar", "p_adjustable"],
                ["d2R_dp" + str(idx_p) + "_dz_bar"],
            )
            d2r_dp_dz_bar_list.append(d2r_dpi_dz_bar_func)

            d2r_dpi_da = csd.jacobian(dr_da, self._p_adjustable_list[idx_p])
            d2r_dpi_da_func = csd.Function(
                "d2R_dp" + str(idx_p) + "_da_func",
                [z_bar, p_fixed_bar, self._p_adjustable],
                [d2r_dpi_da],
                ["z_bar", "p_fixed_bar", "p_adjustable"],
                ["d2R_dp" + str(idx_p) + "_da"],
            )
            d2r_dp_da_list.append(d2r_dpi_da_func)

        n_z_bar = len(z_bar_list)
        d2r_dzdz_j_bar_list = []
        d2r_dadz_j_bar_list = []
        for j in range(n_z_bar):
            d2r_dzdz_j = csd.jacobian(dr_dz_bar, z_bar_list[j])
            d2r_dzdz_j_func = csd.Function(
                "d2r_dzdz_" + str(j) + "_func",
                [z_bar, p_fixed_bar, self._p_adjustable],
                [d2r_dzdz_j],
                ["z_bar", "p_fixed_bar", "p_adjustable"],
                ["d2r_dzdz_" + str(j)],
            )
            d2r_dzdz_j_bar_list.append(d2r_dzdz_j_func)

            d2r_dadz_j = csd.jacobian(dr_da, z_bar_list[j])
            d2r_dadz_j_func = csd.Function(
                "d2r_dadz_" + str(j) + "_func",
                [z_bar, p_fixed_bar, self._p_adjustable],
                [d2r_dadz_j],
                ["z_bar", "p_fixed_bar", "p_adjustable"],
                ["d2r_dadz_" + str(j)],
            )
            d2r_dadz_j_bar_list.append(d2r_dadz_j_func)

        output_dict.update(
            {
                "d2r_dp_da": d2r_dp_da_list,
                "d2r_dp_dz_bar": d2r_dp_dz_bar_list,
                "d2r_dzdz_j_bar": d2r_dzdz_j_bar_list,
                "d2r_dadz_j_bar": d2r_dadz_j_bar_list,
            }
        )

        # for idx_p in range(n_p):
        #     first_sum = 0.0
        #     second_sum = 0.0
        #     for j in range(n_z_bar):
        #         z_bar_j = z_bar[j]
        #         d2r_dzdz_j = csd.jacobian(dr_dz_bar, z_bar_j)
        #         dz_bar_j_dpi = dz_bar_dp[j, idx_p]
        #         first_sum += d2r_dzdz_j @ dz_bar_j_dpi

        #         d2R_dadz_j = csd.jacobian(dr_da, z_bar_j)
        #         second_sum += d2R_dadz_j @ dz_bar_j_dpi

        #     b = -(d2r_dpi_dz_bar + first_sum) @ dz_bar_da - d2r_dpi_da - second_sum
        #     b_list.append(b)
        #     A_list.append(A)

        # big_b = csd.vertcat(*b_list)
        # big_A = csd.diagcat(*A_list)
        # dz_bar_dpi_da = csd.solve(big_A, big_b)
        # dz_bar_dpi_da_func = csd.Function(
        #     "dz_bar_dpi_da_func", [z_bar, p_fixed_bar, self._p_adjustable], [dz_bar_dpi_da]
        # )
        # dz_bar_dp_da.append(dz_bar_dpi_da_func)
        sensitivities = mpc_common.NLPSensitivities.from_dict(output_dict)
        if tau == self._solver_options.mu_target:
            self._sensitivities = sensitivities
        return sensitivities

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
        path_dot_dev_cost = np.zeros(X.shape[1])
        speed_dev_cost = np.zeros(X.shape[1])
        course_rate_cost = np.zeros(X.shape[1])
        speed_rate_cost = np.zeros(X.shape[1])
        crossing_cost = np.zeros(X.shape[1])
        head_on_cost = np.zeros(X.shape[1])
        overtaking_cost = np.zeros(X.shape[1])
        colregs_cost = np.zeros(X.shape[1])
        nx_do = 6
        _, path_var_derivative_refs, _ = self._create_path_parameter_values()
        p_path_values = np.array(
            [
                *self._x_path_coeffs_values.tolist(),
                *self._y_path_coeffs_values.tolist(),
                *self._speed_spl_coeffs_values.tolist(),
                *path_var_derivative_refs.tolist(),
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
                self._params.d_attenuation,
                *self._params.w_colregs.tolist(),
            ]
        )

        nx_do_total = nx_do * self._params.max_num_do_constr_per_zone
        X_do_cr = np.zeros((nx_do_total, X.shape[1]))
        X_do_ho = np.zeros((nx_do_total, X.shape[1]))
        X_do_ot = np.zeros((nx_do_total, X.shape[1]))
        for k in range(X.shape[1]):
            X_do_cr_k = do_cr_parameters[k * nx_do_total : (k + 1) * nx_do_total]
            X_do_ho_k = do_ho_parameters[k * nx_do_total : (k + 1) * nx_do_total]
            X_do_ot_k = do_ot_parameters[k * nx_do_total : (k + 1) * nx_do_total]
            X_do_cr[:, k] = X_do_cr_k
            X_do_ho[:, k] = X_do_ho_k
            X_do_ot[:, k] = X_do_ot_k

        for k in range(X.shape[1]):
            path_dev_cost[k] = self._path_dev_cost(X[:, k], p_path_values)
            path_dot_dev_cost[k] = self._path_dot_dev_cost(X[:, k], p_path_values)
            speed_dev_cost[k] = self._speed_dev_cost(X[:, k], p_path_values)
            if nx_do_total > 0:
                crossing_cost[k] = self._crossing_cost(X[:, k], X_do_cr, p_colregs_values)
                head_on_cost[k] = self._head_on_cost(X[:, k], X_do_ho, p_colregs_values)
                overtaking_cost[k] = self._overtaking_cost(X[:, k], X_do_ot, p_colregs_values)
                colregs_cost[k] = crossing_cost[k] + head_on_cost[k] + overtaking_cost[k]

        for k in range(U.shape[1]):
            course_rate_cost[k] = self._course_rate_cost(U[:, k], p_rate_values)
            speed_rate_cost[k] = self._speed_rate_cost(U[:, k], p_rate_values)

        fig = plt.figure(figsize=(12, 8))
        axes = fig.subplot_mosaic(
            [
                ["p(s_k) - p_k"],
                ["s_dot_cost"],
                ["speed_dev_cost"],
                ["chi_rate_cost"],
                ["U_rate_cost"],
                ["colregs_cost"],
            ]
        )
        axes["p(s_k) - p_k"].plot(path_dev_cost, label=r"$p(s_k) - p_k$ cost")
        axes["p(s_k) - p_k"].set_ylabel("cost")
        axes["p(s_k) - p_k"].set_xlabel("k")
        axes["p(s_k) - p_k"].legend()

        axes["s_dot_cost"].plot(path_dot_dev_cost, label=r"$\dot{s}_{ref} - \dot{s}$ cost")
        axes["s_dot_cost"].set_ylabel("cost")
        axes["s_dot_cost"].set_xlabel("k")
        axes["s_dot_cost"].legend()

        axes["speed_dev_cost"].plot(speed_dev_cost, label=r"$U_{ref}(\omega) - U$ cost")
        axes["speed_dev_cost"].set_ylabel("cost")
        axes["speed_dev_cost"].set_xlabel("k")
        axes["speed_dev_cost"].legend()

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

    def plot_solution_trajectory(self, X: np.ndarray, U: np.ndarray, Sigma: np.ndarray) -> None:
        """Plots the solution trajectory.

        Args:
            X (np.ndarray): State trajectory.
            U (np.ndarray): Input trajectory.
            Sigma (np.ndarray): Slack trajectory.
        """
        speed_refs = np.zeros(X.shape[1])
        mpc_speed_refs = np.zeros(X.shape[1])
        x_path = np.zeros(X.shape[1])
        y_path = np.zeros(X.shape[1])
        _, path_var_derivative_refs, path_var_refs = self._create_path_parameter_values()
        for k in range(X.shape[1]):
            x_path[k] = self._x_path(X[4, k], self._x_path_coeffs_values)
            y_path[k] = self._y_path(X[4, k], self._y_path_coeffs_values)
            x_dot_path = self._x_dot_path(X[4, k], self._x_dot_path_coeffs_values)
            y_dot_path = self._y_dot_path(X[4, k], self._y_dot_path_coeffs_values)
            speed_refs[k] = self._speed_spline(path_var_refs[k], self._speed_spl_coeffs_values)
            mpc_speed_refs[k] = self._speed_spline(X[4, k], self._speed_spl_coeffs_values)

        fig = plt.figure(figsize=(12, 8))
        axes = fig.subplot_mosaic(
            [
                ["x", "y"],
                ["path_dot_variable", "path_variable"],
                ["course", "speed"],
                ["turn_rate", "acceleration"],
                ["path_input", ""],
            ]
        )
        axes["x"].plot(x_path - X[0, :], color="b", label=r"$p_x(s) - x$")
        axes["x"].set_ylabel("[m]")
        # axes["x"].set_xlabel("k")
        axes["x"].legend()

        axes["y"].plot(y_path - X[1, :], color="b", label=r"$p_y(s) - y$")
        axes["y"].set_ylabel("[m]")
        # axes["y"].set_xlabel("k")
        axes["y"].legend()

        # axes["path_dot_variable"].plot(
        #     path_var_derivative_refs - X[5, :], color="b", label=r"$\dot{s}_{ref} - \dot{s}$"
        # )
        axes["path_dot_variable"].plot(X[5, :], color="b", label=r"$\dot{s}$")
        axes["path_dot_variable"].plot(path_var_derivative_refs, color="r", label=r"$\dot{s}_{ref}$")
        axes["path_dot_variable"].set_ylabel("[m/s]")
        # axes["path_dot_variable"].set_xlabel("k")
        axes["path_dot_variable"].legend()

        axes["path_variable"].plot(X[4, :], color="b", label=r"$s$")
        axes["path_variable"].plot(path_var_refs, color="r", label=r"$s_{ref}$")
        axes["path_variable"].set_ylabel("[m]")
        # axes["path_variable"].set_xlabel("k")
        axes["path_variable"].legend()

        chi_diff = (
            np.unwrap(np.concatenate(([X[4, 0]], X[4, :])))[1:] - np.unwrap(np.concatenate(([X[2, 0]], X[2, :])))[1:]
        )
        # axes["course"].plot(np.rad2deg(chi_diff), color="b", label=r"$\chi_d - \chi$")a
        axes["course"].plot(np.rad2deg(X[2, :]), color="b", label=r"$\chi$")
        # axes["course"].plot(np.rad2deg(X[4, :]), color="r", label=r"$\chi_d$")
        axes["course"].set_ylabel("[deg]")
        # axes["course"].set_xlabel("k")
        axes["course"].legend()

        axes["turn_rate"].plot(np.rad2deg(U[0, :]), color="b", label=r"$r$")
        axes["turn_rate"].set_ylabel("[deg/s]")
        # axes["turn_rate"].set_xlabel("k")
        axes["turn_rate"].legend()

        axes["speed"].plot(X[3, :], color="b", label=r"$U$")
        axes["speed"].plot(mpc_speed_refs, color="r", label=r"$U_{d}(\omega)$")
        axes["speed"].plot(speed_refs, color="g", linestyle="--", label=r"$U_{ref}$")
        axes["speed"].set_ylabel("[m/s]")
        # axes["speed"].set_xlabel("k")
        axes["speed"].legend()

        axes["acceleration"].plot(U[1, :], color="b", label=r"$a$")
        axes["acceleration"].set_ylabel("[m/s2]")
        # axes["acceleration"].set_xlabel("k")
        axes["acceleration"].legend()

        axes["path_input"].plot(U[2, :], color="b", label="path input")
        axes["path_input"].set_ylabel("[m/s2]")
        axes["path_input"].set_xlabel("k")
        axes["path_input"].legend()
        fig.tight_layout()
        plt.show(block=False)

    def print_solution_info(self, soln: dict, parameter_values: np.ndarray, stats: dict, t_solve: float) -> None:
        """Prints information about the solution.

        Args:
            soln (dict): NLP solution dictionary.
            parameter_values (np.ndarray): Parameter values used in the NLP solver.
            stats (dict): NLP solver statistics.
            t_solve (float): NLP solver runtime.
        """
        U, X, Sigma = self._extract_trajectories(soln)
        lam_g = soln["lam_g"]
        cost_val = soln["f"].full()[0][0]

        w_sub = np.concatenate((U.T.flatten(), X.T.flatten()), axis=0)
        so_constr_vals = self._static_obstacle_constraints(soln["x"], parameter_values).full()
        do_constr_vals = self._dynamic_obstacle_constraints(soln["x"], parameter_values).full()
        g_ineq_vals = self._inequality_constraints(soln["x"], parameter_values).full()
        g_ineq_bx_vals = self._box_inequality_constraints(soln["x"], parameter_values).full()

        g_eq_jac = self._equality_constraints_jacobian(w_sub, parameter_values).full()
        g_eq_jac_rank = np.linalg.matrix_rank(g_eq_jac)
        max_g_eq_jac_rank = g_eq_jac.shape[1] if g_eq_jac.shape[1] < g_eq_jac.shape[0] else g_eq_jac.shape[0]
        g_ineq_jac_rank = 0
        max_ineq_jac_rank = 0
        if g_ineq_vals.size > 0:
            g_ineq_jac = self._inequality_constraints_jacobian(soln["x"], parameter_values).full()
            g_ineq_jac_rank = np.linalg.matrix_rank(g_ineq_jac)
            max_ineq_jac_rank = (
                g_ineq_jac.shape[1] if g_ineq_jac.shape[1] < g_ineq_jac.shape[0] else g_ineq_jac.shape[0]
            )

        nlp_hess = self._nlp_hess_lag(soln["x"], parameter_values, 1.0, lam_g).full()
        nlp_hess_rank = np.linalg.matrix_rank(nlp_hess)

        arg_max_box_constr = np.argmax(g_ineq_bx_vals)
        max_box_constr = np.max(g_ineq_bx_vals)

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
            f"Mid-level COLAV: \n\t- Runtime: {t_solve} \n\t- Cost: {cost_val} \n\t- Slacks (max, argmax): ({max_sigma}, {arg_max_sigma}) \n\t- Box constraints (max, argmax): ({max_box_constr, arg_max_box_constr}) \n\t- Static obstacle constraints (max, argmax): ({max_so_constr}, {arg_max_so_constr}) \n\t- Dynamic obstacle constraints (max, argmax): ({max_do_constr}, {arg_max_do_constr})\n\t- Equality constraints jac (max_rank, rank): {max_g_eq_jac_rank, g_eq_jac_rank} \n\t- Inequality constraints jac (max_rank, rank): {max_ineq_jac_rank, g_ineq_jac_rank}\n\t- Hessian (max_rank, rank): {nlp_hess.shape[0], nlp_hess_rank}"
        )
