"""
    acados_mpc.py

    Summary:
        Contains a class (impl in Acados) for a mid-level MPC-based COLAV planner.

    Author: Trym Tengesdal
"""

import copy
import pathlib
import shutil
import time
from typing import Dict, Tuple

import casadi as csd
import numpy as np
import rlmpc.common.helper_functions as hf
import rlmpc.common.map_functions as mapf
import rlmpc.common.math_functions as mf
import rlmpc.common.paths as rl_dp
import rlmpc.mpc.common as mpc_common
import rlmpc.mpc.models as models
import rlmpc.mpc.parameters as parameters
import scipy.interpolate as interp
import seacharts.enc as senc
import shapely.geometry as geo
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions
from acados_template.acados_ocp_solver import AcadosOcpSolver


class AcadosMPC:
    def __init__(
        self,
        model: models.MPCModel,
        params: parameters.MidlevelMPCParams,
        solver_options: AcadosOcpOptions,
        identifier: str = "",
    ) -> None:
        self._acados_ocp: AcadosOcp = AcadosOcp()
        self._solver_options = mpc_common.parse_acados_solver_options(solver_options)
        self._acados_ocp_solver: AcadosOcpSolver = None
        self._acados_ocp_solver_nonreg: AcadosOcpSolver = None

        self._prev_sol_status: int = 0
        self._num_consecutive_qp_failures: int = 0
        self.identifier: str = identifier

        self.model = copy.deepcopy(model)
        self._params: parameters.MidlevelMPCParams = copy.deepcopy(params)
        self._so_surfaces: list = []
        self._x_warm_start: np.ndarray = np.array([])
        self._u_warm_start: np.ndarray = np.array([])
        self._initialized = False
        self._map_origin: np.ndarray = np.array([0.0, 0.0])

        self._p_mdl: csd.MX = csd.MX.sym("p_mdl", 0)
        self._p_fixed: csd.MX = csd.MX.sym("p_fixed", 0)
        self._p_fixed_list: list[csd.MX] = []
        self._p_adjustable: csd.MX = csd.MX.sym("p_adjustable", 0)
        self._p_adjustable_list: list[csd.MX] = []
        self._p: csd.MX = csd.MX.sym("p", 0)
        self._p_list: list[csd.MX] = []
        self._adjustable_param_str_list: list[str] = self._params.adjustable_string_list()
        self._all_adjustable_param_str_list: list[str] = self._params.adjustable_string_list()
        self._fixed_param_str_list: list[str] = [
            name for name in self._all_adjustable_param_str_list if name not in self._adjustable_param_str_list
        ]
        self._parameter_values: list = []
        self.verbose: bool = True

        self._p_fixed_values: np.ndarray = np.array([])
        self._p_adjustable_values: np.ndarray = np.array([])
        self._nlp_perturbation: csd.MX = csd.MX.sym("nlp_perturbation", 0)

        self._min_depth: int = 1
        self._t_prev: float = 0.0
        self._xs_prev: np.ndarray = np.array([])
        self._prev_cost: float = np.inf
        self._X_do: list = []

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
        self._s: float = 0.001
        self._s_dot: float = 0.0
        self._s_final_value: float = 0.0
        self._path_linestring: geo.LineString = geo.LineString()

        self._s_refs: np.ndarray = np.array([])
        self._s_dot_refs: np.ndarray = np.array([])

        self._idx_slacked_bx_constr: np.ndarray = np.array([])
        self._action_indices = [0, 1, 2]

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
        self.decision_trajectories: csd.Function = csd.Function("decision_trajectories", [], [])
        self.decision_variables: csd.Function = csd.Function("decision_variables", [], [])
        self._static_obstacle_constraints: csd.Function = csd.Function("static_obstacle_constraints", [], [])
        self._dynamic_obstacle_constraints: csd.Function = csd.Function("dynamic_obstacle_constraints", [], [])
        self._equality_constraints: csd.Function = csd.Function("equality_constraints", [], [])
        self._equality_constraints_jacobian: csd.Function = csd.Function("equality_constraints_jacobian", [], [])
        self._state_box_inequality_constraints: csd.Function = csd.Function("state_box_inequality_constraints", [], [])
        self._input_box_inequality_constraints: csd.Function = csd.Function("input_box_inequality_constraints", [], [])
        self._inequality_constraints: csd.Function = csd.Function("inequality_constraints", [], [])
        self._inequality_constraints_jacobian: csd.Function = csd.Function("inequality_constraints_jacobian", [], [])
        self._box_inequality_constraints: csd.Function = csd.Function("box_inequality_constraints", [], [])

    def reset(self) -> None:
        """Resets the MPC."""
        self._initialized = False
        self._xs_prev = np.array([])
        self._prev_cost = 1e6
        self._prev_sol_status = 0
        self._num_consecutive_qp_failures = 0
        self._t_prev = 0.0
        if self._acados_ocp_solver is not None:
            self._acados_ocp_solver.reset()

    def set_action_indices(self, action_indices: list):
        self._action_indices = action_indices

    def set_adjustable_param_str_list(self, adjustable_param_list: list[str]):
        """Sets the adjustable parameter list for the MPC.

        Args:
            adjustable_param_list (list[str]): List of adjustable parameter strings.
        """
        self._adjustable_param_str_list = adjustable_param_list
        self._fixed_param_str_list = [
            name for name in self._all_adjustable_param_str_list if name not in self._adjustable_param_str_list
        ]
        self._p_adjustable_values = self._params.adjustable(self._adjustable_param_str_list)

    def set_param_subset(self, param_subset: Dict[str, np.ndarray | float]):
        """Sets the parameter subset for the MPC.

        Args:
            param_subset (Dict[str, np.ndarray | float]): Dictionary containing the parameter subset.
        """
        self._params.set_parameter_subset(param_subset)
        self._p_adjustable_values = self._params.adjustable(self._adjustable_param_str_list)

    def get_adjustable_params(self) -> np.ndarray:
        """Returns the RL-tuneable parameters in the MPC.

        Returns:
            np.ndarray: Array of parameters.
        """
        mdl_adjustable_params = np.array([])
        mpc_adjustable_params = self._params.adjustable(name_list=self._adjustable_param_str_list)
        return np.concatenate((mdl_adjustable_params, mpc_adjustable_params))

    def get_fixed_params(self) -> np.ndarray:
        """Returns the fixed parameter values for the NLP problem.

        Returns:
            np.ndarray: Fixed parameter values for the NLP problem.
        """
        return self._p_fixed_values

    def get_antigrounding_surface_functions(self) -> list:
        """Returns the anti-grounding surface functions.

        Returns:
            list: List of anti-grounding surface functions.
        """
        return self._so_surfaces

    @property
    def path_linestring(self) -> geo.LineString:
        return self._path_linestring

    @property
    def map_origin(self) -> np.ndarray:
        return self._map_origin

    @property
    def so_surfaces(self) -> list:
        return self._so_surfaces

    @property
    def min_depth(self) -> int:
        return self._min_depth

    def _set_path_information(
        self, nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline, float]
    ) -> None:
        """Sets the path information for the MPC.

        Args:
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, inter.PchipInterpolator, interp.BSpline, float]): Tuple containing the nominal path splines in x, y, heading and the nominal speed reference. The last element is the path length.
        """
        x_spline, y_spline, _, speed_spline, s_final = nominal_path
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

        self._path_linestring = mapf.create_path_linestring_from_splines(
            [x_spline, y_spline], [x_spline.c, y_spline.c], s_final
        )

        self._s = 0.000000001
        self._s_final_value = s_final
        self.model.set_min_path_variable(self._s)
        self.model.set_max_path_variable(s_final)  # margin
        self._create_path_variable_references()
        # print("Path information set. | s_final: ", s_final)

    def set_params(self, params: parameters.MidlevelMPCParams) -> None:
        """Sets the parameters of the mid-level MPC.

        Args:
            - params (parameters.MidlevelMPCParams): Parameters of the mid-level MPC.
        """
        self._params = params

    def print_warm_start_info(self, xs: np.ndarray) -> None:
        """Prints the warm start info.

        Args:
            - xs (np.ndarray): Current state of the system.
        """
        N = self._acados_ocp.dims.N
        g_do_constr_vals = []
        g_so_constr_vals = []
        g_bx_constr_vals = []
        g_bu_constr_vals = []
        lbu, ubu, lbx, ubx = self.model.get_input_state_bounds()

        for i in range(N + 1):
            p_i = self._parameter_values[i]
            u_i = self._acados_ocp_solver.get(i, "u")
            x_i = self._acados_ocp_solver.get(i, "x")
            g_bx_constr_vals.extend((x_i - ubx).tolist())
            g_bx_constr_vals.extend((lbx - x_i).tolist())
            if i < N:
                g_bu_constr_vals.extend((u_i - ubu).tolist())
                g_bu_constr_vals.extend((lbu - u_i).tolist())
            # g_eq_vals.append(self.model.dynamics(x_i, u_i, csd.vertcat([])).full().flatten())
            g_do_constr_vals.extend(
                self._dynamic_obstacle_constraints(x_i, self._X_do[i], self._params.r_safe_do).full().flatten().tolist()
            )
            g_so_constr_vals.extend(self._static_obstacle_constraints(x_i).full().flatten().tolist())
        if len(g_do_constr_vals) == 0:
            g_do_constr_vals = [0.0]
        if len(g_so_constr_vals) == 0:
            g_so_constr_vals = [0.0]
        g_bx_constr_vals = np.array(g_bx_constr_vals).flatten()
        g_bu_constr_vals = np.array(g_bu_constr_vals).flatten()
        g_do_constr_vals = np.array(g_do_constr_vals).flatten()
        g_so_constr_vals = np.array(g_so_constr_vals).flatten()
        # g_eq_vals = np.array(g_eq_vals).flatten()

        # if g_eq_vals.max() > 1e-6:
        #     print(
        #         f"Warm start is infeasible wrt equality constraints at rows: {np.argwhere(np.abs(g_eq_vals) > 1e-6).flatten().T}!"
        #     )
        if g_bx_constr_vals.max() > 1e-6:
            print(
                f"Warm start is infeasible wrt state box inequality constraints at rows: {np.argwhere(g_bx_constr_vals > 1e-6).flatten().T}!"
            )
        if g_bu_constr_vals.max() > 1e-6:
            print(
                f"Warm start is infeasible wrt input box inequality constraints at rows: {np.argwhere(g_bu_constr_vals > 1e-6).flatten().T}!"
            )
        if g_do_constr_vals.max() > 1e-6:
            print(
                f"Warm start is infeasible wrt dynamic obstacle inequality constraints at rows: {np.argwhere(g_do_constr_vals > 1e-6).flatten().T}!"
            )
        if g_so_constr_vals.max() > 1e-6:
            print(
                f"Warm start is infeasible wrt static obstacle inequality constraints at rows: {np.argwhere(g_so_constr_vals > 1e-6).flatten().T}!"
            )
        print(f"Initial state constraint diff = {self._x_warm_start[:, 0] - xs}")

    def plan(
        self,
        t: float,
        xs: np.ndarray,
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        warm_start: dict,
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
            - enc (senc.ENC): ENC object containing the map info.
            - warm_start (dict): Warm start solution to use.
            - perturb_nlp (bool, optional): Whether to perturb the NLP. Defaults to False.
            - perturb_sigma (float, optional): What standard deviation to use for generating the perturbation. Defaults to 0.001.
            - show_plots (bool, optional): Whether to show plots. Defaults to False.
            - **kwargs: Additional keyword arguments such as an optional previous solution to use.

        Returns:
            - dict: Dictionary containing the optimal trajectory, inputs, slacks and solver stats.

        """
        if not self._initialized:
            self._xs_prev = xs
            self._xs_prev[2] = xs[2] + np.arctan2(xs[4], xs[3])
            self._prev_cost = np.inf
            self._t_prev = t
        else:
            self._xs_prev = warm_start["X"][:, 0] - np.array(
                [self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]
            )

        chi = xs[2] + np.arctan2(xs[4], xs[3])
        xs_unwrapped = xs.copy()
        xs_unwrapped[2] = np.unwrap(np.array([self._xs_prev[2], chi]))[1]
        xs_unwrapped[3] = np.sqrt(xs[3] ** 2 + xs[4] ** 2)
        self._s, self._s_dot = self.compute_path_variable_info(xs_unwrapped)
        xs_unwrapped[4] = self._s
        xs_unwrapped[5] = self._s_dot
        self._xs_prev = xs_unwrapped
        self._x_warm_start = warm_start["X"]
        self._u_warm_start = warm_start["U"]

        self._update_ocp(self._acados_ocp_solver, xs_unwrapped, do_cr_list, do_ho_list, do_ot_list)
        # self.print_warm_start_info(xs_unwrapped)
        try:
            status = self._acados_ocp_solver.solve()
            status_str = mpc_common.map_acados_error_code(status)
        except Exception as _:  # pylint: disable=broad-except
            c = mpc_common.map_acados_error_code(status)
            print(f"[ACADOS] OCP solution: {status} error: {c}")

        if status != 0:
            self._acados_ocp_solver.print_statistics()
        t_solve = self._acados_ocp_solver.get_stats("time_tot")
        cost_val = self._acados_ocp_solver.get_cost()
        n_iter = self._acados_ocp_solver.get_stats("sqp_iter")
        final_residuals = self._acados_ocp_solver.get_stats("residuals")
        success = True if status == 0 else False

        # self._acados_ocp_solver.dump_last_qp_to_json("last_qp.json")
        inputs, trajectory, slacks, lam_g = self._get_solution(self._acados_ocp_solver)
        so_constr_vals, do_constr_vals = self._get_obstacle_constraint_values(trajectory)
        self._x_warm_start = trajectory.copy()
        self._u_warm_start = inputs.copy()

        # self._update_ocp(self._acados_ocp_solver_nonreg, xs_unwrapped, do_cr_list, do_ho_list, do_ot_list)
        # status = self._acados_ocp_solver_nonreg.solve()
        # status_str = mpc_common.map_acados_error_code(status)
        # print(f"[ACADOS] Non-regularized OCP solution: {status_str}")

        # inputs, trajectory, slacks, lam_g = self._get_solution(self._acados_ocp_solver_nonreg)
        # so_constr_vals, do_constr_vals = self._get_obstacle_constraint_values(trajectory)
        # self._x_warm_start = trajectory.copy()
        # self._u_warm_start = inputs.copy()
        if self.verbose:
            np.set_printoptions(precision=3)
            print(
                f"[ACADOS] Mid-level CAS NMPC: \n\t- Status: {status_str} \n\t- Num iter: {n_iter} \n\t- Runtime: {t_solve:.3f} \n\t- Cost: {cost_val:.3f} \n\t- Slacks (max, argmax): ({slacks.max():.3f}, {np.argmax(slacks)}) \n\t- Static obstacle constraints (max, argmax): ({so_constr_vals.max():.3f}, {np.argmax(so_constr_vals)}) \n\t- Dynamic obstacle constraints (max, argmax): ({do_constr_vals.max():.3f}, {np.argmax(do_constr_vals)})) \n\t- Final residuals: {final_residuals}"
            )
        w = np.concatenate((inputs.flatten(), trajectory.flatten(), slacks.flatten())).reshape(-1, 1)
        soln = {"x": w, "lam_g": lam_g, "lam_x": np.zeros(w.shape, dtype=np.float32)}
        self._p_fixed_values = self.create_all_fixed_parameter_values(xs_unwrapped, do_cr_list, do_ho_list, do_ot_list)
        self._t_prev = t

        if status_str == "QPFailure" and status == self._prev_sol_status:
            self._num_consecutive_qp_failures += 1
        self._prev_sol_status = status

        output = {
            "soln": soln,
            "optimal": success,
            "p": self._p_adjustable_values.astype(np.float32),
            "p_fixed": self._p_fixed_values.astype(np.float32),
            "trajectory": trajectory,
            "inputs": inputs,
            "slacks": slacks,
            "so_constr_vals": so_constr_vals,
            "do_constr_vals": do_constr_vals,
            "t_solve": t_solve,
            "cost_val": cost_val,
            "n_iter": n_iter,
            "final_residuals": final_residuals,
            "num_consecutive_qp_failures": self._num_consecutive_qp_failures,
        }
        return output

    def _get_obstacle_constraint_values(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the constraint values at the current solution.

        Args:
            - trajectory (np.ndarray): Trajectory solution.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Equality constraints, static and dynamic obstacle inequality constraints.
        """
        so_constraints = []
        do_constraints = []
        for i in range(self._acados_ocp.dims.N + 1):
            x_i = trajectory[:, i]
            X_do_i = self._X_do[i]
            so_constraints.append(self._static_obstacle_constraints(x_i).full().flatten())
            do_constraints.append(
                self._dynamic_obstacle_constraints(x_i, X_do_i, self._params.r_safe_do).full().flatten()
            )
        so_constraint_arr = np.array(so_constraints).T
        do_constraint_arr = np.array(do_constraints).T
        if so_constraint_arr.size == 0:
            so_constraint_arr = np.array([0.0])
        if do_constraint_arr.size == 0:
            do_constraint_arr = np.array([0.0])
        return so_constraint_arr, do_constraint_arr

    def _get_solution(self, solver: AcadosOcpSolver) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the solution from the solver.

        Args:
            - solver (AcadosOcpSolver): Solver object.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Inputs, states, lower and upper slack variables.
        """
        nx, nu, nh = self._acados_ocp.dims.nx, self._acados_ocp.dims.nu, self._acados_ocp.dims.nh
        max_num_so_constr = self._params.max_num_so_constr

        trajectory = np.zeros((self._acados_ocp.dims.nx, self._acados_ocp.dims.N + 1), dtype=np.float32)
        inputs = np.zeros((self._acados_ocp.dims.nu, self._acados_ocp.dims.N), dtype=np.float32)
        slacks_bx, slacks_so, slacks_do = [], [], []

        start_lbu = 0

        lam_eq, lam_bu, lam_bx, lam_hs_bx, lam_hs_so, lam_hs_do, lam_h = [], [], [], [], [], [], []
        for i in range(self._acados_ocp.dims.N + 1):
            trajectory[:, i] = solver.get(i, "x")

            start_lbx = start_lbu + nu if i < self._acados_ocp.dims.N else 0
            start_lh = start_lbx + nx

            start_ubu = start_lh + nh
            start_ubx = start_ubu + nu if i < self._acados_ocp.dims.N else start_ubu
            start_uh = start_ubx + nx

            start_lsbx = start_uh + nh
            start_usbx = start_lsbx + nx if i > 0 else start_lsbx
            start_ush = start_usbx + nx if i > 0 else start_usbx

            # Extract relevant inequality multipliers for the NLP stage FIX
            lam = solver.get(i, "lam")
            len_bu = nu if i < self._acados_ocp.dims.N else 0
            lbu = lam[start_lbu : start_lbu + len_bu]  # lower bound u_min <= u
            lbx = lam[start_lbx : start_lbx + nx]  # lower bound x_min - sigma <= x
            ubu = lam[start_ubu : start_ubu + len_bu]  # upper bound u_min <= u
            ubx = lam[start_ubx : start_ubx + nx]  # upper bound x <= x_max + sigma
            uh = lam[start_uh : start_uh + nh]  # upper bound for h(t) < 0
            len_sbx_i = nx if i > 0 else 0
            lsbx = lam[start_lsbx : start_lsbx + len_sbx_i]  # lower slack bound for x_min - sigma <= x
            usbx = lam[start_usbx : start_usbx + len_sbx_i]  # Lower bound on slacks x <= x_max + sigma
            ush = lam[start_ush : start_ush + nh]  # Lower bound on slacks for h(t) < 0 + sigma

            lam_bu.extend(lbu.tolist() + ubu.tolist())
            if i == 0:
                lam_eq.extend((lbx - ubx).tolist())
            else:
                lam_bx.extend(lbx.tolist() + ubx.tolist())
            lam_hs_bx.extend(lsbx.tolist() + usbx.tolist())
            lam_hs_so.extend(ush[:max_num_so_constr].tolist())
            lam_hs_do.extend(ush[max_num_so_constr:].tolist())
            lam_h.extend(uh.tolist())

            # if i == 0 or i == 1 or i == self._acados_ocp.dims.N:
            #     print(
            #         f"dim lam_g_ineq_0 = {lbu.size + ubu.size + lbx.size + ubx.size + uh.size + lsbx.size + usbx.size + ush.size}"
            #     )
            if i == 0:
                su = solver.get(i, "su")
                slacks_so.extend(su[:max_num_so_constr].tolist())
                slacks_do.extend(su[max_num_so_constr:].tolist())
            else:  # only extract upper slacks for nonlinear path constraints, and all slacks for the rest
                lower_slacks = solver.get(i, "sl")
                upper_slacks = solver.get(i, "su")
                slacks_bx.extend(lower_slacks[0:nx].tolist() + upper_slacks[0:nx].tolist())
                slacks_so.extend(upper_slacks[nx : nx + max_num_so_constr].tolist())
                slacks_do.extend(upper_slacks[nx + max_num_so_constr :].tolist())

            if i < self._acados_ocp.dims.N:
                lam_eq.extend(solver.get(i, "pi").tolist())
                inputs[:, i] = solver.get(i, "u").T
        slacks = np.concatenate((slacks_bx, slacks_so, slacks_do)).reshape(-1, 1).astype(np.float32)
        lam_eq = np.array(lam_eq, dtype=np.float32).reshape(-1, 1)
        lam_ineq = (
            np.concatenate((lam_bu, lam_bx, lam_hs_bx, lam_hs_so, lam_hs_do, lam_h)).reshape(-1, 1).astype(np.float32)
        )
        lam_g = np.concatenate((lam_eq.flatten(), lam_ineq.flatten())).reshape(-1, 1)
        return inputs, trajectory, slacks, lam_g

    def _update_ocp(
        self,
        solver: AcadosOcpSolver,
        xs: np.ndarray,
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
    ) -> None:
        """Updates the OCP (cost and constraints) with the current info available

        Args:
            - solver (AcadosOcpSolver): Solver object.
            - xs (np.ndarray): Current state [x, y, chi, U, s, s_dot]^T of the ownship.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
        """
        solver.constraints_set(0, "lbx", xs)
        solver.constraints_set(0, "ubx", xs)
        self._parameter_values = []
        self._X_do = []
        for i in range(self._acados_ocp.dims.N + 1):
            solver.set(i, "x", self._x_warm_start[:, i])
            if i < self._acados_ocp.dims.N:
                solver.set(i, "u", self._u_warm_start[:, i])
            p_i = self.create_parameter_values(xs, do_cr_list, do_ho_list, do_ot_list, i)
            if p_i.size != self._p_adjustable.shape[0] + self._p_fixed.shape[0] and self.verbose:
                print(f"Parameter size mismatch: {p_i.size} vs {len(self._p_list)}")
            self._parameter_values.append(p_i)
            solver.set(i, "p", p_i)
        # print("OCP updated")

    def construct_ocp(
        self,
        nominal_path: Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline, float],
        so_list: list,
        enc: senc.ENC,
        map_origin: np.ndarray = np.array([0.0, 0.0]),
        min_depth: int = 5,
        debug: bool = False,
    ) -> None:
        """Constructs the OCP for the NMPC problem using ACADOS.

         Class constructs an ACADOS tailored OCP on the form:
            min     ∫ Lc(x, u, p) dt + Tc_theta(xf)  (from 0 to Tf)
            s.t.    xdot = f_expl(x, u)
                    lbx <= x <= ubx ∀ x
                    lbu <= u <= ubu ∀ u
                    lbh - slh <= h(x, u, p) <= ubh + suh
                    slh, suh >= 0

            where x, u and p are the state, input and parameter vector, respectively.

        Args:
            - nominal_path (Tuple[interp.BSpline, interp.BSpline, interp.PchipInterpolator, interp.BSpline, float]): Tuple containing the nominal path splines in x, y, heading and the nominal speed reference. The last element is the path length.
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.
            - map_origin (np.ndarray, optional): Origin of the map. Defaults to np.array([0.0, 0.0]).
            - min_depth (int, optional): Minimum allowable depth for the vessel. Defaults to 5.
            - debug (bool, optional): Whether to print debug info. Defaults to False.

        """
        self._initialized = False
        self._set_path_information(nominal_path)
        N = int(self._params.T / self._params.dt)

        self._acados_ocp = AcadosOcp()
        self._acados_ocp.solver_options = self._solver_options
        self._min_depth = min_depth
        self._map_origin = map_origin
        self._acados_ocp.model = self.model.as_acados()
        self._acados_ocp.dims.N = N
        self._acados_ocp.solver_options.qp_solver_cond_N = self._acados_ocp.dims.N
        self._acados_ocp.solver_options.tf = self._params.T

        nx = self._acados_ocp.model.x.size()[0]
        nu = self._acados_ocp.model.u.size()[0]
        self._acados_ocp.dims.nx = nx
        self._acados_ocp.dims.nu = nu

        x = self._acados_ocp.model.x
        u = self._acados_ocp.model.u
        self._p_adjustable = csd.vertcat(self._acados_ocp.model.p)
        p_fixed, p_adjustable = [], []  # NLP parameters

        self._nlp_perturbation = csd.MX.sym("nlp_perturbation", nu, 1)
        p_fixed.append(self._nlp_perturbation)
        p_fixed.append(self._p_mdl)

        self._idx_slacked_bx_constr = np.array([0, 1, 2, 3, 4, 5])

        # Path following, speed deviation, chattering and fuel cost parameters
        dim_Q_p = self._params.Q_p.shape[0]
        Q_p_vec = csd.MX.sym("Q_vec", dim_Q_p, 1)
        alpha_app_course = csd.MX.sym("alpha_app_course", 2, 1)
        alpha_app_speed = csd.MX.sym("alpha_app_speed", 2, 1)
        K_app_course = csd.MX.sym("K_app_course", 1, 1)
        K_app_speed = csd.MX.sym("K_app_speed", 1, 1)

        s_dot_ref = csd.MX.sym("s_ref", 1, 1)
        p_fixed.append(self._x_path_coeffs)
        p_fixed.append(self._y_path_coeffs)
        p_fixed.append(self._x_dot_path_coeffs)
        p_fixed.append(self._y_dot_path_coeffs)
        p_fixed.append(s_dot_ref)

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

        # Safety zone parameters
        r_safe_do = csd.MX.sym("r_safe_do", 1)

        p_adjustable.append(Q_p_vec)
        p_adjustable.append(alpha_app_course)
        p_adjustable.append(alpha_app_speed)
        p_adjustable.append(K_app_course)
        p_adjustable.append(K_app_speed)
        p_adjustable.append(alpha_cr)
        p_adjustable.append(y_0_cr)
        p_adjustable.append(alpha_ho)
        p_adjustable.append(x_0_ho)
        p_adjustable.append(alpha_ot)
        p_adjustable.append(x_0_ot)
        p_adjustable.append(y_0_ot)
        p_adjustable.append(d_attenuation)
        p_adjustable.append(colregs_weights)
        p_adjustable.append(r_safe_do)
        p_adjustable, p_fixed = self.prune_adjustable_params(p_adjustable, p_fixed)

        approx_inf = 2000.0
        lbu, ubu, lbx, ubx = self.model.get_input_state_bounds()

        # Input constraints
        self._acados_ocp.constraints.idxbu = np.array(range(nu))
        self._acados_ocp.constraints.lbu = lbu
        self._acados_ocp.constraints.ubu = ubu

        # State constraints
        self._acados_ocp.constraints.idxbx = np.array(range(nx))
        self._acados_ocp.constraints.x0 = np.zeros(nx)
        self._acados_ocp.constraints.lbx = lbx[self._acados_ocp.constraints.idxbx]
        self._acados_ocp.constraints.ubx = ubx[self._acados_ocp.constraints.idxbx]
        self._acados_ocp.constraints.idxsbx = self._idx_slacked_bx_constr

        self._acados_ocp.constraints.idxbx_e = np.array(range(nx))
        self._acados_ocp.constraints.lbx_e = lbx[self._acados_ocp.constraints.idxbx_e]
        self._acados_ocp.constraints.ubx_e = ubx[self._acados_ocp.constraints.idxbx_e]
        self._acados_ocp.constraints.idxsbx_e = self._idx_slacked_bx_constr

        n_colregs_zones = 3
        n_path_constr = self._params.max_num_so_constr + self._params.max_num_do_constr_per_zone * n_colregs_zones
        nx_do = 6
        X_do = csd.MX.sym("X_do", nx_do * self._params.max_num_do_constr_per_zone * n_colregs_zones)
        so_surfaces, _ = mapf.compute_surface_approximations_from_polygons(
            so_list, enc, safety_margins=[0.0], map_origin=self._map_origin, show_plots=debug
        )
        p_fixed.append(X_do)
        so_surfaces = so_surfaces[0]
        self._so_surfaces = so_surfaces

        so_constr_list = []
        do_constr_list = []
        if n_path_constr:
            self._acados_ocp.constraints.lh_0 = -approx_inf * np.ones(n_path_constr)
            self._acados_ocp.constraints.lh = -approx_inf * np.ones(n_path_constr)
            self._acados_ocp.constraints.lh_e = -approx_inf * np.ones(n_path_constr)
            self._acados_ocp.constraints.uh_0 = np.zeros(n_path_constr)
            self._acados_ocp.constraints.uh = np.zeros(n_path_constr)
            self._acados_ocp.constraints.uh_e = np.zeros(n_path_constr)

            # Slacks on dynamic obstacle and static obstacle constraints
            self._acados_ocp.constraints.idxsh_0 = np.array(range(n_path_constr))
            self._acados_ocp.constraints.idxsh = np.array(range(n_path_constr))
            self._acados_ocp.constraints.idxsh_e = np.array(range(n_path_constr))

            con_h_expr = []
            so_constr_list = self._create_static_obstacle_constraint(x, so_surfaces=so_surfaces)
            con_h_expr.extend(so_constr_list)

            do_constr_list = self._create_dynamic_obstacle_constraint(x, X_do, nx_do, r_safe_do)
            con_h_expr.extend(do_constr_list)

            self._acados_ocp.model.con_h_expr_0 = csd.vertcat(*con_h_expr)
            self._acados_ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
            self._acados_ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

        ns = n_path_constr + self._idx_slacked_bx_constr.size
        self._acados_ocp.cost.Zl_0 = 0 * self._params.w_L2 * np.ones(n_path_constr)
        self._acados_ocp.cost.Zl = 0 * self._params.w_L2 * np.ones(ns)
        self._acados_ocp.cost.Zl_e = 0 * self._params.w_L2 * np.ones(ns)
        self._acados_ocp.cost.Zu_0 = 0 * self._params.w_L2 * np.ones(n_path_constr)
        self._acados_ocp.cost.Zu = 0 * self._params.w_L2 * np.ones(ns)
        self._acados_ocp.cost.Zu_e = 0 * self._params.w_L2 * np.ones(ns)

        # We use L1 norm for the slack variables
        self._acados_ocp.cost.zl_0 = self._params.w_L1 * np.ones(n_path_constr)
        self._acados_ocp.cost.zl = self._params.w_L1 * np.ones(ns)
        self._acados_ocp.cost.zl_e = self._params.w_L1 * np.ones(ns)
        self._acados_ocp.cost.zu_0 = self._params.w_L1 * np.ones(n_path_constr)
        self._acados_ocp.cost.zu = self._params.w_L1 * np.ones(ns)
        self._acados_ocp.cost.zu_e = self._params.w_L1 * np.ones(ns)

        # Cost function
        # self._acados_ocp.model.cost_expr_ext_cost = u.T @ np.diag([0.00001, 0.00001, 0.00001]) @ u
        # self._acados_ocp.model.cost_expr_ext_cost_e = 0.0
        self._acados_ocp.cost.cost_type_0 = "EXTERNAL"
        self._acados_ocp.cost.cost_type = "EXTERNAL"
        self._acados_ocp.cost.cost_type_e = "EXTERNAL"

        x_path = self._x_path(x[4], self._x_path_coeffs)
        y_path = self._y_path(x[4], self._y_path_coeffs)
        path_ref = csd.vertcat(x_path, y_path, s_dot_ref)
        path_following_cost, _, _ = mpc_common.path_following_cost_huber(x, path_ref, Q_p_vec)
        rate_cost, _, _ = mpc_common.rate_cost(
            u[0],
            u[1],
            csd.vertcat(alpha_app_course, alpha_app_speed),
            csd.vertcat(K_app_course, K_app_speed),
            r_max=ubu[0],
            a_max=ubu[1],
        )
        x_dot_path = self._x_dot_path(x[4], self._x_dot_path_coeffs)
        y_dot_path = self._y_dot_path(x[4], self._y_dot_path_coeffs)
        speed_ref = x[5] * csd.sqrt(1e-8 + x_dot_path**2 + y_dot_path**2)
        speed_dev_cost = Q_p_vec[1] * (x[3] - speed_ref) ** 2

        X_do_cr = X_do[: nx_do * self._params.max_num_do_constr_per_zone]
        X_do_ho = X_do[
            nx_do * self._params.max_num_do_constr_per_zone : nx_do * 2 * self._params.max_num_do_constr_per_zone
        ]
        X_do_ot = X_do[nx_do * 2 * self._params.max_num_do_constr_per_zone :]
        colregs_cost, _, _, _ = mpc_common.colregs_cost(
            x,
            X_do_cr,
            X_do_ho,
            X_do_ot,
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
        self._acados_ocp.model.cost_expr_ext_cost_0 = path_following_cost + speed_dev_cost + rate_cost + colregs_cost
        self._acados_ocp.model.cost_expr_ext_cost = path_following_cost + speed_dev_cost + rate_cost + colregs_cost
        self._acados_ocp.model.cost_expr_ext_cost_e = path_following_cost + speed_dev_cost

        # Parameters consist of RL adjustable parameters and fixed parameters
        # (either nominal path or dynamic obstacle related).
        self._p_fixed = csd.vertcat(*p_fixed)
        self._p_adjustable = csd.vertcat(*p_adjustable)
        self._p = csd.vertcat(*p_adjustable, *p_fixed)
        self._p_fixed_list = p_fixed
        self._p_adjustable_list = p_adjustable
        self._p_adjustable_values = self.get_adjustable_params()

        self._acados_ocp.model.p = csd.vertcat(self._p_adjustable, self._p_fixed)
        self._acados_ocp.dims.np = self._acados_ocp.model.p.size()[0]
        self._acados_ocp.parameter_values = self.create_parameter_values(
            xs=np.zeros(nx), do_cr_list=[], do_ho_list=[], do_ot_list=[], stage_idx=0
        )

        # remove files in the code export directory
        if rl_dp.acados_code_gen.exists():
            shutil.rmtree(rl_dp.acados_code_gen)
            rl_dp.acados_code_gen.mkdir(parents=True, exist_ok=True)
        solver_json = (
            str(rl_dp.acados_code_gen) + "/acados_ocp_" + self._acados_ocp.model.name + "_" + self.identifier + ".json"
        )

        self._acados_ocp.model.name += "_" + self.identifier
        self._acados_ocp.code_export_directory = rl_dp.acados_code_gen.as_posix()
        self._acados_ocp_solver = None
        self._acados_ocp_solver = AcadosOcpSolver(self._acados_ocp, json_file=solver_json, build=True)

        self._static_obstacle_constraints = csd.Function("so_constr", [x], [csd.vertcat(*so_constr_list)])
        self._dynamic_obstacle_constraints = csd.Function(
            "do_constr", [x, X_do, r_safe_do], [csd.vertcat(*do_constr_list)]
        )

        # formulate non-regularized ocp
        # self._acados_ocp.solver_options.regularize_method = "NO_REGULARIZE"
        # solver_json = "acados_ocp_" + self._acados_ocp.model.name + ".json"
        # self._acados_ocp_solver_nonreg = AcadosOcpSolver(self._acados_ocp, json_file=solver_json)

    def _create_static_obstacle_constraint(
        self,
        x_k: csd.MX,
        so_surfaces: list,
    ) -> list:
        """Creates the static obstacle constraints for the NLP at the current stage, based on the chosen static obstacle constraint type.

        Args:
            - x_k (csd.MX): State vector at the current stage in the OCP.
            - so_surfaces (list): Parametric surface approximations for the static obstacles, if parametric surface constraints are used.

        Returns:
            list: List of static obstacle constraints at the current stage in the OCP.
        """
        so_constr_list = []
        if self._params.max_num_so_constr == 0:
            return so_constr_list
        assert so_surfaces is not None, "Parametric surfaces must be provided for this constraint type."
        n_so = len(so_surfaces)
        for j in range(self._params.max_num_so_constr):
            if j < n_so:
                so_constr_list.append(so_surfaces[j](x_k[0:2].reshape((1, 2))))
                # vertices = mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * r_safe_so + x_k[0:2]
                # vertices = vertices.reshape((-1, 2))
                # for i in range(vertices.shape[0]):
                #     so_constr_list.append(csd.vec(so_surfaces[j](vertices[i, :])))
            else:
                so_constr_list.append(-1.0)
        return so_constr_list

    def _create_dynamic_obstacle_constraint(self, x_k: csd.MX, X_do_k: csd.MX, nx_do: int, r_safe_do: csd.MX) -> list:
        """Creates the dynamic obstacle constraints for the NLP at the current stage.

        Args:
            x_k (csd.MX): State vector at the current stage in the OCP.
            X_do_k (csd.MX): Decision variables of the dynamic obstacles (in all colregs zones) at the current stage in the OCP.
            nx_do (int): Dimension of fixed parameter vector for a dynamic obstacle.
            r_safe_do (csd.MX): Safety distance to dynamic obstacles.

        Returns:
            list: List of dynamic obstacle constraints at the current stage in the OCP.
        """
        do_constr_list = []
        epsilon = 1e-4
        n_do = int(X_do_k.shape[0] / nx_do)
        for i in range(n_do):
            x_aug_do_i = X_do_k[nx_do * i : nx_do * (i + 1)]
            x_do_i = x_aug_do_i[0:4]
            l_do_i = x_aug_do_i[4]
            Rchi_do_i = mf.Rpsi2D_casadi(x_do_i[2])
            p_diff_do_frame = Rchi_do_i.T @ (x_k[0:2] - x_do_i[0:2])
            weights = hf.casadi_matrix_from_nested_list(
                [[1.0 / (0.5 * l_do_i + r_safe_do) ** 2, 0.0], [0.0, 1.0 / (0.5 * l_do_i + r_safe_do) ** 2]]
            )
            # do_constr_list.append(1.0 - p_diff_do_frame.T @ weights @ p_diff_do_frame)
            h_do = csd.log(1.0 + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon)
            do_constr_list.append(h_do)
        return do_constr_list

    def create_parameter_values(
        self,
        xs: np.ndarray,
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        stage_idx: int,
        perturb_nlp: bool = False,
        perturb_sigma: float = 0.001,
    ) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - xs (np.ndarray): State vector at the current stage in the OCP.
            - do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            - do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            - do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            - stage_idx (int): Stage index for the shooting node to consider
            - perturb_nlp (bool, optional): Whether to perturb the NLP problem. Defaults to False.
            - perturb_sigma (float, optional): Standard deviation of the perturbation. Defaults to 0.001.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.


        Returns:
            - np.ndarray: Parameter vector to be used as input to solver
        """
        adjustable_params = self._params.adjustable(self._adjustable_param_str_list)
        if stage_idx == 0 and self.verbose:
            print(f"Adjustable params: {adjustable_params}")

        _, nu = self._acados_ocp.dims.nx, self._acados_ocp.dims.nu
        fixed_parameter_values = []
        d = np.zeros((nu, 1))
        if perturb_nlp:
            d = np.random.normal(0.0, perturb_sigma, size=(nu, 1))
        fixed_parameter_values.extend(d.flatten().tolist())  # d

        path_parameter_values = self._create_path_parameter_values(stages=stage_idx)
        fixed_parameter_values.extend(path_parameter_values)

        non_adjustable_mpc_params = self._params.adjustable(name_list=self._fixed_param_str_list)

        do_cr_parameter_values = self._create_do_parameter_values(xs, do_cr_list, stage_idx)
        do_ho_parameter_values = self._create_do_parameter_values(xs, do_ho_list, stage_idx)
        do_ot_parameter_values = self._create_do_parameter_values(xs, do_ot_list, stage_idx)
        self._X_do.append(np.array(do_cr_parameter_values + do_ho_parameter_values + do_ot_parameter_values))

        n_dos = len(do_cr_list) + len(do_ho_list) + len(do_ot_list)
        p_goal = np.array(list(self.path_linestring.coords[-1]))
        d2goal = np.linalg.norm(xs[0:2] - p_goal)
        if n_dos == 0 or d2goal < 150.0:
            non_adjustable_mpc_params[4] = 10.0
            non_adjustable_mpc_params[5] = 5.0

        fixed_parameter_values.extend(non_adjustable_mpc_params.tolist())
        fixed_parameter_values.extend(do_cr_parameter_values)
        fixed_parameter_values.extend(do_ho_parameter_values)
        fixed_parameter_values.extend(do_ot_parameter_values)
        self._p_fixed_values = np.array(fixed_parameter_values)

        return np.concatenate((adjustable_params, np.array(fixed_parameter_values)))

    def create_all_fixed_parameter_values(
        self,
        state: np.ndarray,
        do_cr_list: list,
        do_ho_list: list,
        do_ot_list: list,
        perturb_nlp: bool = False,
        perturb_sigma: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Creates the fixed parameter values for the NLP problem, which are not adjusted during the optimization.

        Args:
            state (np.ndarray): Current state of the system on the form (x, y, chi, U, s, s_dot)^T.
            do_cr_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the crossing zone.
            do_ho_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the head-on zone.
            do_ot_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for the overtaking zone.
            perturb_nlp (bool, optional): Whether to perturb the NLP problem. Defaults to False.
            perturb_sigma (float, optional): Standard deviation of the perturbation. Defaults to 0.001.

        Returns:
            np.ndarray: Fixed parameter values for the NLP problem.
        """
        fixed_parameter_values: list = []
        n_colregs_zones = 3
        nx, nu = self.model.dims()

        d = np.zeros((nu, 1))
        if perturb_nlp:
            d = np.random.normal(0.0, perturb_sigma, size=(nu, 1))
        fixed_parameter_values.extend(d.flatten().tolist())
        fixed_parameter_values.extend(state.flatten().tolist())  # x0

        path_parameter_values = self._create_path_parameter_values(stages=np.arange(self._acados_ocp.dims.N + 1))
        fixed_parameter_values.extend(path_parameter_values)

        non_adjustable_mpc_params = self._params.adjustable(name_list=self._fixed_param_str_list)
        fixed_parameter_values.extend(non_adjustable_mpc_params.tolist())

        max_num_so_constr = (
            self._params.max_num_so_constr
        )  # min(len(self._so_surfaces), self._params.max_num_so_constr)
        n_bx_slacks = len(self._idx_slacked_bx_constr)
        slack_size = 2 * n_bx_slacks + max_num_so_constr + n_colregs_zones * self._params.max_num_do_constr_per_zone
        W = self._params.w_L1 * np.ones(slack_size)
        fixed_parameter_values.extend(W.tolist())

        N = self._acados_ocp.dims.N
        all_do_cr_parameter_values = []
        all_do_ho_parameter_values = []
        all_do_ot_parameter_values = []
        for k in range(N + 1):
            all_do_cr_parameter_values.extend(self._create_do_parameter_values(state, do_cr_list, k))
            all_do_ho_parameter_values.extend(self._create_do_parameter_values(state, do_ho_list, k))
            all_do_ot_parameter_values.extend(self._create_do_parameter_values(state, do_ot_list, k))
        fixed_parameter_values.extend(all_do_cr_parameter_values)
        fixed_parameter_values.extend(all_do_ho_parameter_values)
        fixed_parameter_values.extend(all_do_ot_parameter_values)

        return np.array(fixed_parameter_values)

    def compute_path_variable_info(self, xs: np.ndarray) -> Tuple[float, float]:
        """Computes the path variable and its derivative from the current state.

        Args:
            xs (np.ndarray): State of the system.

        Returns:
            Tuple[float, float]: Path variable and its derivative.
        """
        s = mapf.find_closest_arclength_to_point(xs[:2], self.path_linestring)
        if s < 0.000001:
            s = 0.000001
        s_dot = self.compute_path_variable_derivative(s)
        return s, s_dot

    def _create_path_variable_references(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates the path variable and its derivative references.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Path variable and its derivative references.
        """
        N = int(self._params.T / self._params.dt)
        s_ref_k = self._s
        dt = self._params.dt
        s_ref_list = []
        s_dot_ref_list = []
        for _ in range(N + 1):
            s_ref_list.append(s_ref_k)
            s_dot_ref_k = self.compute_path_variable_derivative(s_ref_k)
            s_dot_ref_list.append(s_dot_ref_k)
            k1 = self.compute_path_variable_derivative(s_ref_k)
            k2 = self.compute_path_variable_derivative(s_ref_k + 0.5 * dt * k1)
            k3 = self.compute_path_variable_derivative(s_ref_k + 0.5 * dt * k2)
            k4 = self.compute_path_variable_derivative(s_ref_k + dt * k3)
            s_ref_k = s_ref_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self._s_refs = np.array(s_ref_list)
        self._s_dot_refs = np.array(s_dot_ref_list)

    def _create_path_parameter_values(self, stages: int | np.ndarray) -> list:
        """Creates the parameter values for the path constraints.

        Args:
            - stage (int | np.ndarray): Stage indices for the shooting node to consider

        Returns:
            list: List of path parameters to be used as input to solver at the current stage
        """
        path_parameter_values = []
        path_parameter_values.extend(self._x_path_coeffs_values.tolist())
        path_parameter_values.extend(self._y_path_coeffs_values.tolist())
        path_parameter_values.extend(self._x_dot_path_coeffs_values.tolist())
        path_parameter_values.extend(self._y_dot_path_coeffs_values.tolist())
        s_dot_refs = self._s_dot_refs[stages]
        if isinstance(stages, np.ndarray):
            s_dot_refs = s_dot_refs.tolist()
        else:
            s_dot_refs = [s_dot_refs]
        path_parameter_values.extend(s_dot_refs)
        return path_parameter_values

    def _create_do_parameter_values(self, xs: np.ndarray, do_list: list, stage: int) -> list:
        """Creates the parameter values for the dynamic obstacle constraints.

        Args:
            - xs (np.ndarray): Current state of the system
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width) for a certain zone.
            - stage (int): Stage index for the shooting node to consider

        Returns:
            list: List of dynamic obstacle parameters for the considered zone to be used as input to solver
        """
        csog_state = np.array([xs[0], xs[1], xs[2], np.sqrt(xs[3] ** 2 + xs[4] ** 2)])
        n_do = len(do_list)
        nx_do = 6
        X_do = np.zeros(nx_do * self._params.max_num_do_constr_per_zone)
        t = stage * self._params.dt
        for i in range(self._params.max_num_do_constr_per_zone):
            X_do[nx_do * i : nx_do * (i + 1)] = [csog_state[0] - 1e3, csog_state[1] - 1e3, 0.0, 0.0, 10.0, 3.0]
            if i < n_do:
                (ID, do_state, cov, length, width) = do_list[i]
                chi = np.arctan2(do_state[3], do_state[2])
                U = np.sqrt(do_state[2] ** 2 + do_state[3] ** 2)
                X_do[nx_do * i : nx_do * (i + 1)] = np.array(
                    [
                        do_state[0] + t * U * np.cos(chi),
                        do_state[1] + t * U * np.sin(chi),
                        chi,
                        U,
                        length,
                        width,
                    ]
                )
        return X_do.tolist()

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

    def prune_adjustable_params(self, p_adjustable: list, p_fixed: list) -> Tuple[list, list]:
        """Prunes the adjustable parameters,  moves non-considered parameters to the fixed parameter list.

        Args:
            p_adjustable (list): List of adjustable parameters.
            p_fixed (list): List of fixed parameters.

        Returns:
            np.ndarray: Pruned adjustable parameters.
        """
        p_adjustable_upd = []
        for p_adj, name in zip(p_adjustable, self._all_adjustable_param_str_list):
            if name not in self._adjustable_param_str_list:
                p_fixed.append(p_adj)
            else:
                p_adjustable_upd.append(p_adj)
        return p_adjustable_upd, p_fixed
