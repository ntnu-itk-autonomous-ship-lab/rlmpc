"""
    acados_mpc.py

    Summary:
        Contains a class (impl in Acados) for an NMPC trajectory tracking/path following controller with incorporated collision avoidance.

    Author: Trym Tengesdal
"""
from typing import Optional, Tuple

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.mpc.common as mpc_common
import rl_rrt_mpc.mpc.models as models
import rl_rrt_mpc.mpc.parameters as parameters
import seacharts.enc as senc
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions
from acados_template.acados_ocp_solver import AcadosOcpSolver


class AcadosMPC:
    def __init__(self, model: models.MPCModel, params: parameters.TTMPCParams, solver_options: AcadosOcpOptions) -> None:
        self._acados_ocp: AcadosOcp = AcadosOcp()
        self._acados_ocp.solver_options = mpc_common.parse_acados_solver_options(solver_options)
        self._model = model

        self._params0: parameters.TTMPCParams = params
        self._params: parameters.TTMPCParams = params

        self._x_warm_start: np.ndarray = np.array([])
        self._u_warm_start: np.ndarray = np.array([])
        self._initialized = False
        self._map_origin: np.ndarray = np.array([0.0, 0.0])

        self._p_mdl: csd.MX = csd.MX.sym("p_mdl", 0)
        self._p_fixed: csd.MX = csd.MX.sym("p_fixed", 0)
        self._p_adjustable: csd.MX = csd.MX.sym("p_adjustable", 0)
        self._p: csd.MX = csd.vertcat(self._p_fixed, self._p_adjustable)

        self._p_fixed_so_values: np.ndarray = np.zeros(0)
        self._p_adjustable_values: np.ndarray = np.zeros(0)
        self._p_mdl_values: np.ndarray = np.zeros(0)
        self._num_fixed_params: int = 0
        self._num_adjustable_params: int = 0

        self._static_obstacle_constraints: csd.Function = csd.Function("static_obstacle_constraints", [], [])
        self._dynamic_obstacle_constraints: csd.Function = csd.Function("dynamic_obstacle_constraints", [], [])
        self._equality_constraints: csd.Function = csd.Function("equality_constraints", [], [])
        self._inequality_constraints: csd.Function = csd.Function("inequality_constraints", [], [])

        self._min_depth: int = 5
        self._t_prev: float = 0.0
        self._xs_prev: np.ndarray = np.array([])

    @property
    def params(self):
        return self._params

    def update_adjustable_params(self, params: list) -> None:
        """Updates the RL-tuneable parameters in the NMPC.

        Args:
            params (list): List of parameters to update. The order of the parameters are:
                - Q
                - R
                - d_safe_so
                - d_safe_do

        Returns:
            - list: List of newly updated parameters.
        """
        nx = self._acados_ocp.model.x.size()[0]
        nu = self._acados_ocp.model.u.size()[0]
        if self._params.path_following:
            dim_Q = 2
        else:
            dim_Q = nx
        self._params.Q = np.reshape(params[0 : dim_Q * dim_Q], (dim_Q, dim_Q))
        self._params.R = np.reshape(params[dim_Q * dim_Q : dim_Q * dim_Q + nu * nu], (nu, nu))
        self._params.d_safe_so = params[dim_Q * dim_Q + nu * nu]
        self._params.d_safe_do = params[dim_Q * dim_Q + nu * nu + 1]

    def get_adjustable_params(self) -> np.ndarray:
        """Returns the RL-tuneable parameters in the NMPC.

        Returns:
            np.ndarray: Array of parameters. The order of the parameters are:
                - model parameters (if any)
                - Q (flattened)
                - R (flattened)
                - d_safe_so
                - d_safe_do
        """
        if isinstance(self._model, models.KinematicCSOG):
            mdl_params = self._model.params()
            mdl_adjustable_params = np.concatenate((mdl_params.T_chi, mdl_params.T_U))
        else:
            mdl_adjustable_params = np.array([])
        mdl_adjustable_params = self._model.get_adjustable_params()
        mpc_adjustable_params = np.concatenate((self._params.Q.flatten(), self._params.R.flatten(), np.array([self._params.d_safe_so, self._params.d_safe_do])))
        return np.concatenate((mdl_adjustable_params, mpc_adjustable_params))

    def _set_initial_warm_start(self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray]) -> None:
        """Sets the initial warm start state (and input) trajectory for the NMPC.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
        """
        N = int(self._params.T / self._params.dt)
        self._x_warm_start = nominal_trajectory[:6, :]
        psi = nominal_trajectory[2, :].tolist()
        self._x_warm_start[2, :] = np.unwrap(np.concatenate(([psi[0]], psi)))[1:]
        if nominal_inputs is not None and nominal_inputs.size > 0:
            self._u_warm_start = nominal_inputs
        else:
            self._u_warm_start = np.zeros((2, N))

    def _try_to_create_warm_start_solution(
        self,
        u_mod: np.ndarray,
        offset: int,
        n_shifts: int,
        enc: senc.ENC,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Creates a shifted warm start trajectory from the previous trajectory and the new control input.
        Args:
            - u_mod (np.ndarray): The new control input to apply at the end of the previous trajectory.
            - offset (int): The offset from the last sample in the previous trajectory, to apply the modified input vector.
            - n_shifts (int): Number of shifts to perform on the previous trajectory.
            - enc (senc.ENC): The ENC object containing the map.

        Returns:
            Tuple[np.ndarray, np.ndarray, bool]: The new warm start and a boolean indicating if the warm start was successful.
        """
        # Simulate the system from t_N to t_N+n_shifts with the last input
        if offset == 0:
            inputs_past_N = np.tile(u_mod, (n_shifts, 1)).T
            states_past_N = self._model.euler_n_step(self._x_warm_start[:, -1], u_mod, self._p_mdl_values, self._params.dt, n_shifts)
        else:
            inputs_past_N = np.tile(u_mod, (n_shifts + offset, 1)).T
            states_past_N = self._model.euler_n_step(self._x_warm_start[:, -offset], u_mod, self._p_mdl_values, self._params.dt, n_shifts + offset)

        if offset == 0:
            u_warm_start = np.concatenate((self._u_warm_start[:, n_shifts:], inputs_past_N), axis=1)
            x_warm_start = np.concatenate((self._x_warm_start[:, n_shifts:], states_past_N), axis=1)
        else:
            u_warm_start = np.concatenate((self._u_warm_start[:, n_shifts:-offset], inputs_past_N), axis=1)
            x_warm_start = np.concatenate((self._x_warm_start[:, n_shifts:-offset], states_past_N), axis=1)

        psi = x_warm_start[2, :].tolist()
        x_warm_start[2, :] = np.unwrap(np.concatenate(([psi[0]], psi)))[1:]

        pos_past_N = states_past_N[:2, :] + self._map_origin.reshape(2, 1)
        pos_past_N[0, :] = states_past_N[1, :] + self._map_origin[1]
        pos_past_N[1, :] = states_past_N[0, :] + self._map_origin[0]
        min_dist, _, _ = mapf.compute_closest_grounding_dist(pos_past_N, self._min_depth, enc)
        if min_dist <= self._params.d_safe_so:
            return x_warm_start, u_warm_start, False

        # hf.plot_trajectory(x_warm_start + np.array([self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0]).reshape(6, 1), enc, "black")
        return x_warm_start, u_warm_start, True

    def _shift_warm_start(self, xs: np.ndarray, dt: float, enc: senc.ENC) -> None:
        """Shifts the warm start decision trajectory (X and U) dt units ahead. Apply ad hoc maneuvering if the shifted trajectory is in collision.

        Args:
            - xs (np.ndarray): Current state of the system.
            - dt (float): Time to shift the warm start decision trajectory.
            - enc (np.ndarray): Electronic Navigation Chart (ENC) of the environment.
        """
        lbu, ubu, _, _ = self._model.get_input_state_bounds()
        n_attempts = 8
        n_shifts = int(dt / self._params.dt)
        # offsets = [0, 0, int(2.5 * n_shifts), int(2.5 * n_shifts), int(2.5 * n_shifts)]
        # u_attempts = [
        #     self._u_warm_start[:, -1],
        #     np.array([self._u_warm_start[0, -1] * 0.5, 0.0]),
        #     np.array([self._u_warm_start[0, -offsets[1]] * 0.5, 0.8 * lbu[1]]),
        #     np.array([self._u_warm_start[0, -offsets[2]] * 0.5, 0.8 * ubu[1]]),
        #     np.array([0.0, 0.0]),
        # ]
        offsets = [0, 0, int(2.0 * n_shifts), int(2.0 * n_shifts), 0, int(4.0 * n_shifts), int(4.0 * n_shifts), int(4.0 * n_shifts)]
        u_attempts = [
            self._u_warm_start[:, -1],
            np.array([self._u_warm_start[0, -1] * 0.5, 0.0]),
            np.array([self._u_warm_start[0, -offsets[2]] * 0.5, 0.3 * lbu[1]]),
            np.array([self._u_warm_start[0, -offsets[3]] * 0.5, 0.3 * ubu[1]]),
            np.array([self._u_warm_start[0, -offsets[4]] * 0.25, 0.0]),
            np.array([self._u_warm_start[0, -offsets[5]] * 0.5, 0.8 * lbu[1]]),
            np.array([self._u_warm_start[0, -offsets[6]] * 0.5, 0.8 * ubu[1]]),
            np.array([0.0, 0.0]),
        ]
        success = False
        for i in range(n_attempts):
            x_warm_start, u_warm_start, success = self._try_to_create_warm_start_solution(u_attempts[i], offsets[i], n_shifts, enc)
            if success:
                break

        if not success:
            print("Warning: Could not create a feasible warm start solution. Attempting with non-feasible solution.")
        self._x_warm_start = x_warm_start
        self._u_warm_start = u_warm_start

    def plan(self, t: float, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list, enc: senc.ENC) -> dict:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of static obstacle Polygon objects.
            - enc (np.ndarray): Electronic Navigation Chart (ENC) of the environment.

        Returns:
            - dict: Dictionary containing the optimal trajectory, inputs, lower slacks, upper slacks and solver stats.
        """
        if not self._initialized:
            self._set_initial_warm_start(nominal_trajectory, nominal_inputs)
            self._p_fixed_so_values = self._create_fixed_so_parameter_values(nominal_trajectory, xs, so_list, enc)
            self._xs_prev = xs
            self._initialized = True

        psi = xs[2]
        xs_unwrapped = xs.copy()
        xs_unwrapped[2] = np.unwrap(np.array([self._xs_prev[2], psi]))[1]
        self._xs_prev = xs_unwrapped
        dt = t - self._t_prev
        if dt > 0.0:
            self._shift_warm_start(xs_unwrapped, dt, enc)

        self._update_ocp(nominal_trajectory, nominal_inputs, xs_unwrapped, do_list, so_list)
        self._acados_ocp_solver.solve_for_x0(x0_bar=xs_unwrapped)
        self._acados_ocp_solver.print_statistics()
        t_solve = self._acados_ocp_solver.get_stats("time_tot")[0]
        cost_val = self._acados_ocp_solver.get_cost()
        n_iter = self._acados_ocp_solver.get_stats("sqp_iter")[0]
        final_residuals = self._acados_ocp_solver.get_stats("residuals")

        # self._acados_ocp_solver.dump_last_qp_to_json("last_qp.json")
        inputs, trajectory, lower_slacks, upper_slacks = self._get_solution(xs_unwrapped)
        so_constr_vals, do_constr_vals = self._get_obstacle_constraint_values(trajectory)

        print(
            f"NMPC: | Runtime: {t_solve} | Cost: {cost_val} | su (max, argmax): ({np.max(upper_slacks)}, {np.argmax(upper_slacks)}) | so_constr (max, argmax): ({np.max(so_constr_vals)}, {np.argmax(so_constr_vals)}) | do_constr (max, argmax): ({np.argmax(do_constr_vals)}, {np.argmax(do_constr_vals)}))"
        )
        self._x_warm_start = trajectory.copy()
        self._u_warm_start = inputs.copy()
        self._t_prev = t
        output = {
            "trajectory": trajectory,
            "inputs": inputs,
            "lower_slacks": lower_slacks,
            "upper_slacks": upper_slacks,
            "so_constr_vals": so_constr_vals,
            "do_constr_vals": do_constr_vals,
            "t_solve": t_solve,
            "cost_val": cost_val,
            "n_iter": n_iter,
            "final_residuals": final_residuals,
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
            so_constraints.append(self._static_obstacle_constraints(x_i).full().flatten())
            do_constraints.append(self._dynamic_obstacle_constraints(x_i, self._params.d_safe_do).full().flatten())
        so_constraint_arr = np.array(so_constraints).T
        do_constraint_arr = np.array(do_constraints).T
        if so_constraint_arr.size == 0:
            so_constraint_arr = np.array([0.0])
        if do_constraint_arr.size == 0:
            do_constraint_arr = np.array([0.0])
        return so_constraint_arr, do_constraint_arr

    def _get_solution(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the solution from the solver.

        Args:
            - xs (np.ndarray): Current state of the system.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Inputs, states, lower and upper slack variables.
        """
        lower_slacks = np.zeros((self._acados_ocp.dims.ns, self._acados_ocp.dims.N + 1))
        upper_slacks = np.zeros((self._acados_ocp.dims.ns, self._acados_ocp.dims.N + 1))
        trajectory = np.zeros((self._acados_ocp.dims.nx, self._acados_ocp.dims.N + 1))
        inputs = np.zeros((self._acados_ocp.dims.nu, self._acados_ocp.dims.N))
        for i in range(self._acados_ocp.dims.N + 1):
            lower_slacks[:, i] = self._acados_ocp_solver.get(i, "sl")
            upper_slacks[:, i] = self._acados_ocp_solver.get(i, "su")
            trajectory[:, i] = self._acados_ocp_solver.get(i, "x")
            if i < self._acados_ocp.dims.N:
                inputs[:, i] = self._acados_ocp_solver.get(i, "u").T
        psi = trajectory[2, :]
        psi = np.unwrap(np.concatenate(([xs[2]], psi)))[1:]
        trajectory[2, :] = psi
        if lower_slacks.size == 0:
            lower_slacks = np.array([0.0])
            upper_slacks = np.array([0.0])
        return inputs, trajectory, lower_slacks, upper_slacks

    def _update_ocp(self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list) -> None:
        """Updates the OCP (cost and constraints) with the current info available

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
        """
        # self._acados_ocp_solver.constraints_set(0, "lbx", xs)
        # self._acados_ocp_solver.constraints_set(0, "ubx", xs)
        for i in range(self._acados_ocp.dims.N + 1):
            self._acados_ocp_solver.set(i, "x", self._x_warm_start[:, i])
            if i < self._acados_ocp.dims.N:
                self._acados_ocp_solver.set(i, "u", self._u_warm_start[:, i])
            p_i = self.create_parameter_values(nominal_trajectory, nominal_inputs, xs, do_list, so_list, i)
            self._acados_ocp_solver.set(i, "p", p_i)
        # print("OCP updated")

    def construct_ocp(
        self,
        nominal_trajectory: np.ndarray,
        nominal_inputs: Optional[np.ndarray],
        xs: np.ndarray,
        do_list: list,
        so_list: list,
        enc: senc.ENC,
        map_origin: np.ndarray = np.array([0.0, 0.0]),
        min_depth: int = 5,
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
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            - so_list (list): List of static obstacle Polygon objects
            - enc (senc.ENC): ENC object.
            - map_origin (np.ndarray, optional): Origin of the map. Defaults to np.array([0.0, 0.0]).
            - min_depth (int, optional): Minimum allowable depth for the vessel. Defaults to 5.

        """
        self._min_depth = min_depth
        self._map_origin = map_origin
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
        self._p_adjustable = csd.vertcat(self._acados_ocp.model.p)

        self._acados_ocp.cost.cost_type = "EXTERNAL"
        self._acados_ocp.cost.cost_type_e = "EXTERNAL"

        dim_Q = 2
        Qscaling = np.eye(2)
        if not self._params.path_following:  # trajectory tracking
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
        R_vec = csd.MX.sym("R_vec", nu * nu, 1)
        Q_vec = csd.MX.sym("Q_vec", dim_Q * dim_Q, 1)
        Qmtrx = (
            hf.casadi_matrix_from_vector(
                Q_vec,
                dim_Q,
                dim_Q,
            )
            @ Qscaling
        )
        self._p_adjustable = csd.vertcat(self._p_adjustable, Q_vec, R_vec)
        self._num_adjustable_params += dim_Q * dim_Q + nu * nu

        x_ref = csd.MX.sym("x_ref", dim_Q)
        u_ref = csd.MX.sym("u_ref", nu)
        gamma = csd.MX.sym("gamma", 1)
        self._acados_ocp.model.cost_expr_ext_cost = gamma * ((x[0:dim_Q] - x_ref).T @ Qmtrx @ (x[0:dim_Q] - x_ref))  # + (u - u_ref).T @ Rmtrx @ (u - u_ref))
        self._acados_ocp.model.cost_expr_ext_cost_e = gamma * (x[0:dim_Q] - x_ref).T @ Qmtrx @ (x[0:dim_Q] - x_ref)
        self._p_fixed = csd.vertcat(x_ref, u_ref, gamma)
        self._num_fixed_params += dim_Q + nu + 1

        approx_inf = 1e6
        lbu, ubu, lbx, ubx = self._model.get_input_state_bounds()

        # Input constraints
        self._acados_ocp.constraints.idxbu = np.array(range(nu))
        self._acados_ocp.constraints.lbu = lbu
        self._acados_ocp.constraints.ubu = ubu

        # State constraints
        self._acados_ocp.constraints.x0 = xs
        self._acados_ocp.constraints.idxbx = np.array(range(nx))
        self._acados_ocp.constraints.lbx = lbx[self._acados_ocp.constraints.idxbx]
        self._acados_ocp.constraints.ubx = ubx[self._acados_ocp.constraints.idxbx]

        self._acados_ocp.constraints.idxbx_e = np.array(range(nx))
        self._acados_ocp.constraints.lbx_e = lbx[self._acados_ocp.constraints.idxbx_e]
        self._acados_ocp.constraints.ubx_e = ubx[self._acados_ocp.constraints.idxbx_e]

        # Dynamic and static obstacle constraints
        d_safe_so = csd.MX.sym("d_safe_so", 1)
        d_safe_do = csd.MX.sym("d_safe_do", 1)
        self._p_adjustable = csd.vertcat(self._p_adjustable, d_safe_so, d_safe_do)
        self._num_adjustable_params += 2

        # n_ship_vertices = ship_vertices.shape[1]
        #        n_path_constr = self._params.max_num_so_constr * n_ship_vertices + self._params.max_num_do_constr
        n_path_constr = self._params.max_num_so_constr + self._params.max_num_do_constr

        if n_path_constr:
            self._acados_ocp.constraints.lh = -approx_inf * np.ones(n_path_constr)
            self._acados_ocp.constraints.lh_e = self._acados_ocp.constraints.lh
            self._acados_ocp.constraints.uh = np.zeros(n_path_constr)
            self._acados_ocp.constraints.uh_e = self._acados_ocp.constraints.uh

            # Slacks on dynamic obstacle and static obstacle constraints
            self._acados_ocp.constraints.idxsh = np.array(range(n_path_constr))
            self._acados_ocp.constraints.idxsh_e = np.array(range(n_path_constr))

            self._acados_ocp.cost.Zl = 0 * self._params.w_L2 * np.ones(n_path_constr)
            self._acados_ocp.cost.Zl_e = 0 * self._params.w_L2 * np.ones(n_path_constr)
            self._acados_ocp.cost.Zu = self._params.w_L2 * np.ones(n_path_constr)
            self._acados_ocp.cost.Zu_e = self._params.w_L2 * np.ones(n_path_constr)
            self._acados_ocp.cost.zl = 0 * self._params.w_L1 * np.ones(n_path_constr)
            self._acados_ocp.cost.zl_e = 0 * self._params.w_L1 * np.ones(n_path_constr)
            self._acados_ocp.cost.zu = self._params.w_L1 * np.ones(n_path_constr)
            self._acados_ocp.cost.zu_e = self._params.w_L1 * np.ones(n_path_constr)

            con_h_expr = []
            so_constr_list = self._create_static_obstacle_constraint(x, so_list, d_safe_so, ship_vertices, enc)
            con_h_expr.extend(so_constr_list)

            do_constr_list = self._create_dynamic_obstacle_constraint(x, d_safe_do)
            con_h_expr.extend(do_constr_list)

            self._acados_ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
            self._acados_ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

        # Parameters consist of RL adjustable parameters and fixed parameters
        # (either nominal trajectory or dynamic obstacle related).
        self._acados_ocp.model.p = csd.vertcat(self._p_adjustable, self._p_fixed)
        self._acados_ocp.dims.np = self._acados_ocp.model.p.size()[0]

        self._p_fixed_so_values = self._create_fixed_so_parameter_values(nominal_trajectory, xs, so_list, enc)
        self._acados_ocp.parameter_values = self.create_parameter_values(nominal_trajectory, nominal_inputs, xs, do_list, so_list, 0, enc)

        solver_json = "acados_ocp_" + self._acados_ocp.model.name + ".json"
        self._acados_ocp.code_export_directory = dp.acados_code_gen.as_posix()
        self._acados_ocp_solver: AcadosOcpSolver = AcadosOcpSolver(self._acados_ocp, json_file=solver_json)

    def _create_static_obstacle_constraint(
        self,
        x_k: csd.MX,
        so_list: list,
        d_safe_so: csd.MX,
        ship_vertices: np.ndarray,
        enc: senc.ENC,
    ) -> list:
        """Creates the static obstacle constraints for the NLP at the current stage, based on the chosen static obstacle constraint type.

        Args:
            - x_k (csd.MX): State vector at the current stage in the OCP.
            - so_list (list): List of static obstacles.
            - d_safe_so (csd.MX): Safety distance to static obstacles.
            - ship_vertices (np.ndarray): Array of ship vertices.
            - enc (senc.ENC): ENC object.
        Returns:
            list: List of static obstacle constraints at the current stage in the OCP.
        """
        epsilon = 1e-6
        so_constr_list: list = []
        if self._params.max_num_so_constr == 0:
            return so_constr_list

        if self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            A_so_constr = csd.MX.sym("A_so_constr", self._params.max_num_so_constr, 2)
            b_so_constr = csd.MX.sym("b_so_constr", self._params.max_num_so_constr, 1)
            self._p_fixed = csd.vertcat(self._p_fixed, csd.reshape(A_so_constr, -1, 1))
            self._p_fixed = csd.vertcat(self._p_fixed, b_so_constr)
            self._num_fixed_params += self._params.max_num_so_constr * 3
            so_constr_list.append(
                # A_so_constr @ x_k[0:2]
                # - b_so_constr
                csd.vec(A_so_constr @ (mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * d_safe_so + x_k[0:2]) - b_so_constr)
            )
        else:
            if self._params.so_constr_type == parameters.StaticObstacleConstraint.CIRCULAR:
                so_pars = csd.MX.sym("so_pars", 3, self._params.max_num_so_constr)  # (x_c, y_c, r) x self._params.max_num_so_constr
                self._p_fixed = csd.vertcat(self._p_fixed, csd.reshape(so_pars, -1, 1))
                self._num_fixed_params += self._params.max_num_so_constr * 3
                for j in range(self._params.max_num_so_constr):
                    x_c, y_c, r_c = so_pars[0, j], so_pars[1, j], so_pars[2, j]
                    so_constr_list.append(csd.log(r_c**2 + epsilon) - csd.log(((x_k[0] - x_c) ** 2) + (x_k[1] - y_c) ** 2 + epsilon))

            elif self._params.so_constr_type == parameters.StaticObstacleConstraint.ELLIPSOIDAL:
                so_pars = csd.MX.sym("so_pars", 2 + 2 * 2, self._params.max_num_so_constr)  # (x_c, y_c, A_c.flatten().tolist()) x self._params.max_num_so_constr
                self._p_fixed = csd.vertcat(self._p_fixed, csd.reshape(so_pars, -1, 1))
                self._num_fixed_params += self._params.max_num_so_constr * 6
                for j in range(self._params.max_num_so_constr):
                    x_e, y_e, A_e = so_pars[0, j], so_pars[1, j], so_pars[2:, j]
                    A_e = csd.reshape(A_e, 2, 2)
                    p_diff_do_frame = x_k[0:2] - csd.vertcat(x_e, y_e)
                    weights = A_e / d_safe_so**2
                    so_constr_list.append(csd.log(1 + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon))

            elif self._params.so_constr_type == parameters.StaticObstacleConstraint.PARAMETRICSURFACE:
                so_surfaces = mapf.compute_surface_approximations_from_polygons(so_list, enc, safety_margins=[self._params.d_safe_so], map_origin=self._map_origin)[0]
                n_so = len(so_surfaces)
                for j in range(self._params.max_num_so_constr):
                    if j < n_so:
                        so_constr_list.append(so_surfaces[j](x_k[0:2].reshape((1, 2))))
                        # vertices = mf.Rpsi2D_casadi(x_k[2]) @ ship_vertices * d_safe_so + x_k[0:2]
                        # vertices = vertices.reshape((-1, 2))
                        # so_constr_list.append(csd.vec(so_surfaces[j](vertices)))
                    else:
                        so_constr_list.append(0.0)

        self._static_obstacle_constraints = csd.Function("static_obstacle_constraints", [x_k], [csd.vertcat(*so_constr_list)], ["x_k"], ["so_constr"])
        return so_constr_list

    def _create_dynamic_obstacle_constraint(self, x_k: csd.MX, d_safe_do: csd.MX) -> list:
        """Creates the dynamic obstacle constraints for the NLP at the current stage.

        Args:
            - x_k (csd.MX): State vector at the current stage in the OCP.
            - d_safe_do (csd.MX): Safety distance to dynamic obstacles.

        Returns:
            list: List of dynamic obstacle constraints at the current stage in the OCP.
        """
        do_constr_list = []
        epsilon = 1e-6
        for i in range(self._params.max_num_do_constr):
            x_do_i = csd.MX.sym("x_do_" + str(i), 4)
            l_do_i = csd.MX.sym("l_do_" + str(i), 1)
            w_do_i = csd.MX.sym("w_do_" + str(i), 1)
            Rchi_do_i = mf.Rpsi2D_casadi(x_do_i[2])
            p_diff_do_frame = Rchi_do_i @ (x_k[0:2] - x_do_i[0:2])
            weights = hf.casadi_matrix_from_nested_list([[1.0 / (l_do_i + d_safe_do) ** 2, 0.0], [0.0, 1.0 / (w_do_i + d_safe_do) ** 2]])
            do_constr_list.append(csd.log(1 + epsilon) - csd.log(p_diff_do_frame.T @ weights @ p_diff_do_frame + epsilon))
        self._dynamic_obstacle_constraints = csd.Function(
            "dynamic_obstacle_constraints", [x_k, d_safe_do], [csd.vertcat(*do_constr_list)], ["x_k", "d_safe_do"], ["do_constr"]
        )
        return do_constr_list

    def create_parameter_values(
        self, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list, stage_idx: int, enc: senc.ENC = None
    ) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): State vector at the current stage in the OCP.
            - do_list (list): List of dynamic obstacles.
            - so_list (list): List of static obstacles.
            - stage_idx (int): Stage index for the shooting node to consider
            - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

        Returns:
            - np.ndarray: Parameter vector to be used as input to solver
        """
        nu = self._acados_ocp.model.u.size()[0]
        N = self._acados_ocp.dims.N
        adjustable_params = self.get_adjustable_params()

        fixed_parameter_values = []
        if self._params.path_following:
            x_ref_stage = nominal_trajectory[0:2, stage_idx]
        else:
            x_ref_stage = nominal_trajectory[:6, stage_idx]

        u_ref_stage = np.zeros(nu)
        if stage_idx < N and nominal_inputs is not None and not self._params.path_following:  # Time parameterized trajectory tracking
            u_ref_stage = nominal_inputs[:, stage_idx]

        fixed_parameter_values.extend(x_ref_stage.tolist())
        fixed_parameter_values.extend(u_ref_stage.tolist())
        fixed_parameter_values.append(self._params.gamma)

        fixed_so_parameter_values = self._update_so_parameter_values(nominal_trajectory, xs, so_list, enc)
        fixed_parameter_values.extend(fixed_so_parameter_values)

        do_parameter_values = self._create_do_parameter_values(nominal_trajectory, xs, do_list, stage_idx, enc)
        fixed_parameter_values.extend(do_parameter_values)

        return np.concatenate((adjustable_params, np.array(fixed_parameter_values)))

    def _create_fixed_so_parameter_values(self, nominal_trajectory: np.ndarray, xs: np.ndarray, so_list: list, enc: senc.ENC) -> np.ndarray:
        """Creates the fixed parameter values for the static obstacle constraints.

        Args:
            - so_list (list): List of static obstacles.
            - xs (np.ndarray): State vector at the current stage in the OCP.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow.
            - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

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
        return np.array(fixed_so_parameter_values)

    def _update_so_parameter_values(self, nominal_trajectory: np.ndarray, xs: np.ndarray, so_list: list, enc: senc.ENC) -> list:
        """Updates the parameter values for the static obstacle constraints in case of changing constraints.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow.
            - xs (np.ndarray): State vector at the current stage in the OCP.
            - so_list (list): List of static obstacles.
            - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

        Returns:
            np.ndarray: Updated fixed parameter vector for static obstacles to be used as input to solver
        """
        if self._params.so_constr_type == parameters.StaticObstacleConstraint.APPROXCONVEXSAFESET:
            assert len(so_list) == 2, "Approximate convex safe set constraint requires constraint variables A and b"
            A, b = so_list[0], so_list[1]
            self._p_fixed_so_values = np.concatenate((A.flatten(), b.flatten()), axis=0)
        return self._p_fixed_so_values.tolist()

    def _create_do_parameter_values(self, nominal_trajectory: np.ndarray, xs: np.ndarray, do_list: list, stage_idx: int, enc: senc.ENC) -> list:
        """Updates the parameter values for the dynamic obstacle constraints in case of changing constraints.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow.
            - xs (np.ndarray): State vector at the current stage in the OCP.
            - do_list (list): List of dynamic obstacles.
            - stage_idx (int): Stage index for the shooting node to consider
            - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

        Returns:
            np.ndarray: Fixed parameter vector for dynamic obstacles to be used as input to solver
        """
        do_parameter_values = []
        dt = self._params.dt
        n_do = len(do_list)
        for i in range(self._params.max_num_do_constr):
            t = stage_idx * dt
            if i < n_do:
                (ID, state, cov, length, width) = do_list[i]
                do_parameter_values.extend([state[0] + t * state[2], state[1] + t * state[3], state[2], state[3], length, width])
            else:
                do_parameter_values.extend([0.0, 0.0, 0.0, 0.0, 5.0, 2.0])
        return do_parameter_values
