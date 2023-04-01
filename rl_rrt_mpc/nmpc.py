"""
    mpc.py

    Summary:
        Contains a class for an NMPC trajectory tracking/path following controller.

    Author: Trym Tengesdal
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.models as models
import seacharts.enc as senc
from acados_template.acados_ocp import AcadosOcp, AcadosOcpOptions
from acados_template.acados_ocp_solver import AcadosOcpSolver

MAX_NUM_DO_CONSTRAINTS: int = 15
MAX_NUM_SO_CONSTRAINTS: int = 300


@dataclass
class NMPCParams:
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
    solver_options: AcadosOcpOptions = AcadosOcpOptions()

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
            solver_options=AcadosOcpOptions(),
        )

        if config.path_following and config.Q.shape[0] != 2:
            raise ValueError("Q must be a 2x2 matrix when path_following is True.")

        if not config.path_following and config.Q.shape[0] != 6:
            raise ValueError("Q must be a 6x6 matrix when path_following is False (trajectory tracking).")

        config.solver_options.nlp_solver_type = config_dict["solver_options"]["nlp_solver_type"]
        config.solver_options.nlp_solver_max_iter = config_dict["solver_options"]["nlp_solver_max_iter"]
        config.solver_options.nlp_solver_tol_eq = config_dict["solver_options"]["nlp_solver_tol_eq"]
        config.solver_options.nlp_solver_tol_ineq = config_dict["solver_options"]["nlp_solver_tol_ineq"]
        config.solver_options.nlp_solver_tol_comp = config_dict["solver_options"]["nlp_solver_tol_comp"]
        config.solver_options.nlp_solver_tol_stat = config_dict["solver_options"]["nlp_solver_tol_stat"]
        config.solver_options.nlp_solver_ext_qp_res = config_dict["solver_options"]["nlp_solver_ext_qp_res"]
        config.solver_options.qp_solver = config_dict["solver_options"]["qp_solver_type"]
        config.solver_options.qp_solver_iter_max = config_dict["solver_options"]["qp_solver_iter_max"]
        config.solver_options.qp_solver_warm_start = config_dict["solver_options"]["qp_solver_warm_start"]
        config.solver_options.hessian_approx = config_dict["solver_options"]["hessian_approx_type"]
        config.solver_options.globalization = config_dict["solver_options"]["globalization"]
        config.solver_options.levenberg_marquardt = config_dict["solver_options"]["levenberg_marquardt"]
        config.solver_options.print_level = config_dict["solver_options"]["print_level"]
        return config


class NMPC:
    def __init__(self, model: models.TelemetronAcados, params: Optional[NMPCParams] = NMPCParams()) -> None:
        self._ocp: AcadosOcp = AcadosOcp()
        self._model = model
        if params:
            self._params0: NMPCParams = params
            self._params: NMPCParams = params

        nx, nu = self._model.dims
        self._x_warm_start: np.ndarray = np.zeros(nx)
        self._u_warm_start: np.ndarray = np.zeros(nu)
        self._initialized = False
        self._map_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # In east-north coordinates
        self._map_origin: Tuple[float, float] = (0.0, 0.0)  # In east-north coordinates

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
            list: List of newly updated parameters.
        """
        nx = self._ocp.model.x.size()[0]
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
        nx = self._ocp.model.x.size()[0]
        return [*self._params.Q.reshape((nx * nx)).tolist(), self._params.gamma, self._params.d_safe_so, self._params.d_safe_do]

    def _set_initial_warm_start(self, nominal_trajectory: np.ndarray | list, nominal_inputs: np.ndarray) -> None:
        """Sets the initial warm start state (and input) trajectory for the NMPC.

        Args:
            nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Assume that the positions are relative to the coordinate/map origin. Either as np.ndarray or as list of splines for (x, y, Optional[psi], Optional[U], Optional[r]), where the optional elements depend on if path following is used or not.
            nominal_inputs (np.ndarray): Nominal reference inputs used if time parameterized trajectory tracking is selected.
        """
        if isinstance(nominal_trajectory, list):
            # eval nominal traj at current path var up until horizon T, using
        self._x_warm_start = nominal_trajectory
        self._u_warm_start = nominal_inputs

    def plan(
        self, t: float, nominal_trajectory: np.ndarray | list, nominal_inputs: np.ndarray, xs: np.ndarray, do_list: list, so_list: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - nominal_trajectory (np.ndarray | list): Nominal reference trajectory to track. Assume that the positions are relative to the coordinate/map origin. Either as np.ndarray or as list of splines for (x, y, Optional[psi], Optional[U], Optional[r]), where the optional elements depend on if path following is used or not.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width). Assume that the position parts are relative to the coordinate/map origin.
            - so_list (list): List of static obstacle Polygon objects. Assume that the positions are relative to the coordinate/map origin.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Optimal trajectory and inputs for the ownship.
        """
        if not self._initialized:
            self._set_initial_warm_start(nominal_trajectory, nominal_inputs)
            self._initialized = True

        self._update_ocp(t, nominal_trajectory, nominal_inputs, xs, do_list, so_list)
        status = self._ocp_solver.solve()
        self._ocp_solver.print_statistics()
        t_solve = self._ocp_solver.get_stats("time_tot")
        cost_val = self._ocp_solver.get_cost()

        trajectory = np.zeros((self._ocp.dims.nx, self._ocp.dims.N + 1))
        inputs = np.zeros((self._ocp.dims.nu, self._ocp.dims.N))
        for i in range(self._ocp.dims.N + 1):
            trajectory[:, i] = self._ocp_solver.get(i, "x")
            if i < self._ocp.dims.N:
                inputs[:, i] = self._ocp_solver.get(i, "u").T
        print(f"NMPC: | Runtime: {t_solve} | Cost: {cost_val}")
        self._x_warm_start = trajectory.copy()
        self._u_warm_start = inputs.copy()
        return trajectory[:, : self._ocp.dims.N], inputs[:, : self._ocp.dims.N]

    def _update_ocp(self, t: float, nominal_trajectory: np.ndarray, nominal_inputs: np.ndarray, xs: np.ndarray, do_list: list, so_list: list) -> None:
        """Updates the OCP (cost and constraints) with the current info available

        Args:
            t (float): Current time.
            nominal_trajectory (np.ndarray): Nominal reference trajectory to track.
            nominal_inputs (np.ndarray): Nominal reference inputs to track.
            xs (np.ndarray): Current state.
            do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            so_list (list): List of static obstacle Polygon objects
        """
        adjustable_params = self.get_adjustable_params()
        self._ocp_solver.constraints_set(0, "lbx", xs)
        self._ocp_solver.constraints_set(0, "ubx", xs)
        for i in range(self._ocp.dims.N + 1):
            self._ocp_solver.set(i, "x", self._x_warm_start[:, i])
            if i < self._ocp.dims.N:
                self._ocp_solver.set(i, "u", self._u_warm_start[:, i])
            p_i = self.create_parameter_values(adjustable_params, nominal_trajectory, do_list, so_list, i)
            self._ocp_solver.set(i, "p", p_i)
        print("OCP updated")

    def construct_ocp(self, nominal_trajectory: Optional[np.ndarray], do_list: list, so_list: list, enc: senc.ENC) -> None:
        """Constructs the OCP for the NMPC problem.

        Args:
            nominal_trajectory (Optional[np.ndarray]): Nominal time-parameterized reference trajectory to track.
            nominal_path (Optional[Path]): Nominal path to follow. Excludes the nominal trajectory argument.
            do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width)
            so_list (list): List of static obstacle Polygon objects
            enc senc.ENC: ENC object.

        """
        self._map_bbox = enc.bbox
        self._map_bbox = (0.0, 0.0, float(enc.bbox[2] - enc.bbox[0]), float(enc.bbox[3] - enc.bbox[1]))  # In relative coordinates
        min_Fx = self._model.params.Fx_limits[0]
        max_Fx = self._model.params.Fx_limits[1]
        min_Fy = self._model.params.Fy_limits[0]
        max_Fy = self._model.params.Fy_limits[1]
        lever_arm = self._model.params.l_r
        max_turn_rate = self._model.params.r_max
        max_speed = self._model.params.U_max

        self._ocp.model = self._model.as_acados()
        self._ocp.solver_options = self._params.solver_options
        self._ocp.dims.N = int(self._params.T / self._params.dt)
        self._ocp.solver_options.qp_solver_cond_N = self._ocp.dims.N
        self._ocp.solver_options.tf = self._params.T

        nx = self._ocp.model.x.size()[0]
        nu = self._ocp.model.u.size()[0]
        self._ocp.dims.nx = nx
        self._ocp.dims.nu = nu

        x = self._ocp.model.x
        u = self._ocp.model.u

        Qvec = csd.MX.sym("Q", 36)
        gamma = csd.MX.sym("gamma", 1)
        x_ref = csd.MX.sym("x_ref", 6)
        fixed_params = x_ref

        self._ocp.cost.cost_type = "EXTERNAL"
        self._ocp.cost.cost_type_e = "EXTERNAL"
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
        self._ocp.model.cost_expr_ext_cost = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)
        self._ocp.model.cost_expr_ext_cost_e = gamma * (x_ref - x).T @ Qmtrx @ (x_ref - x)

        # # soften
        # ocp.constraints.idxsh = np.array([0])
        # ocp.cost.zl = 1e5 * np.array([1])
        # ocp.cost.zu = 1e5 * np.array([1])
        # ocp.cost.Zl = 1e5 * np.array([1])
        # ocp.cost.Zu = 1e5 * np.array([1])

        approx_inf = 1e10

        # Input constraints
        self._ocp.constraints.idxbu = np.array(range(nu))
        self._ocp.constraints.lbu = np.array(
            [
                min_Fx,
                min_Fy,
                lever_arm * min_Fy,
            ]
        )
        self._ocp.constraints.ubu = np.array([max_Fx, max_Fy, lever_arm * max_Fy])

        # State constraints
        lbx = np.array([0.0, 0.0, -np.pi, 0.0, -0.6 * max_speed, -max_turn_rate])
        ubx = np.array([self._map_bbox[3], self._map_bbox[2], np.pi, max_speed, 0.6 * max_speed, max_turn_rate])
        self._ocp.constraints.idxbx_0 = np.array(range(nx))
        self._ocp.constraints.lbx_0 = lbx
        self._ocp.constraints.ubx_0 = ubx

        self._ocp.constraints.idxbx = np.array([0, 1, 3, 4, 5])
        self._ocp.constraints.lbx = lbx[self._ocp.constraints.idxbx]
        self._ocp.constraints.ubx = ubx[self._ocp.constraints.idxbx]

        self._ocp.constraints.idxbx_e = np.array([0, 1, 3, 4, 5])
        self._ocp.constraints.lbx_e = lbx[self._ocp.constraints.idxbx_e]
        self._ocp.constraints.ubx_e = ubx[self._ocp.constraints.idxbx_e]

        # Dynamic and static obstacle constraints
        d_safe_so = csd.MX.sym("d_safe_so", 1)
        d_safe_do = csd.MX.sym("d_safe_do", 1)

        self._ocp.constraints.lh = np.zeros(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._ocp.constraints.lh_e = self._ocp.constraints.lh
        self._ocp.constraints.uh = approx_inf * np.ones(MAX_NUM_SO_CONSTRAINTS + MAX_NUM_DO_CONSTRAINTS)
        self._ocp.constraints.uh_e = self._ocp.constraints.uh

        con_h_expr = []

        # Static obstacle polygon constraints
        so_surfaces = hf.compute_surface_approximations_from_polygons(so_list, enc)
        n_so = len(so_surfaces)
        for j in range(MAX_NUM_SO_CONSTRAINTS):
            if j < n_so:
                con_h_expr.append(so_surfaces[j](x[:2]))
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

        # Parameters consist of RL adjustable parameters, and fixed parameters (either nominal trajectory or dynamic obstacle related). The model parameters are considered fixed.
        adjustable_params = csd.vertcat(Qvec, gamma, d_safe_so, d_safe_do)
        self._ocp.model.p = csd.vertcat(adjustable_params, fixed_params)
        self._ocp.dims.np = self._ocp.model.p.size()[0]

        self._ocp.model.con_h_expr = csd.vertcat(*con_h_expr)
        self._ocp.model.con_h_expr_e = csd.vertcat(*con_h_expr)

        initial_adjustable_params = [*self._params.Q.reshape(nx**2).tolist(), self._params.gamma, self._params.d_safe_so, self._params.d_safe_do]
        self._ocp.parameter_values = self.create_parameter_values(initial_adjustable_params, nominal_trajectory, do_list, so_list, 0)

        solver_json = "acados_ocp_" + self._ocp.model.name + ".json"
        self._ocp.code_export_directory = "../generated_ocp_" + self._ocp.model.name
        self._ocp_solver: AcadosOcpSolver = AcadosOcpSolver(self._ocp, json_file=solver_json)

    def create_parameter_values(self, adjustable_params: list, nominal_trajectory: np.ndarray, do_list: list, so_list: list, stage_idx: int) -> np.ndarray:
        """Creates the parameter vector values for a stage in the OCP, which is used in the cost function and constraints.

        Args:
            adjustable_params (list): List of adjustable parameter values
            nominal_trajectory (np.ndarray): Nominal trajectory to be followed. Assume that the position part is relative to the map origin.
            do_list (list): List of dynamic obstacles. Assume that the position part is relative to the map origin.
            so_list (list): List of static obstacles. Assume that the position part is relative to the map origin.
            stage_idx (int): Stage index for the shooting node to consider

        Returns:
            np.ndarray: Parameter vector to be used as input to solver
        """
        parameter_values = np.concatenate((np.array(adjustable_params), np.array(nominal_trajectory[:, stage_idx])))
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
                parameter_values = np.concatenate((parameter_values, np.array([0.0, 0.0, 0.0, 0.0, 5.0, 2.0])))
        return parameter_values
