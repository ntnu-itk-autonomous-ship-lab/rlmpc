"""
    models.py

    Summary:
        Contains (Acados and Casadi) models used in the MPC formulation.

    Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Tuple

import casadi as csd
import colav_simulator.core.models as cs_models
import numpy as np
import rlmpc.common.math_functions as mf
from acados_template import AcadosModel


class MPCModel(ABC):
    @abstractmethod
    def params(self):
        "Return the model parameters"

    @abstractmethod
    def dims(self):
        "Return the state and input vector dimensions as tuple"

    @abstractmethod
    def as_acados(self) -> AcadosModel:
        "Returns an AcadosModel object for the given model."

    @abstractmethod
    def as_casadi(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX]:
        """Returns casadi relevant symbolics for the model"""

    @abstractmethod
    def get_input_state_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns input and state constraint boxes relevant for the model."""

    @abstractmethod
    def setup_equations_of_motion(self, **kwargs) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]:
        """Forms the equations of motion for the model"""


@dataclass
class AugmentedKinematicCSOGParams:
    name: str = "AugmentedKinematicCSOG"
    draft: float = 2.0
    length: float = 10.0
    ship_vertices: np.ndarray = field(default_factory=lambda: np.empty(2))
    width: float = 3.0
    T_chi: float = 3.0
    T_U: float = 5.0
    r_max: float = float(np.deg2rad(4))
    U_min: float = 0.0
    U_max: float = 15.0
    U_dot_max: float = 1.0

    @classmethod
    def from_dict(self, params_dict: dict):
        params = AugmentedKinematicCSOGParams(
            draft=params_dict["draft"],
            length=params_dict["length"],
            width=params_dict["width"],
            ship_vertices=np.empty(2),
            T_chi=params_dict["T_chi"],
            T_U=params_dict["T_U"],
            r_max=np.deg2rad(params_dict["r_max"]),
            U_min=params_dict["U_min"],
            U_max=params_dict["U_max"],
            U_dot_max=params_dict["U_dot_max"],
        )
        params.ship_vertices = np.array(
            [
                [params.length / 2.0, -params.width / 2.0],
                [params.length / 2.0, params.width / 2.0],
                [-params.length / 2.0, params.width / 2.0],
                [-params.length / 2.0, -params.width / 2.0],
            ]
        ).T
        return params

    def to_dict(self):
        output_dict = asdict(self)
        output_dict["ship_vertices"] = self.ship_vertices.tolist()
        output_dict["r_max"] = np.rad2deg(self.r_max)
        return output_dict


class AugmentedKinematicCSOG(MPCModel):
    """Casadi+Acados model for the kinematic Course and Speed over Ground model, with augmented state:

    xdot = U cos(chi)
    ydot = U sin(chi)
    chidot = (chi_d - chi) / T_chi
    Udot = (U_d - U) / T_U
    chi_d_dot = u1
    U_d_dot = u2

    i.e. xs = [x, y, chi, U, chi_d, U_d], and the inputs are u = [chi_d_dot, U_d_dot].

    This allows for constraining the speed and course rate, and penalizing high course and speed changes.

    """

    def __init__(self, params: AugmentedKinematicCSOGParams = AugmentedKinematicCSOGParams()):
        self._acados_model = AcadosModel()
        self._params = params
        self.f_impl, self.f_expl, self.xdot, self.x, self.u, self.p = self.setup_equations_of_motion()
        self.dynamics = csd.Function("dynamics", [self.x, self.u, self.p], [self.f_expl], ["x", "u", "p"], ["f_expl"])

        # Input and state bounds
        U_min = self._params.U_min
        U_max = self._params.U_max
        U_dot_max = self._params.U_dot_max
        r_max = self._params.r_max
        approx_inf = 1e10
        self.lbu = np.array(
            [
                -r_max,
                -U_dot_max,
            ]
        )
        self.ubu = np.array([r_max, U_dot_max])

        self.lbx = np.array([-approx_inf, -approx_inf, -approx_inf, U_min, -approx_inf, U_min])
        self.ubx = np.array([approx_inf, approx_inf, approx_inf, U_max, approx_inf, U_max])

    def params(self) -> AugmentedKinematicCSOGParams:
        return self._params

    def dims(self) -> Tuple[int, int]:
        return 6, 2

    def setup_equations_of_motion(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]:
        """Forms the equations of motion for the kinematic model

        Returns:
            Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]: Returns the dynamics equation in implicit (xdot - f(x, u)) and explicit (f(x, u)) format, plus the state derivative, state, input and parameter symbolic vectors
        """
        nx, nu = self.dims()
        x = csd.MX.sym("x", nx)
        u = csd.MX.sym("u", nu)
        xdot = csd.MX.sym("x_dot", nx)

        T_U = csd.MX.sym("T_U", 1)
        T_chi = csd.MX.sym("T_chi", 1)
        p = csd.vertcat(T_chi, T_U)

        kinematics = csd.vertcat(
            x[3] * csd.cos(x[2]), x[3] * csd.sin(x[2]), (x[4] - x[2]) / T_chi, (x[5] - x[3]) / T_U, u[0], u[1]
        )
        f_expl = kinematics
        f_impl = xdot - f_expl
        return f_impl, f_expl, xdot, x, u, p

    def euler_n_step(self, xs: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float, N: int) -> np.ndarray:
        """Simulate N Euler steps for the Telemetron vessel

        Args:
            - xs (np.ndarray): State vector
            - u (np.ndarray): Input vector (chi_d_dot, U_d_dot)
            - p (np.ndarray): Parameter vector
            - dt (float): Time step
            - N (int): Number of steps to simulate

        Returns:
            np.ndarray: Next state vector
        """
        soln = np.zeros((self.x.shape[0], N))
        xs_k = xs
        for k in range(N):
            soln[:, k] = xs_k
            xdot = self.dynamics(xs_k, u, p).full().flatten()
            dxs = xdot * dt
            xs_k = mf.sat(xs_k + dxs, self.lbx, self.ubx)
        return soln

    def erk4_n_step(self, xs: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float, N: int) -> np.ndarray:
        """Simulate N explicit runge kutta 4 steps for the model


        Args:
            xs (np.ndarray): State vector
            u (np.ndarray): Input vector
            p (np.ndarray): Parameter vector
            dt (float): Time step
            N (int): Number of time steps

        Returns:
            np.ndarray: Next state vectors
        """
        soln = np.zeros((self.x.shape[0], N))
        xs_k = xs
        for k in range(N):
            soln[:, k] = xs_k
            k1 = self.dynamics(xs_k, u, p).full().flatten()
            k2 = self.dynamics(xs_k + 0.5 * dt * k1, u, p).full().flatten()
            k3 = self.dynamics(xs_k + 0.5 * dt * k2, u, p).full().flatten()
            k4 = self.dynamics(xs_k + dt * k3, u, p).full().flatten()
            xs_k = xs_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            xs_k = mf.sat(xs_k, self.lbx, self.ubx)
        return soln

    def get_input_state_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.lbu, self.ubu, self.lbx, self.ubx

    def as_casadi(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX]:
        return self.f_expl, self.x, self.u, self.p

    def as_acados(self) -> AcadosModel:
        self._acados_model.f_impl_expr = self.f_impl
        self._acados_model.f_expl_expr = self.f_expl
        self._acados_model.x = self.x
        self._acados_model.xdot = self.xdot
        self._acados_model.u = self.u
        self._acados_model.p = self.p
        self._acados_model.name = "augmented_kinematic_csog"
        return self._acados_model

    @property
    def acados_model(self):
        return self._acados_model


@dataclass
class KinematicCSOGWithAccelerationAndPathtimingParams:
    name: str = "KinematicCSOGModelWithPathtimingAndAcceleration"
    draft: float = 2.0
    length: float = 15.0
    ship_vertices: np.ndarray = field(default_factory=lambda: np.empty(2))
    width: float = 3.0
    r_max: float = float(np.deg2rad(5))
    U_min: float = 0.0
    U_max: float = 10.0
    a_max: float = 0.5
    s_min: float = 0.0
    s_max: float = 1.0
    s_dot_max: float = 1e10

    @classmethod
    def from_dict(self, params_dict: dict):
        params = KinematicCSOGWithAccelerationAndPathtimingParams(
            draft=params_dict["draft"],
            length=params_dict["length"],
            width=params_dict["width"],
            ship_vertices=np.empty(2),
            r_max=np.deg2rad(params_dict["r_max"]),
            U_min=params_dict["U_min"],
            U_max=params_dict["U_max"],
            a_max=params_dict["a_max"],
            s_min=params_dict["s_min"],
            s_max=params_dict["s_max"],
            s_dot_max=params_dict["s_dot_max"],
        )
        params.ship_vertices = np.array(
            [
                [params.length / 2.0, -params.width / 2.0],
                [params.length / 2.0, params.width / 2.0],
                [-params.length / 2.0, params.width / 2.0],
                [-params.length / 2.0, -params.width / 2.0],
            ]
        ).T
        return params

    def to_dict(self):
        output_dict = asdict(self)
        output_dict["ship_vertices"] = self.ship_vertices.tolist()
        output_dict["r_max"] = float(np.rad2deg(self.r_max))
        output_dict["s_min"] = float(self.s_min)
        output_dict["s_max"] = float(self.s_max)
        return output_dict


class KinematicCSOGWithAccelerationAndPathtiming(MPCModel):
    """Casadi+Acados model for the kinematic Course and Speed over Ground model with turn rate and acceleration as input, with augmented state:

    xdot = U cos(chi)
    ydot = U sin(chi)
    chidot = rr
    Udot = a
    s_dot = s_dot
    s_ddot = u_p

    i.e. xs = [x, y, chi, U, s, s_dot], and the input is u = [r, a, u_p]
    calculated as

    This, like for the AugmentedKinematicCSOG, allows for constraining the acceleration and course rate,
    and penalizing high course and speed changes.
    """

    def __init__(
        self,
        params: KinematicCSOGWithAccelerationAndPathtimingParams = KinematicCSOGWithAccelerationAndPathtimingParams(),
    ):
        self._acados_model = AcadosModel()
        self._params = params
        self.setup_equations_of_motion()

        # Input and state bounds
        U_min = -self._params.U_max
        U_max = self._params.U_max
        a_max = self._params.a_max
        r_max = self._params.r_max
        s_min = self._params.s_min
        s_max = self._params.s_max
        s_dot_max = self._params.s_dot_max
        approx_inf = 1e10
        self.lbu = np.array([-r_max, -a_max, -approx_inf])
        self.ubu = np.array([r_max, a_max, approx_inf])
        self.lbx = np.array([-approx_inf, -approx_inf, -approx_inf, -U_max, s_min, 0.0])
        self.ubx = np.array([approx_inf, approx_inf, approx_inf, U_max, s_max, s_dot_max])

    def params(self) -> KinematicCSOGWithAccelerationAndPathtimingParams:
        return self._params

    def dims(self) -> Tuple[int, int]:
        return 6, 3

    def set_min_path_variable(self, s_min: float):
        self._params.s_min = s_min
        self.lbx[4] = s_min

    def set_max_path_variable(self, s_max: float):
        self._params.s_max = s_max
        self.ubx[4] = s_max

    def setup_equations_of_motion(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]:
        """Forms the equations of motion for the kinematic model"""
        nx, nu = self.dims()
        x = csd.MX.sym("x", nx)
        u = csd.MX.sym("u", nu)
        xdot = csd.MX.sym("x_dot", nx)

        p = csd.vertcat([])

        kinematics = csd.vertcat(x[3] * csd.cos(x[2]), x[3] * csd.sin(x[2]), u[0], u[1], x[5], u[2])
        f_expl = kinematics
        f_impl = xdot - f_expl

        self.f_expl = f_expl
        self.f_impl = f_impl
        self.xdot = xdot
        self.x = x
        self.u = u
        self.p = p
        self.dynamics = csd.Function("dynamics", [self.x, self.u, self.p], [self.f_expl], ["x", "u", "p"], ["f_expl"])
        return f_impl, f_expl, xdot, x, u, p

    def euler_n_step(self, xs: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float, N: int) -> np.ndarray:
        """Simulate N Euler steps for the model

        Args:
            - xs (np.ndarray): State vector
            - u (np.ndarray): Input vector (chi_d_dot, U_d_dot), either nu x N or nu x 1
            - p (np.ndarray): Parameter vector
            - dt (float): Time step
            - N (int): Number of steps to simulate, typically equal to N_mpc + 1

        Returns:
            np.ndarray: Next state vector
        """
        nu, N_u = u.shape
        assert N_u == 1 or N_u == N - 1, "u must be either nu x 1 or nu x N - 1"
        soln = np.zeros((self.x.shape[0], N))
        xs_k = xs
        u_k = u[:, 0]
        for k in range(N):
            soln[:, k] = xs_k
            if N_u > 1 and k < N_u:
                u_k = u[:, k]
            xdot = self.dynamics(xs_k, u_k, p).full().flatten()
            dxs = xdot * dt
            xs_k = mf.sat(xs_k + dxs, self.lbx, self.ubx)
        return soln

    def erk4_n_step(self, xs: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float, N: int) -> np.ndarray:
        """Simulate N explicit runge kutta 4 steps for the model

        Args:
            xs (np.ndarray): State vector
            u (np.ndarray): Input vectors, either nu x N or nu x 1
            p (np.ndarray): Parameter vector
            dt (float): Time step
            N (int): Number of time steps, typically equal to N_mpc + 1

        Returns:
            np.ndarray: Next state vectors
        """
        nu, N_u = u.shape
        assert N_u == 1 or N_u == N - 1, "u must be either nu x 1 or nu x N - 1"
        soln = np.zeros((self.x.shape[0], N))
        xs_k = xs
        u_k = u[:, 0]
        for k in range(N):
            soln[:, k] = xs_k
            if N_u > 1 and k < N_u:
                u_k = u[:, k]

            k1 = self.dynamics(xs_k, u_k, p).full().flatten()
            k2 = self.dynamics(xs_k + 0.5 * dt * k1, u_k, p).full().flatten()
            k3 = self.dynamics(xs_k + 0.5 * dt * k2, u_k, p).full().flatten()
            k4 = self.dynamics(xs_k + dt * k3, u_k, p).full().flatten()
            xs_k = xs_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # xs_k = mf.sat(xs_k, self.lbx, self.ubx)
        return soln

    def get_input_state_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.lbu, self.ubu, self.lbx, self.ubx

    def as_casadi(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX]:
        return self.f_expl, self.x, self.u, self.p

    def as_acados(self) -> AcadosModel:
        self._acados_model.f_impl_expr = self.f_impl
        self._acados_model.f_expl_expr = self.f_expl
        self._acados_model.x = self.x
        self._acados_model.xdot = self.xdot
        self._acados_model.u = self.u
        self._acados_model.p = self.p
        self._acados_model.name = "kinematic_csog_with_acceleration_and_path_timing"
        return self._acados_model

    @property
    def acados_model(self):
        return self._acados_model


class Telemetron(MPCModel):
    def __init__(self, params: cs_models.TelemetronParams = cs_models.TelemetronParams()):
        self._acados_model = AcadosModel()
        self._params = params
        self.f_impl, self.f_expl, self.xdot, self.x, self.u, self.p = self.setup_equations_of_motion()
        self.dynamics = csd.Function("dynamics", [self.x, self.u, self.p], [self.f_expl], ["x", "u", "p"], ["f_expl"])
        # Input and state bounds
        min_Fx = self._params.Fx_limits[0]
        max_Fx = self._params.Fx_limits[1]
        min_Fy = self._params.Fy_limits[0]
        max_Fy = self._params.Fy_limits[1]
        max_turn_rate = self._params.r_max
        max_speed = self._params.U_max
        self.lbu = np.array(
            [
                min_Fx,
                min_Fy,
            ]
        )
        self.ubu = np.array([max_Fx, max_Fy])

        approx_inf = 1e10
        self.lbx = np.array([-approx_inf, -approx_inf, -approx_inf, 0.0, -max_speed, -max_turn_rate])
        self.ubx = np.array([approx_inf, approx_inf, approx_inf, max_speed, max_speed, max_turn_rate])

    def params(self) -> cs_models.TelemetronParams:
        return self._params

    def dims(self) -> Tuple[int, int]:
        return 6, 2

    def setup_equations_of_motion(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]:
        """Forms the equations of motion for the Telemetron vessel

        Returns:
            Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]: Returns the dynamics equation in implicit (xdot - f(x, u)) and explicit (f(x, u)) format, plus the state derivative, state, input and parameter symbolic vectors
        """
        x = csd.MX.sym("x", 6)
        u = csd.MX.sym("u", 2)  # Fx, Fy as inputs
        xdot = csd.MX.sym("x_dot", 6)

        M = self._params.M_rb + self._params.M_a
        Minv = np.linalg.inv(self._params.M_rb + self._params.M_a)

        C = mf.Cmtrx_casadi(csd.MX(M), x[3:6])
        D = mf.Dmtrx_casadi(csd.MX(self._params.D_l), csd.MX(self._params.D_q), csd.MX(self._params.D_c), x[3:6])

        Rpsi = mf.Rpsi_casadi(x[2])

        kinematics = Rpsi @ x[3:6]
        B = np.array([[1, 0], [0, 1], [0, -self._params.l_r]])
        kinetics = Minv @ (-C @ x[3:6] - D @ x[3:6] + B @ u)

        p = csd.MX.sym("p", 0)

        f_expl = csd.vertcat(kinematics, kinetics)
        f_impl = xdot - f_expl
        return f_impl, f_expl, xdot, x, u, p

    def euler_n_step(self, xs: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float, N: int) -> np.ndarray:
        """Simulate N Euler steps for the Telemetron vessel

        Args:
            - xs (np.ndarray): State vector
            - u (np.ndarray): Input vector
            - p (np.ndarray): Parameter vector
            - dt (float): Time step
            - N (int): Number of steps to simulate

        Returns:
            np.ndarray: Next state vector
        """
        soln = np.zeros((self.x.shape[0], N))
        xs_k = xs
        for k in range(N):
            soln[:, k] = xs_k
            xdot = self.dynamics(xs_k, u, p).full().flatten()
            dxs = xdot * dt
            xs_k = mf.sat(xs_k + dxs, self.lbx, self.ubx)
        return soln

    def erk4_n_step(self, xs: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float, N: int) -> np.ndarray:
        """Simulate N explicit runge kutta 4 steps for the model


        Args:
            xs (np.ndarray): State vector
            u (np.ndarray): Input vector
            p (np.ndarray): Parameter vector
            dt (float): Time step
            N (int): Number of time steps

        Returns:
            np.ndarray: Next state vectors
        """
        soln = np.zeros((self.x.shape[0], N))
        xs_k = xs
        for k in range(N):
            soln[:, k] = xs_k
            k1 = self.dynamics(xs_k, u, p).full().flatten()
            k2 = self.dynamics(xs_k + 0.5 * dt * k1, u, p).full().flatten()
            k3 = self.dynamics(xs_k + 0.5 * dt * k2, u, p).full().flatten()
            k4 = self.dynamics(xs_k + dt * k3, u, p).full().flatten()
            xs_k = xs_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            xs_k = mf.sat(xs_k, self.lbx, self.ubx)
        return soln

    def get_input_state_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.lbu, self.ubu, self.lbx, self.ubx

    def as_casadi(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX]:
        return self.f_expl, self.x, self.u, self.p

    def as_acados(self) -> AcadosModel:
        self._acados_model.f_impl_expr = self.f_impl
        self._acados_model.f_expl_expr = self.f_expl
        self._acados_model.x = self.x
        self._acados_model.xdot = self.xdot
        self._acados_model.u = self.u
        self._acados_model.name = "telemetron"
        return self._acados_model

    @property
    def acados_model(self):
        return self._acados_model


@dataclass
class DoubleIntegratorParams:
    s_min: float = 0.0
    s_max: float = 1.0
    s_dot_min: float = 0.0
    s_dot_max: float = 1e10

    @classmethod
    def from_dict(self, params_dict: dict):
        params = DoubleIntegratorParams(
            s_min=params_dict["s_min"],
            s_max=params_dict["s_max"],
            s_dot_min=params_dict["s_dot_min"],
            s_dot_max=params_dict["s_dot_max"],
        )
        return params

    def to_dict(self):
        output_dict = asdict(self)
        return output_dict


class DoubleIntegrator(MPCModel):
    """Typically used for path timing model in the MPC.

    s_dot (x1_dot) = s_dot (x2)
    s_ddot (x2_dot) = u
    """

    def __init__(self, params: DoubleIntegratorParams = DoubleIntegratorParams()):
        self._acados_model = AcadosModel()
        self._params = params
        self.f_impl, self.f_expl, self.xdot, self.x, self.u, self.p = self.setup_equations_of_motion()
        self.dynamics = csd.Function(
            "dynamics", [self.x, self.u, self.p], [self.f_expl], ["x_p", "u_p", "p_p"], ["f_expl_p"]
        )

        # Input and state bounds
        approx_inf = 1e10
        self.lbu = np.array(
            [
                -approx_inf,
            ]
        )
        self.ubu = np.array([approx_inf])

        self.lbx = np.array([self._params.s_min, self._params.s_dot_min])
        self.ubx = np.array([self._params.s_max, self._params.s_dot_max])

    def set_min_path_variable(self, s_min: float):
        self._params.s_min = s_min
        self.lbx[0] = s_min

    def set_max_path_variable(self, s_max: float):
        self._params.s_max = s_max
        self.ubx[0] = s_max

    def params(self) -> DoubleIntegratorParams:
        return self._params

    def dims(self) -> Tuple[int, int]:
        return 2, 1

    def setup_equations_of_motion(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]:
        """Forms the equations of motion for the double integrator

        Returns:
            Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]: Returns the dynamics equation in implicit (xdot - f(x, u)) and explicit (f(x, u)) format, plus the state derivative, state, input and parameter symbolic vectors
        """
        x_s = csd.MX.sym("x_s", 2)
        u_s = csd.MX.sym("u_s", 1)
        x_s_dot = csd.MX.sym("x_s_dot", 2)

        p = csd.vertcat([])

        f_expl = csd.vertcat(x_s[1], u_s)
        f_impl = x_s_dot - f_expl
        return f_impl, f_expl, x_s_dot, x_s, u_s, p

    def euler_n_step(self, xs: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float, N: int) -> np.ndarray:
        """Simulate N Euler steps for the double integrator

        Args:
            - xs (np.ndarray): State vector
            - u (np.ndarray): Input vector
            - p (np.ndarray): Parameter vector
            - dt (float): Time step
            - N (int): Number of steps to simulate

        Returns:
            np.ndarray: Next state vector
        """
        soln = np.zeros((self.x.shape[0], N))
        xs_k = xs
        for k in range(N):
            soln[:, k] = xs_k.reshape(-1)
            xdot = self.dynamics(xs_k, u, p).full().flatten()
            dxs = xdot * dt
            xs_k = mf.sat(xs_k + dxs, self.lbx, self.ubx)
        return soln

    def erk4_n_step(self, xs: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float, N: int) -> np.ndarray:
        """Simulate N explicit runge kutta 4 steps for the model


        Args:
            xs (np.ndarray): State vector
            u (np.ndarray): Input vector
            p (np.ndarray): Parameter vector
            dt (float): Time step
            N (int): Number of time steps

        Returns:
            np.ndarray: Next state vectors
        """
        soln = np.zeros((self.x.shape[0], N))
        xs_k = xs
        for k in range(N):
            soln[:, k] = xs_k.reshape(-1)
            k1 = self.dynamics(xs_k, u, p).full().flatten()
            k2 = self.dynamics(xs_k + 0.5 * dt * k1, u, p).full().flatten()
            k3 = self.dynamics(xs_k + 0.5 * dt * k2, u, p).full().flatten()
            k4 = self.dynamics(xs_k + dt * k3, u, p).full().flatten()
            xs_k = xs_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            xs_k = mf.sat(xs_k, self.lbx, self.ubx)
        return soln

    def get_input_state_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.lbu, self.ubu, self.lbx, self.ubx

    def as_casadi(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX]:
        return self.f_expl, self.x, self.u, self.p

    def as_acados(self) -> AcadosModel:
        self._acados_model.f_impl_expr = self.f_impl
        self._acados_model.f_expl_expr = self.f_expl
        self._acados_model.x = self.x
        self._acados_model.xdot = self.xdot
        self._acados_model.u = self.u
        self._acados_model.p = self.p
        self._acados_model.name = "double_integrator"
        return self._acados_model

    @property
    def acados_model(self):
        return self._acados_model
