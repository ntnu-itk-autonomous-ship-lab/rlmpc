"""
    models.py

    Summary:
        Contains (Acados and Casadi) models used in the MPC formulation.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from typing import Tuple

import casadi as csd
import colav_simulator.core.models as cs_models
import numpy as np
import rl_rrt_mpc.common.math_functions as mf
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
    def as_casadi(self) -> Tuple[csd.MX, csd.MX, csd.MX]:
        """Returns casadi relevant symbolics for the model"""

    @abstractmethod
    def get_input_state_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns input and state constraint boxes relevant for the model."""


class Telemetron(MPCModel):
    def __init__(self, params: cs_models.TelemetronParams = cs_models.TelemetronParams()):
        self._acados_model = AcadosModel()
        self._params = params

    def params(self) -> cs_models.TelemetronParams:
        return self._params

    def dims(self) -> Tuple[int, int]:
        return 6, 3

    def setup_equations_of_motion(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]:
        """Forms the equations of motion for the Telemetron vessel

        Returns:
            Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]: Returns the dynamics equation in implicit (xdot - f(x, u)) and explicit (f(x, u)) format, plus the state, state derivative and input symbolic vectors
        """
        # Input
        u = csd.MX.sym("u", 3)
        # Pose [x, y, psi] and its derivative
        eta = csd.MX.sym("eta", 3)
        eta_dot = csd.MX.sym("eta_dot", 3)
        # BODY Velocity [u, v, r] and its derivative
        nu = csd.MX.sym("nu", 3)
        nu_dot = csd.MX.sym("nu_dot", 3)
        # State vector and its derivative
        x = csd.vertcat(eta, nu)
        xdot = csd.vertcat(eta_dot, nu_dot)

        M = self._params.M_rb + self._params.M_a
        Minv = np.linalg.inv(self._params.M_rb + self._params.M_a)

        C = mf.Cmtrx_casadi(csd.MX(M), nu)
        D = mf.Dmtrx_casadi(csd.MX(self._params.D_l), csd.MX(self._params.D_q), csd.MX(self._params.D_c), nu)

        Rpsi = mf.Rpsi_casadi(eta[2])

        kinematics = Rpsi @ nu
        kinetics = Minv @ (-C @ nu - D @ nu + u)

        f_expl = csd.vertcat(kinematics, kinetics)
        f_impl = xdot - f_expl
        return f_impl, f_expl, x, xdot, u

    def get_input_state_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the bounds for the state and input vectors.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Returns the lower and upper bounds for the input and state vectors, respectively
        """
        # Input and state bounds
        min_Fx = self._params.Fx_limits[0]
        max_Fx = self._params.Fx_limits[1]
        min_Fy = self._params.Fy_limits[0]
        max_Fy = self._params.Fy_limits[1]
        lever_arm = self._params.l_r
        max_turn_rate = self._params.r_max
        max_speed = self._params.U_max
        lbu = np.array(
            [
                min_Fx,
                min_Fy,
                lever_arm * min_Fy,
            ]
        )
        ubu = np.array([max_Fx, max_Fy, lever_arm * max_Fy])

        approx_inf = 1e10
        lbx = np.array([-approx_inf, approx_inf, -np.pi, 0.0, -0.6 * max_speed, -max_turn_rate])
        ubx = np.array([-approx_inf, approx_inf, np.pi, max_speed, 0.6 * max_speed, max_turn_rate])
        return lbu, ubu, lbx, ubx

    def as_casadi(self) -> Tuple[csd.MX, csd.MX, csd.MX]:
        """Returns casadi relevant symbolics for the Telemetron

        Returns:
            Tuple[csd.MX, csd.MX, csd.MX]: Returns the Telemetron model dynamics f(x, u), the state and input vector symbolics
        """
        _, f_expl, x, _, u = self.setup_equations_of_motion()
        return f_expl, x, u

    def as_acados(self) -> AcadosModel:
        """Returns an AcadosModel object for the Telemetron

        Returns:
            AcadosModel: Telemetron model as acados compatible object
        """
        f_impl, f_expl, x, xdot, u = self.setup_equations_of_motion()

        self._acados_model.f_impl_expr = f_impl
        self._acados_model.f_expl_expr = f_expl
        self._acados_model.x = x
        self._acados_model.xdot = xdot
        self._acados_model.u = u
        self._acados_model.name = "telemetron"
        return self._acados_model

    @property
    def acados_model(self):
        return self._acados_model
