"""
    models.py

    Summary:
        Contains (Acados and Casadi) models used in the MPC formulation.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

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
        return 6, 2

    def setup_equations_of_motion(self) -> Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]:
        """Forms the equations of motion for the Telemetron vessel

        Returns:
            Tuple[csd.MX, csd.MX, csd.MX, csd.MX, csd.MX]: Returns the dynamics equation in implicit (xdot - f(x, u)) and explicit (f(x, u)) format, plus the state derivative, state and input symbolic vectors
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

        f_expl = csd.vertcat(kinematics, kinetics)
        f_impl = xdot - f_expl
        return f_impl, f_expl, xdot, x, u

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
        max_turn_rate = self._params.r_max
        max_speed = self._params.U_max
        lbu = np.array(
            [
                min_Fx,
                min_Fy,
            ]
        )
        ubu = np.array([max_Fx, max_Fy])

        approx_inf = 1e10
        lbx = np.array([-approx_inf, -approx_inf, -approx_inf, 0.0, -max_speed, -max_turn_rate])
        ubx = np.array([approx_inf, approx_inf, approx_inf, max_speed, max_speed, max_turn_rate])
        return lbu, ubu, lbx, ubx

    def as_casadi(self) -> Tuple[csd.MX, csd.MX, csd.MX]:
        """Returns casadi relevant symbolics for the Telemetron

        Returns:
            Tuple[csd.MX, csd.MX, csd.MX]: Returns the Telemetron model dynamics f(x, u), the state and input vector symbolics
        """
        _, f_expl, _, x, u = self.setup_equations_of_motion()
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
