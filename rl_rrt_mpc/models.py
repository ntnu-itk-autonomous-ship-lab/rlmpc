"""
    models.py

    Summary:
        Contains (Acados) models used in the MPC formulation.

    Author: Trym Tengesdal
"""
import casadi as csd
import colav_simulator.core.models as models
import numpy as np
import rl_rrt_mpc.common.math_functions as mf
from acados_template import AcadosModel


class TelemetronAcados:
    def __init__(self, params: models.TelemetronParams):
        self._model = AcadosModel()
        self._params = params

    def create_model(self) -> AcadosModel:

        # Input
        u = csd.SX.sym("u", 3)
        # Pose [x, y, psi] and its derivative
        eta = csd.SX.sym("eta", 3)
        eta_dot = csd.SX.sym("eta_dot", 3)
        # BODY Velocity [u, v, r] and its derivative
        nu = csd.SX.sym("nu", 3)
        nu_dot = csd.SX.sym("nu_dot", 3)
        # State vector and its derivative
        x = csd.vertcat(eta, nu)
        xdot = csd.vertcat(eta_dot, nu_dot)

        M = self._params.M_rb + self._params.M_a
        Minv = np.linalg.inv(self._params.M_rb + self._params.M_a)

        C = mf.Cmtrx_casadi(M, nu)
        D = mf.Dmtrx_casadi(csd.SX(self._params.D_l), csd.SX(self._params.D_q), csd.SX(self._params.D_c), nu)

        Rpsi = mf.Rpsi_casadi(eta[2])

        kinematics = Rpsi @ nu
        kinetics = Minv * (u - D * nu - C * nu)

        f_expl = csd.vertcat(kinematics, kinetics)
        f_impl = xdot - f_expl

        self._model.f_impl_expr = f_impl
        self._model.f_expl_expr = f_expl
        self._model.x = x
        self._model.xdot = xdot
        self._model.u = u
        self._model.name = "telemetron"
        return self._model

    @property
    def model(self):
        return self._model
