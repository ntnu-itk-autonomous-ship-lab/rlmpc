"""
    models.py

    Summary:
        Contains models used in the MPC formulation, and conversion function to AcadosModel format.

    Author: Trym Tengesdal
"""
import casadi as csd
from acados_template import AcadosModel


def export_model(IModel) -> AcadosModel:

    model_name = "pendulum_ode"

    # constants
    M = 1.0  # mass of the cart [kg] -> now estimated
    m = 0.1  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    l = 0.8  # length of the rod [m]

    # set up states & controls
    x1 = csd.SX.sym("x1")
    theta = csd.SX.sym("theta")
    v1 = csd.SX.sym("v1")
    dtheta = csd.SX.sym("dtheta")

    x = csd.vertcat(x1, theta, v1, dtheta)

    F = csd.SX.sym("F")
    u = csd.vertcat(F)

    # xdot
    x1_dot = csd.SX.sym("x1_dot")
    theta_dot = csd.SX.sym("theta_dot")
    v1_dot = csd.SX.sym("v1_dot")
    dtheta_dot = csd.SX.sym("dtheta_dot")

    xdot = csd.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # dynamics
    cos_theta = csd.cos(theta)
    sin_theta = csd.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta
    f_expl = csd.vertcat(
        v1,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F) / denominator,
        (-m * l * cos_theta * sin_theta * dtheta * dtheta + F * cos_theta + (M + m) * g * sin_theta) / (l * denominator),
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model
