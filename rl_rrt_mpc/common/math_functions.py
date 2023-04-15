"""
    math_functions.py

    Summary:
        Contains various commonly used mathematical functions.

    Author: Trym Tengesdal
"""
import math
from typing import Tuple

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf


def wrap_min_max(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """Wraps input x to [x_min, x_max)

    Args:
        x (float or np.ndarray): Unwrapped value
        x_min (float or np.ndarray): Minimum value
        x_max (float or np.ndarray): Maximum value

    Returns:
        float or np.ndarray: Wrapped value
    """
    if isinstance(x, np.ndarray):
        return x_min + np.mod(x - x_min, x_max - x_min)
    else:
        return x_min + (x - x_min) % (x_max - x_min)


def wrap_angle_to_pmpi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [-pi, pi)

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, -np.pi * np.ones(angle.size), np.pi * np.ones(angle.size))
    else:
        return wrap_min_max(angle, -np.pi, np.pi)


def wrap_angle_to_02pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [0, 2pi)

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, np.zeros(angle.size), 2 * np.pi * np.ones(angle.size))
    else:
        return wrap_min_max(angle, 0, 2 * np.pi)


def wrap_angle_diff_to_pmpi(a_1: float | np.ndarray, a_2: float | np.ndarray) -> float | np.ndarray:
    """Wraps angle difference a_1 - a_2 to within [-pi, pi)

    Args:
        a_1 (float or np.ndarray): Angle in radians
        a_2 (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle difference
    """
    diff = wrap_angle_to_pmpi(a_1) - wrap_angle_to_pmpi(a_2)
    if isinstance(diff, np.ndarray):
        return wrap_min_max(diff, -np.pi * np.ones(diff.size), np.pi * np.ones(diff.size))
    else:
        return wrap_min_max(diff, -np.pi, np.pi)


def wrap_angle_diff_to_02pi(a_1: float | np.ndarray, a_2: float | np.ndarray) -> float | np.ndarray:
    """Wraps angle difference a_1 - a_2 to within [0, 2pi)

    Args:
        a_1 (float or np.ndarray): Angle in radians
        a_2 (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle difference
    """
    diff = wrap_angle_to_02pi(a_1) - wrap_angle_to_02pi(a_2)
    if isinstance(diff, np.ndarray):
        return wrap_min_max(diff, np.zeros(diff.size), 2 * np.pi * np.ones(diff.size))
    else:
        return wrap_min_max(diff, 0, 2 * np.pi)


def angle_between_vectors(vector1: Tuple[float, float], vector2: Tuple[float, float]) -> float:
    """Calculates the angle between two vectors.

    Args:
        vector1 (Tuple[float, float]): Vector 1
        vector2 (Tuple[float, float]): Vector 2

    Returns:
        float: Angle between the two vectors.
    """
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    acos_feed = inner_product / (len1 * len2)
    if acos_feed < -1.0:
        acos_feed = -1.0
    elif acos_feed > 1.0:
        acos_feed = 1.0
    return math.acos(acos_feed)


def knots2mps(knots: float) -> float:
    """Converts from knots to meters per second.

    Args:
        knots (float): Knots to convert.

    Returns:
        float: Knots converted to m/s.
    """
    mps = knots * 1.852 / 3.6
    return mps


def cm2inch(cm: float) -> float:  # inch to cm
    """Converts from cm to inches.

    Args:
        cm (float): centimetres to convert.

    Returns:
        float: Resulting inches.
    """
    return cm / 2.54


def mps2knots(mps):
    """Converts from meters per second to knots.

    Args:
        mps (float): Mps to convert.

    Returns:
        float: m/s converted to knots.
    """
    knots = mps * 3.6 / 1.852
    return knots


def rotate_vec_2D(vec: np.ndarray, angle: float) -> np.ndarray:
    """Rotates a 2D vector by angle.

    Args:
        vec (np.ndarray): 2D vector to rotate.
        angle (float): Angle to rotate by.

    Returns:
        np.ndarray: Rotated vector.
    """
    r_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rot_vec = np.dot(r_mat, vec)
    return rot_vec


def normalize_vec(v: np.ndarray):
    """Normalize vector v to length 1.

    Args:
        v (np.ndarray): Vector to normalize

    Returns:
        np.ndarray: Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm < 0.000001:
        return v
    else:
        return v / norm


def sat(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    """
    return np.clip(x, x_min, x_max)


def Rzyx(phi, theta, psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R = np.ndarray(
        [
            [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
            [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
            [-sth, cth * sphi, cth * cphi],
        ]
    )

    return R


def Rpsi(psi) -> np.ndarray:
    """
    R = Rpsi(psi) computes the 3x3 rotation matrix of an angle psi about the z-axis
    """
    Rmtrx = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    return Rmtrx


def Rpsi_casadi(psi: float) -> csd.MX:
    """Same as Rpsi but for casadi."""
    return hf.casadi_matrix_from_nested_list([[csd.cos(psi), -csd.sin(psi), 0], [csd.sin(psi), csd.cos(psi), 0], [0, 0, 1]])


def Rpsi2D(psi: float) -> np.ndarray:
    """
    R = Rpsi2D(psi) computes the 2D rotation matrix.
    Rmtrx = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    """
    return np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])


def Rpsi2D_casadi(psi: csd.MX) -> csd.MX:
    """Same as Rpsi2D but for casadi."""
    return hf.casadi_matrix_from_nested_list([[csd.cos(psi), -csd.sin(psi)], [csd.sin(psi), csd.cos(psi)]])


def Cmtrx(Mmtrx: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Calculates coriolis matrix C(v)

    Assumes decoupled surge and sway-yaw dynamics.
    See eq. (7.12) - (7.15) in Fossen2011

    Args:
        Mmtrx (np.ndarray): Mass matrix.
        nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T

    Returns:
        np.ndarray: Coriolis matrix C(v)
    """
    c13 = -(Mmtrx[1, 1] * nu[1] + Mmtrx[1, 2] * nu[2])
    c23 = Mmtrx[0, 0] * nu[0]

    return np.array([[0, 0, c13], [0, 0, c23], [-c13, -c23, 0]])


def Cmtrx_casadi(Mmtrx: csd.MX, nu: csd.MX) -> csd.MX:
    """Same as Cmtrx but for casadi"""
    c13 = -(Mmtrx[1, 1] * nu[1] + Mmtrx[1, 2] * nu[2])
    c23 = Mmtrx[0, 0] * nu[0]
    return hf.casadi_matrix_from_nested_list([[0, 0, c13], [0, 0, c23], [-c13, -c23, 0]])


def Dmtrx(D_l: np.ndarray, D_q: np.ndarray, D_c: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Calculates damping matrix D

    Assumes decoupled surge and sway-yaw dynamics.
    See eq. (7.24) in Fossen2011+

    Args:
        D_l (np.ndarray): Linear damping matrix.
        D_q (np.ndarray): Quadratic damping matrix.
        D_c (np.ndarray): Cubic damping matrix.
        nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T

    Returns:
        np.ndarray: Damping matrix D = D_l + D_q(nu) + D_c(nu)
    """
    return D_l + D_q * np.abs(nu) + D_c * (nu * nu)


def Dmtrx_casadi(D_l: csd.MX, D_q: csd.MX, D_c: csd.MX, nu: csd.MX) -> csd.MX:
    """Same as Dmtrx but for casadi"""
    return D_l + D_q * csd.fabs(nu) + D_c * (nu * nu)


def Smtrx(a: np.ndarray) -> np.ndarray:
    """S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b.

    Args:
        a (np.ndarray): 3x1 vector.

    Returns:
        np.ndarray: 3x3 skew-symmetric matrix.
    """

    S = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return S


def compute_angle_between_vectors(vector1, vector2):
    """Computes the angle between two vectors.

    Args:
        vector1 (_type_): First vector.
        vector2 (_type_): Second vector.

    Returns:
        float: Angle between the two vectors.
    """
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    acos_feed = inner_product / (len1 * len2)
    if acos_feed < -1:
        acos_feed = -1
    elif acos_feed > 1:
        acos_feed = 1
    return math.acos(acos_feed)
