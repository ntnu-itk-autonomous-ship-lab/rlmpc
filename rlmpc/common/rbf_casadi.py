"""
    rbf_casadi.py

    Summary:
        Contains Radial Basis Functions (RBF) interpolators implemented in CasADi. Parts of the code are copied/changed from scipy functionality.

    Author: Trym Tengesdal, Scipy authors.
"""

import casadi as csd
import numpy as np


def linear(r: csd.MX) -> csd.MX:
    return -r


def thin_plate_spline(r: csd.MX) -> csd.MX:
    return r**2 * csd.log(r + 1e-07)


def cubic(r: csd.MX) -> csd.MX:
    return r**3


def quintic(r: csd.MX) -> csd.MX:
    return -(r**5)


def multiquadric(r: csd.MX) -> csd.MX:
    return -csd.sqrt(r**2 + 1)


def inverse_multiquadric(r: csd.MX) -> csd.MX:
    return 1 / csd.sqrt(r**2 + 1)


def inverse_quadratic(r: csd.MX) -> csd.MX:
    return 1 / (r**2 + 1)


def gaussian(r: csd.MX) -> csd.MX:
    return csd.exp(-(r**2))


NAME_TO_FUNC = {
    "linear": linear,
    "thin_plate_spline": thin_plate_spline,
    "cubic": cubic,
    "quintic": quintic,
    "multiquadric": multiquadric,
    "inverse_multiquadric": inverse_multiquadric,
    "inverse_quadratic": inverse_quadratic,
    "gaussian": gaussian,
}


def kernel_vector(x: csd.MX, y: np.ndarray, kernel_func, out: csd.MX) -> csd.MX:
    """Evaluate RBFs, with centers at `y`, at the point `x`."""
    for i in range(y.shape[0]):
        out[i] = kernel_func(csd.norm_2(x - y[i].reshape(1, 2)))
        # out[i] = csd.if_else(out[i] < 1e-5, 0.0, out[i])
    return out


def kernel_vector2(x: csd.MX, y: np.ndarray, out: csd.MX) -> csd.MX:
    """Evaluate RBFs, with centers at `y`, at the point `x`."""
    for i in range(y.shape[0]):
        out[i] = thin_plate_spline(csd.norm_2(x - y[i].reshape(1, 2)))
        # out[i] = csd.if_else(out[i] < 1e-5, 0.0, out[i])
    return out


def polynomial_vector(x: csd.MX, powers: np.ndarray, out: csd.MX) -> csd.MX:
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""
    for i in range(powers.shape[0]):
        out[i] = (x[0] ** powers[i, 0]) * (x[1] ** powers[i, 1])
    return out


class RBFInterpolator:
    """A class for radial basis function interpolation using CasADi.

    The interpolant is constructed by summing a radial basis function interpolant + a polynomial interpolant:

    f(x) = sum_i=1^n * phi(||x - x_i||) + p(x)
    """

    def __init__(
        self,
        y: np.ndarray,
        d: np.ndarray,
        coeffs: np.ndarray,
        powers: np.ndarray,
        shift: np.ndarray,
        scale: np.ndarray,
        kernel: str = "thin_plate_spline",
        epsilon: float = 1.0,
    ) -> None:
        """Initialize the RBFInterpolatorCasadi class.

        Args:
            y (np.ndarray): Array of data points we know the function values at.
            d (np.ndarray): Corresponding function values at the data points.
            coeffs (np.ndarray): Coefficients for the RBF interpolant.
            powers (np.ndarray): Exponents for the polynomial interpolant.
            shift (np.ndarray): Shifts the polynomial domain for numerical stability.
            scale (np.ndarray): Scales the polynomial domain for numerical stability. Defaults to None.
            kernel (str): Name of the RBF kernel used. Defaults to "thin_plate_spline".
            epsilon (float): Scaling parameter for the RBF kernel.
        """
        self.shift: np.ndarray = shift.reshape(1, 2)
        self.scale: np.ndarray = scale.reshape(1, 2)
        self.coeffs: np.ndarray = coeffs
        self.powers: np.ndarray = powers
        self.y: np.ndarray = y
        self.d: np.ndarray = d
        self.p: int = self.y.shape[0]
        self.r: int = self.powers.shape[0]
        self.kernel_func = NAME_TO_FUNC[kernel]
        self.epsilon: float = epsilon
        self.yeps: np.ndarray = y * epsilon
        self.vec: csd.MX = csd.MX.zeros(self.p + self.r)

    def __call__(self, x: csd.MX) -> csd.MX:
        """Evaluate the interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) csd.MX array
            Evaluation point coordinates. Q==1 is the number of points to evaluate.

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`.

        """
        xeps = x * self.epsilon
        xhat = (x - self.shift) / self.scale

        self.vec[: self.p] = kernel_vector(xeps, self.yeps, self.kernel_func, self.vec[: self.p])
        # self.vec[: self.p] = kernel_vector2(xeps, self.yeps, self.vec[: self.p])
        self.vec[self.p :] = polynomial_vector(xhat, self.powers, self.vec[self.p :])
        return self.vec.T @ self.coeffs
