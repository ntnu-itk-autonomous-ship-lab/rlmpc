"""
    rbf_casadi.py

    Summary:
        Contains Radial Basis Functions (RBF) interpolators implemented in CasADi. Parts of the code are copied/changed from scipy functionality.

    Author: Trym Tengesdal, Scipy authors.
"""
import casadi as csd
import numpy as np
from scipy.special import comb

# These RBFs are implemented.
_AVAILABLE = {"linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian"}


# The shape parameter does not need to be specified when using these RBFs.
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}


# For RBFs that are conditionally positive definite of order m, the interpolant
# should include polynomial terms with degree >= m - 1. Define the minimum
# degrees here. These values are from Chapter 8 of Fasshauer's "Meshfree
# Approximation Methods with MATLAB". The RBFs that are not in this dictionary
# are positive definite and do not need polynomial terms.
_NAME_TO_MIN_DEGREE = {"multiquadric": 0, "linear": 0, "thin_plate_spline": 1, "cubic": 1, "quintic": 2}


def linear(r: csd.MX) -> csd.MX:
    return -r


def thin_plate_spline(r: csd.MX) -> csd.MX:
    if r == 0:
        return 0.0
    else:
        return r**2 * csd.log(r)


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


def kernel_vector(x: csd.MX, y: np.ndarray, kernel_func) -> csd.MX:
    """Evaluate RBFs, with centers at `y`, at the point `x`."""
    out = csd.MX.zeros(y.shape[0], 1)
    for i in range(y.shape[0]):
        out[i] = kernel_func(np.linalg.norm(x - y[i]))
    return out


def polynomial_vector(x: csd.MX, powers: np.ndarray) -> csd.MX:
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""
    out = csd.MX.zeros(powers.shape[0], dtype=float)
    for i in range(powers.shape[0]):
        out[i] = np.prod(x ** powers[i])
    return out


def kernel_matrix(x, kernel_func) -> csd.MX:
    """Evaluate RBFs, with centers at `x`, at `x`."""
    out = csd.MX.zeros((x.shape[0], x.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        for j in range(i + 1):
            out[i, j] = kernel_func(np.linalg.norm(x[i] - x[j]))
            out[j, i] = out[i, j]
    return out


def polynomial_matrix(x, powers) -> csd.MX:
    """Evaluate monomials, with exponents from `powers`, at `x`."""
    out = csd.MX.zeros((x.shape[0], powers.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        for j in range(powers.shape[0]):
            out[i, j] = np.prod(x[i] ** powers[j])
    return out


def _kernel_matrix(x: csd.MX, kernel: str) -> csd.MX:
    """Return RBFs, with centers at `x`, evaluated at `x`."""
    kernel_func = NAME_TO_FUNC[kernel]
    out = kernel_matrix(x, kernel_func)
    return out


def _polynomial_matrix(x: csd.MX, powers: np.ndarray) -> csd.MX:
    """Return monomials, with exponents from `powers`, evaluated at `x`."""
    out = polynomial_matrix(x, powers)
    return out


def _monomial_powers(ndim: int, degree: int) -> np.ndarray:
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = comb(degree + ndim, ndim, exact=True)
    out = np.zeros((nmonos, ndim), dtype=int)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out[count, var] += 1

            count += 1

    return out


def _build_evaluation_coefficients(x: csd.MX, y: np.ndarray, kernel: str, epsilon: float, powers: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> csd.MX:
    """Construct the coefficients needed to evaluate
    the RBF.

    Parameters
    ----------
    x : (Q, N) float ndarray
        Evaluation point coordinates. Q is the number of evaluation points. N is the number of dimensions.
    y : (P, N) float ndarray
        Data point coordinates. P is the number of data points.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    shift : (N,) float ndarray
        Shifts the polynomial domain for numerical stability.
    scale : (N,) float ndarray
        Scales the polynomial domain for numerical stability.

    Returns
    -------
    (Q, P + R) csd.MX array

    """
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]

    yeps = y * epsilon
    xeps = x * epsilon
    xhat = (x - shift) / scale

    vec = np.empty((q, p + r), dtype=float)
    for i in range(q):
        vec[i, :p] = kernel_vector(xeps[i], yeps, kernel_func)
        vec[i, p:] = polynomial_vector(xhat[i], powers)
    return vec


class RBFInterpolatorCasadi:
    """A class for radial basis function interpolation using CasADi."""

    def __init__(
        self,
        y: np.ndarray,
        d: np.ndarray,
        coeffs: np.ndarray,
        powers: np.ndarray,
        shift: np.ndarray,
        scale: np.ndarray,
        smoothing: float = 0.0,
        kernel: str = "thin_plate_spline",
        epsilon: float | None = None,
    ) -> None:
        """Initialize the RBFInterpolatorCasadi class.

        Args:
            y (np.ndarray): Array of data points we know the function values at.
            d (np.ndarray): Corresponding function values at the data points.
            coeffs (np.ndarray): Coefficients for the RBF interpolant.
            powers (np.ndarray): Exponents for the polynomial interpolant.
            shift (np.ndarray): Shifts the polynomial domain for numerical stability.
            scale (np.ndarray): Scales the polynomial domain for numerical stability. Defaults to None.
            smoothing (float): Smoothing parameter for the RBF interpolant. Defaults to 0.0.
            kernel (str): Name of the RBF kernel used. Defaults to "thin_plate_spline".
            epsilon (float | None): Scaling parameter for the RBF kernel. Defaults to None.
        """
        self.shift = shift
        self.scale = scale
        self.coeffs = coeffs
        self.powers = powers
        self.y = y
        self.d = d
        d_shape = d.shape[1:]
        self.d_shape = d_shape
        self.neighbors = None
        self.smoothing = smoothing
        self.kernel = kernel

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError("`epsilon` must be specified if `kernel` is not one of " f"{_SCALE_INVARIANT}.")
        else:
            epsilon = float(epsilon)
        self.epsilon = epsilon

    def _chunk_evaluator(self, x: csd.MX, y: np.ndarray, shift: np.ndarray, scale: np.ndarray, coeffs: np.ndarray, memory_budget: int = 1000000) -> csd.MX:
        """
        Evaluate the interpolation while controlling memory consumption.
        We chunk the input if we need more memory than specified.

        Parameters
        ----------
        x : (Q, N) csd.MX array. Q is the number of points to evaluate, N is the number of dimensions.
            array of points on which to evaluate
        y: (P, N) float ndarray
            array of points on which we know function values. P is the number of points.
        shift: (N, ) ndarray
            Domain shift used to create the polynomial matrix.
        scale : (N,) float ndarray
            Domain scaling used to create the polynomial matrix.
        coeffs: (P+R, S) float ndarray
            Coefficients in front of basis functions
        memory_budget: int
            Total amount of memory (in units of sizeof(float)) we wish
            to devote for storing the array of coefficients for
            interpolated points. If we need more memory than that, we
            chunk the input.

        Returns
        -------
        (Q, S) csd.MX array
        Interpolated array
        """
        nx, _ = x.shape
        if self.neighbors is None:
            nnei = len(y)
        else:
            nnei = self.neighbors
        # in each chunk we consume the same space we already occupy
        chunksize = memory_budget // ((self.powers.shape[0] + nnei)) + 1
        if chunksize <= nx:
            out = csd.MX.sym("out", (nx, self.d.shape[1]))
            for i in range(0, nx, chunksize):
                vec = _build_evaluation_coefficients(x[i : i + chunksize, :], y, self.kernel, self.epsilon, self.powers, shift, scale)
                out[i : i + chunksize, :] = np.dot(vec, coeffs)
        else:
            vec = _build_evaluation_coefficients(x, y, self.kernel, self.epsilon, self.powers, shift, scale)
            out = csd.dot(vec, coeffs)
        return out

    def __call__(self, x: csd.MX) -> csd.MX:
        """Evaluate the interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) csd.MX array
            Evaluation point coordinates. Q is the number of points to evaluate.

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`.

        """
        _, N = x.shape
        if N != 2:
            raise ValueError("`x` must be a 2-dimensional array.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError("Expected the second axis of `x` to have length " f"{self.y.shape[1]}.")

        # Our memory budget for storing RBF coefficients is
        # based on how many floats in memory we already occupy
        # If this number is below 1e6 we just use 1e6
        # This memory budget is used to decide how we chunk
        # the inputs
        memory_budget = max(x.size + self.y.size + self.d.size, 1000000)

        # No neighbours are considered for simplicity
        assert self.neighbors is None, "Nearest neighbours are not supported yet"
        out = self._chunk_evaluator(x, self.y, self.shift, self.scale, self.coeffs, memory_budget=memory_budget)
        out = out.reshape((nx,) + self.d_shape)
        return out
