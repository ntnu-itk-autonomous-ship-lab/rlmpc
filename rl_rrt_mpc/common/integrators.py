"""
    integrators.py

    Summary:
        Contains integrator functionality to be used with casadi.
        Inspired by Casadi example code and milliampere MPC code.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod

import casadi as csd
import numpy as np


class Integrator(ABC):
    @abstractmethod
    def __init__(self, x, p, ode, quad, h):
        """
        Prototype class for Optimal Control Problem (OCP) integrators.
        All subclasses should have the following arguments:
        Arguments
        ---------
        x:
            Initial states
        p:
            Parameters
        ode:
            Differential equation to be solved
        quad:
            Quadrature so be solved
        h:
            Integration step length

        """

    @abstractmethod
    def __call__(self, x, p):
        """
        Return the symbolic integration and constraints to add to the NLP
        Arguments
        ---------
        x:
            Initial states
        p:
            Parameters
        Returns
        -------
        tuple:
            Returns tuple containing integral of final state of ode and quad
            as well as lists of states, state bounds, constraint, and constraint bounds
        """


class CasadiInternal(Integrator):
    """
    Internal Casadi integrators
    """

    def __init__(self, x, p, ode, quad, h, integrator="cvodes"):
        """
        Keyword Arguments
        -----------------
        integrator:
            Casadi built-in integrators: cvodes, idas, collocation, oldcollocation, rk
        """

        # Initialize parameters
        dae = {"x": x, "p": p, "ode": ode, "quad": quad}
        opts = {"tf": h}
        self.F = csd.integrator("F", integrator, dae, opts)

    def __call__(self, x, p):
        Xk_end, J, _, _, _, _ = self.F(x, p, [], [], [], [])

        # Not a direct method, so the integration constraints are offloaded to external integrator
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        # Return final contributions, decision variabels and constraints
        return Xk_end, J, w, lbw, ubw, g, lbg, ubg


class ERK4(Integrator):
    """
    Fixed step Explicit Runge-Kutta 4 integrator
    """

    def __init__(self, x, p, ode, quad, h, M=1):
        """
        Keyword Arguments
        -----------------
        M:
            Number integrator steps

        """

        # Make function for ode and quaderature (to be integrated)
        f = csd.Function("f", [x, p], [ode, quad], ["x", "p"], ["ode", "quad"])

        # Initialize parameters
        X0 = csd.MX.sym("X0", x.shape)
        P = csd.MX.sym("P", p.shape)
        H = h
        DT = H / M

        # Set initial state and quadrature
        X = X0
        Q = 0

        # Create symbolic M step RK4 integrator
        for j in range(M):
            k1, k1_q = f(X, P)
            k2, k2_q = f(X + DT / 2 * k1, P)
            k3, k3_q = f(X + DT / 2 * k2, P)
            k4, k4_q = f(X + DT * k3, P)

            dX = DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            X = X + dX
            Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

        self.F = csd.Function("F", [X0, P], [X, Q], ["x0", "p"], ["xf", "qf"])

    def __call__(self, x, p):
        Xk_end, J = self.F(x, p)

        # Is a direct method, but the integration constraints has no integration constraints
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        # Return final contributions, decision variabels and constraints
        return Xk_end, J, w, lbw, ubw, g, lbg, ubg


class DirectCollocation(Integrator):
    """
    Direct collocation integrator
    """

    def __init__(self, x, p, ode, quad, h, d=3, lbx=None, ubx=None):
        """
        Keyword Arguments
        -----------------
        d:
            Number of collocation points
        """

        self.f = csd.Function("f", [x, p], [ode, quad], ["x", "p"], ["ode", "quad"])
        self.d = d
        self.h = h
        self.lbx = lbx
        self.ubx = ubx

    def __call__(self, x, p):
        Xk = x
        Uk = p
        J = 0
        k = "k"

        # Construct polinomial basis function
        B, C, D, P = construct_basis(self.d)

        # Create empty list for decision variabels and constraints
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        Xc = []
        # Create decision variabels for collocation points
        for j in range(self.d):
            Xkj = csd.MX.sym("X_" + str(k) + "_" + str(j), x.shape[0])
            Xc.append(Xkj)
            w.append(Xkj)
            if self.lbx is None:
                lbw.append([-np.inf] * x.shape[0])
            else:
                lbw.append(self.lbx)

            if self.ubx is None:
                ubw.append([np.inf] * x.shape[0])
            else:
                ubw.append(self.ubx)

        # Loop over collocation points to generate constraints
        Xk_end = D[0] * Xk
        for j in range(1, self.d + 1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j] * Xk
            for r in range(self.d):
                xp = xp + C[r + 1, j] * Xc[r]

            # Append collocation equations
            fj, qj = self.f(Xc[j - 1], Uk)
            g.append(self.h * fj - xp)
            lbg.append([0] * x.shape[0])
            ubg.append([0] * x.shape[0])

            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1]

            # Add contribution to quadrature function
            J = J + B[j] * qj * self.h

        # Return final contributions, decision variabels and constraints
        return Xk_end, J, w, lbw, ubw, g, lbg, ubg


def construct_basis(d: int):
    """
    Function constructs Legendre polinomial basis function for the collocation
    with polynomial degree d

    Arguments
    ---------
    d: Degree of polinomial

    Returns
    -------
    tuple:
        B: Coefficients of the collocation equation
        C: Coefficients of continuity equation
        D: Coefficient of quaderature
        p: Polinomial
    """
    # Get collocation points
    tau_root = np.append(0, csd.collocation_points(d, "legendre"))

    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    B = np.zeros(d + 1)
    P = []

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Legendre polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

        # Add polynomial to P
        P.append(p)

    return B, C, D, P
