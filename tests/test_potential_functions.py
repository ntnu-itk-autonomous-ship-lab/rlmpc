"""Test module for gym.py

    Shows how to use the gym environment, and how to save a video + gif of the simulation.
"""
from pathlib import Path

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage{bm}",
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}",
            ]
        ),
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{bm}",
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}",
            ]
        ),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def f(x: float) -> float:
    return x / np.sqrt(x**2 + 1)


def gw_potential(p: np.ndarray, alpha: np.ndarray, y_0: float) -> float:
    """Calculates the potential function for the give-way situation.

    Args:
        p (np.ndarray): Position relative to the TS body-fixed frame.
        alpha (np.ndarray): Parameter adjusting the steepness.
        y_0 (float): Parameter adjusting the attenuation on the side of the ship.

    Returns:
        float: The potential function value.
    """
    return f(alpha[0] * p[0]) * (f(alpha[1] * (p[1] - y_0)) + 1)


def ho_potential(p: np.ndarray, alpha: np.ndarray, x_0: float) -> float:
    """Calculates the potential function for the head-on situation.

    Args:
        p (np.ndarray): Position relative to the TS body-fixed frame.
        alpha (np.ndarray): Parameter adjusting the steepness.
        x_0 (float): Parameter adjusting the attenuation on the ship front.

    Returns:
        float: The potential function value.
    """
    return 0.5 * (f(alpha[0] * (x_0 - p[0])) + 1) * f(alpha[1] * p[1])


def ot_potential(p: np.ndarray, alpha: np.ndarray, x_0: float, y_0: float) -> float:
    """Calculates the potential function for the give-way situation.

    Args:
        p (np.ndarray): Position relative to the TS body-fixed frame.
        alpha (np.ndarray): Parameter adjusting the steepness.
        x_0 (float): Parameter adjusting the attenuation on the ship front.
        y_0 (float): Parameter adjusting the attenuation on the side of the ship.


    Returns:
        float: The potential function value.
    """
    return 0.5 * f(alpha[0] * (x_0 - p[0])) * f(alpha[1] * np.abs(p[1] - y_0))


if __name__ == "__main__":

    npy = 100
    npx = 100
    y = np.linspace(-1500, 1500, npy)
    x = np.linspace(-1500, 1500, npx)
    Y, X = np.meshgrid(y, x, indexing="ij")

    alpha_gw = np.array([1 / 500, 1 / 500])
    y_0_gw = -500.0

    alpha_ho = np.array([1 / 500, 1 / 500])
    x_0_ho = 1000.0

    alpha_ot = np.array([1 / 500, 1 / 500])
    x_0_ot = 500.0
    y_0_ot = 500.0

    gw_surface = np.zeros((npy, npx))
    ho_surface = np.zeros((npy, npx))
    ot_surface = np.zeros((npy, npx))

    heading_gw = -np.pi / 2
    heading_ho = np.pi
    heading_ot = 0.0

    p_ts = np.array([0.0, 0.0])

    for i in range(npy):
        for j in range(npx):
            p = np.array([x[j], y[i]])
            p_rel = mf.Rmtrx2D(heading_gw).T @ (p - p_ts)
            gw_surface[i, j] = gw_potential(p_rel, alpha_gw, y_0_gw)

            p_rel = mf.Rmtrx2D(heading_ho).T @ (p - p_ts)
            ho_surface[i, j] = ho_potential(p_rel, alpha_ho, x_0_ho)

            p_rel = mf.Rmtrx2D(heading_ot).T @ (p - p_ts)
            ot_surface[i, j] = ot_potential(p_rel, alpha_ot, x_0_ot, y_0_ot)

    # Plot the potential functions
    ax1 = plt.figure().add_subplot(111, projection="3d")
    ax1.plot_surface(Y, X, gw_surface, cmap=cm.coolwarm)
    ax1.set_ylabel("North [m]")
    ax1.set_xlabel("East [m]")
    ax1.set_zlabel(r"$h_{gw}(\bm{p}_{rel})$")

    fig2, ax2 = plt.subplots()
    pc2 = ax2.contourf(Y, X, gw_surface, cmap=cm.coolwarm)
    gw_poly = mapf.create_ship_polygon(0, 0, 2.0 * heading_gw, 500, 100)
    x, y = gw_poly.exterior.xy
    cbar2 = fig2.colorbar(pc2)
    ax2.fill(y, x, alpha=1, fc="b", ec="none")
    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")

    ax3 = plt.figure().add_subplot(111, projection="3d")
    ax3.plot_surface(Y, X, ho_surface, cmap=cm.coolwarm)
    ax3.set_ylabel("North [m]")
    ax3.set_xlabel("East [m]")
    ax3.set_zlabel(r"$h_{ho}(\bm{p}_{rel})$")

    fig4, ax4 = plt.subplots()
    pc4 = ax4.contourf(Y, X, ho_surface, cmap=cm.coolwarm)
    ho_poly = mapf.create_ship_polygon(0, 0, heading_ho + np.pi / 2.0, 500, 100)
    x, y = ho_poly.exterior.xy
    cbar4 = fig4.colorbar(pc4)
    ax4.fill(y, x, alpha=1, fc="b", ec="none")
    ax4.set_xlabel("East [m]")
    ax4.set_ylabel("North [m]")

    ax5 = plt.figure().add_subplot(111, projection="3d")
    ax5.plot_surface(Y, X, ot_surface, cmap=cm.coolwarm)
    ax5.set_ylabel("North [m]")
    ax5.set_xlabel("East [m]")
    ax5.set_zlabel(r"$h_{ot}(\bm{p}_{rel})$")

    fig6, ax6 = plt.subplots()
    pc6 = ax6.contourf(Y, X, ot_surface, cmap=cm.coolwarm)
    ot_poly = mapf.create_ship_polygon(0, 0, heading_ot + np.pi / 2.0, 500, 100)
    x, y = ot_poly.exterior.xy
    cbar6 = fig6.colorbar(pc6)
    ax6.fill(y, x, alpha=1, fc="b", ec="none")
    ax6.set_xlabel("East [m]")
    ax6.set_ylabel("North [m]")
    plt.show(block=False)
    print("done")
