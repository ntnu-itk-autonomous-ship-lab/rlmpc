"""
"""

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.mpc.common as mpc_common
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

if __name__ == "__main__":
    inf_val = mpc_common.potential_field_base_function(-np.inf)

    npy = 100
    npx = 100
    y = np.linspace(-1000, 1000, npy)
    x = np.linspace(-1000, 1000, npx)
    Y, X = np.meshgrid(y, x, indexing="ij")

    alpha_cr = [0.03, 0.002]
    y_0_cr = 400.0
    alpha_ho = [0.002, 0.03]
    x_0_ho = 400.0
    alpha_ot = [0.005, 0.01]
    x_0_ot = 300.0
    y_0_ot = 100.0

    # alpha_cr = np.array([1 / 500, 1 / 500])
    # y_0_cr = -500.0

    # alpha_ho = np.array([1 / 500, 1 / 500])
    # x_0_ho = 1000.0

    # alpha_ot = np.array([1 / 500, 1 / 500])
    # x_0_ot = 500.0
    # y_0_ot = 500.0

    cr_surface = np.zeros((npy, npx))
    ho_surface = np.zeros((npy, npx))
    ot_surface = np.zeros((npy, npx))

    heading_cr = -np.pi / 2
    heading_ho = np.pi  # 1.3417258721756462  # np.pi
    heading_ot = 0.0  # 0.0

    d_factor = 400.0

    p_ts = np.array([0.0, 0.0])
    # p_ts = np.array([6574223.5983, -30497.9815])
    for i in range(npy):
        for j in range(npx):
            p = np.array([x[j], y[i]])
            # p = np.array([6574298.6177, -30398.2644])
            p_rel = mf.Rmtrx2D(heading_cr).T @ (p - p_ts)
            d_rel = np.linalg.norm(p_rel)
            cr_surface[i, j] = mpc_common.cr_potential(p_rel, alpha_cr, y_0_cr) * np.exp(-d_rel / d_factor)

            p_rel = mf.Rmtrx2D(heading_ho).T @ (p - p_ts)
            d_rel = np.linalg.norm(p_rel)
            ho_surface[i, j] = mpc_common.ho_potential(p_rel, alpha_ho, x_0_ho) * np.exp(-d_rel / d_factor)

            p_rel = mf.Rmtrx2D(heading_ot).T @ (p - p_ts)
            d_rel = np.linalg.norm(p_rel)
            ot_surface[i, j] = mpc_common.ot_potential(p_rel, alpha_ot, x_0_ot, y_0_ot) * np.exp(-d_rel / d_factor)

    # Plot the potential functions
    colormap = cm.inferno
    # ax1 = plt.figure().add_subplot(111, projection="3d")
    # ax1.plot_surface(Y, X, cr_surface, cmap=colormap)
    # ax1.set_ylabel("North [m]")
    # ax1.set_xlabel("East [m]")
    # ax1.set_zlabel(r"$h_{gw}(\bm{p}_{rel})$")
    ship_length = 250
    ship_width = 50
    fig2, ax2 = plt.subplots()
    pc2 = ax2.contourf(Y, X, cr_surface, cmap=colormap)
    gw_poly = mapf.create_ship_polygon(p_ts[0], p_ts[1], heading_cr, ship_length, ship_width)
    y, x = gw_poly.exterior.xy
    cbar2 = fig2.colorbar(pc2)
    ax2.fill(y, x, alpha=1, fc="g", ec="none")
    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")

    # ax3 = plt.figure().add_subplot(111, projection="3d")
    # ax3.plot_surface(Y, X, ho_surface, cmap=colormap)
    # ax3.set_ylabel("North [m]")
    # ax3.set_xlabel("East [m]")
    # ax3.set_zlabel(r"$h_{ho}(\bm{p}_{rel})$")

    fig4, ax4 = plt.subplots()
    pc4 = ax4.contourf(Y, X, ho_surface, cmap=colormap)
    ho_poly = mapf.create_ship_polygon(p_ts[0], p_ts[1], heading_ho, ship_length, ship_width)
    y, x = ho_poly.exterior.xy
    cbar4 = fig4.colorbar(pc4)
    ax4.fill(y, x, alpha=1, fc="g", ec="none")
    ax4.set_xlabel("East [m]")
    ax4.set_ylabel("North [m]")

    # ax5 = plt.figure().add_subplot(111, projection="3d")
    # ax5.plot_surface(Y, X, ot_surface, cmap=colormap)
    # ax5.set_ylabel("North [m]")
    # ax5.set_xlabel("East [m]")
    # ax5.set_zlabel(r"$h_{ot}(\bm{p}_{rel})$")

    fig6, ax6 = plt.subplots()
    pc6 = ax6.contourf(Y, X, ot_surface, cmap=colormap)
    ot_poly = mapf.create_ship_polygon(p_ts[0], p_ts[1], heading_ot, ship_length, ship_width)
    y, x = ot_poly.exterior.xy
    cbar6 = fig6.colorbar(pc6)
    ax6.fill(y, x, alpha=1, fc="g", ec="none")
    ax6.set_xlabel("East [m]")
    ax6.set_ylabel("North [m]")
    plt.tight_layout()
    plt.show(block=False)
    print("done")
