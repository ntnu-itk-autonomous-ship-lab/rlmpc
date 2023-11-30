"""Test module for gym.py

    Shows how to use the gym environment, and how to save a video + gif of the simulation.
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
    y = np.linspace(-1500, 1500, npy)
    x = np.linspace(-1500, 1500, npx)
    Y, X = np.meshgrid(y, x, indexing="ij")

    alpha_cr = np.array([1 / 500, 1 / 500])
    y_0_cr = -500.0

    alpha_ho = np.array([1 / 500, 1 / 500])
    x_0_ho = 1000.0

    alpha_ot = np.array([1 / 500, 1 / 500])
    x_0_ot = 500.0
    y_0_ot = 500.0

    colregs_weights = [1.0, 1.0, 1.0]

    xs = np.array([6574298.6, -30098.26, -1.78, 4.6, 0.00001, 4.6])
    xs_rel = np.array([0.0, 0.0, -1.78, 4.6, 0.00001, 4.6])
    xs_target = np.array([6574223.59832493 - xs[0], -30497.98151476 - xs[1], 0.7897036, 3.38688909])
    chi_target = np.arctan2(xs_target[3], xs_target[2])
    U_target = np.sqrt(xs_target[2] ** 2 + xs_target[3] ** 2)
    xs_ts = np.array([xs_target[0], xs_target[1], chi_target, U_target, 10.0, 2.0, 1.0])

    xs_ts_inactive = np.array([0.0 - 10000.0, 0.0 - 10000.0, 0.0, 0.0, 10.0, 2.0, 0.0])

    colregs_cost, cr_cost, ho_cost, ot_cost = mpc_common.colregs_cost(
        xs_rel,
        xs_ts_inactive,
        xs_ts,
        xs_ts_inactive,
        7,
        alpha_cr,
        y_0_cr,
        alpha_ho,
        x_0_ho,
        alpha_ot,
        x_0_ot,
        y_0_ot,
        colregs_weights,
    )

    print("done")
