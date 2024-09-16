"""Test module for gym.py

    Shows how to use the gym environment, and how to save a video + gif of the simulation.
"""

from pathlib import Path

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import matplotlib.pyplot as plt
import numpy as np
import rlmpc.mpc.common as mpc_common
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


if __name__ == "__main__":
    n = 200
    a_max = 0.3
    r_max = np.deg2rad(8)
    a = np.linspace(-a_max, a_max, n)
    r = np.linspace(-r_max, r_max, n)

    # alpha_app = [112.5, 0.00006, 8.0, 0.001]
    alpha_app = [40.0, 0.001, 15.0, 0.001]
    # alpha_app = [40.0, 0.003, 40.0, 0.005]
    K_app = [60.0, 30.0]

    r_costs = np.zeros(n)
    a_costs = np.zeros(n)
    for i in range(n):
        rate_cost, r_cost, a_cost = mpc_common.rate_cost(
            r[i], a[i], alpha_app=alpha_app, K_app=K_app, a_max=a_max, r_max=r_max
        )
        r_costs[i] = r_cost
        a_costs[i] = a_cost

    # Plot
    fig = plt.figure(figsize=(12, 8))
    axes = fig.subplot_mosaic(
        [
            ["a"],
            ["r"],
        ]
    )
    axes["a"].plot(a, a_costs, color="b", label=r"a_{cost}")
    axes["a"].set_ylabel("cost")
    axes["a"].set_xlabel("m/s2")
    axes["a"].legend()

    axes["r"].plot(r, r_costs, color="r", label=r"r_{cost}")
    axes["r"].set_ylabel("cost")
    axes["r"].set_xlabel("rad/s")
    axes["r"].legend()

    fig.tight_layout()
    plt.show(block=False)
    print("done")
