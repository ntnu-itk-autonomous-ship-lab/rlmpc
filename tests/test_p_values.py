"""Test module for gym.py

Shows how to use the gym environment, and how to save a video + gif of the simulation.
"""

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
    n = 100
    alpha = 0.05

    # t = x - mu  / s / sqrt(n)

    # case 1:
    #

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
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

    axes["r"].plot(r * 180.0 / np.pi, r_costs, color="r", label=r"r_{cost}")
    axes["r"].set_ylabel("cost")
    axes["r"].set_xlabel("deg/s")
    axes["r"].legend()

    fig.tight_layout()
    plt.show()
    print("done")
