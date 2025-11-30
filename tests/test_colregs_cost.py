import numpy as np
import pytest

import rlmpc.mpc.common as mpc_common


def test_colregs_cost() -> None:
    inf_val = mpc_common.potential_field_base_function(-np.inf)

    npy = 100
    npx = 100
    y = np.linspace(-1500, 1500, npy)
    x = np.linspace(-1500, 1500, npx)
    Y, X = np.meshgrid(y, x, indexing="ij")

    alpha_cr = [0.03, 0.002]
    y_0_cr = 400.0
    alpha_ho = [0.002, 0.03]
    x_0_ho = 400.0
    alpha_ot = [0.005, 0.01]
    x_0_ot = 300.0
    y_0_ot = 100.0
    d_attenuation = 400.0
    colregs_weights = [70.0, 70.0, 25.0]

    xs = np.array([6574298.6, -30098.26, -1.78, 4.6, 0.00001, 4.6])
    xs_rel = np.array([0.0, 0.0, -1.78, 4.6])

    xs_target = np.array(
        [6574223.59832493 - xs[0], -30497.98151476 - xs[1], 0.7897036, 3.38688909]
    )
    chi_target = np.arctan2(xs_target[3], xs_target[2])
    U_target = np.sqrt(xs_target[2] ** 2 + xs_target[3] ** 2)
    xs_ts = np.array([xs_target[0], xs_target[1], chi_target, U_target, 2.0, 1.0])

    xs_ts_inactive = np.array([0.0 - 1e10, 0.0 - 1e10, 0.0, 0.0, 10.0, 2.0])

    xs_do_ho = np.array([-83.9337, -447.2138, 0.7897, 3.3869, 10.0, 3.0])
    U_do_ho = np.sqrt(xs_do_ho[2] ** 2 + xs_do_ho[3] ** 2)
    chi_do_ho = np.arctan2(xs_do_ho[3], xs_do_ho[2])
    xs_do_ho[3] = U_do_ho
    xs_do_ho[2] = chi_do_ho

    colregs_cost, cr_cost, ho_cost, ot_cost = mpc_common.colregs_cost(
        x=xs_rel,
        X_do_cr=xs_ts_inactive,
        X_do_ho=xs_do_ho,
        X_do_ot=xs_ts_inactive,
        nx_do=6,
        alpha_cr=alpha_cr,
        y_0_cr=y_0_cr,
        alpha_ho=alpha_ho,
        x_0_ho=x_0_ho,
        alpha_ot=alpha_ot,
        x_0_ot=x_0_ot,
        y_0_ot=y_0_ot,
        d_attenuation=d_attenuation,
        weights=colregs_weights,
    )
    assert colregs_cost > 0.0
    assert cr_cost == pytest.approx(0.0)
    assert ho_cost == pytest.approx(colregs_cost)
    assert ot_cost == pytest.approx(0.0)


if __name__ == "__main__":
    test_colregs_cost()
