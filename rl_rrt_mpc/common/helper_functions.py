"""
    helper_functions.py

    Summary:
        Contains miscellaneous helper functions for the RL-RRT-MPC COLAV system.

    Author: Trym Tengesdal
"""
from pathlib import Path
from typing import Optional, Tuple

import casadi as csd
import numpy as np
import rl_rrt_mpc.common.file_utils as fu
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.common.paths as dp
import seacharts.enc as senc
import shapely.affinity as affinity
import shapely.geometry as geometry
import yaml
from scipy.stats import chi2


def decision_trajectories_from_solution(soln: np.ndarray, N: int, nu: int, nx: int, ns: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the input sequence U, state sequence X and the slack variable sequence S from the solution vector soln = w = [U.flattened, X.flattened, Sigma.flattened] from the optimization problem.

    Args:
        soln (np.ndarray): A solution vector from the optimization problem.
        N (int): The prediction horizon.
        nu (int): The input dimension
        nx (int): The state dimension
        ns (int): The slack variable dimension

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The input sequence U, state sequence X and the slack variable sequence S.
    """
    U = np.zeros((nu, N))
    X = np.zeros((nx, N + 1))
    Sigma = np.zeros((ns, N + 1))
    for k in range(N + 1):
        if k < N:
            U[:, k] = soln[k * nu : (k + 1) * nu].ravel()
        X[:, k] = soln[N * nu + k * nx : N * nu + (k + 1) * nx].ravel()
        Sigma[:, k] = soln[N * nu + (N + 1) * nx + k * ns : N * nu + (N + 1) * nx + (k + 1) * ns].ravel()
    return U, X, Sigma


def linestring_to_ndarray(line: geometry.LineString) -> np.ndarray:
    """Converts a shapely LineString to a numpy array

    Args:
        - line (LineString): Any LineString object

    Returns:
        np.ndarray: Numpy array containing the coordinates of the LineString
    """
    return np.array(line.coords).transpose()


def ndarray_to_linestring(array: np.ndarray) -> geometry.LineString:
    """Converts a 2D numpy array to a shapely LineString

    Args:
        - array (np.ndarray): Numpy array of 2 x n_samples, containing the coordinates of the LineString

    Returns:
        LineString: Any LineString object
    """
    assert array.shape[0] == 2 and array.shape[1] > 1, "Array must be 2 x n_samples with n_samples > 1"
    return geometry.LineString(list(zip(array[0, :], array[1, :])))


def casadi_matrix_from_nested_list(M: list):
    """Convenience function for making a ca.SX matrix from lists of lists
    (don't know why this doesn't exist already), the alternative is
    ca.vertcat(
        ca.horzcat(a,b,c),
        ca.horzcat(d,e,f),
        ...
    )

    Args:
        M (list): List of lists

    Returns:
        csd.SX: Casadi matrix
    """
    return csd.vertcat(*(csd.horzcat(*row) for row in M))


def casadi_matrix_from_vector(v: csd.SX.sym, n_rows: int, n_cols: int):
    llist = []
    for i in range(n_rows):
        nested_list = []
        for j in range(n_cols):
            nested_list.append(v[i * n_rows + j])
        llist.append(nested_list)
    return casadi_matrix_from_nested_list(llist)


def load_rrt_solution(save_file: Path = dp.rrt_solution) -> dict:
    return fu.read_yaml_into_dict(save_file)


def save_rrt_solution(rrt_solution: dict, save_file: Path = dp.rrt_solution) -> None:
    save_file.touch(exist_ok=True)
    with save_file.open(mode="w") as file:
        yaml.dump(rrt_solution, file)


def translate_dynamic_obstacle_coordinates(dynamic_obstacles: list, x_shift: float, y_shift: float) -> list:
    """Translates the coordinates of a list of dynamic obstacles by (-y_shift, -x_shift)

    Args:
        dynamic_obstacles (list): List of dynamic obstacle objects on the form (ID, state, cov, length, width)
        x_shift (float): Easting shift
        y_shift (float): Northing shift

    Returns:
        list: List of dynamic obstacles with shifted coordinates
    """
    translated_dynamic_obstacles = []
    for (ID, state, cov, length, width) in dynamic_obstacles:
        translated_state = state - np.array([y_shift, x_shift, 0.0, 0.0])
        translated_dynamic_obstacles.append((ID, translated_state, cov, length, width))
    return translated_dynamic_obstacles


def translate_polygons(polygons: list, x_shift: float, y_shift: float) -> list:
    """Shifts the coordinates of a list of polygons by (-x_shift, -y_shift)

    Args:
        polygons (list): List of shapely polygons
        x_shift (float): Shift easting
        y_shift (float): Shift northing

    Returns:
        list: List of shifted polygons
    """
    translated_polygons = []
    for polygon in polygons:
        translated_polygon = affinity.translate(polygon, xoff=-x_shift, yoff=-y_shift)
        translated_polygons.append(translated_polygon)
    return translated_polygons


def create_ellipse(center: np.ndarray, A: Optional[np.ndarray] = None, a: float | None = 1.0, b: float | None = 1.0, phi: float | None = 0.0) -> Tuple[list, list]:
    """Create standard ellipse at center, with input semi-major axis, semi-minor axis and angle.

    Either specified by c, A or c, a, b, phi:

    (p - c)^T A (p - c) = 1

    or

    (p - c)^T R^T D R (p - c) = 1

    with R = R(phi) and D = diag(1 / a^2, 1 / b^2)


    Args:
        - center (np.ndarray): Center of ellipse
        - A (Optional[np.ndarray], optional): Hessian matrix. Defaults to None.
        - a (float | None, optional): Semi-major axis. Defaults to 1.0.
        - b (float | None, optional): Semi-minor axis. Defaults to 1.0.
        - phi (float | None, optional): Angle. Defaults to 0.0.


    Returns:
        Tuple[list, list]: List of x and y coordinates
    """

    if A is not None:
        # eigenvalues and eigenvectors of the covariance matrix
        eigenval, eigenvec = np.linalg.eig(A[0:2, 0:2])

        largest_eigenval = max(eigenval)
        largest_eigenvec_idx = np.argwhere(eigenval == max(eigenval))[0][0]
        largest_eigenvec = eigenvec[:, largest_eigenvec_idx]

        smallest_eigenval = min(eigenval)
        # if largest_eigenvec_idx == 0:
        #     smallest_eigenvec = eigenvec[:, 1]
        # else:
        #     smallest_eigenvec = eigenvec[:, 0]

        angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
        angle = mf.wrap_angle_to_02pi(angle)

        a = np.sqrt(largest_eigenval)
        b = np.sqrt(smallest_eigenval)
    else:
        angle = phi

    # the ellipse in "body" x and y coordinates
    t = np.linspace(0, 2.01 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)

    R = mf.Rpsi2D(angle)

    # Rotate to NED by angle phi, N_ell_points x 2
    ellipse_xy = np.array([x, y])
    for i in range(ellipse_xy.shape[1]):
        ellipse_xy[:, i] = R @ ellipse_xy[:, i] + center

    return ellipse_xy[0, :].tolist(), ellipse_xy[1, :].tolist()


def create_probability_ellipse(P: np.ndarray, probability: float = 0.99) -> Tuple[list, list]:
    """Creates a probability ellipse for a covariance matrix P and a given
    confidence level (default 0.99).

    Args:
        P (np.ndarray): Covariance matrix
        probability (float, optional): Confidence level. Defaults to 0.99.

    Returns:
        np.ndarray: Ellipse data in x and y coordinates
    """

    # eigenvalues and eigenvectors of the covariance matrix
    eigenval, eigenvec = np.linalg.eig(P[0:2, 0:2])

    largest_eigenval = max(eigenval)
    largest_eigenvec_idx = np.argwhere(eigenval == max(eigenval))[0][0]
    largest_eigenvec = eigenvec[:, largest_eigenvec_idx]

    smallest_eigenval = min(eigenval)
    # if largest_eigenvec_idx == 0:
    #     smallest_eigenvec = eigenvec[:, 1]
    # else:
    #     smallest_eigenvec = eigenvec[:, 0]

    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
    angle = mf.wrap_angle_to_02pi(angle)

    # Get the ellipse scaling factor based on the confidence level
    chisquare_val = chi2.ppf(q=probability, df=2)

    a = chisquare_val * np.sqrt(largest_eigenval)
    b = chisquare_val * np.sqrt(smallest_eigenval)

    # the ellipse in "body" x and y coordinates
    t = np.linspace(0, 2.01 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)

    R = mf.Rpsi2D(angle)

    # Rotate to NED by angle phi, N_ell_points x 2
    ellipse_xy = np.array([x, y])
    for i in range(len(ellipse_xy)):
        ellipse_xy[:, i] = R @ ellipse_xy[:, i]

    return ellipse_xy[0, :].tolist(), ellipse_xy[1, :].tolist()


def plot_trajectory(trajectory: np.ndarray, enc: senc.ENC, color: str) -> None:
    enc.start_display()
    trajectory_line = []
    for k in range(trajectory.shape[1]):
        trajectory_line.append((trajectory[1, k], trajectory[0, k]))
    enc.draw_line(trajectory_line, color=color, width=0.5, thickness=0.5, marker_type=None)


def plot_dynamic_obstacles(dynamic_obstacles: list, enc: senc.ENC, T: float, dt: float) -> None:
    N = int(T / dt)
    enc.start_display()
    for (ID, state, cov, length, width) in dynamic_obstacles:
        ellipse_x, ellipse_y = create_probability_ellipse(cov, 0.99)
        ell_geometry = geometry.Polygon(zip(ellipse_y + state[1], ellipse_x + state[0]))
        enc.draw_polygon(ell_geometry, color="orange", alpha=0.3)

        for k in range(0, N, 10):
            do_poly = create_ship_polygon(
                state[0] + k * dt * state[2], state[1] + k * dt * state[3], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0
            )
            enc.draw_polygon(do_poly, color="red")
        do_poly = create_ship_polygon(state[0], state[1], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0)
        enc.draw_polygon(do_poly, color="red")


def plot_rrt_tree(node_list: list, enc: senc.ENC) -> None:
    enc.start_display()
    for node in node_list:
        enc.draw_circle((node["state"][1], node["state"][0]), 5.0, color="black", fill=True, thickness=1.0, edge_style=None)
        for sub_node in node_list:
            if node["id"] == sub_node["id"] or sub_node["parent_id"] != node["id"]:
                continue
            enc.draw_line([(node["state"][1], node["state"][0]), (sub_node["state"][1], sub_node["state"][0])], color="white", width=0.5, thickness=0.5, marker_type=None)


def create_ship_polygon(x: float, y: float, heading: float, length: float, width: float, length_scaling: float = 1.0, width_scaling: float = 1.0) -> geometry.Polygon:
    """Creates a ship polygon from the ship`s position, heading, length and width.

    Args:
        x (float): The ship`s north position
        y (float): The ship`s east position
        heading (float): The ship`s heading
        length (float): Length of the ship
        width (float): Width of the ship
        length_scaling (float, optional): Length scale factor. Defaults to 1.0.
        width_scaling (float, optional): Length scale factor. Defaults to 1.0.

    Returns:
        np.ndarray: Ship polygon
    """
    eff_length = length * length_scaling
    eff_width = width * width_scaling

    x_min, x_max = x - eff_length / 2.0, x + eff_length / 2.0 - eff_width
    y_min, y_max = y - eff_width / 2.0, y + eff_width / 2.0
    left_aft, right_aft = (y_min, x_min), (y_max, x_min)
    left_bow, right_bow = (y_min, x_max), (y_max, x_max)
    coords = [left_aft, left_bow, (y, x + eff_length / 2.0), right_bow, right_aft]
    poly = geometry.Polygon(coords)
    return affinity.rotate(poly, -heading, origin=(y, x), use_radians=True)
