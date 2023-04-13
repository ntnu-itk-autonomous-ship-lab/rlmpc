"""
    helper_functions.py

    Summary:
        Contains miscellaneous helper functions for the RL-RRT-MPC COLAV system.

    Author: Trym Tengesdal
"""
from pathlib import Path
from typing import Optional, Tuple

import casadi as csd
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.file_utils as fu
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.common.paths as dp
import scipy.interpolate as scipyintp
import seacharts.enc as senc
import shapely.affinity as affinity
import yaml
from matplotlib import cm
from scipy.stats import chi2
from shapely.geometry import Polygon


def compute_surface_approximations_from_polygons(polygons: list, enc: Optional[senc.ENC] = None, show_plots: bool = True) -> list:
    """Computes smooth 2D surface approximations from the input polygon list.

    Args:
        polygons (list): List of shapely polygons
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        list: List of surface approximations for each polygon.
    """
    surfaces = []
    npx_min = 15
    npy_min = 15
    for j, polygon in enumerate(polygons):
        # Sjå på CDL for å forenkle problemet.
        # Finne "kystlinjepolygons"
        # rekne ut hensiktsmessig npx og npy basert på n_vertices og polygon.bounds
        n_vertices = len(polygon.exterior.coords)
        npx = int(max(npx_min, n_vertices / 2))
        npy = int(max(npy_min, n_vertices / 2))
        poly_min_east, poly_min_north, poly_max_east, poly_max_north = polygon.buffer(2.0).bounds
        north_coords = np.linspace(start=poly_min_north, stop=poly_max_north, num=npx)
        east_coords = np.linspace(start=poly_min_east, stop=poly_max_east, num=npy)
        X, Y = np.meshgrid(north_coords, east_coords, indexing="ij")
        map_coords = np.hstack((Y.reshape(-1, 1), X.reshape(-1, 1)))
        poly_path = mpath.Path(polygon.buffer(0.2).exterior.coords)
        mask = poly_path.contains_points(points=map_coords, radius=0.1)
        mask = mask.astype(float).reshape((npy, npx))
        mask[mask > 0.0] = -1.0
        polygon_surface = csd.interpolant("so_surface" + str(j), "bspline", [east_coords, north_coords], mask.ravel(order="F"))
        polygon_surface2_tck = scipyintp.bisplrep(Y, X, mask, s=0)
        surfaces.append(polygon_surface)
        surface_points = np.zeros((npy, npx))

        if show_plots:
            assert enc is not None
            poly = affinity.translate(polygon, xoff=enc.origin[0], yoff=enc.origin[1])
            enc.draw_polygon(poly, color="red")
            for i, east_coord in enumerate(east_coords):
                for j, north_coord in enumerate(north_coords):
                    surface_points[i, j] = polygon_surface([east_coord, north_coord])

            ax = plt.figure().add_subplot(111, projection="3d")
            ax.plot_surface(Y, X, surface_points, rcount=200, ccount=200, cmap=cm.coolwarm)
            ax.set_xlabel("East")
            ax.set_ylabel("North")
            ax.set_zlabel("Mask")

            extra_north_coords = np.linspace(start=poly_min_north, stop=poly_max_north, num=500)
            extra_east_coords = np.linspace(start=poly_min_east, stop=poly_max_east, num=500)
            yY, xX = np.meshgrid(extra_east_coords, extra_north_coords, indexing="ij")
            surface_coords2 = scipyintp.bisplev(yY[:, 0], xX[0, :], polygon_surface2_tck)

            # yY_new = yY[:-1, :-1] + np.diff(yY[:2, 0])[0] / 2.0
            # xX_new = xX[:-1, :-1] + np.diff(xX[0, :2])[0] / 2.0
            # surface_coords2 = scipyintp.bisplev(yY_new[:, 0], xX_new[0, :], polygon_surface2_tck)
            # plt.figure()
            # plt.pcolormesh(yY, xX, surface_coords2, shading="flat", cmap=cm.coolwarm)
            # plt.colorbar()
            ax2 = plt.figure().add_subplot(111, projection="3d")
            ax2.plot_surface(yY, xX, surface_coords2, rcount=200, ccount=200, cmap=cm.coolwarm)
            plt.show()
    return surfaces


def compute_splines_from_polygons(polygons: list, enc: Optional[senc.ENC] = None) -> Tuple[list, list]:
    """Computes splines from a list of polygons

    Args:
        polygons (list): List of shapely polygons
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

    Returns:
        Tuple[list, list]: List of tuples with splines for x and y, and similarly for the derivatives
    """
    splines = []
    spline_derivatives = []
    for polygon in polygons:
        east, north = polygon.exterior.xy
        if len(east) < 3:
            continue

        linspace = np.linspace(0.0, 1.0, len(east))
        spline_x = scipyintp.PchipInterpolator(linspace, north, extrapolate=False)
        spline_y = scipyintp.PchipInterpolator(linspace, east, extrapolate=False)
        splines.append((spline_x, spline_y))
        spline_derivatives.append((spline_x.derivative(), spline_y.derivative()))
        # if enc is not None:
        #     enc.start_display()
        #     x_spline_vals = spline_x(linspace)
        #     y_spline_vals = spline_y(linspace)
        #     pairs = list(zip(y_spline_vals, x_spline_vals))
        #     enc.draw_line(pairs, color="black", width=0)

    return splines, spline_derivatives


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


def shift_dynamic_obstacle_coordinates(dynamic_obstacles: list, x_shift: float, y_shift: float) -> list:
    """Shifts the coordinates of a list of dynamic obstacles by (-y_shift, -x_shift)

    Args:
        dynamic_obstacles (list): List of dynamic obstacle objects on the form (ID, state, cov, length, width)
        x_shift (float): Easting shift
        y_shift (float): Northing shift

    Returns:
        list: List of dynamic obstacles with shifted coordinates
    """
    shifted_dynamic_obstacles = []
    for (ID, state, cov, length, width) in dynamic_obstacles:
        shifted_state = state - np.array([y_shift, x_shift, 0.0, 0.0])
        shifted_dynamic_obstacles.append((ID, shifted_state, cov, length, width))
    return shifted_dynamic_obstacles


def shift_polygon_coordinates(polygons: list, x_shift: float, y_shift: float) -> list:
    """Shifts the coordinates of a list of polygons by (-x_shift, -y_shift)

    Args:
        polygons (list): List of shapely polygons
        x_shift (float): Shift easting
        y_shift (float): Shift northing

    Returns:
        list: List of shifted polygons
    """
    shifted_polygons = []
    for polygon in polygons:
        shifted_polygon = affinity.translate(polygon, xoff=-x_shift, yoff=-y_shift)
        shifted_polygons.append(shifted_polygon)
    return shifted_polygons


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


def plot_trajectory(trajectory: np.ndarray, times: np.ndarray, enc: senc.ENC, color: str) -> None:
    enc.start_display()
    trajectory_line = []
    for k in range(trajectory.shape[1]):
        trajectory_line.append((trajectory[1, k], trajectory[0, k]))
    enc.draw_line(trajectory_line, color=color, width=1.0, thickness=1.0, marker_type=None)


def plot_dynamic_obstacles(dynamic_obstacles: list, enc: senc.ENC, T: float, dt: float) -> None:
    N = int(T / dt)
    enc.start_display()
    for (ID, state, cov, length, width) in dynamic_obstacles:
        ellipse_x, ellipse_y = create_probability_ellipse(cov, 0.99)
        ell_geometry = Polygon(zip(ellipse_y + state[1], ellipse_x + state[0]))
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
            enc.draw_line([(node["state"][1], node["state"][0]), (sub_node["state"][1], sub_node["state"][0])], color="white", width=1.0, thickness=1.0, marker_type=None)


def create_ship_polygon(x: float, y: float, heading: float, length: float, width: float, length_scaling: float = 1.0, width_scaling: float = 1.0) -> Polygon:
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
    poly = Polygon(coords)
    return affinity.rotate(poly, -heading, origin=(y, x), use_radians=True)
