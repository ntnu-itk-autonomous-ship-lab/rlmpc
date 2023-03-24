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
import rl_rrt_mpc.common.paths as dp
import seacharts.enc as senc
import shapely.affinity as affinity
import yaml
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Polygon


def compute_splines_from_polygons(polygons: list, enc: Optional[senc.ENC] = None) -> Tuple[list, list]:
    """Computes splines from a list of polygons

    Args:
        polygons (list): List of shapely polygons
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

    Returns:
        list: List of tuples with splines for x and y, and similarly for the derivatives
    """
    splines = []
    spline_derivatives = []
    for polygon in polygons:
        east, north = polygon.exterior.xy
        if len(east) < 3:
            continue

        linspace = np.linspace(0.0, 1.0, len(east))
        spline_x = PchipInterpolator(linspace, north, extrapolate=False)
        spline_y = PchipInterpolator(linspace, east, extrapolate=False)
        splines.append((spline_x, spline_y))
        spline_derivatives.append((spline_x.derivative(), spline_y.derivative()))
        if enc is not None:
            enc.start_display()
            x_spline_vals = spline_x(linspace)
            y_spline_vals = spline_y(linspace)
            pairs = list(zip(y_spline_vals, x_spline_vals))
            enc.draw_line(pairs, color="black", width=0)

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


def casadi_matrix_from_vector(v: csd.SX.sym):
    len_v = v.shape[0]
    n_rows = int(np.sqrt(len_v))
    llist = []
    for i in range(n_rows):
        nested_list = []
        for j in range(n_rows):
            nested_list.append(v[i * n_rows + j])
        llist.append(nested_list)
    return casadi_matrix_from_nested_list(llist)


def load_rrt_solution(save_file: Path = dp.rrt_solution) -> dict:
    return fu.read_yaml_into_dict(save_file)


def save_rrt_solution(rrt_solution: dict, save_file: Path = dp.rrt_solution) -> None:
    save_file.touch(exist_ok=True)
    with save_file.open(mode="w") as file:
        yaml.dump(rrt_solution, file)


def plot_rrt_solution(trajectory: np.ndarray, times: np.ndarray, enc: senc.ENC) -> None:
    enc.start_display()
    trajectory_line = []
    for k in range(trajectory.shape[1]):
        trajectory_line.append((trajectory[1, k], trajectory[0, k]))
    enc.draw_line(trajectory_line, color="magenta", width=1.0, thickness=1.0, marker_type=None)


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
