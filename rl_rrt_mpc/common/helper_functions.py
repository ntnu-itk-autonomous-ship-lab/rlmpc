"""
    helper_functions.py

    Summary:
        Contains miscellaneous helper functions for the RL-RRT-MPC COLAV system.

    Author: Trym Tengesdal
"""
from typing import Optional, Tuple

import seacharts.enc as senc
import shapely.affinity as affinity
from shapely.geometry import Polygon


def plot_rrt_solution(states: list, times: list, enc: senc.ENC) -> None:
    enc.start_display()
    states_line = []
    for state in states:
        states_line.append((state[1], state[0]))
    enc.draw_line(states_line, color="magenta", width=1.0, thickness=1.0, marker_type=None)


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
