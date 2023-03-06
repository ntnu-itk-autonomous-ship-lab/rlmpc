"""
    helper_functions.py

    Summary:
        Contains miscellaneous helper functions for the RL-RRT-MPC COLAV system.

    Author: Trym Tengesdal
"""
from typing import Optional, Tuple

import seacharts.enc as senc


def plot_rrt_tree(node_list: list, enc: senc.ENC) -> None:

    enc.start_display()

    for node in node_list:
        enc.draw_circle((node["state"][1], node["state"][0]), 5.0, color="black", fill=True, thickness=1.0, edge_style=None)
        for sub_node in node_list:
            if node["id"] == sub_node["id"] or sub_node["parent_id"] != node["id"]:
                continue

            enc.draw_line(
                [(node["state"][1], node["state"][0]), (sub_node["state"][1], sub_node["state"][0])], color="white", width=1.0, thickness=1.0, marker_type=None
            )
