"""
    mpc_interface.py

    Summary:
        Contains the interface class for MPC-based COLAV algorithms.


    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import seacharts.enc as senc


class IMPC(ABC):
    @abstractmethod
    def construct_ocp(
        self,
        nominal_trajectory: np.ndarray,
        nominal_inputs: Optional[np.ndarray],
        xs: np.ndarray,
        do_list: list,
        so_list: list,
        enc: senc.ENC,
        map_origin: np.ndarray = np.array([0.0, 0.0]),
        min_depth: int = 5,
    ) -> None:
        """Constructs the Optimal Control Problem (OCP) for the MPC COLAV algorithm.

        Args:
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track or path to follow.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state of the ownship.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of static obstacle Polygon objects.
            - enc (senc.ENC): ENC object containing information about the ENC.
            - map_origin (np.ndarray, optional): Origin of the map. Defaults to np.array([0.0, 0.0]).
            - min_depth (int, optional): Minimum allowable depth for the vessel. Defaults to 5.
        """

    @abstractmethod
    def plan(
        self, t: float, nominal_trajectory: np.ndarray, nominal_inputs: Optional[np.ndarray], xs: np.ndarray, do_list: list, so_list: list, enc: senc.ENC, **kwargs
    ) -> dict:
        """Plans a static and dynamic obstacle free trajectory for the ownship.

        Args:
            - t (float): Current time.
            - nominal_trajectory (np.ndarray): Nominal reference trajectory to track (position in NED and velocity in BODY) or path to follow.
            - nominal_inputs (Optional[np.ndarray]): Nominal reference inputs used if time parameterized trajectory tracking is selected.
            - xs (np.ndarray): Current state.
            - do_list (list): List of dynamic obstacle info on the form (ID, state, cov, length, width).
            - so_list (list): List of ALL static obstacle Polygon objects.
            - enc (senc.ENC): Electronic Navigational Chart object.
            - **kwargs: Additional keyword arguments which depends on the static obstacle constraint type used.

        Returns:
            - dict: Dictionary containing the optimal trajectory, inputs, slacks and solver stats.
        """
