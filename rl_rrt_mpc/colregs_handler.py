"""
    colregs_handler.py

    Summary:
        Class for a COLREGS handler.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Tuple

import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as cs_mhm
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf


# class syntax
class COLREGSSituation(Enum):
    """Enum for the different possible COLREGS situations"""

    OTGW = 0  # The own-ship is overtaking the obstacle
    OTSO = 1  # The obstacle is overtaking the own-ship
    HO = 2  # Head-on situation
    CRGW = 3  # Crossing situation with own-ship as give-way
    CRSO = 4  # Crossing situation with own-ship as stand-on
    NAR = 5  # Not applicable (no situation), meaning the obstacle is not relevant to the own-ship


@dataclass
class COLREGSHandlerParams:
    theta_critical_ot: float = np.deg2rad(45.0)  # contact angle threshold for OT situation
    theta_critical_ho: float = np.deg2rad(13.0)  # contact and bearing angle threshold for HO situation
    theta_critical_cr: float = np.deg2rad(-13.0)  # contact and bearing angle threshold for CR situation
    theta_ot_min: float = np.deg2rad(112.5)  # minimum bearing angle for OT situation
    theta_ot_max: float = np.deg2rad(247.5)  # maximum bearing angle for OT situation

    d_activation: float = 5000.0  # distance threshold for activation of COLREGS handler
    t_cpa_entry: float = 100.0  # time to CPA threshold for entry into COLREGS situation
    d_cpa_entry: float = 100.0  # distance to CPA threshold for entry into COLREGS situation

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Creates a parameters object from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Converts the parameters to a dictionary."""
        return asdict(self)


class COLREGSHandler:
    def __init__(self, params: COLREGSHandlerParams = COLREGSHandlerParams()) -> None:
        self._params = params

        self._do_labels = []
        self._do_situations = []
        self._do_cr_list = []
        self._do_ho_list = []
        self._do_ot_list = []

    def handle(self, xs: np.ndarray, do_list: list) -> Tuple[list, list, list]:
        """Handles the current situation with own-ship state xs and list of dynamic obstacles in do_list.

        This means:
            - Separate dynamic obstacles by their relevance to the own-ship, into three lists: crossing, head-on and overtaking.
            - Ignore stand-on obstacles.
            - For each list, sort the dynamic obstacles by their distance/danger level to the own-ship.
            -

        Args:
            xs (np.ndarray): State of the own-ship on the form [x, y, U, chi]^T
            do_list (list): List of dynamic obstacles on the form (ID, state, cov, length, width)

        Returns:
            dict:
        """
        do_cr_list = []
        do_ho_list = []
        do_ot_list = []

        p_os = np.array([xs[0], xs[1]])
        v_os = np.array([xs[2] * np.cos(xs[3]), xs[2] * np.sin(xs[3])])
        chi_os = xs[3]
        for i, (ID, do_state, cov, length, width) in enumerate(do_list):

            p_do = do_state[0:2]
            v_do = do_state[2:]
            do_is_relevant = self.check_if_do_is_relevant(p_os, v_os, p_do, v_do)

            if ID not in self._do_labels and not do_is_relevant:
                continue

            if ID in self._do_labels and do_is_relevant:
                # Update the state of the dynamic obstacle in the list
                continue

            situation, do_passed_by, os_passed_by = self.determine_applicable_rules(xs, do_state)
            print(f"DO{i} | Situation: {situation.name}, do_passed_by: {do_passed_by}, os_passed_by: {os_passed_by}")

            if situation == COLREGSSituation.NAR or do_passed_by or os_passed_by:
                # remove the dynamic obstacle from the list
                self._do_labels.remove(ID)
                self._do_situations.remove(situation)

            if situation == COLREGSSituation.HO:
                do_ho_list.append((ID, do_state, cov, length, width))
            elif situation == COLREGSSituation.CRGW:
                do_cr_list.append((ID, do_state, cov, length, width))
            elif situation == COLREGSSituation.OTGW:
                do_ot_list.append((ID, do_state, cov, length, width))
            self._do_situations.append(situation)
            self._do_labels.append(i)

        return do_cr_list, do_ho_list, do_ot_list

    def check_if_do_is_relevant(self, p_os: np.ndarray, v_os: np.ndarray, p_do: np.ndarray, v_do: np.ndarray) -> bool:
        """Checks if the dynamic obstacle is relevant to the own-ship.

        Args:
            p_os (np.ndarray): Position of the own-ship on the form [x, y]^T
            v_os (np.ndarray): Velocity of the own-ship on the form [u, v]^T
            p_do (np.ndarray): Position of the dynamic obstacle on the form [x, y]^T
            v_do (np.ndarray): Velocity of the dynamic obstacle on the form [u, v]^T

        Returns:
            bool: True if the dynamic obstacle is relevant, False otherwise.
        """
        t_cpa, d_cpa = mf.cpa(p_os, v_os, p_do, v_do)
        return (t_cpa < self._params.t_cpa_entry) and (d_cpa < self._params.d_cpa_entry) and (np.linalg.norm(p_do - p_os) < self._params.d_activation)

    def determine_applicable_rules(self, xs: np.ndarray, do_state: np.ndarray) -> Tuple[COLREGSSituation, int, int]:
        """Determine applicable COLREGS rule for vessel with regards to obstacle at sample index i.

        Args:
            xs (np.ndarray): State of the own-ship on the form [x, y, U, chi]^T
            do_state (np.ndarray): State of the obstacle on the form [x, y, U, chi]^T

        Returns:
            Tuple[COLREGSSituation, int, int]: Tuple of the applicable rule, whether the obstacle is passed and whether the ownship is passed.
        """
        U_os = xs[2]
        chi_os = xs[3]
        U_do = do_state[2]
        chi_do = do_state[3]
        v_do = np.array([U_do * np.sin(chi_do), U_do * np.cos(chi_do)])
        v_os = np.array([U_os * np.sin(chi_os), U_os * np.cos(chi_os)])
        dist2do = do_state[0:2] - xs[0:2]
        los = dist2do / np.linalg.norm(dist2do)

        # Relative bearing (obstacle as seen from own-ship)
        beta_180 = mf.wrap_angle_diff_to_pmpi(np.arctan2(dist2do[0], dist2do[1]), chi_os)
        beta = mf.wrap_angle_to_02pi(beta_180)

        # Contact angle (own-ship as seen from obstacle)
        alpha = mf.wrap_angle_diff_to_pmpi(np.arctan2(-dist2do[0], -dist2do[1]), chi_do)
        alpha_360 = mf.wrap_angle_to_02pi(alpha)

        situation = COLREGSSituation.NAR
        # OTSO if theta_ot_min < beta < theta_ot_max and abs(alpha) < theta_critical_ot and U_os < U_do
        if (beta > self._params.theta_ot_min) and (beta < self._params.theta_ot_max) and (abs(alpha) < self._params.theta_critical_ot) and (U_os < U_do):
            situation = COLREGSSituation.OTSO
        # OTGW if theta_ot_min < alpha < theta_ot_max and abs(beta) < theta_critical_ot and U_os > U_do
        elif (alpha_360 > self._params.theta_ot_min) and (alpha_360 < self._params.theta_ot_max) and (abs(beta_180) < self._params.theta_critical_ot) and (U_os > U_do):
            situation = COLREGSSituation.OTGW
        # HO if abs(beta) < theta_critical_ho and abs(alpha) < theta_critical_ho
        elif (abs(beta_180) < self._params.theta_critical_ho) and (abs(alpha) < self._params.theta_critical_ho):
            situation = COLREGSSituation.HO
        # CRSO if -theta_ot_min < alpha < theta_critical_ot and -theta_ot_min < beta < theta_critical_cr
        elif (0.0 < alpha_360 < self._params.theta_ot_min) and (-self._params.theta_ot_min < beta_180 < self._params.theta_critical_cr):
            situation = COLREGSSituation.CRSO
        elif (0.0 < beta < self._params.theta_ot_min) and (-self._params.theta_ot_min < alpha < self._params.theta_critical_cr):
            situation = COLREGSSituation.CRGW

        do_passed_by = False
        os_passed_by = False
        if np.dot(v_do, -los) < np.cos(self._params.theta_ot_min) * np.linalg.norm(v_do):
            os_passed_by = True

        if np.dot(v_os, los) < np.cos(self._params.theta_ot_min) * np.linalg.norm(v_os):
            do_passed_by = True
        return situation, do_passed_by, os_passed_by
