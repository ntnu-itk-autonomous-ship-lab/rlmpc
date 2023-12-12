"""
    colregs_handler.py

    Summary:
        Class for a COLREGS handler.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Tuple

import colav_simulator.common.math_functions as mf
import numpy as np


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
    d_critical_so: float = 100.0  # critical distance threshold where stand-on duties are aborted

    d_activation: float = 5000.0  # distance threshold for activation of COLREGS handler
    t_cpa_entry: float = 100.0  # time to CPA threshold for entry into COLREGS situation
    d_cpa_entry: float = 900.0  # distance to CPA threshold for entry into COLREGS situation

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Creates a parameters object from a dictionary."""
        params = cls(**config_dict)
        params.theta_critical_ot = np.deg2rad(params.theta_critical_ot)
        params.theta_critical_ho = np.deg2rad(params.theta_critical_ho)
        params.theta_critical_cr = np.deg2rad(params.theta_critical_cr)
        params.theta_ot_min = np.deg2rad(params.theta_ot_min)
        params.theta_ot_max = np.deg2rad(params.theta_ot_max)
        return params

    def to_dict(self) -> dict:
        """Converts the parameters to a dictionary."""
        output = asdict(self)
        output["theta_critical_ot"] = np.rad2deg(output["theta_critical_ot"])
        output["theta_critical_ho"] = np.rad2deg(output["theta_critical_ho"])
        output["theta_critical_cr"] = np.rad2deg(output["theta_critical_cr"])
        output["theta_ot_min"] = np.rad2deg(output["theta_ot_min"])
        output["theta_ot_max"] = np.rad2deg(output["theta_ot_max"])
        return output


class COLREGSHandler:
    def __init__(self, params: COLREGSHandlerParams = COLREGSHandlerParams()) -> None:
        self._params = params

        self._do_labels: list = []
        self._do_situations: list = []
        self._do_cr_list: list = []
        self._do_ho_list: list = []
        self._do_ot_list: list = []

    def handle(self, xs: np.ndarray, do_list: list) -> Tuple[list, list, list]:
        """Handles the current situation with own-ship state xs and list of dynamic obstacles in do_list.

        This means:
            - Separate dynamic obstacles by their relevance to the own-ship, into three lists: crossing, head-on and overtaking.
            - Ignore stand-on obstacles.
            - For each list, sort the dynamic obstacles by their distance/danger level to the own-ship.
            -

        Args:
            xs (np.ndarray): State of the own-ship on the form [x, y, chi, U]^T
            do_list (list): List of dynamic obstacles on the form (ID, state, cov, length, width)

        Returns:
            Tuple[list, list, list]: Tuple of the crossing, head-on and overtaking dynamic obstacle lists.
        """

        p_os = np.array([xs[0], xs[1]])
        v_os = np.array([xs[2] * np.cos(xs[3]), xs[2] * np.sin(xs[3])])
        for i, (ID, do_state, do_cov, length, width) in enumerate(do_list):
            p_do = do_state[0:2]
            v_do = do_state[2:]
            dist2do = float(np.linalg.norm(p_do - p_os))
            do_is_relevant = self.check_if_do_is_relevant(p_os, v_os, p_do, v_do)

            if ID not in self._do_labels and not do_is_relevant:
                continue

            situation, do_passed_by, os_passed_by = self.determine_applicable_rules(xs, do_state)

            if ID in self._do_labels and (do_passed_by or os_passed_by):
                print(f"Removed DO{i} | do_passed_by: {do_passed_by}, os_passed_by: {os_passed_by}")
                self._remove_do(ID)
                continue

            if ID in self._do_labels and do_is_relevant:
                self._update_do(ID, (ID, do_state, do_cov, length, width))
                continue

            if situation == COLREGSSituation.HO:
                self._do_ho_list.append((ID, do_state, do_cov, length, width))
            elif situation == COLREGSSituation.CRGW:
                self._do_cr_list.append((ID, do_state, do_cov, length, width))
            elif situation == COLREGSSituation.CRSO and dist2do < self._params.d_critical_so:
                self._do_cr_list.append((ID, do_state, do_cov, length, width))
            elif situation == COLREGSSituation.OTGW:
                self._do_ot_list.append((ID, do_state, do_cov, length, width))
            elif situation == COLREGSSituation.OTSO and dist2do < self._params.d_critical_so:
                self._do_ot_list.append((ID, do_state, do_cov, length, width))
            else:
                continue

            self._do_situations.append((ID, situation))
            self._do_labels.append(ID)
            print(
                f"DO{i} Added | Start situation: {situation.name}, do_passed_by: {do_passed_by}, os_passed_by: {os_passed_by}"
            )

        # sort do lists by distance to own-ship
        self._do_cr_list.sort(key=lambda x: np.linalg.norm(x[1][0:2] - p_os))
        self._do_ho_list.sort(key=lambda x: np.linalg.norm(x[1][0:2] - p_os))
        self._do_ot_list.sort(key=lambda x: np.linalg.norm(x[1][0:2] - p_os))
        # if len(self._do_cr_list) > 0:
        #     print(f"CR DOs: {self._do_cr_list}")
        # if len(self._do_ho_list) > 0:
        #     print(f"HO DOs: {self._do_ho_list}")
        # if len(self._do_ot_list) > 0:
        #     print(f"OT DOs: {self._do_ot_list}")

        return self._do_cr_list, self._do_ho_list, self._do_ot_list

    def _update_do(self, ID: int, do_info: Tuple[int, np.ndarray, np.ndarray, float, float]) -> None:
        """Updates the state of the dynamic obstacle with ID.

        Args:
            ID (int): ID of the dynamic obstacle to be updated.
            do_info (Tuple[int, np.ndarray, np.ndarray, float, float]): New information about the dynamic obstacle.
        """
        _, situation = self._get_do_situation(ID)
        if situation == COLREGSSituation.CRGW:
            for i, (do_ID, _, _, _, _) in enumerate(self._do_cr_list):
                if ID == do_ID:
                    self._do_cr_list[i] = do_info
                    break
        elif situation == COLREGSSituation.HO:
            for i, (do_ID, _, _, _, _) in enumerate(self._do_ho_list):
                if ID == do_ID:
                    self._do_ho_list[i] = do_info
                    break
        elif situation == COLREGSSituation.OTGW:
            for i, (do_ID, _, _, _, _) in enumerate(self._do_ot_list):
                if ID == do_ID:
                    self._do_ot_list[i] = do_info
                    break

    def _remove_do(self, ID: int) -> None:
        """Removes a dynamic obstacle from the lists.

        Args:
            ID (int): ID of the dynamic obstacle to be removed.
        """
        sit_idx, situation = self._get_do_situation(ID)
        if situation == COLREGSSituation.CRGW:
            for i, (do_ID, _, _, _, _) in enumerate(self._do_cr_list):
                if ID == do_ID:
                    self._do_cr_list.pop(i)
                    break
        elif situation == COLREGSSituation.HO:
            for i, (do_ID, _, _, _, _) in enumerate(self._do_ho_list):
                if ID == do_ID:
                    self._do_ho_list.pop(i)
                    break
        elif situation == COLREGSSituation.OTGW:
            for i, (do_ID, _, _, _, _) in enumerate(self._do_ot_list):
                if ID == do_ID:
                    self._do_ot_list.pop(i)
                    break
        self._do_labels.remove(ID)
        self._do_situations.pop(sit_idx)

    def _get_do_situation(self, ID: int) -> Tuple[int, COLREGSSituation]:
        """Returns the COLREGS situation of the dynamic obstacle with ID.

        Args:
            ID (int): ID of the dynamic obstacle.

        Returns:
            Optional[COLREGSSituation]: The COLREGS situation of the dynamic obstacle.
        """
        for i, (do_ID, situation) in enumerate(self._do_situations):
            if do_ID == ID:
                return i, situation
        raise ValueError(f"Dynamic obstacle with ID {ID} not found in the list of dynamic obstacles.")

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
        return (
            (t_cpa < self._params.t_cpa_entry)
            and (d_cpa < self._params.d_cpa_entry)
            and (np.linalg.norm(p_do - p_os) < self._params.d_activation)
        )

    def determine_applicable_rules(self, xs: np.ndarray, do_state: np.ndarray) -> Tuple[COLREGSSituation, int, int]:
        """Determine applicable COLREGS rule for vessel with regards to obstacle at sample index i.

        Args:
            xs (np.ndarray): State of the own-ship on the form [x, y, chi, U]^T
            do_state (np.ndarray): State of the obstacle on the form [x, y, Vx, Vy]^T

        Returns:
            Tuple[COLREGSSituation, int, int]: Tuple of the applicable rule, whether the obstacle is passed and whether the ownship is passed.
        """
        U_os = xs[3]
        chi_os = xs[2]
        U_do = np.linalg.norm(do_state[2:])
        chi_do = mf.wrap_angle_to_pmpi(np.arctan2(do_state[3], do_state[2]))
        v_do = np.array([U_do * np.cos(chi_do), U_do * np.sin(chi_do)])
        v_os = np.array([U_os * np.cos(chi_os), U_os * np.sin(chi_os)])
        dist2do = do_state[0:2] - xs[0:2]
        los = dist2do / np.linalg.norm(dist2do)

        # Relative bearing (obstacle as seen from own-ship)
        beta_180 = mf.wrap_angle_diff_to_pmpi(np.arctan2(dist2do[1], dist2do[0]), chi_os)
        beta = mf.wrap_angle_to_02pi(beta_180)

        # Contact angle (own-ship as seen from obstacle)
        alpha = mf.wrap_angle_diff_to_pmpi(np.arctan2(-dist2do[1], -dist2do[0]), chi_do)
        alpha_360 = mf.wrap_angle_to_02pi(alpha)

        situation = COLREGSSituation.NAR
        # OTSO if theta_ot_min < beta < theta_ot_max and abs(alpha) < theta_critical_ot and U_os < U_do
        if (
            (beta > self._params.theta_ot_min)
            and (beta < self._params.theta_ot_max)
            and (abs(alpha) < self._params.theta_critical_ot)
            and (U_os < U_do)
        ):
            situation = COLREGSSituation.OTSO
        # OTGW if theta_ot_min < alpha < theta_ot_max and abs(beta) < theta_critical_ot and U_os > U_do
        elif (
            (alpha_360 > self._params.theta_ot_min)
            and (alpha_360 < self._params.theta_ot_max)
            and (abs(beta_180) < self._params.theta_critical_ot)
            and (U_os > U_do)
        ):
            situation = COLREGSSituation.OTGW
        # HO if abs(beta) < theta_critical_ho and abs(alpha) < theta_critical_ho
        elif (abs(beta_180) < self._params.theta_critical_ho) and (abs(alpha) < self._params.theta_critical_ho):
            situation = COLREGSSituation.HO
        # CRSO if -theta_ot_min < alpha < theta_critical_ot and -theta_ot_min < beta < theta_critical_cr
        elif (0.0 < alpha_360 < self._params.theta_ot_min) and (
            -self._params.theta_ot_min < beta_180 < self._params.theta_critical_cr
        ):
            situation = COLREGSSituation.CRSO
        elif (0.0 <= beta <= self._params.theta_ot_min) and (
            -self._params.theta_ot_min <= alpha <= self._params.theta_critical_cr
        ):
            situation = COLREGSSituation.CRGW

        # print(
        #     f"distance os <-> do: {np.linalg.norm(dist2do)}, bearing os -> do: {180.0 * beta / np.pi}, bearing do -> os: {180.0 * alpha / np.pi}"
        # )

        do_passed_by = False
        os_passed_by = False
        if np.dot(v_do, -los) < np.cos(self._params.theta_ot_min) * np.linalg.norm(v_do):
            os_passed_by = True

        if np.dot(v_os, los) < np.cos(self._params.theta_ot_min) * np.linalg.norm(v_os):
            do_passed_by = True
        return situation, do_passed_by, os_passed_by
