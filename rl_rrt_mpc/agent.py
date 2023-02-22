"""
    rl_rrt_mpc.py

    Summary:
        Contains the main RL-RRT-MPC COLAV system class.

    Author: Trym Tengesdal
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import colav_simulator.core.colav.colav_interface as ci
import numpy as np
import rrt_rust as rrt
import seacharts.enc as senc


@dataclass
class RLRRTMPCParams:

    rl: Any
    rrt: Any
    mpc: Any

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RLRRTMPCParams(rl=None, rrt=None, mpc=None)
        config.rl = config_dict["rl"]
        config.rrt = config_dict["rrt"]
        config.mpc = config_dict["mpc"]
        return config


class RLRRTMPC(ci.ICOLAV):
    def __init__(self, config: RLRRTMPCParams | None) -> None:

        if config:
            self._rl = config.rl
            self._rrt = config.rrt
            self._mpc = config.mpc
        self._references = np.empty(9)
        self._initialized = False
        self._t_prev = 0.0

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[senc.ENC] = None,
        goal_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self._initialized:
            self._t_prev = t
            self._initialized = True

        references = np.zeros(9)
        return references

    def get_current_plan(self) -> np.ndarray:
        return self._references
