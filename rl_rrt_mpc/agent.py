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
import rl_rrt_mpc.common.config_parsing as cp
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.mpc as mpc
import rl_rrt_mpc.rl as rl
import rrt_rust as rrt
import seacharts.enc as senc


@dataclass
class RRTParams:
    max_iter: int = 1000
    max_nodes: int = 10000
    max_time: float = 10.0
    step_size: float = 0.1
    alpha: float = 1.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTParams(
            max_iter=config_dict["max_iter"],
            max_nodes=config_dict["max_nodes"],
            max_time=config_dict["max_time"],
            step_size=config_dict["step_size"],
            alpha=config_dict["alpha"],
        )
        return config

    def to_dict(self):
        return {
            "max_iter": self.max_iter,
            "max_nodes": self.max_nodes,
            "max_time": self.max_time,
            "step_size": self.step_size,
            "alpha": self.alpha,
        }


@dataclass
class Config:
    rl: rl.RLParams
    rrt: dict
    mpc: Any

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(rl=rl.RLParams.from_dict(config_dict["rl"]), rrt=RRTParams.from_dict(config_dict["rrt"]), mpc=mpc.MPCParams.from_dict(config_dict["mpc"]))
        return config


class RLRRTMPCBuilder:
    @classmethod
    def build(cls, config: Config) -> Tuple[rl.RL, rrt.RRT, mpc.MPC]:

        rl_obj = rl.RL(config.rl)
        print(config.rrt)
        rrt_obj = rrt.RRT(config.rrt)
        mpc_obj = mpc.MPC(config.mpc)
        return rl_obj, rrt_obj, mpc_obj


class RLRRTMPC(ci.ICOLAV):
    def __init__(self, config: Config | None, config_file: Optional[Path] = dp.rl_rrt_mpc_config) -> None:

        if config:
            self._config: Config = config
        else:
            self._config = cp.extract(Config, config_file, dp.rl_rrt_mpc_schema)

        self._rl, self._rrt, self._mpc = RLRRTMPCBuilder.build(self._config)

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
        **kwargs
    ) -> np.ndarray:
        """Implements the ICOLAV plan interface function. Relies on getting the own-ship minimum depth
        in order to extract relevant grounding hazards.
        """
        if not self._initialized:
            self._t_prev = t
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards(kwargs["os_length"], enc)

            if False and enc is not None:
                enc.start_display()
                for hazard in relevant_grounding_hazards:
                    enc.draw_polygon(hazard, color="red")

            self._rrt.transfer_enc_data(relevant_grounding_hazards)

        references = np.zeros((9, 1))
        return references

    def get_current_plan(self) -> np.ndarray:
        return self._references


# hazard.geoms[i].exterior.
