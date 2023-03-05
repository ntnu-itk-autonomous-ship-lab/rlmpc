"""
    rl_rrt_mpc.py

    Summary:
        Contains the main RL-RRT-MPC COLAV system class.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import colav_simulator.core.colav.colav_interface as ci
import informed_rrt_star_rust as rrt
import numpy as np
import rl_rrt_mpc.common.config_parsing as cp
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.mpc as mpc
import rl_rrt_mpc.rl as rl
import seacharts.enc as senc


@dataclass
class RRTParams:
    max_nodes: int = 2000
    max_iter: int = 10000
    iter_between_direct_goal_growth: int = 100
    min_node_dist: float = 5.0
    goal_radius: float = 100.0
    step_size: float = 0.1
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 5.0
    max_node_dist: float = 300.0
    gamma: float = 1000.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


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
    def build(cls, config: Config) -> Tuple[rl.RL, rrt.InformedRRTStar, mpc.MPC]:

        rl_obj = rl.RL(config.rl)
        rrt_obj = rrt.InformedRRTStar(config.rrt)
        mpc_obj = mpc.MPC(config.mpc)
        return rl_obj, rrt_obj, mpc_obj


class RLRRTMPC(ci.ICOLAV):
    def __init__(self, config: Optional[Config] = None, config_file: Optional[Path] = dp.rl_rrt_mpc_config) -> None:

        if config:
            self._config: Config = config
        else:
            self._config = cp.extract(Config, config_file, dp.rl_rrt_mpc_schema)

        self._rl, self._rrt, self._mpc = RLRRTMPCBuilder.build(self._config)

        self._references = np.empty(9)
        self._initialized = False
        self._t_prev = 0.0
        self._min_depth: int = 0
        self._plan = np.empty(9)

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[senc.ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Implements the ICOLAV plan interface function. Relies on getting the own-ship minimum depth
        in order to extract relevant grounding hazards.
        """
        assert goal_state is not None, "Goal state must be provided to the RL-RRT-MPC"
        if not self._initialized:
            self._min_depth = mapf.find_minimum_depth(kwargs["os_draft"], enc)
            self._t_prev = t
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards(self._min_depth, enc)

            if enc is not None:
                enc.start_display()
                for hazard in relevant_grounding_hazards:
                    enc.draw_polygon(hazard, color="red")

            self._rrt.transfer_enc_data(relevant_grounding_hazards)
            self._rrt.set_init_state(ownship_state.tolist())
            self._rrt.set_goal_state(goal_state.tolist())

            rrtresult: dict = self._rrt.grow_towards_goal(ownship_state.tolist(), ownship_state[2], [])

        references = np.zeros((9, 1))
        return references

    def get_current_plan(self) -> np.ndarray:
        return self._references


# hazard.geoms[i].exterior.
