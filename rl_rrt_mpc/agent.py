"""
    rl_rrt_mpc.py

    Summary:
        Contains the main RL-RRT-MPC COLAV system class.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.core.colav.colav_interface as ci
import informed_rrt_star_rust as rrt
import numpy as np
import rl_rrt_mpc.common.config_parsing as cp
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.map_functions as mapf
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.models as models
import rl_rrt_mpc.mpc as mpc
import rl_rrt_mpc.rl as rl
import seacharts.enc as senc
from shapely import strtree


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
    max_nn_node_dist: float = 300.0
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
    rrt: RRTParams
    mpc: mpc.MPCParams

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(
            rl=rl.RLParams.from_dict(config_dict["rl"]),
            rrt=RRTParams.from_dict(config_dict["rrt"]),
            mpc=mpc.MPCParams.from_dict(config_dict["mpc"]),
        )
        return config


class RLRRTMPCBuilder:
    @classmethod
    def build(cls, config: Config) -> Tuple[rl.RL, rrt.InformedRRTStar, mpc.MPC]:

        rl_obj = rl.RL(config.rl)
        rrt_obj = rrt.InformedRRTStar(config.rrt)
        mpc_obj = mpc.MPC(models.TelemetronAcados(), config.mpc)
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
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])

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
            self._geometry_tree, poly_list = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)

            self._rrt.transfer_enc_data(relevant_grounding_hazards)
            self._rrt.set_init_state(ownship_state.tolist())
            self._rrt.set_goal_state(goal_state.tolist())

            U_d = ownship_state[3]  # Constant desired speed given by the initial own-ship speed
            # rrtresult: dict = self._rrt.grow_towards_goal(ownship_state.tolist(), U_d, [])
            rrtresult = hf.load_rrt_solution()
            states = rrtresult["states"]
            times = rrtresult["times"]
            inputs = []
            tree_list = self._rrt.get_tree_as_list_of_dicts()

            if enc is not None:
                enc.start_display()
                # for hazard in relevant_grounding_hazards:
                #     enc.draw_polygon(hazard, color="red")

                hf.plot_rrt_tree(tree_list, enc)
                hf.plot_rrt_solution(states, times, enc)
                ship_poly = hf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 5, 2)
                enc.draw_circle((ownship_state[1], ownship_state[0]), radius=40, color="yellow")
                enc.draw_polygon(ship_poly, color="pink")
                enc.draw_circle((goal_state[1], goal_state[0]), radius=40, color="cyan")
                hf.save_rrt_solution(states, inputs, times)

            polygons_considered_in_mpc = mapf.extract_polygons_near_trajectory(states, self._geometry_tree, buffer=self._config.mpc.reference_traj_bbox_buffer, enc=enc)
            self._mpc.construct_ocp(do_list=do_list, so_list=polygons_considered_in_mpc, enc=enc)

        references = self._mpc.plan(t=t, nominal_trajectory=states, nominal_inputs=[], xs=ownship_state, do_list=do_list, so_list=polygons_considered_in_mpc)
        return references

    def get_current_plan(self) -> np.ndarray:
        return self._references
