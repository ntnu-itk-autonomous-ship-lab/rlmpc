
# rlmpc
This repository contains
- Functionality for ship collision avoidance motion planning using NMPC
- Variational autoencoders for feature extraction from Electronic Navigational Charts (ENCs) and target tracks
- A Soft Actor Critic (SAC) implementation with an NMPC actor (in progress), based on a modded version of stable baselines3

using the colav-simulation framework in <https://github.com/NTNU-Autoship-Internal/colav_simulator>.

<p align="center">
    <img src="https://github.com/NTNU-Autoship-Internal/rlmpc/blob/main/mpc_teaser.gif?raw=true" width="1000px"><br/>
    <em>Example run of an MPC-based COLAV planner in the simulator.</em>
</p>
<p align="center">
    <img src="https://github.com/NTNU-Autoship-Internal/rlmpc/blob/main/simple_planning_example_ep1.gif?raw=true" width="1000px"><br/>
    <em>Example run of an anti-grounding tracking NMPC controller in the simulator.</em>
</p>

[![platform](https://img.shields.io/badge/platform-linux-lightgrey)]()
[![python version](https://img.shields.io/badge/python-3.10-blue)]()

## Main Dependencies
See `setup.cfg` file. Nonlisted dependencies here include

- Acados for solving optimal control problems
- Casadi -''-, <=3.6.5
- The `colav-simulator` repo in <https://github.com/NTNU-Autoship-Internal/colav_simulator>.
- The `rrt-rs` library for Rapidly-exploring Random Trees at <https://github.com/NTNU-Autoship-Internal/rrt-rs>.

## Citation
If you are using code from this repository in your work, please use the following citation:

```bibtex
@Article{Tengesdal2024sacn,
  author  = {Tengesdal, T and Menges, D. and Gros, S. and Johansen, T. A.},
  journal = {IEEE Access},
  title   = {Soft Actor Critic Reinforcement Learning for Adaptable Collision-free Ship Trajectory Planning},
  year={2024},
  volume={},
  number={},
  pages={},
  doi={},
  note={Unpublished},
}
```

## Usage
See the test files for usage.
