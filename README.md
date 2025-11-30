
# rlmpc

This repository contains research code behind
- A trajectory tracking NMPC with anti-grounding functionality, described in <https://ieeexplore.ieee.org/abstract/document/10644772>.
- A mid-level path-following NMPC with speed profile consideration, that considers dynamic ship collision avoidance and grounding avoidance.
- Variational autoencoders for feature extraction from Electronic Navigational Charts (ENCs) and dynamc obstacle target tracks.
- A Soft Actor Critic (SAC) implementation coupled with the mid-level NMPC for situation-dependent tuning of the NMPC, based on a modded version of stable baselines3 where the MPC Optimal Control Problem sensitivities are used in the SAC actor gradient calculations. Unpublished work in progress. Note that the NMPC formulation in Acados is highly sensitive to changes in the parameters, so take great care in how the cost function and constraints are setup to ensure the (S)QP-solvers do not fail too often. The most robust way to use this functionality would be by just formulating a linear MPC.

Coupled with the colav-simulation framework in <https://github.com/NTNU-TTO/colav-simulator>. Example training/nmpc run scripts found under `run_examples`. Developed mainly with Python 3.10 and 3.11. Note that some modernization/fixes is needed to make the code fully compatible with never acados versions following the latest porting to use `uv` and newer dependency versions.

<p align="center">
    <img src="https://github.com/ntnu-itk-autonomous-ship-lab/rlmpc/blob/main/simple_planning_example_ep1.gif?raw=true" width="1000px"><br/>
    <em>Example run of an anti-grounding tracking NMPC controller in the simulator.</em>
</p>
<p align="center">
    <img src="https://github.com/NTNU-TTO/colav-simulator/blob/main/gym_env_teaser.gif?raw=true" width="1000px"><br/>
    <em>Example visualization of a DRL-based MPC algorithm run in multiple evaluation episodes using the COLAVEnvironment Gymnasium functionality.</em>
</p>

[![platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)]()
[![python version](https://img.shields.io/badge/python-3.10-blue)]()

## Main Dependencies
Git dependencies here include

- `Acados` for solving optimal control problems: <https://github.com/acados/acados>.
- The `seacharts` package in <https://github.com/trymte/seacharts>.
- The `colav-simulator` repo in <https://github.com/NTNU-TTO/colav-simulator>.
- The `rrt-star-lib` library for Rapidly-exploring Random Trees at <https://github.com/ntnu-itk-autonomous-ship-lab/rrt-rs>.

## Installation

Use the convenience script in `install_project.sh` to install everything:
```bash
chmod +x install_project.sh
./install_project.sh
```

Check the installation by e.g. running the anti-grounding MPC script 
```bash
python3 run_examples/run_antigrounding_nmpc_near_land_tracking.py
```

Be warned that the `run_training_*` scripts are not fully cleaned up for ease of use yet, so use with care.

## Citation
If you are using code from this repository in your work, please use the following citation for the machine-learning related parts
```bibtex
@Article{Tengesdal2024sacn,
  author  = {Tengesdal, T and Menges, D. and Gros, S. and Johansen, T. A.},
  journal = {},
  title   = {Soft Actor Critic Reinforcement Learning for Adaptable Collision-free Ship Trajectory Planning},
  year={2024},
  volume={},
  number={},
  pages={},
  doi={},
  note={Unpublished},
}
```
and the below one for the anti-grounding NMPC:
```bibtex
@INPROCEEDINGS{10644772,
  author={Tengesdal, Trym and Gros, Sébastien and Johansen, Tor A.},
  booktitle={2024 American Control Conference (ACC)}, 
  title={Real-time Feasible Usage of Radial Basis Functions for Representing Unstructured Environments in Optimal Ship Control}, 
  year={2024},
  volume={},
  number={},
  pages={4050-4057},
  keywords={Grounding;Trajectory tracking;Navigation;Optimal control;Hazards;Real-time systems;Trajectory;Planning;Marine vehicles;Predictive control},
  doi={10.23919/ACC60939.2024.10644772}}
```

## Usage
See the run examples and test files for usage.
