
# RL-RRT-MPC
This repository implements an Informed RRT*-based planning algorithm for a maritime vessel navigating at sea, with a Model Predictive Control for further trajectory planning.
RL is used to update the algorithm parameters.


## Installation

To install and use this COLAV system, install `acados` and the corresponding Python package `acados_template`, and then perform the following commands
```bash
cd rl_rrt_mpc
pip install -e .
cd informed_rrt_star_rust
maturin develop
```


## Usage
See the test file for example usage.
