
# RL-RRT-MPC
This repository implements a two-layer trajectory planning algorithm consisting of an RRT for top-level static obstacle collision-free motion planning and an RL-MPC for mid-level local collision-free trajectory planning.


[![platform](https://img.shields.io/badge/platform-linux-lightgrey)]()
[![python version](https://img.shields.io/badge/python-3.10-blue)]()
[![python version](https://img.shields.io/badge/python-3.11-blue)]()

## Dependencies
Install these first.

- acados_template (and acados locally)
- maturin
- matplotlib
- shapely
- numpy
- rrt-rs: https://github.com/NTNU-Autoship-Internal/rrt-rs
- seacharts: https://github.com/trymte/seacharts
- colav_simulator: https://github.com/NTNU-Autoship-Internal/colav_simulator

## Installation and usage in Python

- 1: Install python dependencies.
- 2: Install rrt-rs.
- 3: Install package locally with `pip install -e .`


## Usage
See the test file for example usage.
