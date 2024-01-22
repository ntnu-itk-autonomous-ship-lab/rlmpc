"""
    paths.py

    Summary:
        Contains paths to default configuration files and schemas.

    Author: Trym Tengesdal
"""
import pathlib

root = pathlib.Path(__file__).parents[2]
config = root / "config"
package = root / "rl_rrt_mpc"
data = root / "data"
figures = root / "figures"
acados_code_gen = root / "acados_code_gen"
scenarios = root / "scenarios"

schemas = package / "schemas"
rl_rrt_mpc_schema = schemas / "rl_rrt_mpc.yaml"
rlmpc_schema = schemas / "rlmpc.yaml"
ttmpc_schema = schemas / "ttmpc.yaml"

rl_rrt_mpc_config = config / "rl_rrt_mpc.yaml"
rlmpc_config = config / "rlmpc.yaml"
ttmpc_config = config / "ttmpc.yaml"

rrt_solution = data / "rrt_solution.yaml"
