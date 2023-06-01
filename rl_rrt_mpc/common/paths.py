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

schemas = package / "schemas"
rl_rrt_mpc_schema = schemas / "rl_rrt_mpc.yaml"
rl_mpc_schema = schemas / "rl_mpc.yaml"

rl_rrt_mpc_config = config / "rl_rrt_mpc.yaml"
rl_mpc_config = config / "rl_mpc.yaml"
rrt_solution = data / "rrt_solution.yaml"
