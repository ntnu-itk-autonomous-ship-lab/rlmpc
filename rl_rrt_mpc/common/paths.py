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

schemas = package / "schemas"
rl_rrt_mpc_schema = schemas / "rl_rrt_mpc.yaml"

rl_rrt_mpc_config = config / "rl_rrt_mpc.yaml"
