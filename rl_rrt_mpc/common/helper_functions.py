"""
    helper_functions.py

    Summary:
        Contains miscellaneous helper functions for the RL-RRT-MPC COLAV system.

    Author: Trym Tengesdal
"""
import inspect
import pathlib
from contextlib import contextmanager
from typing import Any, Optional, Tuple, Union

import casadi as csd
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.integrators as sim_integrators
import colav_simulator.core.models as sim_models
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.file_utils as fu
import rl_rrt_mpc.common.math_functions as mf
import rl_rrt_mpc.common.paths as dp
import seacharts.enc as senc
import shapely.affinity as affinity
import shapely.geometry as geometry
import tqdm
import yaml
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.stats import chi2


@contextmanager
def redirect_stdout__to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        to_print = "".join(map(repr, args))
        tqdm.tqdm.write(to_print, **kwargs)

    try:
        # Globally replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


def tqdm_context(*args, **kwargs):
    with redirect_stdout__to_tqdm():
        postfix_dict = kwargs.pop("postfix_dict", {})
        additional_info_flag = kwargs.pop("additional_info_flag", False)
        position = kwargs.pop("pos", 0)
        kwargs.update({"position": position})
        kwargs.update({"leave": False})

        t_main = tqdm.tqdm(*args, **kwargs)
        t_main.postfix_dict = postfix_dict
        if additional_info_flag:
            yield t_main
        for x in t_main:
            t_main.set_postfix(**t_main.postfix_dict)
            t_main.refresh()
            yield x


def create_los_based_trajectory(
    xs: np.ndarray,
    waypoints: np.ndarray,
    speed_plan: np.ndarray,
    los: guidances.LOSGuidance,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a trajectory based on the provided LOS guidance, controller and model.

    Args:
        - xs (np.ndarray): State vector
        - waypoints (np.ndarray): Waypoints
        - speed_plan (np.ndarray): Speed plan
        - los (guidances.LOSGuidance): LOS guidance object
        - dt (float): Time step

    Returns:
        np.ndarray: Trajectory
    """
    model = sim_models.Telemetron()
    controller = controllers.FLSH(model.params)
    trajectory = []
    inputs = []
    xs_k = xs
    t = 0.0
    reached_goal = False
    t_braking = 30.0
    t_brake_start = 0.0
    while t < 2000.0:
        trajectory.append(xs_k)
        references = los.compute_references(waypoints, speed_plan, None, xs_k, dt)
        if reached_goal:
            references[3:] = np.tile(0.0, (references[3:].size, 1))
        u = controller.compute_inputs(references, xs_k, dt)
        inputs.append(u)
        w = None
        xs_k = sim_integrators.erk4_integration_step(model.dynamics, model.bounds, xs_k, u, w, dt)

        dist2goal = np.linalg.norm(xs_k[0:2] - waypoints[:, -1])
        t += dt
        if dist2goal < 70.0 and not reached_goal:
            reached_goal = True
            t_brake_start = t

        if reached_goal and t - t_brake_start > t_braking:
            break

    return np.array(trajectory).T, np.array(inputs)[:, :2].T


def interpolate_solution(
    trajectory: np.ndarray, inputs: np.ndarray, t: float, t_prev: float, T_mpc: float, dt_mpc: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolates the solution from the MPC to the time step in the simulation.

    Args:
        - trajectory (np.ndarray): The solution state trajectory.
        - inputs (np.ndarray): The solution input trajectory.
        - t (float): The current time step.
        - t_prev (float): The previous time step.
        - T_mpc (float): The MPC horizon.
        - dt_mpc (float): The MPC time step.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The interpolated solution state trajectory and input trajectory.
    """
    intp_trajectory = trajectory
    intp_inputs = inputs
    if dt_mpc > t - t_prev or dt_mpc < t - t_prev:
        nx = trajectory.shape[0]
        nu = inputs.shape[0]
        dt_sim = np.max([t - t_prev, 0.5])
        sim_times = np.arange(0.0, T_mpc, dt_sim)
        mpc_times = np.arange(0.0, T_mpc, dt_mpc)
        n_samples = len(sim_times)
        intp_trajectory = np.zeros((nx, n_samples))
        intp_inputs = np.zeros((nu, n_samples))
        for dim in range(nx):
            intp_trajectory[dim, :] = interp1d(mpc_times, trajectory[dim, :], kind="linear", fill_value="extrapolate")(
                sim_times
            )
        for dim in range(nu):
            intp_inputs[dim, :] = interp1d(mpc_times, inputs[dim, :], kind="linear", fill_value="extrapolate")(
                sim_times
            )

        return intp_trajectory, intp_inputs
    return intp_trajectory, intp_inputs


def shift_nominal_plan(
    nominal_trajectory: np.ndarray, nominal_inputs: np.ndarray, ownship_state: np.ndarray, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Updates the nominal trajectory and inputs to the MPC based on the current ownship state. This is done by
    find closest point on nominal trajectory to the current state and then shifting the nominal trajectory to this point

    Args:
        - nominal_trajectory (np.ndarray): The nominal trajectory.
        - nominal_inputs (np.ndarray): The nominal inputs.
        - ownship_state (np.ndarray): The ownship state.
        - N (int): MPC horizon length in samples

    Returns:
        Tuple[np.ndarray, np.ndarray]: The shifted nominal trajectory and inputs.
    """
    nx = ownship_state.size
    nu = nominal_inputs.shape[0]
    closest_idx = int(
        np.argmin(
            np.linalg.norm(
                nominal_trajectory[:2, :] - np.tile(ownship_state[:2], (len(nominal_trajectory[0, :]), 1)).T, axis=0
            )
        )
    )
    shifted_nominal_trajectory = nominal_trajectory[:, closest_idx:]
    shifted_nominal_inputs = nominal_inputs[:, closest_idx:]
    n_samples = shifted_nominal_trajectory.shape[1]
    if n_samples == 0:  # Done with following nominal trajectory, stop
        shifted_nominal_trajectory = np.tile(
            np.array([ownship_state[0], ownship_state[1], ownship_state[2], 0.0, 0.0, 0.0]), (N + 1, 1)
        ).T
        shifted_nominal_inputs = np.zeros((nu, N))
    elif n_samples < N + 1:
        shifted_nominal_trajectory = np.zeros((nx, N + 1))
        shifted_nominal_trajectory[:, :n_samples] = nominal_trajectory[:, closest_idx : closest_idx + n_samples]
        shifted_nominal_trajectory[:, n_samples:] = np.tile(
            nominal_trajectory[:, closest_idx + n_samples - 1], (N + 1 - n_samples, 1)
        ).T
        shifted_nominal_inputs = np.zeros((nu, N))
        shifted_nominal_inputs[:, : n_samples - 1] = nominal_inputs[:, closest_idx : closest_idx + n_samples - 1]
        shifted_nominal_inputs[:, n_samples - 1 :] = np.tile(
            nominal_inputs[:, closest_idx + n_samples - 2], (N - n_samples + 1, 1)
        ).T
    else:
        shifted_nominal_trajectory = shifted_nominal_trajectory[:, : N + 1]
        shifted_nominal_inputs = shifted_nominal_inputs[:, :N]
    return shifted_nominal_trajectory, shifted_nominal_inputs


def decision_trajectories_from_solution(
    soln: np.ndarray, N: int, nu: int, nx: int, ns: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the input sequence U, state sequence X and the slack variable sequence S from the solution vector soln = w = [U.flattened, X.flattened, Sigma.flattened] from the optimization problem.

    Args:
        soln (np.ndarray): A solution vector from the optimization problem.
        N (int): The prediction horizon.
        nu (int): The input dimension
        nx (int): The state dimension
        ns (int): The slack variable dimension

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The input sequence U, state sequence X and the slack variable sequence S.
    """
    U = np.zeros((nu, N))
    X = np.zeros((nx, N + 1))
    Sigma = np.zeros((ns, N + 1))
    for k in range(N + 1):
        if k < N:
            U[:, k] = soln[k * nu : (k + 1) * nu].ravel()
        X[:, k] = soln[N * nu + k * nx : N * nu + (k + 1) * nx].ravel()
        Sigma[:, k] = soln[N * nu + (N + 1) * nx + k * ns : N * nu + (N + 1) * nx + (k + 1) * ns].ravel()
    return U, X, Sigma


def linestring_to_ndarray(line: geometry.LineString) -> np.ndarray:
    """Converts a shapely LineString to a numpy array

    Args:
        - line (LineString): Any LineString object

    Returns:
        np.ndarray: Numpy array containing the coordinates of the LineString
    """
    return np.array(line.coords).transpose()


def ndarray_to_linestring(array: np.ndarray) -> geometry.LineString:
    """Converts a 2D numpy array to a shapely LineString

    Args:
        - array (np.ndarray): Numpy array of 2 x n_samples, containing the coordinates of the LineString

    Returns:
        LineString: Any LineString object
    """
    assert array.shape[0] == 2 and array.shape[1] > 1, "Array must be 2 x n_samples with n_samples > 1"
    return geometry.LineString(list(zip(array[0, :], array[1, :])))


def casadi_potential_field_base_function(x: csd.MX) -> csd.MX:
    """Potential field base function f(x) = x / sqrt(x^2 + 1)

    Args:
        x (csd.MX): Input

    Returns:
        csd.MX: Output f(x)
    """
    return x / csd.sqrt(x**2 + 1)


def casadi_matrix_from_nested_list(M: list) -> csd.MX:
    """Convenience function for making a casadi matrix from lists of lists
    (don't know why this doesn't exist already), the alternative is

    Args:
        M (list): List of lists

    Returns:
        csd.MX: Casadi matrix
    """
    return csd.vertcat(*(csd.horzcat(*row) for row in M))


def casadi_diagonal_matrix_from_vector(v: csd.MX) -> csd.MX:
    """Creates a diagonal matrix from a vector.

    Args:
        v (csd.MX): Vector symbolic representing diagonal entries
    """
    n = v.shape[0]
    llist = []
    for i in range(n):
        nested_list = []
        for j in range(n):
            if i == j:
                nested_list.append(v[i])
            else:
                nested_list.append(0.0)
        llist.append(nested_list)
    return casadi_matrix_from_nested_list(llist)


def casadi_matrix_from_vector(v: csd.MX, n_rows: int, n_cols: int) -> csd.MX:
    """Creates a matrix from a vector.

    Args:
        v (csd.MX): Vector symbolic representing matrix entries
        n_rows (int): Rows in matrix
        n_cols (int): Columns in matrix

    Returns:
        csd.MX: Casadi matrix
    """
    llist = []
    for i in range(n_rows):
        nested_list = []
        for j in range(n_cols):
            nested_list.append(v[i * n_rows + j])
        llist.append(nested_list)
    return casadi_matrix_from_nested_list(llist)


def load_rrt_solution(save_file: pathlib.Path = dp.rrt_solution) -> dict:
    return fu.read_yaml_into_dict(save_file)


def save_rrt_solution(rrt_solution: dict, save_file: pathlib.Path = dp.rrt_solution) -> None:
    save_file.touch(exist_ok=True)
    with save_file.open(mode="w") as file:
        yaml.dump(rrt_solution, file)


def translate_dynamic_obstacle_coordinates(dynamic_obstacles: list, x_shift: float, y_shift: float) -> list:
    """Translates the coordinates of a list of dynamic obstacles by (-y_shift, -x_shift)

    Args:
        dynamic_obstacles (list): List of dynamic obstacle objects on the form (ID, state, cov, length, width)
        x_shift (float): Easting shift
        y_shift (float): Northing shift

    Returns:
        list: List of dynamic obstacles with shifted coordinates
    """
    translated_dynamic_obstacles = []
    for ID, state, cov, length, width in dynamic_obstacles:
        translated_state = state - np.array([y_shift, x_shift, 0.0, 0.0])
        translated_dynamic_obstacles.append((ID, translated_state, cov, length, width))
    return translated_dynamic_obstacles


def translate_polygons(polygons: list, x_shift: float, y_shift: float) -> list:
    """Shifts the coordinates of a list of polygons by (-x_shift, -y_shift)

    Args:
        polygons (list): List of shapely polygons
        x_shift (float): Shift easting
        y_shift (float): Shift northing

    Returns:
        list: List of shifted polygons
    """
    translated_polygons = []
    for polygon in polygons:
        translated_polygon = affinity.translate(polygon, xoff=-x_shift, yoff=-y_shift)
        translated_polygons.append(translated_polygon)
    return translated_polygons


def create_ellipse(
    center: np.ndarray,
    A: Optional[np.ndarray] = None,
    a: float | None = 1.0,
    b: float | None = 1.0,
    phi: float | None = 0.0,
) -> Tuple[list, list]:
    """Create standard ellipse at center, with input semi-major axis, semi-minor axis and angle.

    Either specified by c, A or c, a, b, phi:

    (p - c)^T A (p - c) = 1

    or

    (p - c)^T R^T D R (p - c) = 1

    with R = R(phi) and D = diag(1 / a^2, 1 / b^2)


    Args:
        - center (np.ndarray): Center of ellipse
        - A (Optional[np.ndarray], optional): Hessian matrix. Defaults to None.
        - a (float | None, optional): Semi-major axis. Defaults to 1.0.
        - b (float | None, optional): Semi-minor axis. Defaults to 1.0.
        - phi (float | None, optional): Angle. Defaults to 0.0.


    Returns:
        Tuple[list, list]: List of x and y coordinates
    """

    if A is not None:
        # eigenvalues and eigenvectors of the covariance matrix
        eigenval, eigenvec = np.linalg.eig(A[0:2, 0:2])

        largest_eigenval = max(eigenval)
        largest_eigenvec_idx = np.argwhere(eigenval == max(eigenval))[0][0]
        largest_eigenvec = eigenvec[:, largest_eigenvec_idx]

        smallest_eigenval = min(eigenval)
        # if largest_eigenvec_idx == 0:
        #     smallest_eigenvec = eigenvec[:, 1]
        # else:
        #     smallest_eigenvec = eigenvec[:, 0]

        angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
        angle = mf.wrap_angle_to_02pi(angle)

        a = np.sqrt(largest_eigenval)
        b = np.sqrt(smallest_eigenval)
    else:
        angle = phi

    # the ellipse in "body" x and y coordinates
    t = np.linspace(0, 2.01 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)

    R = mf.Rpsi2D(angle)

    # Rotate to NED by angle phi, N_ell_points x 2
    ellipse_xy = np.array([x, y])
    for i in range(ellipse_xy.shape[1]):
        ellipse_xy[:, i] = R @ ellipse_xy[:, i] + center

    return ellipse_xy[0, :].tolist(), ellipse_xy[1, :].tolist()


def create_probability_ellipse(P: np.ndarray, probability: float = 0.99) -> Tuple[list, list]:
    """Creates a probability ellipse for a covariance matrix P and a given
    confidence level (default 0.99).

    Args:
        P (np.ndarray): Covariance matrix
        probability (float, optional): Confidence level. Defaults to 0.99.

    Returns:
        np.ndarray: Ellipse data in x and y coordinates
    """

    # eigenvalues and eigenvectors of the covariance matrix
    eigenval, eigenvec = np.linalg.eig(P[0:2, 0:2])

    largest_eigenval = max(eigenval)
    largest_eigenvec_idx = np.argwhere(eigenval == max(eigenval))[0][0]
    largest_eigenvec = eigenvec[:, largest_eigenvec_idx]

    smallest_eigenval = min(eigenval)
    # if largest_eigenvec_idx == 0:
    #     smallest_eigenvec = eigenvec[:, 1]
    # else:
    #     smallest_eigenvec = eigenvec[:, 0]

    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
    angle = mf.wrap_angle_to_02pi(angle)

    # Get the ellipse scaling factor based on the confidence level
    chisquare_val = chi2.ppf(q=probability, df=2)

    a = chisquare_val * np.sqrt(largest_eigenval)
    b = chisquare_val * np.sqrt(smallest_eigenval)

    # the ellipse in "body" x and y coordinates
    t = np.linspace(0, 2.01 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)

    R = mf.Rpsi2D(angle)

    # Rotate to NED by angle phi, N_ell_points x 2
    ellipse_xy = np.array([x, y])
    for i in range(len(ellipse_xy)):
        ellipse_xy[:, i] = R @ ellipse_xy[:, i]

    return ellipse_xy[0, :].tolist(), ellipse_xy[1, :].tolist()


def plot_surface_approximation_stuff(
    radial_basis_function: csd.Function,
    radial_basis_function_gradient: csd.Function,
    surface_data_points: Tuple[list, list],
    surface_data_point_mask: list,
    surface_data_points_before_buffering: Tuple[list, list],
    original_polygon: geometry.Polygon,
    polygon: geometry.Polygon,
    polygon_safety: geometry.Polygon,
    polygon_index: int,
    relevant_coastline_safety: geometry.Polygon,
    d_safe: float,
    map_origin: np.ndarray,
    enc: senc.ENC,
) -> Tuple[list, list]:
    """Plots the surface approximation stuff. This is used for debugging purposes, not optimized for readability.

    Args:
        radial_basis_function (csd.Function): Radial basis function
        radial_basis_function_gradient (csd.Function): Gradient of radial basis function
        surface_data_points (Tuple[list, list]): Data points used for interpolation
        surface_data_point_mask (list): Mask of data points used for interpolation
        surface_data_points_before_buffering (Tuple[list, list]): Data points used for interpolation before buffering
        original_polygon (geometry.Polygon): Original polygon
        polygon (geometry.Polygon): Polygon clipped inside trajectory envelope
        polygon_safety (geometry.Polygon): Polygon clipped inside trajectory envelope and safety buffered
        polygon_index (int): Index of polygon
        relevant_coastline_safety (geometry.Polygon): Relevant coastline of the clipped safety polygon
        d_safe (float): Safety distance
        map_origin (np.ndarray): Map origin in NE
        enc (senc.ENC): ENC object

    Returns:
        Tuple[list, list]: List of figures and axes
    """
    x_surface, y_surface = surface_data_points
    cap_style = 2
    join_style = 2
    ax1 = plt.figure().add_subplot(111, projection="3d")
    # ax2 = plt.figure().add_subplot(111, projection="3d")
    # ax3 = plt.figure().add_subplot(111, projection="3d")
    # ax3 = plt.figure().add_subplot(111)
    # ax5 = plt.figure().add_subplot(111, projection="3d")
    poly_min_east, poly_min_north, poly_max_east, poly_max_north = polygon.buffer(
        d_safe + 10.0, cap_style=cap_style, join_style=join_style
    ).bounds

    (
        coastline_min_east,
        coastline_min_north,
        coastline_max_east,
        coastline_max_north,
    ) = relevant_coastline_safety.bounds
    # if polygon_index == 1:
    #     translated_polygon = translate_polygons([polygon], -map_origin[1], -map_origin[0])[0]
    #     enc.draw_polygon(
    #         translated_polygon.buffer(d_safe, cap_style=cap_style, join_style=join_style), color="black", fill=False
    #     )
    #    save_path = dp.figures
    #     enc.save_image(name="enc_island_polygon", path=save_path, extension="pdf")
    #     enc.save_image(name="enc_island_polygon", path=save_path, scale=2.0)

    # if polygon_index == 8:
    #     polygon_diff = ops.split(
    #         relevant_coastline_safety.buffer(10.0, cap_style=cap_style, join_style=join_style),
    #         geometry.LineString(original_polygon.exterior.coords),
    #     )
    #     geom = polygon_diff.geoms[1]
    #     translated_geom = translate_polygons([geom], -map_origin[1], -map_origin[0])[0]
    #     enc.draw_polygon(translated_geom, color="black", fill=False)

    y_poly_safety, x_poly_safety = polygon_safety.exterior.coords.xy

    # Compute error approximation
    compute_err_approx = False
    if compute_err_approx:
        n_points = 200
        grid_resolution_y = 0.5
        grid_resolution_x = 0.5
        buffer = 5.0
        npy = int((poly_max_east + 2 * buffer - poly_min_east) / grid_resolution_y)
        npx = int((poly_max_north + 2 * buffer - poly_min_north) / grid_resolution_x)
        north_coords = np.linspace(start=poly_min_north - buffer, stop=poly_max_north + buffer, num=npx)
        east_coords = np.linspace(start=poly_min_east - buffer, stop=poly_max_east + buffer, num=npy)

        Y, X = np.meshgrid(east_coords, north_coords, indexing="ij")
        map_coords = np.hstack((Y.reshape(-1, 1), X.reshape(-1, 1)))

        poly_path = mpath.Path(np.array([y_poly_safety, x_poly_safety]).T)
        mask = poly_path.contains_points(points=map_coords, radius=0.00001)
        mask = mask.astype(float).reshape((npy, npx))
        mask[mask > 0.0] = 1.0

        epsilon = 1e-3
        dist_surface_points = np.zeros((npy, npx))
        diff_surface_points = np.zeros((npy, npx))

        for i, east_coord in enumerate(east_coords):
            if polygon_index == 0 and east_coord < coastline_min_east:
                continue
            for ii, north_coord in enumerate(north_coords):
                if polygon_index == 0 and north_coord < coastline_min_north:
                    continue

                if (
                    polygon_index == 0
                    and north_coord < coastline_min_north + 200.0
                    and east_coord < coastline_min_east + 200.0
                ):
                    continue

                if (
                    polygon_index == 0
                    and north_coord < coastline_min_north + 20.0
                    and east_coord > coastline_max_east - 60.0
                ):
                    continue

                if polygon_index == 8 and not geometry.Point(east_coord, north_coord).within(geom):
                    continue

                if polygon_index == 8 and north_coord < 215.0 and east_coord < 324.8:
                    continue

                if polygon_index == 8 and north_coord < -12.0 and east_coord > 1257.0:
                    continue

                if (
                    mask[i, ii] > 0.0
                    and radial_basis_function(np.array([north_coord, east_coord]).reshape((1, 2))) <= 0.0 + epsilon
                ) or (
                    mask[i, ii] <= 0.0
                    and radial_basis_function(np.array([north_coord, east_coord]).reshape((1, 2))) > 0.0 + epsilon
                ):
                    # if mask[i, ii] - radial_basis_function(np.array([north_coord, east_coord]).reshape((1, 2))) > 0.0:
                    #    print("Error: ", mask[i, ii] - radial_basis_function(np.array([north_coord, east_coord]).reshape((1, 2))))
                    d2poly = polygon_safety.distance(geometry.Point(east_coord, north_coord))
                    dist_surface_points[i, ii] = d2poly
                    diff_surface_points[i, ii] = radial_basis_function(
                        np.array([north_coord, east_coord]).reshape((1, 2))
                    )
        print("polygon_index = {polygon_index} |Max distance of error: ", np.max(dist_surface_points))

        n_points = len(x_surface)
        actual_dataset_error = np.zeros(n_points)
        for i, (north_coord, east_coord) in enumerate(zip(x_surface, y_surface)):
            point = np.array([north_coord + 0.000001, east_coord + 0.000001]).reshape(1, 2)
            actual_dataset_error[i] = surface_data_point_mask[i] - radial_basis_function(point).full()
        mean_error = np.mean(dist_surface_points)
        max_error = np.max(dist_surface_points)
        idx_max_error = np.argmax(actual_dataset_error)
        std_error = np.std(dist_surface_points)
        print(
            f"polygon_index = {polygon_index} | Num interpolation data points: {len(x_surface)} | Num original poly points: {len(x_poly_safety)}"
        )
        print(
            f"Dataset: Mean 0point crossing error: {mean_error}, Max, idx max error: ({max_error}, {idx_max_error}), Std error: {std_error}"
        )

        Y, X = np.meshgrid(east_coords + map_origin[1], north_coords + map_origin[0], indexing="ij")
        # Y, X = np.meshgrid(east_coords, north_coords, indexing="ij")
        # ax5.plot_surface(Y, X, dist_surface_points, rcount=100, ccount=100, cmap=cm.coolwarm)
        # # ax5.contourf(Y, X, mask.T, zdir="z", offset=50.0, cmap=cm.coolwarm)
        # ax5.set_xlabel("East [m]")
        # ax5.set_ylabel("North [m]")
        # ax5.set_zlabel("Distance [m]")

        y_surface_orig, x_surface_orig = surface_data_points_before_buffering
        fig6, ax6 = plt.subplots()
        pc6 = ax6.pcolormesh(Y, X, dist_surface_points, shading="gouraud", rasterized=True)
        ax6.plot(y_surface_orig + map_origin[1], x_surface_orig + map_origin[0], "k")
        # ax6.plot(y_surface_orig, x_surface_orig, "k")
        cbar6 = fig6.colorbar(pc6)
        cbar6.set_label("Distance [m]")
        ax6.set_xlabel("East [m]")
        ax6.set_ylabel("North [m]")

    buffer = 1.0
    n_points = 50
    extra_north_coords = np.linspace(start=poly_min_north - buffer, stop=poly_max_north + buffer, num=n_points)
    extra_east_coords = np.linspace(start=poly_min_east - buffer, stop=poly_max_east + buffer, num=n_points)

    surface_points = np.zeros((n_points, n_points))
    surface_grad_points = np.zeros((n_points, n_points, 2))
    for i, east_coord in enumerate(extra_east_coords):
        for ii, north_coord in enumerate(extra_north_coords):
            point = np.array([north_coord, east_coord]).reshape(1, 2)
            surface_points[i, ii] = radial_basis_function(point).full()
            surface_grad_points[i, ii, :] = radial_basis_function_gradient(point).full().flatten()
    yY, xX = np.meshgrid(extra_east_coords + map_origin[1], extra_north_coords + map_origin[0], indexing="ij")

    print(f"Number of gradient NaNs: {np.count_nonzero(np.isnan(surface_grad_points))}")

    fig1 = ax1.figure
    ax1.plot_surface(yY, xX, surface_points, cmap=cm.coolwarm)
    ax1.set_ylabel("North [m]")
    ax1.set_xlabel("East [m]")
    ax1.set_zlabel(r"$h_j(\bm{\zeta})$")
    # fig1.savefig("surface_approx.pdf", bbox_inches="tight", dpi=50)

    fig2, ax2 = plt.subplots()
    ax2.pcolormesh(yY, xX, surface_points, shading="gouraud")
    p = ax2.scatter(y_surface + map_origin[1], x_surface + map_origin[0], c=np.array(surface_data_point_mask), ec="k")
    cbar4 = fig2.colorbar(p)
    cbar4.set_label(r"$h_j(\bm{\zeta})$")
    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")
    # fig2.savefig("colormesh_island_polygon.pdf", bbox_inches="tight", dpi=50)

    # ax2.plot_surface(yY, xX, surface_grad_points[:, :, 0], rcount=200, ccount=200, cmap=cm.coolwarm)
    # ax2.set_xlabel("East")
    # ax2.set_ylabel("North")
    # ax2.set_zlabel("Mask")
    # ax2.set_title("Spline surface gradient x")

    # ax3.plot_surface(yY, xX, surface_grad_points[:, :, 1], rcount=200, ccount=200, cmap=cm.coolwarm)
    # ax3.set_xlabel("East")
    # ax3.set_ylabel("North")
    # ax3.set_zlabel("Mask")
    # ax3.set_title("Spline surface gradient y")
    plt.show(block=False)
    # ax1.clear()
    # ax2.clear()
    # ax3.clear()
    # ax5.clear()

    return [fig1, fig2], [ax1, ax2]


def create_arc_length_spline(x: list, y: list) -> Tuple[interp1d, interp1d, list]:
    """Creates a spline for the arc length of the input x and y coordinates.

    Args:
        - x (list): List of x coordinates.
        - y (list): List of y coordinates.

    Returns:
        Tuple[interp1d, interp1d, list]: Tuple of arc length splines for x and y coordinates.
    """
    # Interpolate the data to get more points => higher accuracy in the arc length spline
    n_points = len(x)
    y_interp = interp1d(np.arange(n_points), y, kind="linear")
    x_interp = interp1d(np.arange(n_points), x, kind="linear")

    n_expanded_points = 500
    y_expanded = list(y_interp(np.linspace(0, n_points - 1, n_expanded_points)))
    x_expanded = list(x_interp(np.linspace(0, n_points - 1, n_expanded_points)))
    arc_length = [0.0]
    for i in range(1, n_expanded_points):
        pi = np.array([x_expanded[i - 1], y_expanded[i - 1]])
        pj = np.array([x_expanded[i], y_expanded[i]])
        arc_length.append(np.linalg.norm(pi - pj))
    arc_length = np.cumsum(arc_length)
    y_interp_arc_length = interp1d(arc_length, y_expanded, kind="linear")
    x_interp_arc_length = interp1d(arc_length, x_expanded, kind="linear")
    return x_interp_arc_length, y_interp_arc_length, arc_length
