"""
    map_functions.py

    Summary:
        Contains various commonly used map functions, including
        latlon to local UTM (ENU) coordinate transformation functions etc..

    Author: Trym Tengesdal
"""
import unittest
from typing import Optional, Tuple

import casadi as csd
import geopandas as gpd
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import rl_rrt_mpc.common.helper_functions as hf
import rl_rrt_mpc.common.paths as dp
import rl_rrt_mpc.common.rbf_casadi as rbf_casadi
import rl_rrt_mpc.common.smallestenclosingcircle as smallestenclosingcircle
import rl_rrt_mpc.gmm_em as gmm_em
import scipy.cluster.vq as scipyvq
import scipy.interpolate as scipyintp
import seacharts.enc as senc
import shapely.affinity as affinity
import shapely.geometry as geometry
from matplotlib import cm

# import triangle as tr
from osgeo import osr
from shapely import ops, strtree


def local2latlon(x: float | list | np.ndarray, y: float | list | np.ndarray, utm_zone: int) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
    """Transform coordinates from x (east), y (north) to latitude, longitude.

    Args:
        x (float | list): East coordinate(s) in a local UTM coordinate system.
        y (float | list): North coordinate(s) in a local UTM coordinate system.
        utm_zone (int): UTM zone.

    Raises:
        ValueError: If the input string is not correct.

    Returns:
        Tuple[float | list | np.ndarray, float | list | np.ndarray]: Tuple of latitude and longitude coordinates.
    """
    to_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        from_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        from_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "utm_zone" is not correct. Supported zones sofar are 32 and 33.')

    src = osr.SpatialReference()
    src.ImportFromEPSG(from_zone)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(to_zone)
    transform = osr.CoordinateTransformation(src, tgt)

    if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
        coordinates = transform.TransformPoints(list(zip(x, y)))
        lat = [coord[0] for coord in coordinates]
        lon = [coord[1] for coord in coordinates]
    else:
        lat, lon, _ = transform.TransformPoint(x, y)

    return lat, lon


def latlon2local(lat: float | list | np.ndarray, lon: float | list | np.ndarray, utm_zone: int) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
    """Transform coordinates from latitude, longitude to UTM32 or UTM33

    Args:
        lat (float | list): Latitude coordinate(s)
        lon (float | list): Longitude coordinate(s)
        utm_zone (str): UTM zone.

    Raises:
        ValueError: If the input string is not correct.

    Returns:
        Tuple[float | list | np.ndarray, float | list | np.ndarray]: Tuple of east and north coordinates.
    """
    from_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        to_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        to_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "utm_zone" is not correct. Supported zones sofar are 32 and 33.')

    src = osr.SpatialReference()
    src.ImportFromEPSG(from_zone)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(to_zone)
    transform = osr.CoordinateTransformation(src, tgt)

    if isinstance(lat, (list, np.ndarray)) and isinstance(lon, (list, np.ndarray)):
        coordinates = transform.TransformPoints(list(zip(lat, lon)))
        x = [coord[0] for coord in coordinates]
        y = [coord[1] for coord in coordinates]
    else:
        x, y, _ = transform.TransformPoint(lat, lon)

    return x, y


def dist_between_latlon_coords(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes the distance between two latitude, longitude coordinates.

    Args:
        lat1 (float): Latitude of first coordinate.
        lon1 (float): Longitude of first coordinate.
        lat2 (float): Latitude of second coordinate.
        lon2 (float): Longitude of second coordinate.

    Returns:
        float: Distance between the two coordinates in meters
    """
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).m


def latlon2local_haversine(lon0: float, lat0: float, lon1: float, lat1: float) -> list:
    """Transform the coordinates from latitude, longitude to the local coordinate system using the Haversine formula.

    :param lon0: Origon on the x-axis
    :param lat0: 0rigon on the y-axis
    :param lon1: longitude to transform
    :param lat1: latitude to transform

    :return  in metres"""
    if lon0 < 0 or lat0 < 0 or lon1 < 0 or lat1 < 0:
        raise ValueError("Input coordinates (lat lon) are negative!")

    r = 6362132.0
    lat0, lon0, lat1, lon1 = map(np.radians, [lat0, lon0, lat1, lon1])

    dlon = lon1 - lon0
    dlat = 0.0
    # x-distance (longitude)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    x = r * c

    dlon = 0.0
    dlat = lat1 - lat0
    # y-distance (latitude)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    y = r * c

    return [x, y]


def create_ship_polygon(x: float, y: float, heading: float, length: float, width: float, scale: float = 1.0) -> geometry.Polygon:
    """Creates a ship polygon from the ship`s position, heading, length and width.

    Args:
        x (float): The ship`s north position
        y (float): The ship`s east position
        heading (float): The ship`s heading
        length (float): Length of the ship
        width (float): Width of the ship
        scale (float, optional): Scale factor. Defaults to 1.0.

    Returns:
        np.ndarray: Ship polygon
    """
    eff_length = length * scale
    eff_width = width * scale

    x_min, x_max = x - eff_length / 2.0, x + eff_length / 2.0 - eff_width
    y_min, y_max = y - eff_width / 2.0, y + eff_width / 2.0
    left_aft, right_aft = (y_min, x_min), (y_max, x_min)
    left_bow, right_bow = (y_min, x_max), (y_max, x_max)
    coords = [left_aft, left_bow, (y, x + eff_length / 2.0), right_bow, right_aft]
    poly = geometry.Polygon(coords)
    return affinity.rotate(poly, -heading, origin=(y, x), use_radians=True)


def find_minimum_depth(vessel_draft, enc: senc.ENC):
    """Find the minimum seabed depth for the given vessel draft (for it to avoid grounding)

    Args:
        vessel_draft (float): The vessel`s draft.

    Returns:
        float: The minimum seabed depth required for a safe journey for the vessel.
    """
    lowest_possible_depth = 0
    for depth in enc.seabed:
        if vessel_draft <= depth:
            lowest_possible_depth = depth
            break
    return lowest_possible_depth


def extract_relevant_grounding_hazards(vessel_min_depth: int, enc: senc.ENC) -> geometry.MultiPolygon:
    """Extracts the relevant grounding hazards from the ENC as a list of polygons.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    # return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]
    relevant_hazards = [enc.land.geometry.union(enc.shore.geometry).union(dangerous_seabed)]
    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        filtered_relevant_hazards.append(geometry.MultiPolygon(geometry.Polygon(p.exterior) for p in hazard.geoms))
    return filtered_relevant_hazards


def create_point_list_from_polygons(polygons: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x and y coordinates of the polygons.
    """
    px, py, ls = [], [], []
    for i, poly in enumerate(polygons):
        y, x = poly.exterior.coords.xy
        a = np.array(x.tolist())
        b = np.array(y.tolist())
        la, lx = len(a), len(px)
        c = [(i + lx, (i + 1) % la + lx) for i in range(la - 1)]
        px += a.tolist()
        py += b.tolist()
        ls += c

    points = np.array([px, py]).T
    P1, P2 = points[ls][:, 0], points[ls][:, 1]
    return P1, P2


def fill_rtree_with_geometries(geometries: list) -> Tuple[strtree.STRtree, list]:
    """Fills an rtree with the given multipolygon geometries. Used for fast spatial queries.

    Args:
        - geometries (list): The geometries to fill the rtree with.

    Returns:
        Tuple[strtree.STRtree, list]: The rtree containing the geometries, and the Polygon objects used to build it.
    """
    poly_list = []
    for poly in geometries:
        assert isinstance(poly, geometry.MultiPolygon), "Only MultiPolygon members are supported"
        for sub_poly in poly.geoms:
            poly_list.append(sub_poly)
    return strtree.STRtree(poly_list), poly_list


def generate_enveloping_polygon(trajectory: np.ndarray, buffer: float) -> geometry.Polygon:
    """Creates an enveloping polygon around the trajectory of the vessel, buffered by the given amount.

    Args:
        - trajectory (np.ndarray): Trajectory with columns [x, y, psi, u, v, r]
        - buffer (float): Buffer size

    Returns:
        geometry.Polygon: The query polygon
    """
    point_list = []
    for k in range(trajectory.shape[1]):
        point_list.append((trajectory[1, k], trajectory[0, k]))
    trajectory_linestring = geometry.LineString(point_list).buffer(buffer)
    return trajectory_linestring


def extract_polygons_near_trajectory(
    trajectory: np.ndarray, geometry_tree: strtree.STRtree, buffer: float, enc: Optional[senc.ENC] = None, show_plots: bool = False
) -> Tuple[list, geometry.Polygon]:
    """Extracts the polygons that are relevant for the trajectory of the vessel, inside a corridor of the given buffer size.

    Args:
        - trajectory (np.ndarray): Trajectory with columns [x, y, psi, u, v, r]
        - geometry_tree (strtree.STRtree): The rtree containing the relevant grounding hazard polygons.
        - buffer (float): Buffer size
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - show_plots (bool, optional): Whether to show plots or not. Defaults to False.

    Returns:
        Tuple[list, geometry.Polygon]: List of tuples of relevant polygons inside query/envelope polygon and the corresponding original polygon they belong to. Also returns the query polygon.
    """
    enveloping_polygon = generate_enveloping_polygon(trajectory, buffer)
    polygons_near_trajectory = geometry_tree.query(enveloping_polygon)
    poly_list = []
    for poly in polygons_near_trajectory:
        relevant_poly_list = []
        intersection_poly = enveloping_polygon.intersection(poly)
        if intersection_poly.area == 0.0 and intersection_poly.length == 0.0:
            continue

        if isinstance(intersection_poly, geometry.MultiPolygon):
            for sub_poly in intersection_poly.geoms:
                relevant_poly_list.append(sub_poly)
        else:
            relevant_poly_list.append(intersection_poly)
        poly_list.append((relevant_poly_list, poly))

    if enc is not None and show_plots:
        enc.start_display()
        enc.draw_polygon(enveloping_polygon, color="yellow", alpha=0.2)
        # for poly_sublist, _ in poly_list:
        #     for poly in poly_sublist:
        #         enc.draw_polygon(poly, color="red", fill=False)

    return poly_list, enveloping_polygon


def extract_safe_sea_area(min_depth: int, enveloping_polygon: geometry.Polygon, enc: Optional[senc.ENC] = None) -> geometry.MultiPolygon:
    """Extracts the safe sea area from the ENC as a list of polygons.

    This includes sea polygons that are above the vessel`s minimum depth.

    Args:
        - polygon_list (list): The list of polygons to check for safe sea area.
        - enveloping_polygon (geometry.Polygon): The query polygon.
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.

    Returns:
        list: The safe sea area.
    """
    safe_sea = enc.seabed[min_depth].geometry.intersection(enveloping_polygon)
    # if enc is not None:
    #     enc.draw_polygon(safe_sea, color="orange", alpha=0.5)
    return safe_sea


def extract_boundary_polygons_inside_envelope(
    poly_tuple_list: list, enveloping_polygon: geometry.Polygon, enc: Optional[senc.ENC] = None, show_plots: bool = True
) -> list:
    """Extracts the boundary trianguled polygons that are relevant for the trajectory of the vessel, inside a corridor of the given buffer size.

    Args:
        - poly_tuple_list (list): List of tuples with relevant polygons inside query/envelope polygon and the corresponding original polygon they belong to.
        - enveloping_polygon (geometry.Polygon): The query polygon.
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - show_plots (bool, optional): Whether to show plots or not. Defaults to False.

    Returns:
        list: List of boundary polygons.
    """
    boundary_polygons = []
    for relevant_poly_list, original_polygon in poly_tuple_list:
        for relevant_polygon in relevant_poly_list:
            triangle_boundaries = extract_triangle_boundaries_from_polygon(relevant_polygon, enveloping_polygon, original_polygon)
            if not triangle_boundaries:
                continue

            if enc is not None and show_plots:
                # enc.draw_polygon(poly, color="pink", alpha=0.3)
                for tri in triangle_boundaries:
                    enc.draw_polygon(tri, color="red", fill=False)

            boundary_polygons.extend(triangle_boundaries)
    return boundary_polygons


def extract_triangle_boundaries_from_polygon(polygon: geometry.Polygon, planning_area_envelope: geometry.Polygon, original_polygon: geometry.Polygon) -> list:
    """Extracts the triangles that comprise the boundary of the polygon.

    Triangles are filtered out if they have two vertices on the envelope boundary and is inside of the original polygon.

    Args:
        - polygon (geometry.Polygon): The polygon in consideration inside the envelope polygon.
        - planning_area_envelope (geometry.Polygon): A polygon representing the relevant area the vessel is planning to navigate in.
        - original_polygon (geometry.Polygon): The original polygon that the relevant polygon belongs to.

    Returns:
        list: List of shapely polygons representing the boundary triangles for the polygon.
    """
    cdt = constrained_delaunay_triangulation_custom(polygon)
    # return cdt
    envelope_boundary = geometry.LineString(planning_area_envelope.exterior.coords).buffer(0.0001)
    original_polygon_boundary = geometry.LineString(original_polygon.exterior.coords).buffer(0.0001)
    boundary_triangles = []
    if len(cdt) == 1:
        return cdt

    for tri in cdt:
        v_count = 0
        idx_prev = 0
        for idx, v in enumerate(tri.exterior.coords):
            if v_count == 2 and idx_prev == idx - 1 and tri not in boundary_triangles:
                boundary_triangles.append(tri)
                break
            v_point = geometry.Point(v)
            if original_polygon_boundary.contains(v_point):
                v_count += 1
                idx_prev = idx

    return boundary_triangles


# def constrained_delaunay_triangulation(polygon: geometry.Polygon) -> list:
#     """Uses the triangle library to compute a constrained delaunay triangulation.

#     Args:
#         polygon (geometry.Polygon): The polygon to triangulate.

#     Returns:
#         list: List of triangles as shapely polygons.
#     """
#     x, y = polygon.exterior.coords.xy
#     vertices = np.array([list(a) for a in zip(x, y)])
#     cdt = tr.triangulate({"vertices": vertices})
#     triangle_indices = cdt["triangles"]
#     triangles = [geometry.Polygon([cdt["vertices"][i] for i in tri]) for tri in triangle_indices]

#     cdt_triangles = []
#     for tri in triangles:
#         intersection_poly = tri.intersection(polygon)

#         if isinstance(intersection_poly, geometry.Point) or isinstance(intersection_poly, geometry.LineString):
#             continue

#         if intersection_poly.area == 0.0:
#             continue

#         # cdt_triangles.append(tri)
#         if isinstance(intersection_poly, geometry.MultiPolygon) or isinstance(intersection_poly, geometry.GeometryCollection):
#             for sub_poly in intersection_poly.geoms:
#                 if sub_poly.area == 0.0 or isinstance(sub_poly, geometry.Point) or isinstance(sub_poly, geometry.LineString):
#                     continue
#                 cdt_triangles.append(sub_poly)
#         else:
#             cdt_triangles.append(intersection_poly)
#     return cdt_triangles


def constrained_delaunay_triangulation_custom(polygon: geometry.Polygon) -> list:
    """Converts a polygon to a list of triangles. Basically constrained delaunay triangulation.

    Args:
        - polygon (geometry.Polygon): The polygon to triangulate.

    Returns:
        list: List of triangles as shapely polygons.
    """
    res_intersection_gdf = gpd.GeoDataFrame(geometry=[polygon])
    # Create ID to identify overlapping polygons
    res_intersection_gdf["TRI_ID"] = res_intersection_gdf.index
    # List to keep triangulated geometries
    triangles = []
    # List to keep the original IDs
    triangle_ids = []
    # Triangulate single or multi-polygons
    for i, _ in res_intersection_gdf.iterrows():
        tri_ = ops.triangulate(res_intersection_gdf.geometry.values[i])
        triangles.append(tri_)
        for _ in range(0, len(tri_)):
            triangle_ids.append(res_intersection_gdf.TRI_ID.values[i])
    # Check if it is a single or multi-polygon
    len_list = len(triangles)
    triangles = np.array(triangles).flatten().tolist()
    # unlist geometries for multi-polygons
    if len_list > 1:
        triangles = [item for sublist in triangles for item in sublist]
    # Create triangulated polygons
    filtered_triangles = gpd.GeoDataFrame(triangles)
    filtered_triangles = filtered_triangles.set_geometry(triangles)
    del filtered_triangles[0]
    # Assign original IDs to each triangle
    filtered_triangles["TRI_ID"] = triangle_ids
    # Create new ID for each triangle
    filtered_triangles["LINK_ID"] = filtered_triangles.index
    # Create centroids from all triangles
    filtered_triangles["centroid"] = filtered_triangles.centroid
    filtered_triangles_centroid = filtered_triangles.set_geometry("centroid")
    del filtered_triangles_centroid["geometry"]
    del filtered_triangles["centroid"]
    # Find triangle centroids inside original polygon
    filtered_triangles_join = gpd.sjoin(
        filtered_triangles_centroid[["centroid", "TRI_ID", "LINK_ID"]], res_intersection_gdf[["geometry", "TRI_ID"]], how="inner", predicate="within"
    )
    # Remove overlapping from other triangles (Necessary for multi-polygons overlapping or close to each other)
    filtered_triangles_join = filtered_triangles_join[filtered_triangles_join["TRI_ID_left"] == filtered_triangles_join["TRI_ID_right"]]
    # Remove overload triangles from same filtered_triangless
    filtered_triangles = filtered_triangles[filtered_triangles["LINK_ID"].isin(filtered_triangles_join["LINK_ID"])]
    filtered_triangles = filtered_triangles.geometry.values
    # double check
    cdt_triangles = []
    for tri in triangles:
        intersection_poly = tri.intersection(polygon)
        if isinstance(intersection_poly, geometry.Point) or isinstance(intersection_poly, geometry.LineString):
            continue
        if intersection_poly.area == 0.0:
            continue

        if isinstance(intersection_poly, geometry.MultiPolygon) or isinstance(intersection_poly, geometry.GeometryCollection):
            for sub_poly in intersection_poly.geoms:
                if sub_poly.area == 0.0 or isinstance(sub_poly, geometry.Point) or isinstance(sub_poly, geometry.LineString):
                    continue
                cdt_triangles.append(sub_poly)
        else:
            cdt_triangles.append(intersection_poly)
    return cdt_triangles


def k_means_clustering_for_polygon(n_clusters: int, polygon: geometry.Polygon, enc: Optional[senc.ENC] = None, show_plots: bool = True) -> list:
    """Performs k-means clustering on the input polygon

    Args:
        n_clusters (int): Number of clusters.
        polygons (Polygon): Shapely polygon
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        list: List of clusters.
    """
    clusters = scipyvq.kmeans2(np.array(polygon.exterior.coords.xy).T, n_clusters, minit="points")[0]
    return clusters


def compute_smallest_enclosing_circle_for_polygons(polygons: list, enc: Optional[senc.ENC] = None, show_plots: bool = True) -> list:
    """Computes the smallest enclosing circle for each polygon in the the input list.

    Args:
        polygons (list): List of shapely polygons
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        list: List of smallest enclosing circles (center.north, center.east, radius) for each polygon.
    """
    circles = []
    for polygon in polygons:
        points = [(p[1], p[0]) for p in polygon.exterior.coords]
        circle = smallestenclosingcircle.make_circle(points)
        circles.append(circle)
        if enc is not None and show_plots:
            enc.draw_circle((circle[1], circle[0]), circle[2], color="red", fill=False)
    return circles


def compute_mvee(points, tol: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1

    Args:
        - points: A np array of points, each row is a point.
        - tol: The tolerance for convergence of the algorithm.

    Returns:
        tuple: A tuple of the ellipse parameters (c (center), a (major axis), b (minor axis), phi (ellipse angle)).
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N) / N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * np.linalg.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
    c = u * points
    A = np.linalg.inv(points.T * np.diag(u) * points - c.T * c) / d
    return c, np.asarray(A)


def compute_smallest_enclosing_ellipse_for_polygons(polygons: list, enc: Optional[senc.ENC] = None, show_plots: bool = True) -> list:
    """Computes smallest enclosing ellipse for each polygon in the list.

    Args:
        polygons (list): List of shapely polygons
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        list: List of ellipse approximations for each polygon.
    """
    ellipses = []
    for poly in polygons:
        y, x = poly.exterior.coords.xy
        c, A = compute_mvee(np.array([x, y]).T)
        ellipses.append((c, A))
        ell_x, ell_y = hf.create_ellipse(c, np.asarray(np.linalg.inv(A)))
        ell = geometry.Polygon(zip(ell_y, ell_x))
        if enc is not None and show_plots:
            enc.draw_polygon(ell, color="red", fill=False)
    return ellipses


def compute_multi_circular_approximations_from_polygons(polygons: list, enc: Optional[senc.ENC] = None, show_plots: bool = True) -> list:
    """Computes multiple circular approximations from the input polygon list.

    Args:
        polygons (list): List of shapely polygons
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        list: List of circular approximations for each polygon.
    """
    circles = []
    for polygon in polygons:
        clusters = k_means_clustering_for_polygon(n_clusters=3, polygon=polygon, enc=enc, show_plots=show_plots)
    return circles


def compute_multi_ellipsoidal_approximations_from_polygons(
    poly_tuple_list: list, planning_area_envelope: geometry.Polygon, enc: Optional[senc.ENC] = None, show_plots: bool = True
) -> list:
    """Computes ellipsoidal approximations from the input polygon list.

    Args:
        - poly_tuple_list (list): List of tuples with relevant polygons inside query/envelope polygon and the corresponding original polygon they belong to.
        - planning_area_envelope (geometry.Polygon): Planning area envelope.
        - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        - show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        list: List of ellipsoidal approximations for each polygon.
    """
    envelope_boundary = geometry.LineString(planning_area_envelope.exterior.coords).buffer(0.0001)
    ellipses = []
    ellipses_per_m2 = 5e-4

    for (polygons, original_poly) in poly_tuple_list:
        original_polygon_boundary = geometry.LineString(original_poly.exterior.coords).buffer(0.0001)
        for polygon in polygons:
            centroid = [polygon.centroid.y, polygon.centroid.x]
            min_y, min_x, max_y, max_x = polygon.bounds
            num_ellipses = max(int(polygon.area * ellipses_per_m2), 1)
            init_mu = np.zeros((num_ellipses, 2))
            init_sigma = np.zeros((num_ellipses, 2, 2))
            for i in range(num_ellipses):
                init_mu[i, :] = centroid + np.random.uniform(low=np.array([min_x, min_y]) - centroid, high=np.array([max_x, max_y]) - centroid, size=(2,))
                init_sigma[i, :, :] = 1e4 * np.eye(2)
            gmm_em_object = gmm_em.GMM_EM(k=num_ellipses, dim=2, init_mu=init_mu, init_sigma=init_sigma, init_pi=None)

            # Remove points that are on the enveloping polygon boundary
            y, x = polygon.exterior.coords.xy
            relevant_boundary_points = []
            for (xcoord, ycoord) in zip(x, y):
                if original_polygon_boundary.contains(geometry.Point(ycoord, xcoord)):
                    relevant_boundary_points.append([xcoord, ycoord])
            relevant_boundary_points = np.array(relevant_boundary_points)

            if enc is not None and show_plots:
                enc.draw_polygon(envelope_boundary, color="green", fill=False)
                enc.draw_polygon(polygon, color="red", fill=False)
                enc.draw_polygon(original_polygon_boundary, color="blue")

            gmm_em_object.init_em(X=relevant_boundary_points)
            mu_c, sigma_c, _ = gmm_em_object.run(num_iters=50)
            for i in range(num_ellipses):
                ellipses.append((mu_c[i, :].T, sigma_c[i, :, :]))
                ell_x, ell_y = hf.create_ellipse(center=mu_c[i, :], A=np.squeeze(sigma_c[i, :, :]))
                ell = geometry.Polygon(zip(ell_y, ell_x))
                if enc is not None and show_plots:
                    enc.draw_polygon(ell, color="orange", fill=False)

    return ellipses


def compute_surface_approximations_from_polygons(
    polygons: list, enc: Optional[senc.ENC] = None, safety_margins: Optional[list] = None, scale_data: bool = True, show_plots: bool = False
) -> list:
    """Computes smooth 2D surface approximations from the input polygon list.

    Args:
        polygons (list): List of shapely polygons
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        safety_margins (Optional[list], optional): List of safety margins to buffer the polygon. Defaults to None.
        scale_data (bool, optional): Whether to scale the data to within -1 and 1, for better numerical conditioning. Defaults to True.
        show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        list: List of surface approximations for each polygon.
    """
    surfaces_list = []
    cap_style = 2
    join_style = 2
    code_gen = csd.CodeGenerator("surface_functions")
    if show_plots:
        ax = plt.figure().add_subplot(111, projection="3d")
        # ax2 = plt.figure().add_subplot(111, projection="3d")
        # ax.axis("equal")
        # ax3 = plt.figure().add_subplot(111)

    if safety_margins is None:
        safety_margins = [0.0]

    j = 0
    for d_safe in safety_margins:
        surfaces = []
        safety_margin_str = "safety_margin_" + str(int(d_safe))
        for polygons, original_poly in polygons:
            original_polygon_boundary = geometry.LineString(original_poly.exterior.coords).buffer(0.5, cap_style=cap_style, join_style=join_style)
            original_polygon_boundary_d_safe = geometry.LineString(original_poly.buffer(d_safe, cap_style=cap_style, join_style=join_style).exterior.coords).buffer(
                0.5, cap_style=cap_style, join_style=join_style
            )
            for polygon in polygons:
                coastline_original = polygon.intersection(original_polygon_boundary)
                n_orig_boundary_points = len(coastline_original.exterior.coords.xy[0])
                coastline = polygon.buffer(d_safe, cap_style=cap_style, join_style=join_style).intersection(original_polygon_boundary_d_safe)
                y_poly_unstructured, x_poly_unstructured = coastline.exterior.coords.xy

                for i in range(len(y_poly_unstructured) - 1):
                    pi = np.array([x_poly_unstructured[i], y_poly_unstructured[i]])
                    pj = np.array([x_poly_unstructured[i + 1], y_poly_unstructured[i + 1]])
                    d2next = np.linalg.norm(pi - pj)
                    if d2next > 35.0:
                        # insert a point in between the two points
                        p_mid = (pi + pj) / 2.0
                        x_poly_unstructured.insert(i + 1, p_mid[0])
                        y_poly_unstructured.insert(i + 1, p_mid[1])
                    # print(f"Distance between vertex {i} and {i+1}: {d2next}")

                mask_unstructured = [1.0] * len(y_poly_unstructured)
                n_boundary_points = len(y_poly_unstructured)
                # print(f"n_boundary_points before: {n_orig_boundary_points} | after: {n_boundary_points}")

                # Add buffer points just outside the polygon coastline, where the mask is zero or negative (no collision)
                step_buffer = 1000.0
                n_levels = 1
                for level in range(n_levels):
                    buff_l = 0.1 + level * step_buffer
                    try:
                        y_poly, x_poly = polygon.buffer(d_safe + buff_l, cap_style=cap_style, join_style=join_style).exterior.coords.xy
                        for (xcoord, ycoord) in zip(x_poly, y_poly):
                            if original_polygon_boundary_d_safe.buffer(buff_l, cap_style=cap_style, join_style=join_style).contains(geometry.Point(ycoord, xcoord)):
                                x_poly_unstructured.append(xcoord)
                                y_poly_unstructured.append(ycoord)
                                mask_unstructured.append(-10.0)
                    except AttributeError:
                        break
                    # if enc is not None and show_plots:
                    #     enc.draw_polygon(coastline.buffer(buff_l, cap_style=cap_style, join_style=join_style), color="orange", fill=False)

                relevant_boundary = polygon.buffer(d_safe + 100.0).intersection(
                    geometry.LineString(original_poly.buffer(d_safe + 100.0).exterior.coords).buffer(1.0, cap_style=cap_style, join_style=join_style)
                )
                y_boundary, x_boundary = relevant_boundary.exterior.coords.xy
                n_boundary_points = 8
                if len(y_boundary) < n_boundary_points:
                    n_boundary_points = len(y_boundary)
                elif len(y_boundary) > 300:
                    n_boundary_points = 20
                step = int(len(y_boundary) / n_boundary_points)
                for i in range(0, len(y_boundary), step):
                    x_poly_unstructured.append(x_boundary[i])
                    y_poly_unstructured.append(y_boundary[i])
                    mask_unstructured.append(-100.0)

                poly_min_east, poly_min_north, poly_max_east, poly_max_north = polygon.buffer(d_safe + 100.0, cap_style=cap_style, join_style=join_style).bounds

                # rbf = scipyintp.RBFInterpolator(
                #     np.array([x_poly_unstructured, y_poly_unstructured]).T, np.array(mask_unstructured), kernel="gaussian", epsilon=0.08, smoothing=1e-3
                # )

                if scale_data:
                    map_xmin, map_ymin, map_xmax, map_ymax = enc.bbox
                    for i in range(len(x_poly_unstructured)):  # pylint: disable=consider-using-enumerate
                        x_poly_unstructured[i] = (x_poly_unstructured[i] - map_ymin) / (map_ymax - map_ymin)
                        y_poly_unstructured[i] = (y_poly_unstructured[i] - map_xmin) / (map_xmax - map_xmin)

                rbf = scipyintp.RBFInterpolator(
                    np.array([x_poly_unstructured, y_poly_unstructured]).T, np.array(mask_unstructured), kernel="thin_plate_spline", epsilon=10.0, smoothing=1e-5
                )
                rbf_csd = rbf_casadi.RBFInterpolator(
                    np.array([x_poly_unstructured, y_poly_unstructured]).T,
                    np.array(mask_unstructured),
                    rbf._coeffs,
                    rbf.powers,
                    rbf._shift,
                    rbf._scale,
                    "thin_plate_spline",
                    rbf.epsilon,
                )
                x = csd.MX.sym("x", 2)
                intp = rbf_csd(x.reshape((1, 2)))
                rbf_surface_func = csd.Function("so_surface_func_" + str(j) + "_" + safety_margin_str, [x.reshape((1, 2))], [intp])
                surfaces.append(rbf_surface_func)

                grad_rbf = csd.gradient(rbf_surface_func(x.reshape((1, 2))), x.reshape((1, 2)))
                grad_rbf_func = csd.Function("grad_f", [x.reshape((1, 2))], [grad_rbf])

                code_gen.add(rbf_surface_func)
                j += 1

                if enc is not None and show_plots:
                    # if j == 2:
                    #     enc.draw_polygon(polygon.buffer(d_safe, cap_style=cap_style, join_style=join_style), color="black", fill=False)
                    #     save_path = dp.figures
                    #     enc.save_image(name="enc_island_polygon", path=save_path, extension="pdf")
                    #     enc.save_image(name="enc_island_polygon", path=save_path, scale=2.0)
                    assert enc is not None
                    buffer = 0.0
                    n_points = 150

                    if scale_data:
                        extra_north_coords = np.linspace(
                            start=(poly_min_north - buffer - map_ymin) / (map_ymax - map_ymin),
                            stop=(poly_max_north + buffer - map_ymin) / (map_ymax - map_ymin),
                            num=n_points,
                        )
                        extra_east_coords = np.linspace(
                            start=(poly_min_east - buffer - map_xmin) / (map_xmax - map_xmin),
                            stop=(poly_max_east + buffer - map_xmin) / (map_xmax - map_xmin),
                            num=n_points,
                        )
                    else:
                        extra_north_coords = np.linspace(start=poly_min_north - buffer, stop=poly_max_north + buffer, num=n_points)
                        extra_east_coords = np.linspace(start=poly_min_east - buffer, stop=poly_max_east + buffer, num=n_points)

                    surface_points = np.zeros((n_points, n_points))
                    surface_grad_points = np.zeros((n_points, n_points, 2))
                    for i, east_coord in enumerate(extra_east_coords):
                        for ii, north_coord in enumerate(extra_north_coords):
                            point = np.array([north_coord, east_coord]).reshape(1, 2)
                            surface_points[i, ii] = rbf_surface_func(point)
                            surface_grad_points[i, ii, :] = grad_rbf_func(point).full().flatten()

                        # ax3.plot(extra_east_coords, surface_points2[i, :], "b")
                        # ax3.plot(np.array(y_poly_unstructured) - poly_min_east, mask_unstructured, "ro")
                        # ax3.set_ylim([-3.1, 1.1])
                        # plt.show(block=False)
                        # ax3.clear()

                    print(f"Number of gradient NaNs: {np.count_nonzero(np.isnan(surface_grad_points))}")
                    yY, xX = np.meshgrid(extra_east_coords, extra_north_coords, indexing="ij")

                    fig, ax4 = plt.subplots()
                    ax4.pcolormesh(yY, xX, surface_points, shading="gouraud")
                    p = ax4.scatter(y_poly_unstructured, x_poly_unstructured, c=np.array(mask_unstructured), s=50, ec="k")
                    fig.colorbar(p)
                    ax4.set_xlabel("East")
                    ax4.set_ylabel("North")

                    ax.plot_surface(yY, xX, surface_points, rcount=200, ccount=200, cmap=cm.coolwarm)
                    ax.set_ylabel("North")
                    ax.set_xlabel("East")
                    ax.set_zlabel("Mask")
                    # ax.set_title("Spline surface")

                    # ax2.plot_surface(xX, yY, surface_grad_points[:, :, 0], rcount=200, ccount=200, cmap=cm.coolwarm)
                    # ax2.set_xlabel("North")
                    # ax2.set_ylabel("East")
                    # ax2.set_zlabel("Mask")
                    # ax2.set_title("Spline surface gradient x")
                    plt.show(block=False)
                    ax.clear()
                    # ax2.clear()
        surfaces_list.append(surfaces)
        code_gen.generate()
    return surfaces_list


def compute_splines_from_polygons(polygons: list, enc: Optional[senc.ENC] = None, show_plots: bool = False) -> Tuple[list, list]:
    """Computes splines from a list of polygons

    Args:
        polygons (list): List of shapely polygons
        enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

    Returns:
        Tuple[list, list]: List of tuples with splines for x and y, and similarly for the derivatives
    """
    splines = []
    spline_derivatives = []
    for polygon in polygons:
        east, north = polygon.exterior.xy
        if len(east) < 3:
            continue

        linspace = np.linspace(0.0, 1.0, len(east))
        spline_x = scipyintp.PchipInterpolator(linspace, north, extrapolate=False)
        spline_y = scipyintp.PchipInterpolator(linspace, east, extrapolate=False)
        splines.append((spline_x, spline_y))
        spline_derivatives.append((spline_x.derivative(), spline_y.derivative()))
        if enc is not None and show_plots:
            enc.start_display()
            x_spline_vals = spline_x(linspace)
            y_spline_vals = spline_y(linspace)
            pairs = list(zip(y_spline_vals, x_spline_vals))
            enc.draw_line(pairs, color="black", width=0)

    return splines, spline_derivatives


def compute_closest_grounding_dist(vessel_trajectory: np.ndarray, minimum_vessel_depth: int, enc: senc.ENC, show_plots: bool = False) -> Tuple[float, np.ndarray, int]:
    """Computes the closest distance to grounding for the given vessel trajectory.

    Args:
        - vessel_trajectory (np.ndarray): The vessel`s trajectory, 2 x n_samples.
        - minimum_vessel_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enc (senc.ENC): The ENC to check for grounding.
        - show_plots (bool, optional): Whether to show plots or not. Defaults to False.

    Returns:
        Tuple[float, int]: The closest distance to grounding, corresponding distance vector and the index of the trajectory point.
    """
    dangerous_seabed = extract_relevant_grounding_hazards(minimum_vessel_depth, enc)
    vessel_traj_linestring = hf.ndarray_to_linestring(vessel_trajectory)
    # if enc and show_plots:
    #     enc.start_display()
    #     for hazard in dangerous_seabed:
    #         enc.draw_polygon(hazard, color="red")
    # intersection_points = find_intersections_line_polygon(vessel_traj_linestring, dangerous_seabed, enc)

    # Will find the first gronding point.
    min_dist = 1e12
    for idx, point in enumerate(vessel_traj_linestring.coords):
        for hazard in dangerous_seabed:
            dist = hazard.distance(geometry.Point(point))
            if dist < min_dist:
                min_dist = dist
                min_idx = idx

    closest_point = geometry.Point(vessel_traj_linestring.coords[min_idx])
    nearest_poly_points = []
    for hazard in dangerous_seabed:
        nearest_point = ops.nearest_points(closest_point, hazard)[1]
        nearest_poly_points.append(nearest_point)

    epsilon = 0.01
    for i, point in enumerate(nearest_poly_points):
        points = [
            (np.asarray(closest_point.coords.xy[0])[0], np.asarray(closest_point.coords.xy[1])[0]),
            (np.asarray(point.coords.xy[0])[0], np.asarray(point.coords.xy[1])[0]),
        ]

        if enc and show_plots:
            enc.draw_line(points, color="cyan", marker_type="o")

        min_dist_vec = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
        if np.linalg.norm(min_dist_vec) <= min_dist + epsilon and np.linalg.norm(min_dist_vec) >= min_dist - epsilon:
            break

    # if enc and show_plots:
    #     enc.close_display()
    return min_dist, min_dist_vec, min_idx


def find_intersections_line_polygon(
    line: geometry.LineString, polygon: geometry.Polygon | geometry.MultiPolygon, enc: Optional[senc.ENC] = None, show_plots: bool = False
) -> np.ndarray:
    """Finds the intersection points between a line and a polygon.

    Args:
        line (geometry.Linestring): Line to intersect with polygon
        polygon (geometry.Polygon | geometry.MultiPolygon): Polygon to check for intersection with line

    Returns:
        list: List of intersection points
    """

    intersection_points = polygon.intersection(line)
    coords = []
    if intersection_points.type == "LineString":
        x, y = intersection_points.coords.xy
        coords = [list(x), list(y)]
    elif intersection_points.type == "MultiLineString":
        for line in intersection_points.geoms:
            if enc and show_plots:
                enc.draw_line(line.coords, color="red", marker_type="o")
            x, y = line.coords.xy
            coords.append([list(x), list(y)])
    return np.array(coords, dtype=object)


class TestMapFunctions(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
