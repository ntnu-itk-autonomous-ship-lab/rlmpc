"""
    map_functions.py

    Summary:
        Contains various commonly used map functions, including
        latlon to local UTM (ENU) coordinate transformation functions etc..

    Author: Trym Tengesdal
"""

from typing import Optional, Tuple

import casadi as csd
import geopandas as gpd
import geopy.distance
import numpy as np
import rlmpc.common.gmm_em as gmm_em
import rlmpc.common.helper_functions as hf
import rlmpc.common.paths as dp
import rlmpc.common.rbf_casadi as rbf_casadi
import rlmpc.common.smallestenclosingcircle as smallestenclosingcircle
import scipy.interpolate as scipyintp
import scipy.spatial as scipy_spatial
import seacharts.enc as senc
import shapely.affinity as affinity
import shapely.geometry as geometry

# import triangle as tr
from osgeo import osr
from shapely import ops, strtree


def local2latlon(
    x: float | list | np.ndarray, y: float | list | np.ndarray, utm_zone: int
) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
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


def latlon2local(
    lat: float | list | np.ndarray, lon: float | list | np.ndarray, utm_zone: int
) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
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


def create_ship_polygon(
    x: float, y: float, heading: float, length: float, width: float, scale: float = 1.0
) -> geometry.Polygon:
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


def extract_relevant_grounding_hazards(
    vessel_min_depth: int, enc: senc.ENC, buffer: Optional[float] = None
) -> geometry.MultiPolygon:
    """Extracts the relevant grounding hazards from the ENC as a multipolygon.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.
        buffer (Optional[float], optional): Buffer size. Defaults to None.

    Returns:
        geometry.MultiPolygon: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    # return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]
    relevant_hazards = [enc.land.geometry.union(enc.shore.geometry).union(dangerous_seabed)]
    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        poly = geometry.MultiPolygon(geometry.Polygon(p.exterior) for p in hazard.geoms)
        if buffer is not None:
            poly = poly.buffer(buffer)
        filtered_relevant_hazards.append(poly)
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
    trajectory: np.ndarray,
    geometry_tree: strtree.STRtree,
    buffer: float,
    enc: Optional[senc.ENC] = None,
    show_plots: bool = False,
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


def extract_vertices_from_polygon_list(polygons: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x (north) and y (east) coordinates of the polygons.
    """
    px, py = [], []
    for i, poly in enumerate(polygons):
        if isinstance(poly, geometry.MultiPolygon):
            for sub_poly in poly:
                y, x = sub_poly.exterior.coords.xy
                px.extend(x[:-1].tolist())
                py.extend(y[:-1].tolist())
        elif isinstance(poly, geometry.Polygon):
            y, x = poly.exterior.coords.xy
            px.extend(x[:-1].tolist())
            py.extend(y[:-1].tolist())
        else:
            continue
    return np.array(px), np.array(py)


def extract_safe_sea_area(
    min_depth: int,
    enveloping_polygon: geometry.Polygon,
    enc: Optional[senc.ENC] = None,
    as_polygon_list: bool = False,
    show_plots: bool = False,
) -> geometry.MultiPolygon | list:
    """Extracts the safe sea area from the ENC as a list of polygons.

    This includes sea polygons that are above the vessel`s minimum depth.

    Args:
        - min_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enveloping_polygon (geometry.Polygon): The query polygon.
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - as_polygon_list (bool, optional): Option for returning the safe sea area as a list of polygons. Defaults to False.
        - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        MultiPolygon | list: The safe sea area.
    """
    safe_sea = enc.seabed[min_depth].geometry.intersection(enveloping_polygon)
    if enc is not None and show_plots:
        enc.start_display()
        enc.draw_polygon(safe_sea, color="green", alpha=0.25, fill=False)

    if as_polygon_list:
        if isinstance(safe_sea, geometry.MultiPolygon):
            return [poly for poly in safe_sea.geoms]
        elif isinstance(safe_sea, geometry.Polygon):
            return [safe_sea]
        else:
            return []
    return safe_sea


def create_free_boundary_points_from_enc(enc: senc.ENC, hazards: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates an array of points on the ENC boundary which is free from grounding hazards.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        hazards (list): List of relevant grounding hazards.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and y coordinates of the free boundary points.
    """
    (xmin, ymin, xmax, ymax) = enc.bbox
    n_pts_per_side = 50
    x = np.linspace(xmin, xmax, n_pts_per_side)
    y = np.linspace(ymin, ymax, n_pts_per_side)
    points = []
    for i in range(n_pts_per_side):
        p = geometry.Point(x[i], ymin)
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = geometry.Point(x[i], ymax)
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = geometry.Point(xmin, y[i])
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = geometry.Point(xmax, y[i])
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)
    # [enc.draw_circle((p.x, p.y), radius=0.5, color="yellow") for p in points]
    Y = np.array([p.x for p in points])
    X = np.array([p.y for p in points])
    return X, Y


def bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> geometry.Polygon:
    """Converts a bounding box to a polygon.

    Args:
        bbox (Tuple[float, float, float, float]): The bounding box.

    Returns:
        geometry.Polygon: The polygon.
    """
    (xmin, ymin, xmax, ymax) = bbox
    return geometry.Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])


def point_in_polygon_list(point: geometry.Point, polygons: list) -> bool:
    """Checks if a point is in a list of polygons.

    Args:
        point (Point): The point to check.
        polygons (list): List of polygons.

    Returns:
        bool: True if the point is in a hazard, False otherwise.
    """
    for poly in polygons:
        if point.within(poly) or point.touches(poly):
            return True
    return False


def point_in_point_list(point: geometry.Point, points: list) -> bool:
    """Checks if a point is in a list of points.

    Args:
        point (Point): The point to check.
        points (list): List of points.

    Returns:
        bool: True if the point is in a hazard, False otherwise.
    """
    for p in points:
        if point.within(p) or point.touches(p):
            return True
    return False


def create_safe_sea_voronoi_diagram(enc: senc.ENC, vessel_min_depth: int = 5) -> Tuple[scipy_spatial.Voronoi, list]:
    """Creates a Voronoi diagram of the safe sea region (i.e. its vertices).

    Args:
        enc (ENC): The Electronic Navigational Chart object.
        vessel_min_depth (float): The safe minimum depth for the vessel to voyage in.

    Returns:
        scipy_spatial.Voronoi: The Voronoi diagram of the safe sea region.
    """
    bbox = enc.bbox
    enc_bbox_poly = bbox_to_polygon(bbox)
    safe_sea = extract_safe_sea_area(vessel_min_depth, enc_bbox_poly, enc, as_polygon_list=True, show_plots=True)
    polygons = []
    for sea_poly in safe_sea:
        if isinstance(sea_poly, geometry.MultiPolygon):
            for poly in sea_poly:
                polygons.append(poly)
        elif isinstance(sea_poly, geometry.Polygon):
            polygons.append(sea_poly)
        else:
            continue
    px, py = extract_vertices_from_polygon_list(polygons)
    points = np.vstack((py, px)).T
    vor = scipy_spatial.Voronoi(points)
    region_polygons = create_region_polygons_from_voronoi(vor, enc=enc)
    for point in points:
        enc.draw_circle((point[0], point[1]), radius=0.4, color="red")

    # # Keep all voronoi region boundary points that are in the safe sea area
    # safe_points = []
    # for region in vor.regions:
    #     region_vertices = vor.vertices[region]
    #     for vertex in region_vertices:
    #         point = Point(vertex)
    #         if point_in_polygon_list(point, polygons):
    #             safe_points.append((point.x, point.y))
    # settt = set(safe_points)
    # safe_points = list(settt)
    # for point in safe_points:
    #     enc.draw_circle((point[0], point[1]), radius=0.4, color="magenta")
    return vor, region_polygons


def create_safe_sea_triangulation(enc: senc.ENC, vessel_min_depth: int = 5, show_plots: bool = False) -> list:
    """Creates a constrained delaunay triangulation of the safe sea region.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        vessel_min_depth (int, optional): The safe minimum depth for the vessel to voyage in. Defaults to 5.

    Returns:
        list: List of triangles.
    """
    safe_sea_poly_list = extract_safe_sea_area(
        vessel_min_depth, bbox_to_polygon(enc.bbox), enc, as_polygon_list=True, show_plots=show_plots
    )
    cdt_list = []
    largest_poly_area = 0.0
    for poly in safe_sea_poly_list:
        cdt = constrained_delaunay_triangulation_custom(poly)
        if poly.area > largest_poly_area:
            largest_poly_area = poly.area
            cdt_largest = cdt
        if show_plots:
            enc.draw_polygon(poly, color="orange", alpha=0.5)
            for triangle in cdt:
                enc.draw_polygon(triangle, color="black", fill=False)
        cdt_list.append(cdt)
    # for triangle in cdt_largest:
    #    print("n_tri_vertices: ", len(triangle.exterior.coords))
    return cdt_largest


def create_region_polygons_from_voronoi(vor: scipy_spatial.Voronoi, enc: Optional[senc.ENC] = None) -> list:
    """Creates a list of polygons from the Voronoi diagram.

    Args:
        vor (scipy_spatial.Voronoi): The Voronoi diagram.
        enc (Optional[ENC], optional): The Electronic Navigational Chart object. Defaults to None.

    Returns:
        list: List of polygons.
    """
    polygons = []
    for region in vor.regions:
        if not region:
            continue
        region_vertices = vor.vertices[region]
        if region_vertices.shape[0] < 3:
            continue
        region_poly = geometry.Polygon(region_vertices)
        if region_poly.area < 1.0:
            continue
        polygons.append(region_poly)
        if enc:
            enc.start_display()
            enc.draw_polygon(region_poly, color="yellow", alpha=0.5)
    return polygons


def extract_boundary_polygons_inside_envelope(
    poly_tuple_list: list, enveloping_polygon: geometry.Polygon, enc: Optional[senc.ENC] = None, show_plots: bool = True
) -> list:
    """Extracts the boundary trianguled polygons that are relevant for the trajectory of the vessel, inside the given envelope polygon.

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
            triangle_boundaries = extract_triangle_boundaries_from_polygon(
                relevant_polygon, enveloping_polygon, original_polygon
            )
            if not triangle_boundaries:
                continue

            if enc is not None and show_plots:
                # enc.draw_polygon(poly, color="pink", alpha=0.3)
                for tri in triangle_boundaries:
                    enc.draw_polygon(tri, color="red", fill=False)

            boundary_polygons.extend(triangle_boundaries)
    return boundary_polygons


def extract_triangle_boundaries_from_polygon(
    polygon: geometry.Polygon, planning_area_envelope: geometry.Polygon, original_polygon: geometry.Polygon
) -> list:
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
        filtered_triangles_centroid[["centroid", "TRI_ID", "LINK_ID"]],
        res_intersection_gdf[["geometry", "TRI_ID"]],
        how="inner",
        predicate="within",
    )
    # Remove overlapping from other triangles (Necessary for multi-polygons overlapping or close to each other)
    filtered_triangles_join = filtered_triangles_join[
        filtered_triangles_join["TRI_ID_left"] == filtered_triangles_join["TRI_ID_right"]
    ]
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

        if isinstance(intersection_poly, geometry.MultiPolygon) or isinstance(
            intersection_poly, geometry.GeometryCollection
        ):
            for sub_poly in intersection_poly.geoms:
                if (
                    sub_poly.area == 0.0
                    or isinstance(sub_poly, geometry.Point)
                    or isinstance(sub_poly, geometry.LineString)
                ):
                    continue
                cdt_triangles.append(sub_poly)
        else:
            cdt_triangles.append(intersection_poly)
    return cdt_triangles


def compute_smallest_enclosing_circle_for_polygons(
    polygons: list,
    enc: Optional[senc.ENC] = None,
    map_origin: np.ndarray = np.array([0.0, 0.0]),
    show_plots: bool = True,
) -> list:
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
            enc.draw_circle((circle[1] + map_origin[1], circle[0] + map_origin[0]), circle[2], color="red", fill=False)
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


def compute_smallest_enclosing_ellipse_for_polygons(
    polygons: list,
    enc: Optional[senc.ENC] = None,
    map_origin: np.ndarray = np.array([0.0, 0.0]),
    show_plots: bool = True,
) -> list:
    """Computes smallest enclosing ellipse for each polygon in the list.

    Args:
        - polygons (list): List of shapely polygons
        - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        - map_origin: Origin of the map (north, east) in meters. Defaults to np.array([0.0, 0.0]).
        - show_plots (bool, optional): Whether to show plots. Defaults to False.

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
            enc.draw_polygon(hf.translate_polygons([ell], -map_origin[1], -map_origin[0]), color="red", fill=False)
    return ellipses


def compute_multi_ellipsoidal_approximations_from_polygons(
    poly_tuple_list: list,
    planning_area_envelope: geometry.Polygon,
    enc: Optional[senc.ENC] = None,
    map_origin: np.ndarray = np.array([0.0, 0.0]),
    show_plots: bool = True,
) -> list:
    """Computes ellipsoidal approximations from the input polygon list.

    Args:
        - poly_tuple_list (list): List of tuples with relevant polygons inside query/envelope polygon and the corresponding original polygon they belong to.
        - planning_area_envelope (geometry.Polygon): Planning area envelope.
        - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        - map_origin (np.ndarray, optional): Origin of the map (north, east) in meters. Defaults to np.array([0.0, 0.0]).
        - show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        list: List of ellipsoidal approximations for each polygon.
    """
    envelope_boundary = geometry.LineString(planning_area_envelope.exterior.coords).buffer(0.0001)
    ellipses = []
    ellipses_per_m2 = 5e-4

    for polygons, original_poly in poly_tuple_list:
        original_polygon_boundary = geometry.LineString(original_poly.exterior.coords).buffer(0.0001)
        for polygon in polygons:
            centroid = [polygon.centroid.y, polygon.centroid.x]
            min_y, min_x, max_y, max_x = polygon.bounds
            num_ellipses = max(int(polygon.area * ellipses_per_m2), 1)
            init_mu = np.zeros((num_ellipses, 2))
            init_sigma = np.zeros((num_ellipses, 2, 2))
            for i in range(num_ellipses):
                init_mu[i, :] = centroid + np.random.uniform(
                    low=np.array([min_x, min_y]) - centroid, high=np.array([max_x, max_y]) - centroid, size=(2,)
                )
                init_sigma[i, :, :] = 1e4 * np.eye(2)
            gmm_em_object = gmm_em.GMM_EM(k=num_ellipses, dim=2, init_mu=init_mu, init_sigma=init_sigma, init_pi=None)

            # Remove points that are on the enveloping polygon boundary
            y, x = polygon.exterior.coords.xy
            relevant_boundary_points = []
            for xcoord, ycoord in zip(x, y):
                if original_polygon_boundary.contains(geometry.Point(ycoord, xcoord)):
                    relevant_boundary_points.append([xcoord, ycoord])
            relevant_boundary_points = np.array(relevant_boundary_points)

            if enc is not None and show_plots:
                enc.draw_polygon(
                    hf.translate_polygons([envelope_boundary], -map_origin[1], -map_origin[0])[0],
                    color="green",
                    fill=False,
                )
                enc.draw_polygon(
                    hf.translate_polygons([polygon], -map_origin[1], -map_origin[0])[0], color="red", fill=False
                )
                enc.draw_polygon(
                    hf.translate_polygons([original_polygon_boundary], -map_origin[1], -map_origin[0])[0], color="blue"
                )

            gmm_em_object.init_em(X=relevant_boundary_points)
            mu_c, sigma_c, _ = gmm_em_object.run(num_iters=50)
            for i in range(num_ellipses):
                ellipses.append((mu_c[i, :].T, sigma_c[i, :, :]))
                ell_x, ell_y = hf.create_ellipse(center=mu_c[i, :], A=np.squeeze(sigma_c[i, :, :]))
                ell = geometry.Polygon(zip(ell_y, ell_x))
                if enc is not None and show_plots:
                    enc.draw_polygon(
                        hf.translate_polygons([ell], -map_origin[1], -map_origin[0])[0], color="orange", fill=False
                    )

    return ellipses


def compute_surface_approximations_from_polygons(
    polygons: list,
    enc: Optional[senc.ENC] = None,
    safety_margins: list = [0.0],
    map_origin: np.ndarray = np.array([0.0, 0.0]),
    show_plots: bool = False,
) -> Tuple[list, list]:
    """Computes smooth 2D surface approximations from the input polygon list.

    Args:
        - polygons (list): List of tuples containing 1) Polygons inside a trajectory buffered region and 2) the original polygon in the map.
        - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.
        - safety_margins (Optional[list], optional): List of safety margins to buffer the polygon. Defaults to None.
        - map_origin (np.ndarray, optional): Map origin. Defaults to np.array([0.0, 0.0]).
        - show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        - Tuple[list, list]: List of casadi and scipy surface approximations for each polygon, respectively.
    """
    surfaces_list = []
    scipy_surfaces_list = []
    cap_style = 2
    join_style = 2
    rel_bbox = list(enc.bbox)
    rel_bbox[0] -= map_origin[1]
    rel_bbox[1] -= map_origin[0]
    rel_bbox[2] -= map_origin[1]
    rel_bbox[3] -= map_origin[0]
    bbox_poly = bbox_to_polygon(tuple(rel_bbox))
    j = 0
    for d_safe in safety_margins:
        d_safe = d_safe + 0.05  # buffer to account for function slope not being infinite
        surfaces = []
        scipy_surfaces = []
        safety_margin_str = "safety_margin_" + str(int(d_safe))
        for polygons, original_poly in polygons:
            original_polygon_boundary = geometry.LineString(original_poly.exterior.coords).buffer(
                0.5, cap_style=cap_style, join_style=join_style
            )
            original_polygon_boundary_d_safe = geometry.LineString(
                original_poly.buffer(d_safe, cap_style=cap_style, join_style=join_style).exterior.coords
            ).buffer(0.5, cap_style=cap_style, join_style=join_style)
            for polygon in polygons:
                # Extract the relevant polygon coastline  and safety buffered polygon coastline
                relevant_coastline = polygon.intersection(original_polygon_boundary).intersection(bbox_poly)
                # Check if the relevant coastline is a multipolygon and choose the largest one
                if isinstance(relevant_coastline, geometry.MultiPolygon):
                    max_area_idx = np.argmax([poly.area for poly in relevant_coastline.geoms])
                    relevant_coastline = relevant_coastline.geoms[max_area_idx]

                if enc is not None and show_plots:
                    enc.start_display()
                    translated_poly = hf.translate_polygons([polygon], -map_origin[1], -map_origin[0])[0]
                    translated_coastline = hf.translate_polygons([relevant_coastline], -map_origin[1], -map_origin[0])[
                        0
                    ]
                    enc.draw_polygon(
                        translated_coastline.buffer(0.0, cap_style=cap_style, join_style=join_style),
                        color="green",
                        fill=False,
                    )
                relevant_coastline_safety = (
                    polygon.buffer(d_safe, cap_style=cap_style, join_style=join_style)
                    .intersection(original_polygon_boundary_d_safe)
                    .intersection(bbox_poly)
                )
                # Again, check if the safety relevant coastline is a multipolygon and choose the largest one
                if isinstance(relevant_coastline_safety, geometry.MultiPolygon):
                    max_area_idx = np.argmax([poly.area for poly in relevant_coastline_safety.geoms])
                    relevant_coastline_safety = relevant_coastline_safety.geoms[max_area_idx]
                y_coastline_orig, x_coastline_orig = relevant_coastline_safety.exterior.coords.xy

                if len(y_coastline_orig) < 3:
                    continue

                x_coastline_interp, y_coastline_interp, arc_length = hf.create_arc_length_spline(
                    x_coastline_orig, y_coastline_orig
                )

                # Tuning parameter
                orig_point_spacing = max(7.0, 0.008 * arc_length[-1])
                # if show_plots:
                #     print(
                #         f"Polygon {j}: Relevant coastline arc length: {arc_length[-1]} | distance spacing: {orig_point_spacing}"
                #     )

                y_surface_points = list(y_coastline_interp(np.arange(0, arc_length[-1], orig_point_spacing)))
                x_surface_points = list(x_coastline_interp(np.arange(0, arc_length[-1], orig_point_spacing)))
                x_surface_data_points = []
                y_surface_data_points = []
                for xcoord, ycoord in zip(x_surface_points, y_surface_points):
                    if point_is_within_distance_of_points_in_list(
                        xcoord,
                        ycoord,
                        points=zip(x_surface_data_points, y_surface_data_points),
                        distance=0.2 * orig_point_spacing,
                    ):
                        continue

                    x_surface_data_points.append(xcoord)
                    y_surface_data_points.append(ycoord)
                    if enc is not None and show_plots:
                        enc.draw_circle(
                            (ycoord + map_origin[1], xcoord + map_origin[0]), radius=0.5, color="blue", fill=False
                        )

                if enc is not None and show_plots:
                    translated_coastline = hf.translate_polygons(
                        [relevant_coastline_safety], -map_origin[1], -map_origin[0]
                    )[0]
                    enc.draw_polygon(
                        translated_coastline.buffer(0.0, cap_style=cap_style, join_style=join_style),
                        color="orange",
                        fill=False,
                    )

                n_surface_data_points = len(y_surface_data_points)
                x_surface_points_before_buffering = x_surface_data_points.copy()
                y_surface_points_before_buffering = y_surface_data_points.copy()
                mask_surface_data_points = [1.0] * len(y_surface_data_points)
                # if show_plots:
                #     print(
                #         f"n_surface_points before: {len(y_coastline_orig)} | after interpolation: {n_surface_data_points}"
                #     )

                # Add buffer points just outside the relevant polygon coastline, where the mask is zero or negative (no collision)
                buffer_distance = 0.5
                y_poly, x_poly = polygon.buffer(
                    d_safe + buffer_distance, cap_style=cap_style, join_style=join_style
                ).exterior.coords.xy
                x_poly_spline, y_poly_spline, arc_length_poly = hf.create_arc_length_spline(x_poly, y_poly)

                surface_value_at_buffer_points = -0.5
                y_poly_spaced = list(y_poly_spline(np.arange(0, arc_length_poly[-1], orig_point_spacing)))
                x_poly_spaced = list(x_poly_spline(np.arange(0, arc_length_poly[-1], orig_point_spacing)))
                y_buffer_points = []
                x_buffer_points = []

                relevant_coastline_safety_buffered = (
                    polygon.buffer(d_safe + buffer_distance, cap_style=cap_style, join_style=join_style)
                    .intersection(
                        original_polygon_boundary_d_safe.buffer(
                            buffer_distance, cap_style=cap_style, join_style=join_style
                        )
                    )
                    .intersection(bbox_poly)
                )
                try:
                    for xcoord, ycoord in zip(x_poly_spaced, y_poly_spaced):
                        # if point_is_within_distance_of_points_in_list(
                        #     xcoord,
                        #     ycoord,
                        #     points=zip(x_buffer_points, y_buffer_points),
                        #     distance=0.5 * orig_point_spacing,
                        # ):
                        #     continue

                        if relevant_coastline_safety_buffered.buffer(0.2).contains(geometry.Point(ycoord, xcoord)):
                            x_buffer_points.append(xcoord)
                            y_buffer_points.append(ycoord)
                            mask_surface_data_points.append(surface_value_at_buffer_points)
                            # if enc is not None:
                            #     enc.draw_circle(
                            #         (ycoord + map_origin[1], xcoord + map_origin[0]),
                            #         radius=0.5,
                            #         color="yellow",
                            #         fill=False,
                            #     )
                except AttributeError:
                    break
                x_surface_data_points.extend(x_buffer_points)
                y_surface_data_points.extend(y_buffer_points)
                # if show_plots:
                #     print(f"n_surface_points after buffer points: {len(y_surface_data_points)}")

                ## Add more buffer points further away from the relevant polygon coastline, where the mask is zero or negative (no collision)
                buffer_distance = 1.0
                surface_value_at_outlier_points = -1.0
                relevant_coastline_extra_buffered = polygon.buffer(d_safe + buffer_distance)

                # if enc is not None:
                #     translated_poly = hf.translate_polygons(
                #         [polygon.buffer(d_safe + buffer_distance, cap_style=cap_style, join_style=join_style)],
                #         -map_origin[1],
                #         -map_origin[0],
                #     )[0]
                #     enc.draw_polygon(translated_poly, color="black", fill=False)
                #     translated_orig_b = hf.translate_polygons(
                #         [
                #             original_polygon_boundary_d_safe.buffer(
                #                 buffer_distance, cap_style=cap_style, join_style=join_style
                #             )
                #         ],
                #         -map_origin[1],
                #         -map_origin[0],
                #     )[0]
                #     enc.draw_polygon(translated_orig_b, color="blue", fill=False)
                #     translated_coastline = hf.translate_polygons(
                #         [relevant_coastline_extra_buffered], -map_origin[1], -map_origin[0]
                #     )[0]
                #     enc.draw_polygon(
                #         translated_coastline.buffer(0.0, cap_style=cap_style, join_style=join_style),
                #         color="green",
                #         fill=False,
                #     )
                # Again, check if the relevant coastline is a multipolygon and choose the largest one
                if isinstance(relevant_coastline_extra_buffered, geometry.MultiPolygon):
                    max_area_idx = np.argmax([poly.area for poly in relevant_coastline_extra_buffered.geoms])
                    relevant_coastline_extra_buffered = relevant_coastline_extra_buffered.geoms[max_area_idx]
                y_buffered_boundary, x_buffered_boundary = relevant_coastline_extra_buffered.exterior.coords.xy

                if False:  # enc is not None and show_plots:
                    translated_extra_buff_coastline = hf.translate_polygons(
                        [relevant_coastline_extra_buffered], -map_origin[1], -map_origin[0]
                    )[0]
                    enc.draw_polygon(
                        translated_extra_buff_coastline.buffer(0.0, cap_style=cap_style, join_style=join_style),
                        color="cyan",
                        fill=False,
                    )

                polygon_safety = polygon.buffer(d_safe, cap_style=cap_style, join_style=join_style)
                (
                    x_extra_boundary_spline,
                    y_extra_boundary_spline,
                    arc_length_extra_boundary,
                ) = hf.create_arc_length_spline(x_buffered_boundary, y_buffered_boundary)
                # Tuning parameter
                extra_buffer_point_distance_spacing = max(50.0, 0.02 * arc_length_extra_boundary[-1])
                y_extra_boundary = list(
                    y_extra_boundary_spline(
                        np.arange(0, arc_length_extra_boundary[-1], extra_buffer_point_distance_spacing)
                    )
                )
                x_extra_boundary = list(
                    x_extra_boundary_spline(
                        np.arange(0, arc_length_extra_boundary[-1], extra_buffer_point_distance_spacing)
                    )
                )
                x_buffer_points = []
                y_buffer_points = []
                for xcoord, ycoord in zip(x_extra_boundary, y_extra_boundary):
                    if point_is_within_distance_of_points_in_list(
                        xcoord,
                        ycoord,
                        points=zip(x_buffer_points, y_buffer_points),
                        distance=0.5 * extra_buffer_point_distance_spacing,
                    ):
                        continue
                    x_buffer_points.append(xcoord)
                    y_buffer_points.append(ycoord)
                    mask_surface_data_points.append(surface_value_at_outlier_points)
                    if False:  # enc is not None and show_plots:
                        enc.draw_circle(
                            (ycoord + map_origin[1], xcoord + map_origin[0]), radius=1.0, color="black", fill=False
                        )
                x_surface_data_points.extend(x_buffer_points)
                y_surface_data_points.extend(y_buffer_points)
                # if show_plots:
                #     print(f"extra_buffer_point_distance_spacing: {extra_buffer_point_distance_spacing}")
                # if show_plots:
                #     print(f"Polygon {j}: num total surface data points: {len(y_surface_data_points)}")
                smoothing = 5.0

                rbf = scipyintp.RBFInterpolator(
                    np.array([x_surface_data_points, y_surface_data_points]).T,
                    np.array(mask_surface_data_points),
                    kernel="thin_plate_spline",
                    epsilon=1.0,
                    smoothing=smoothing,
                )
                rbf_csd = rbf_casadi.RBFInterpolator(
                    np.array([x_surface_data_points, y_surface_data_points]).T,
                    np.array(mask_surface_data_points),
                    rbf._coeffs,
                    rbf.powers,
                    rbf._shift,
                    rbf._scale,
                    "thin_plate_spline",
                    rbf.epsilon,
                )
                x = csd.MX.sym("x", 2)
                intp = rbf_csd(x.reshape((1, 2)))
                rbf_surface_func = csd.Function(
                    "so_surface_func_" + str(j) + "_" + safety_margin_str, [x.reshape((1, 2))], [intp]
                )
                grad_rbf = csd.gradient(rbf_surface_func(x.reshape((1, 2))), x.reshape((1, 2)))
                grad_rbf_func = csd.Function("grad_f", [x.reshape((1, 2))], [grad_rbf])

                scipy_surfaces.append(rbf)
                surfaces.append(rbf_surface_func)

                if False:  # enc is not None and show_plots:
                    hf.plot_surface_approximation_stuff(
                        radial_basis_function=rbf_surface_func,
                        radial_basis_function_gradient=grad_rbf_func,
                        surface_data_points=(x_surface_data_points, y_surface_data_points),
                        surface_data_point_mask=mask_surface_data_points,
                        surface_data_points_before_buffering=(
                            x_surface_points_before_buffering,
                            y_surface_points_before_buffering,
                        ),
                        original_polygon=original_poly,
                        polygon=polygon,
                        polygon_safety=polygon_safety,
                        polygon_index=j,
                        relevant_coastline_safety=relevant_coastline_safety,
                        d_safe=d_safe,
                        map_origin=map_origin,
                        enc=enc,
                    )

                j += 1
        surfaces_list.append(surfaces)
        scipy_surfaces_list.append(scipy_surfaces)
    return surfaces_list, scipy_surfaces_list


def point_is_within_distance_of_points_in_list(x: float, y: float, points: list, distance: float) -> bool:
    """Checks if a point is within a certain distance of any point in a list.

    Args:
        x (float): Point x-coordinate
        y (float): Point y-coordinate
        points (list): List of points
        distance (float): Distance threshold

    Returns:
        bool: True if point is within distance of any point in list, False otherwise.
    """
    return np.any([np.linalg.norm(np.array([x, y]) - np.array([x_, y_])) < distance for x_, y_ in points])


def find_intersections_line_polygon(
    line: geometry.LineString,
    polygon: geometry.Polygon | geometry.MultiPolygon,
    enc: Optional[senc.ENC] = None,
    show_plots: bool = False,
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


def create_path_linestring_from_splines(
    splines: list, spline_coeffs: list, final_path_variable_value: float
) -> geometry.LineString:
    """Creates a path linestring from a list of splines.

    Args:
        splines (list): List of (xy) splines
        spline_coeffs (list): List of spline coefficients
        final_path_variable_value (float): Path variable end value

    Returns:
        geometry.LineString: Path linestring
    """
    path_points = []
    path_values = np.linspace(0.001, final_path_variable_value, 100)
    for s in path_values:
        x = splines[0](s)
        y = splines[1](s)
        path_points.append([x, y])
    path_linestring = geometry.LineString(path_points)
    return path_linestring


def find_closest_arclength_to_point(p: np.ndarray, path_linestring: geometry.LineString) -> float:
    """Finds the closest arclength to a point on a path linestring.

    Args:
        p (np.ndarray): Point
        path_linestring (geometry.LineString): Path linestring

    Returns:
        float: Closest arclength
    """
    closest_arclength = path_linestring.project(geometry.Point(p))
    return closest_arclength
