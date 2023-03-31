"""
    map_functions.py

    Summary:
        Contains various commonly used map functions, including
        latlon to local UTM (ENU) coordinate transformation functions etc..

    Author: Trym Tengesdal
"""
import unittest
from typing import Optional, Tuple

import geopandas as gpd
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import seacharts.enc as senc
import shapely.affinity as affinity
import shapely.geometry as geometry

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


def extract_relevant_grounding_hazards(vessel_min_depth: int, enc: senc.ENC) -> list:
    """Extracts the relevant grounding hazards from the ENC as a list of polygons.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]


def fill_rtree_with_geometries(geometries: list) -> Tuple[strtree.STRtree, list]:
    """Fills an rtree with the given multipolygon geometries. Used for fast spatial queries.

    Args:
        geometries (list): List of shapely Multipolygon geometries

    Returns:
        Tuple[strtree.STRtree, list]: The rtree containing the geometries, and the Polygon objects used to build it.
    """
    poly_list = []
    for geom in geometries:
        assert isinstance(geom, geometry.MultiPolygon), "Only multipolygons are supported"
        for poly in geom.geoms:
            poly_list.append(poly)
    return strtree.STRtree(poly_list), poly_list


def generate_enveloping_polygon(trajectory: np.ndarray, buffer: float) -> geometry.Polygon:
    """Creates an enveloping polygon around the trajectory of the vessel, buffered by the given amount.

    Args:
        trajectory (np.ndarray): Trajectory with columns [x, y, psi, u, v, r]
        buffer (float): Buffer size

    Returns:
        geometry.Polygon: The query polygon
    """
    point_list = []
    for k in range(trajectory.shape[1]):
        point_list.append((trajectory[1, k], trajectory[0, k]))
    trajectory_linestring = geometry.LineString(point_list).buffer(buffer)
    return trajectory_linestring


def extract_polygons_near_trajectory(trajectory: np.ndarray, geometry_tree: strtree.STRtree, buffer: float, enc: Optional[senc.ENC] = None) -> list:
    """Extracts the polygons that are relevant for the trajectory of the vessel, inside a corridor of the given buffer size.

    Args:
        trajectory (np.ndarray): Trajectory with columns [x, y, psi, u, v, r]
        geometry_tree (strtree.STRtree): The rtree containing the relevant grounding hazard polygons.
        buffer (float): Buffer size
        enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.

    Returns:
        list: List of relevant grounding hazard polygons intersecting the trajectory corridor.
    """
    enveloping_polygon = generate_enveloping_polygon(trajectory, buffer)
    polygons_relevant_for_trajectory = geometry_tree.query(enveloping_polygon)
    poly_list = []
    for poly in polygons_relevant_for_trajectory:
        intersection_poly = enveloping_polygon.intersection(poly)

        if intersection_poly.area == 0.0 and intersection_poly.length == 0.0:
            continue

        if isinstance(intersection_poly, geometry.MultiPolygon):
            for sub_poly in intersection_poly.geoms:
                poly_list.append(sub_poly)
        else:
            poly_list.append(intersection_poly)

    if enc is not None:
        enc.start_display()
        enc.draw_polygon(enveloping_polygon, color="yellow", alpha=0.1)
        # for poly in poly_list:
        #     enc.draw_polygon(poly, color="black")

    return poly_list


def extract_triangle_boundaries_from_polygons(polygons: list, enc: Optional[senc.ENC] = None) -> list:
    """Computes CDT for all polygons in the list, and returns a list of triangles comprising the boundary of each polygon.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        list: List of list of shapely polygons representing the boundary triangles for each polygon.
    """
    poly_boundary_list = []
    for poly in polygons:
        cdt = constrained_delaunay_triangulatio_custom(poly)
        boundary_triangles = extract_triangle_boundaries_from_polygon(cdt, poly)
        if len(boundary_triangles) == 0:
            boundary_triangles = cdt
        poly_boundary_list.append(boundary_triangles)
        if enc is not None:
            # for tri in cdt:
            #     enc.draw_polygon(tri, color="orange", alpha=1.0, fill=False)
            # enc.draw_polygon(poly, color="pink", alpha=0.3)
            for tri in boundary_triangles:
                enc.draw_polygon(tri, color="red", fill=False)

    return poly_boundary_list


def extract_triangle_boundaries_from_polygon(cdt: list, polygon: geometry.Polygon) -> list:
    """Extracts the triangles that comprise the boundary of the polygon.

    Args:
        cdt (list): List of shapely polygons representing the CDT for the polygon.
        polygon (geometry.Polygon): The polygon.

    Returns:
        list: List of shapely polygons representing the boundary triangles for the polygon.
    """
    boundary_triangles = []
    if len(cdt) == 1:
        return cdt

    for tri in cdt:
        boundary_line = geometry.LineString(tri.exterior.coords)
        v_count = 0
        idx_prev = 0
        for idx, v in enumerate(polygon.exterior.coords):
            if v_count == 2 and idx_prev == idx - 1 and tri not in boundary_triangles:
                boundary_triangles.append(tri)
                break

            if boundary_line.contains(geometry.Point(v)):
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
        polygon (geometry.Polygon): The polygon to triangulate.

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


def linestring_to_ndarray(line: geometry.LineString) -> np.ndarray:
    """Converts a shapely LineString to a numpy array

    Args:
        line (LineString): Any LineString object

    Returns:
        np.ndarray: Numpy array containing the coordinates of the LineString
    """
    return np.array(line.coords).transpose()


def ndarray_to_linestring(array: np.ndarray) -> geometry.LineString:
    """Converts a 2D numpy array to a shapely LineString

    Args:
        array (np.ndarray): Numpy array of 2 x n_samples, containing the coordinates of the LineString

    Returns:
        LineString: Any LineString object
    """
    assert array.shape[0] == 2 and array.shape[1] > 1, "Array must be 2 x n_samples with n_samples > 1"
    return geometry.LineString(list(zip(array[0, :], array[1, :])))


def compute_closest_grounding_dist(vessel_trajectory: np.ndarray, minimum_vessel_depth: int, enc: senc.ENC, show_enc: bool = False) -> Tuple[float, np.ndarray, int]:
    """Computes the closest distance to grounding for the given vessel trajectory.

    Args:
        vessel_trajectory (np.ndarray): The vessel`s trajectory, 2 x n_samples.
        minimum_vessel_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.

    Returns:
        Tuple[float, int]: The closest distance to grounding, corresponding distance vector and the index of the trajectory point.
    """
    dangerous_seabed = extract_relevant_grounding_hazards(minimum_vessel_depth, enc)
    vessel_traj_linestring = ndarray_to_linestring(vessel_trajectory)
    # if enc and show_enc:
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

        if enc and show_enc:
            enc.draw_line(points, color="cyan", marker_type="o")

        min_dist_vec = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
        if np.linalg.norm(min_dist_vec) <= min_dist + epsilon and np.linalg.norm(min_dist_vec) >= min_dist - epsilon:
            break

    # if enc and show_enc:
    #     enc.close_display()
    return min_dist, min_dist_vec, min_idx


def find_intersections_line_polygon(line: geometry.LineString, polygon: geometry.Polygon | geometry.MultiPolygon, enc: Optional[senc.ENC] = None) -> np.ndarray:
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
            # if enc:
            #     enc.draw_line(line.coords, color="red", marker_type="o")
            x, y = line.coords.xy
            coords.append([list(x), list(y)])
    return np.array(coords, dtype=object)


class TestMapFunctions(unittest.TestCase):
    def test_to_triangle(self):
        polygon = geometry.Polygon([(3.0, 0.0), (2.0, 0.0), (2.0, 0.75), (2.5, 0.75), (2.5, 0.6), (2.25, 0.6), (2.25, 0.2), (3.0, 0.2), (3.0, 0.0)])
        triangles = to_triangles(polygon)

        x, y = polygon.exterior.xy
        plt.plot(x, y, color="b", alpha=0.7, linewidth=3, solid_capstyle="round", zorder=2)
        for triangle in triangles:
            x_t, y_t = triangle.exterior.xy
            plt.plot(x_t, y_t, color="r", alpha=0.7, linewidth=3, solid_capstyle="round", zorder=1)
        plt.show()


if __name__ == "__main__":
    unittest.main()
