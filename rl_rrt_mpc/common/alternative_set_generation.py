"""
    alternative_set_generator.py

    Summary:
        Contains a class for generating a convex safe set around a point

    Author: Aksel Vaaler, Trym Tengesdal
"""
import time
from typing import Optional, Tuple

import numpy as np
import seacharts.enc as senc
import shapely.geometry as geo
from pyexpat.errors import XML_ERROR_UNEXPECTED_STATE
from scipy.spatial import HalfspaceIntersection
from shapely import affinity as aff
from shapely import ops as ops


def reduce_constraints(A_full: np.ndarray, b_full: np.ndarray, n_set_constr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reduces the number of constraints to n_set_constr, by removing the last constraints in A_full and b_full

    Args:
        A_full (np.ndarray): The full constraint matrix
        b_full (np.ndarray): The full constraint vector
        n_set_constr (int): The number of constraints to keep

    Returns:
        Tuple[np.ndarray, np.ndarray]: The reduced constraint matrix and vector
    """
    A = np.zeros((n_set_constr, 2))
    b = np.zeros(n_set_constr)
    A[: A_full[0:n_set_constr].shape[0]] = A_full[0:n_set_constr]
    b[: b_full[0:n_set_constr].shape[0]] = b_full[0:n_set_constr]
    return A, b


# Defines a rectangle of a given width and length,
# centered on the object position, and rotated according to the object heading
def object_horizon(xpos: float, ypos: float, heading: float, width: float, length: float) -> geo.Polygon:
    rect = geo.Polygon(
        [(xpos - width / 2, ypos - length / 2), (xpos - width / 2, ypos + length / 2), (xpos + width / 2, ypos + length / 2), (xpos + width / 2, ypos - length / 2)]
    )
    return aff.rotate(rect, -heading)


# Detects polygonal water area which the ship is currently in, and
# computes the intersection between the object horizon and the area with water,
# aka the current 'safe' area
def safe_area(obj_hor_rectangle, seabed_multipoly):
    for seabed_polygon in seabed_multipoly.geoms:
        if obj_hor_rectangle.intersects(seabed_polygon):
            return obj_hor_rectangle.intersection(seabed_polygon)

    return None


def orthogonal_constraint_line(xpos: float, ypos: float, boundary_point: geo.Point, line_length: float) -> Tuple[geo.LineString, float, float]:
    xdiff = boundary_point.x - xpos
    ydiff = boundary_point.y - ypos
    sq_dist = xdiff**2 + ydiff**2
    epsilon = 1e-5
    dist = np.sqrt(sq_dist)

    # v is a vector in direction orthogonal to vector from ship pos to pos of nearest unsafe point
    # the length of v is equal to 1/2 of the specified desired line length
    vx = -line_length / 2 * ydiff / dist
    vy = line_length / 2 * xdiff / dist

    # x1,y1,x2,y2 specify the start and end points of the constraint line
    x1 = boundary_point.x + vx
    y1 = boundary_point.y + vy
    x2 = boundary_point.x - vx
    y2 = boundary_point.y - vy

    # a and b represent a row of A and B constraint matrices used to
    # apply position constraints in safety filter
    # The below formulation ensures that a*ship_pos <= b always means that
    # the ship is on the "correct side" of the constraint line
    #
    a = np.array([xdiff, ydiff]) / dist
    b = dist
    return geo.LineString([(x1, y1), (x2, y2)]), a, b


def multiline_inside_constraint(line, xpos, ypos, boundary_point, constraint_line):
    const_norm_x = -(boundary_point.x - xpos)
    const_norm_y = -(boundary_point.y - ypos)

    if geo.Point((line.coords[0][0], line.coords[0][1])).intersects(constraint_line):

        line_grad_x = line.coords[1][0] - line.coords[0][0]
        line_grad_y = line.coords[1][1] - line.coords[0][1]

        return (const_norm_x * line_grad_x + const_norm_y * line_grad_y) >= 0

    elif geo.Point((line.coords[-1][0], line.coords[-1][1])).intersects(constraint_line):

        line_grad_x = line.coords[-2][0] - line.coords[-1][0]
        line_grad_y = line.coords[-2][1] - line.coords[-1][1]

        return (const_norm_x * line_grad_x + const_norm_y * line_grad_y) >= 0

    else:
        xdiff = boundary_point.x - xpos
        ydiff = boundary_point.y - ypos
        lxdiff = line.coords[0][0] - xpos
        lydiff = line.coords[0][1] - ypos

        return (xdiff * lxdiff + ydiff * lydiff) <= (xdiff**2 + ydiff**2)


def lines_inside_constraint(safe_bound: geo.MultiLineString, xpos: float, ypos: float, boundary_point: geo.Point, constraint_line: geo.LineString) -> geo.MultiLineString:
    inside_lines = []
    for line in safe_bound.geoms:
        if multiline_inside_constraint(line, xpos, ypos, boundary_point, constraint_line):
            inside_lines.append(line)
    return geo.MultiLineString(inside_lines)


def safe_near_point(safe_bound: geo.MultiLineString, pos: geo.Point, dist: float) -> geo.Point:
    near_point = ops.nearest_points(safe_bound, pos)[0]
    v = near_point.x - pos.x, near_point.y - pos.y
    u = dist * v / (np.sqrt(v[0] ** 2 + v[1] ** 2))

    point = near_point.x - u[0], near_point.y - u[1]

    return geo.Point((point))


def safe_convex_poly(safe_bound: geo.LineString | geo.MultiLineString, pos: geo.Point, enc: Optional[senc.ENC] = None) -> Tuple[list, np.ndarray, np.ndarray]:
    """Computes an approximate convex safe set of the in general non-convedx safe area, with constraints

    A(p - pos) <= b

    where pos is the generator point

    Args:
        - safe_bound (geo.LineString | geo.MultiLineString): Boundary of the safe area
        - pos (geo.Point): Generator point
        - enc (Optional[senc.ENC], optional): ENC object. Defaults to None.

    Returns:
        Tuple[list, np.ndarray, np.ndarray]: List of constraint lines, A and b constraint matrices
    """
    xpos = pos.x
    ypos = pos.y

    if isinstance(safe_bound, geo.LineString):
        safe_bound = geo.MultiLineString([safe_bound])

    constraint_lines = []

    if enc is not None:
        enc.start_display()
        for line in safe_bound.geoms:
            enc.draw_line(line.coords, color="pink")

    near_point = safe_near_point(safe_bound, geo.Point(pos), 5)
    constraint_line, A, b = orthogonal_constraint_line(xpos, ypos, near_point, 1000)
    constraint_lines.append(constraint_line)
    curr_safe = ops.split(safe_bound, constraint_line)
    curr_safe = lines_inside_constraint(curr_safe, xpos, ypos, near_point, constraint_line.buffer(0.1))

    if enc is not None:
        enc.draw_line(curr_safe.coords, color="orange")
    # near_point = safe_near_point(curr_safe, geo.Point(pos), 5)
    # constraint_line, A_row_next, b_row_next = orthogonal_constraint_line(xpos, ypos, near_point, 350)
    # A = np.vstack([A, A_row_next])
    # b = np.vstack([b, b_row_next])
    # constraint_lines.append(constraint_line)
    # curr_safe = ops.split(curr_safe, constraint_line)
    # curr_safe = lines_inside_constraint(curr_safe, xpos, ypos, near_point, constraint_line.buffer(0.1))

    while not curr_safe.is_empty:
        near_point = safe_near_point(curr_safe, geo.Point(pos), 5)
        constraint_line, A_row_next, b_row_next = orthogonal_constraint_line(xpos, ypos, near_point, 1000)
        A = np.vstack([A, A_row_next])
        b = np.vstack([b, b_row_next])
        constraint_lines.append(constraint_line)
        curr_safe = ops.split(curr_safe, constraint_line)
        curr_safe = lines_inside_constraint(curr_safe, xpos, ypos, near_point, constraint_line.buffer(0.1))

        if enc is not None:
            enc.draw_line(curr_safe.coords, color="orange")

    return constraint_lines, A, b


def plot_constraints(A: np.ndarray, b: np.ndarray, p: np.ndarray, enc: senc.ENC) -> None:
    """Plots the constraints in the halfspace intersection

    Args:
        A (np.ndarray): Constraint matrix
        b (np.ndarray): Constraint vector
        p (np.ndarray): Point inside the constrant set
        enc (senc.ENC): Electronic Navigational Chart object.
    """
    A_nonzero = A[~np.any(A == 0.0, axis=1)]
    b_nonzero = b[b != 0.0]
    hs = HalfspaceIntersection(np.concatenate((A_nonzero, -b_nonzero.reshape((-1, 1))), axis=1), p)
    intersections = hs.intersections
    points = [(p_i[1], p_i[0]) for p_i in intersections]
    points.append(points[0])
    set_polygon = geo.Polygon(points)
    enc.start_display()
    enc.draw_polygon(set_polygon, color="green", fill=False)
