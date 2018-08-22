from __future__ import division
from scipy.spatial import Voronoi
import numpy as np
import pyclipper
import multiflexxlib.ub as ub
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def voronoi_polygons(x, y=None, aspect=1, max_cell=0.1):
    # type: (...) -> list
    """
    :param x: 1-D list or np.array: x coordinates on a 2-D plane.
    :param y: 1-D list or np.array: y coordinates on a 2-D plane.
    :param aspect: aspect correction, larger value means each unit of y has greater absolute length.
    :param max_cell: Max size of rectangular bounding box.
    :return:
    """
    if aspect is None:
        aspect = 1
    if y is None:
        x = np.asarray(x)
        y = x[:, 1]
        x = x[:, 0]
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    y = y.copy() * aspect

    if x.shape != y.shape:
        raise TypeError('input data shape mismatch in Voronoi patch calculation.')

    xy = np.vstack((x, y)).T
    xmin, xmax = x.min() - max_cell, x.max() + max_cell
    ymin, ymax = y.min() - max_cell, y.max() + max_cell
    bbox = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
    xy_padded = np.vstack((xy, bbox))
    vor = Voronoi(xy_padded)
    list_of_polygons = _populate_lop(x, y, vor.point_region, vor.regions, vor.vertices, max_cell, aspect)

    return list_of_polygons


def _populate_lop(x, y, point_region, regions, vertices, max_cell, aspect):
    list_of_polygons = []
    for i in range(len(x)):
        region_index = point_region[i]
        vertices_indexes = regions[region_index]
        patch_vert_coords = [vertices[vert_no] for vert_no in vertices_indexes]

        pvca = np.asarray(patch_vert_coords)  # patch_vert_coords_array
        x_min, x_max = x[i] - max_cell, x[i] + max_cell
        y_min, y_max = y[i] - max_cell, y[i] + max_cell
        pvca[:, 0] = pvca[:, 0].clip(min=x_min, max=x_max)
        pvca[:, 1] = pvca[:, 1].clip(min=y_min, max=y_max)
        pvca[:, 1] = pvca[:, 1] / aspect

        list_of_polygons.append(pvca)
    return list_of_polygons


def draw_patches(list_of_polygons, intensity_list=None, cm='inferno', mesh=False, norm=None, zorder=10):
    patches = [Polygon(polygon, lw=0.02, zorder=zorder) for polygon in list_of_polygons]
    pc = PatchCollection(patches)
    if not mesh:
        if intensity_list is None:
            raise ValueError('values not provided for Voronoi plot.')
        pc.set_array(np.asarray(intensity_list))
        pc.set_edgecolor('face')
        pc.set_cmap(cm)
        pc.set_norm(norm)
    else:
        pc.set_edgecolor([0.3, 0.3, 0.5])
        pc.set_facecolor([0, 0, 0, 0])
    return pc


def check_bbox(first, second):
    first = np.asarray(first)
    second = np.asarray(second)
    first_x_min = np.min(first[:, 0])
    first_x_max = np.max(first[:, 0])
    first_y_min = np.min(first[:, 1])
    first_y_max = np.max(first[:, 1])

    second_x_min = np.min(second[:, 0])
    second_x_max = np.max(second[:, 0])
    second_y_min = np.min(second[:, 1])
    second_y_max = np.max(second[:, 1])
    if first_x_min > second_x_max or first_x_max < second_x_min:
        return False
    if first_y_min > second_y_max or first_y_max < second_y_min:
        return False

    return True


def segment_intersect(first, second):
    # type: (...) -> bool
    # bounding box
    if not check_bbox(first, second):
        return False

    if side_of_point(first[0], second) == side_of_point(first[1], second):
        # print('1st seg side check failed')
        return False
    if side_of_point(second[0], first) == side_of_point(second[1], first):
        # print('2nd seg side check failed')
        return False

    return True


def side_of_point(point, segment):
    # type: (...) -> bool
    """
    Calculates if point in 2-D plane is to the left-upper side of line.
    :param point: [x, y] pair defining a point.
    :param segment: row vectors defining a line.
    :return: True if point is to the left-upper side. False if not or point on line.
    """
    point = np.array(point)
    segment = np.array(segment)
    point_shifted = point - segment[0, :]
    segment_shifted = segment - segment[0, :]
    det = np.linalg.det(np.vstack((segment_shifted[1, :], point_shifted)))
    return det > 0


def segment_intersect_polygons(segment, list_of_polygons):
    # TODO: make this faster.
    segment = np.array(segment)
    list_of_polygons = np.array(list_of_polygons)
    results = []
    for polygon in list_of_polygons:
        this_polygon = False
        if not check_bbox(segment, polygon):
            pass  # if bounding box does not intersect, skip individual segment intersection check
        else:
            for i in range(polygon.shape[0]):
                if segment_intersect(segment, np.vstack((polygon[i-1], polygon[i]))):
                    this_polygon = True
                    break
        results.append(this_polygon)

    return results


def projection_on_segment(points, segment, aspect=1):
    """
    Projects a row array of points onto given segment
    :param points: Points to be projected, NxD array.
    :param segment:  Segment to project onto, 2xD array.
    :param aspect: ONLY WORKS IF D=2, >1 means unit length of y axis is greater.
    :return: 1-D array of percentiles
    """
    points = np.array(points)
    segment = np.array(segment)
    if len(points.shape) == 1:
        shape = points.shape[0]
        points = points.reshape(-1, shape)

    if np.all(segment[0] == segment[1]):
        raise ValueError('provided segment has zero length.')
    if aspect != 1 and points.shape[1] != 2:
        raise NotImplementedError('aspect correction for higher dims not implemented.')

    if aspect != 1:
        aspect_multiplier = np.array([1, aspect])
        points = points * aspect_multiplier
        segment = segment * aspect_multiplier
    shift = segment[0]
    points_sh = points - shift
    segment_sh = (segment - shift)[1]
    norm = np.linalg.norm
    percentiles = []
    for point_sh in points_sh:
        cos_theta = np.dot(point_sh, segment_sh) / (norm(segment_sh) * norm(point_sh))
        percentiles.append(norm(point_sh) * cos_theta / norm(segment_sh))

    return np.array(percentiles)


def point_in_polygon(point, polygon):
    stc = pyclipper.scale_to_clipper
    in_polygon = pyclipper.PointInPolygon(stc(point), stc(polygon))
    if in_polygon == 1:
        return True
    else:
        return False


def draw_line(ax, points, ub_matrix, sys='r'):
    # type: (..., (np.ndarray, list), ub.UBMatrix, str) -> ...
    points_p = ub_matrix.convert(points, sys=sys+'p', axis=0)
    return ax.plot(points_p[:, 0], points_p[:, 1])


def find_spurion_cutoff(values, step_length=0.01, gradient_ratio=50):
    values = np.asarray(values)
    value_max = values.max()
    gradient_ratio = gradient_ratio * step_length / 100
    for i in range(int(3 // step_length)):
        delta = np.percentile(a=values, q=100 - i * step_length, interpolation='linear') - \
                np.percentile(a=values, q=100 - (i + 1) * step_length, interpolation='linear')
        if delta / value_max > gradient_ratio:
            for j in range(i, int(3 // step_length)):
                delta = np.percentile(values, q=100 - j * step_length, interpolation='linear') - \
                        np.percentile(values, q=100 - (j + 1) * step_length, interpolation='linear')
                if delta / value_max < gradient_ratio:
                    cutoff = 100 - j * step_length
                    return cutoff
    return 100  # no cutoff found
