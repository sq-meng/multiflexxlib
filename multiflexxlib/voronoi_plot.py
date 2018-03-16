from scipy.spatial import Voronoi
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def generate_vpatch(x, y=None, aspect=1, angle_mode=False) -> list:
    """
    :param x: 1-D list or np.array: x coordinates on a 2-D plane.
    :param y: 1-D list or np.array: y coordinates on a 2-D plane.
    :param aspect: aspect correction, larger value means each unit of y has greater absolute length.
    :param angle_mode: Not implemented.
    :return:
    """
    if aspect is None:
        aspect = 1
    if y is None:
        x = np.array(x)
        y = x[:, 1]
        x = x[:, 0]
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    if not angle_mode:
        y = y.copy() * aspect

    if x.shape != y.shape:
        raise TypeError('input data shape mismatch in Voronoi patch calculation.')

    xy = np.vstack((x, y)).T
    xmin, xmax = x.min() - 0.1, x.max() + 0.1
    ymin, ymax = y.min() - 0.1, y.max() + 0.1
    bbox = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
    xy_padded = np.vstack((xy, bbox))
    vor = Voronoi(xy_padded)

    list_of_polygons = []
    for i in range(len(x)):
        region_index = vor.point_region[i]
        vertice_indexes = vor.regions[region_index]
        patch_vert_coords = [vor.vertices[vert_no] for vert_no in vertice_indexes]
        patch_vert_coords_array = np.array(patch_vert_coords)
        patch_vert_coords_array[:, 1] = patch_vert_coords_array[:, 1] / aspect
        list_of_polygons.append(patch_vert_coords_array)

    return list_of_polygons


def draw_patches(list_of_polygons, intensity_list=None, cm='inferno', empty_face=False, norm=None, alpha=None):
    patches = [Polygon(polygon, lw=0.1) for polygon in list_of_polygons]
    pc = PatchCollection(patches)
    if not empty_face:
        if intensity_list is None:
            raise ValueError('values not provided for Voronoi plot.')
        pc.set_array(np.array(intensity_list))
        pc.set_edgecolor('face')
        pc.set_cmap(cm)
        pc.set_norm(norm)
    else:
        pc.set_edgecolor([0.3, 0.3, 0.5])
        pc.set_facecolor([0, 0, 0, 0])
    return pc


def check_bbox(first, second):
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


def segment_intersect(first, second) -> bool:
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


def side_of_point(point, segment) -> bool:
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
