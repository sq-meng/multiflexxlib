import multiflexxlib.voronoi_plot as vor


def test_point_in_poly():
    poly1 = [[0, 0], [1, 0], [1, 1], [0, 1]]
    poly2 = [[0, 0], [1, 0], [1, 1], [0, 2]]
    point1 = [0.5, 0.5]
    point2 = [-1, 2]
    point3 = [0.1, 1.1]
    assert vor.point_in_polygon(point1, poly1)
    assert vor.point_in_polygon(point1, poly2)
    assert not vor.point_in_polygon(point2, poly2)
    assert vor.point_in_polygon(point3, poly2)