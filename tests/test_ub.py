import multiflexxlib as mfl
import numpy as np


def test_ub_creation():
    u = mfl.UBMatrix(4, [1, 0, 0], [0, 1, 1])
    assert np.all(u.latparam == np.array([4, 4, 4, 90, 90, 90]))
    u = mfl.UBMatrix([4, 4.04, 4.9], [1, 0, 0], [0, 1, 0])
    assert np.all(u.latparam == np.array([4, 4.04, 4.9, 90, 90, 90]))
    u = mfl.UBMatrix([4, 4.04, 4.9, 90, 90, 120], [1, 0, 0], [0, 1, 0])
    assert np.all(u.latparam == np.array([4, 4.04, 4.9, 90, 90, 120]))


def test_ub_ortho():
    u = mfl.UBMatrix(4.0, [1, 0, 0], [0, 0, 1])
    assert u.is_orthogonal
    coords = np.array([[1, 1, 0],
                      [0, 1, 0],
                      [2, 1, 0],
                      [1, -1, 0]])
    assert u.convert(coords, 'rs', axis=0).shape == (4, 3)
    assert u.convert(coords.T, 'rs', axis=1).shape == (3, 4)
    assert np.isclose(u.theta, np.pi/2)


def test_ub_hex_ab():
    u = mfl.UBMatrix([4.0, 4.0, 5.0, 90, 90, 120], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0])
    assert not u.is_orthogonal
    coords = np.array([[1, 1, 0],
                       [0, 1, 0],
                       [2, 1, 0],
                       [1, -1, 0]])
    assert u.convert(coords, 'rs', axis=0).shape == (4, 3)
    assert u.convert(coords.T, 'rs', axis=1).shape == (3, 4)
    assert np.isclose(u.theta, np.pi/3)
    assert np.all(np.isclose(u.plot_x, [1, 0, 0]))
    assert np.all(np.isclose(u.plot_y_actual, [-0.5, 1, 0]))
    assert np.all(np.isclose(u.plot_y_nominal, [0, 1, 0]))


def test_v1_to_v2():
    assert np.isclose(mfl.ub.v1_to_v2([1, 0, 0], [0, 1, 0]), np.pi / 2)
    assert np.isclose(mfl.ub.v1_to_v2([0, 1, 0], [1, 0, 0]), -np.pi / 2)
    assert np.isclose(mfl.ub.v1_to_v2([2, 0, 0], [1, 1, 0]), np.pi / 4)


def test_find_a3_a4():
    u = mfl.UBMatrix([3.9044, 3.9044, 3.9200, 90, 90, 90], [1, 1, 0], [0, 0, 1])
    a3, a4 = mfl.ub.find_a3_a4([1, 1, 0], ub_matrix=u, ki=1.55, kf=1.55, sense=1)
    assert np.isclose(a3, -42.77, atol=0.1)
    assert np.isclose(a4, 94.47, atol=0.1)
    a3, a4 = mfl.ub.find_a3_a4([1, 1, 0], ub_matrix=u, ki=1.55, kf=1.55, sense=-1)
    assert np.isclose(a3, 42.77, atol=0.1)
    assert np.isclose(a4, -94.47, atol=0.1)