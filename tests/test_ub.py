import multiflexxlib as mfl
import numpy as np


def test_ub_ortho():
    u = mfl.UBMatrix([4.0, 4.0, 4.0, 90, 90, 90], [1, 0, 0], [0, 0, 1])
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
