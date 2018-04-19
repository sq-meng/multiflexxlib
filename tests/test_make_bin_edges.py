import multiflexxlib as mfl
import numpy as np
import pytest


def test_walking_bins():
    v1 = [1, 2, 3, 4, 5]
    b1 = mfl.make_bin_edges(v1, tolerance=0.2)
    assert np.all(np.isclose(b1, [0.9, 1.5, 2.5, 3.5, 4.5, 5.1]))
    v2 = [1, 2, 2.1, 3, 4]
    b2 = mfl.make_bin_edges(v2)
    assert np.all(np.isclose(b2, [0.9, 1.5, 2.55, 3.5, 4.1]))
    with pytest.raises(ValueError):
        v3 = [1, 2.1, 2.2, 2.3, 2.4, 3, 4]
        mfl.make_bin_edges(v3)


def test_regular_bins():
    v10 = [1, 2, 3]
    b10 = mfl.make_bin_edges(v10, tolerance=0.5, strategy='regular')
    assert np.all(np.isclose(b10, [0.75, 1.25, 1.75, 2.25, 2.75, 3.25]))


def test_given_bins():
    v100 = np.linspace(1, 10, 10)
    b100 = mfl.make_bin_edges(v100, strategy=v100)
    assert np.all(np.isclose(b100, v100))

def test_exc():
    with pytest.raises(ValueError):
        mfl.make_bin_edges([1,2,3], tolerance=0.2, strategy='wrong')