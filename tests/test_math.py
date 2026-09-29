import numpy as np

from octohand.math3d import LAY_FLAT_X, axis_from_quat, quat_mul, quat_rotate, yaw_quat


def test_lay_flat_points_along_x():
    axis = quat_rotate(LAY_FLAT_X, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(axis, [1.0, 0.0, 0.0], atol=1e-6)


def test_yaw_then_lay_flat_stays_horizontal():
    tilted = quat_mul(yaw_quat(0.7), LAY_FLAT_X)
    axis = axis_from_quat(tilted)
    assert abs(axis[2]) < 1e-6
    assert np.isclose(np.linalg.norm(axis), 1.0)
