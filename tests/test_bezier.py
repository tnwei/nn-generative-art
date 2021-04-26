from code.bezier import gen_bezier_ctrl_points
import numpy as np


def test_ctrl_points_tangent_to_prev_ctrl_points():
    ctrl_points1 = gen_bezier_ctrl_points(num_ctrl_points=5, prev_points=None)
    ctrl_points2 = gen_bezier_ctrl_points(num_ctrl_points=5, prev_points=ctrl_points1)
    assert np.array_equal(
        ctrl_points1[-1, :], ctrl_points2[0, :]
    ), f"{ctrl_points1[-1, :]}, {ctrl_points2[0, :]}"
