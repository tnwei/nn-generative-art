import numpy as np
from math import comb
from typing import Tuple, Optional

# Logic modified from Omar Aflak's gist for Bezier interpolation:
# https://gist.githubusercontent.com/OmarAflak/860ef23fcdd57dfd83470381f5db9b31/raw/367163e81eabdfdbfa26e3e4cc06ac85b54c7cf0/medium_bezier_matrix.py


def get_bezier_mat(n: int) -> np.ndarray:
    coef = [
        [comb(n, i) * comb(i, k) * (-1) ** (i - k) for k in range(i + 1)]
        for i in range(n + 1)
    ]
    # padding with zeros to create a square matrix
    return [row + [0] * (n + 1 - len(row)) for row in coef]


def evaluate_bezier_mat(ctrl_points: np.ndarray, num_points: int) -> np.ndarray:
    n = len(ctrl_points) - 1
    T = lambda t: [t ** i for i in range(n + 1)]
    M = get_bezier_mat(n)
    return np.array(
        [np.dot(np.dot(T(t), M), ctrl_points) for t in np.linspace(0, 1, num_points)]
    )


def bezier_mat(
    ctrl_points: np.ndarray, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    new_points_mat = evaluate_bezier_mat(ctrl_points, num_points)
    nx_mat, ny_mat = new_points_mat[:, 0], new_points_mat[:, 1]
    return nx_mat, ny_mat


def gen_bezier_ctrl_points(
    num_ctrl_points: int, prev_points: Optional[np.ndarray] = None
):
    assert num_ctrl_points > 2, "Continuation of slope needs >2 points to work"
    assert prev_points is None or isinstance(prev_points, np.ndarray)
    if isinstance(prev_points, np.ndarray):
        assert (len(prev_points.shape) == 2) and (prev_points.shape[1] == 2)

    if prev_points is None:
        return np.random.standard_normal(size=(num_ctrl_points, 2))
    else:
        end_slope_vec = prev_points[-1, :] - prev_points[-2, :]
        end_slope_uvec = end_slope_vec / np.linalg.norm(end_slope_vec)
        new_second_row = prev_points[-1, :] + end_slope_uvec * np.random.uniform(
            low=0.1, high=0.5
        )
        return np.concatenate(
            [
                [prev_points[-1, :]],
                [new_second_row],
                np.random.standard_normal(size=(num_ctrl_points - 2, 2)),
            ]
        )
