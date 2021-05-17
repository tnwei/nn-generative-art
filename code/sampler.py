"""
This module includes the latent space sampler object that enables sampling
smooth (enough) trajectories within a designated latent space.
"""

import numpy as np
from typing import NoReturn, Generator, Optional

try:
    from .bezier import gen_bezier_ctrl_points, bezier_mat
except:
    from bezier import gen_bezier_ctrl_points, bezier_mat


def linear_sampler(
    dims: int,
    min_coord: np.ndarray,
    max_coord: np.ndarray,
    init_coord: np.ndarray,
    stepsize: float,
    smooth_start_stop: bool,
) -> Generator[np.ndarray, None, None]:
    current_point = init_coord
    while True:
        # Find next point
        next_point = np.random.uniform(low=min_coord, high=max_coord, size=(dims,))

        # Calculate distance because we want to determine steps and direction
        dist = np.linalg.norm(next_point - current_point)
        stepcount = int(dist / stepsize)
        steps = np.linspace(current_point, next_point, stepcount)

        # Smooth decel and accel
        if smooth_start_stop is True:
            # Decel final 20% of path
            # 80% to 90% runs at 1.5x steps
            # 90% to 100% runs at 3x steps
            decel_point1 = int(stepcount * 0.8)
            decel_point2 = int(stepcount * 0.9)
            decel_path1 = np.linspace(
                steps[decel_point1],
                steps[decel_point2],
                int(1.5 * (decel_point2 - decel_point1)),
            )
            decel_path2 = np.linspace(
                steps[decel_point2], steps[-1], int(3 * (stepcount - decel_point2))
            )

            # Accel first 10% of path
            # 0% to 10% stretched into 3x steps
            # 10% to 20% stretched into 1.5x steps
            # Pretty much reverse of above
            accel_point1 = int(stepcount * 0.1)
            accel_point2 = int(stepcount * 0.2)
            accel_path1 = np.linspace(
                steps[0], steps[accel_point1], int(3 * (accel_point1 - 0))
            )
            accel_path2 = np.linspace(
                steps[accel_point1],
                steps[accel_point2],
                int(1.5 * (accel_point2 - accel_point1)),
            )

            steps = np.concatenate(
                [
                    accel_path1,
                    accel_path2,
                    steps[accel_point2:decel_point1],
                    decel_path1,
                    decel_path2,
                ]
            )
        else:
            pass

        for i in range(steps.shape[0]):
            yield steps[i, :]

        current_point = next_point


class LatentSpaceSampler:
    def __init__(
        self,
        dims: int = 3,
        init_coord: np.ndarray = np.array([0, 0, 0]),
        stepsize: float = 0.01,
        min_coord: np.ndarray = np.array([-1, -1, -1]),
        max_coord: np.ndarray = np.array([1, 1, 1]),
        smooth_start_stop: bool = True,
        keep_past_latent_vecs: Optional[int] = None,
    ) -> NoReturn:
        # Check to ensure all inputs have the correct dimensions
        assert len(init_coord) == dims
        assert len(min_coord) == dims
        assert len(max_coord) == dims

        # Transform all to numpy array
        if not isinstance(init_coord, np.ndarray):
            init_coord = np.array(init_coord)

        if not isinstance(min_coord, np.ndarray):
            min_coord = np.array(min_coord)

        if not isinstance(max_coord, np.ndarray):
            max_coord = np.array(max_coord)

        # Assign variables
        self.dims = dims
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.init_coord = init_coord
        self.stepsize = stepsize
        self.smooth_start_stop = smooth_start_stop
        self.keep_past_latent_vecs = keep_past_latent_vecs
        self.past_latent_vecs = []

        # Initialize the generator
        self.generator = linear_sampler(
            dims=self.dims,
            min_coord=self.min_coord,
            max_coord=self.max_coord,
            init_coord=self.init_coord,
            stepsize=self.stepsize,
            smooth_start_stop=self.smooth_start_stop,
        )

    def update_latent_vecs(self, latent_vec: np.ndarray) -> NoReturn:
        """
        Adds latent_vec to store and trims the list if too long
        """
        if self.keep_past_latent_vecs is None:
            pass
        else:
            self.past_latent_vecs.append(latent_vec)

            if len(self.past_latent_vecs) > self.keep_past_latent_vecs:
                self.past_latent_vecs.pop(0)

    def sample(self) -> np.ndarray:
        """
        Explore latent space by travelling to a randomly selected point.
        This process is repeated until the number of steps as specified by
        `iterations` is exceeded, of which the steps are truncated and
        returned together with the path lengths. If `iterations` is None, continuously produces frames.

        The heavy lifting is done in the generator object created by `_create_generator`, this is simply the sampling function.
        """
        next_latent_vec = next(self.generator)
        self.update_latent_vecs(next_latent_vec)
        return next_latent_vec


# ref type for generator expressions: https://stackoverflow.com/a/42227485/13095028
def bezier_sampler(
    num_ctrl_points: int = 5, steps: int = 50
) -> Generator[np.ndarray, None, None]:
    prev_ctrl_points = None
    while True:
        ctrl_points = gen_bezier_ctrl_points(num_ctrl_points, prev_ctrl_points)
        if prev_ctrl_points is not None:
            assert np.array_equal(
                ctrl_points[0, :], prev_ctrl_points[-1, :]
            ), f"{ctrl_points[0, :]}, {prev_ctrl_points[-1, :]}"
        nx, ny = bezier_mat(ctrl_points, steps)

        for i in range(steps):
            yield np.array([nx[i], ny[i]])

        prev_ctrl_points = ctrl_points.copy()


class BezierSampler:
    def __init__(
        self, num_ctrl_points=5, steps=50, keep_past_latent_vecs: Optional[int] = None
    ):
        self.generator = bezier_sampler(num_ctrl_points, steps)
        self.keep_past_latent_vecs = keep_past_latent_vecs
        self.past_latent_vecs = []

    def sample(self):
        next_latent_vec = next(self.generator)
        self.update_latent_vecs(next_latent_vec)
        return next_latent_vec

    def update_latent_vecs(self, latent_vec: np.ndarray) -> NoReturn:
        """
        Adds latent_vec to store and trims the list if too long
        """
        if self.keep_past_latent_vecs is None:
            pass
        else:
            self.past_latent_vecs.append(latent_vec)

            if len(self.past_latent_vecs) > self.keep_past_latent_vecs:
                self.past_latent_vecs.pop(0)
