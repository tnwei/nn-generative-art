"""
This module includes the latent space sampler object that enables sampling 
smooth (enough) trajectories within a designated latent space.
"""

import numpy as np


class LatentSpaceSampler:
    def __init__(
        self,
        dims=3,
        init_coord=np.array([0, 0, 0]),
        stepsize=0.01,
        min_coord=np.array([-1, -1, -1]),
        max_coord=np.array([1, 1, 1]),
        smooth_start_stop=True,
    ):
        """
        Input
        -----
        dims: int
        init_coord: iterable
        stepsize: int
        min_coord: iterable
        max_coord: iterable
        smooth_start_stop: bool
        """
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

        # Initialize the generator
        self._init_generator()

    def _init_generator(self):
        # Writing it this way to leave flexibility for implementing other generators
        self.generator = self._point2point_latent_explorer_generator(
            dims=self.dims,
            min_coord=self.min_coord,
            max_coord=self.max_coord,
            init_coord=self.init_coord,
            stepsize=self.stepsize,
            smooth_start_stop=self.smooth_start_stop,
        )

    def _point2point_latent_explorer_generator(
        self, dims, min_coord, max_coord, init_coord, stepsize, smooth_start_stop
    ):
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

    def sample(self, iterations=1, return_value_if_one=True):
        """
        Explore latent space by travelling to a randomly selected point.
        This process is repeated until the number of steps as specified by
        `iterations` is exceeded, of which the steps are truncated and
        returned together with the path lengths. If `iterations` is None, continuously produces frames.

        The heavy lifting is done in the generator object created by `_create_generator`, this is simply the sampling function.

        Output
        ------
        pos_arr: np.ndarray
        path_lengths: List[Int]

        """
        if iterations is None:
            iterations = 1

        if iterations == 1 and return_value_if_one is True:
            return next(self.generator)

        ans = []

        for _ in range(iterations):
            ans.append(next(self.generator))

        return ans
