import numpy as np
import scipy.sparse as sp


def find_min_dist_mat(R: np.ndarray) -> np.ndarray:
    """
    R is reference array
    Returns distance array D
    
    Note: cannot use lru_cache from functools because np.ndarray
    is not hashable by Python
    """
    D = np.full(shape=R.shape, fill_value=np.nan)
    sparse_R = sp.coo_matrix(R)

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            # Init min dist to closest reference point, as max possible dist
            min_dist = np.linalg.norm([D.shape[0], D.shape[1]])
            anchor_point = (np.nan, np.nan)
            # Have checked, NaNs are omitted so below returns all points of interest only
            for ri, rj in zip(sparse_R.row, sparse_R.col):
                if np.linalg.norm([i - ri, j - rj]) < min_dist:
                    anchor_point = (ri, rj)
                    min_dist = np.linalg.norm([i - ri, j - rj])

            # Update min_dist in array
            D[i, j] = min_dist

    return D
