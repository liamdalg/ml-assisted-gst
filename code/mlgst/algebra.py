from typing import Tuple

import numpy as np

from numpy.random import default_rng
from numpy import linalg

rng = default_rng()


def random_vector(dim: int) -> np.ndarray:
    """Generate a complex random vector of dimension `dim`."""
    real = rng.standard_normal((dim,))
    imag = rng.standard_normal((dim,))
    return real + (imag * 1j)


def orthogonal_vector(vec: np.ndarray) -> np.ndarray:
    """Generate a random vector which is orthogonal to `vec`."""
    rand_vec = random_vector(vec.shape[0])
    mat = np.column_stack((vec, rand_vec))
    q, _ = linalg.qr(mat)
    return q[:, 1]


def mean_cov(arr: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the mean and co-variance of `arr`, weighted by `weights`.

    Args:
        arr: 2d matrix where each row is an observation.
        weights: vector of weights for each observation.

    Returns:
        Mean vector and covariance matrix.
    """
    mean = np.sum(weights[:, np.newaxis] * arr, axis=0)
    arr_mat = arr[:, :, np.newaxis] @ arr[:, np.newaxis, :]
    mean_mat = mean[:, np.newaxis] * mean
    cov = np.sum(weights[:, np.newaxis, np.newaxis] * arr_mat, axis=0) - mean_mat
    return mean, cov


def cov_norm(arr: np.ndarray, weights: np.ndarray) -> float:
    """Calculate the norm of the covariance matrix of `arr`, weighted by `weights`."""
    _, cov = mean_cov(arr, weights)
    eig = linalg.eigvals(cov)
    return linalg.norm(eig)
