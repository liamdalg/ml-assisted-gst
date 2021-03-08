from typing import List, Tuple

import h5py
import numpy as np

from numpy.random import default_rng
from scipy.linalg import orth
from tqdm import tqdm

from optics import get_unitary

# h5py's default label for complex numbers is ('r', 'i')
# change to ('Re', 'Im') to match Mathematica's format
conf = h5py.get_config()
conf.complex_names = ('Re', 'Im')
rng = default_rng()


def random_vector(dim: int) -> np.ndarray:
    """Generate a complex random vector of dimension `dim`."""
    real = rng.standard_normal((dim,))
    imag = rng.standard_normal((dim,))
    return real + (imag * 1j)


def orthogonal_vector(vec: np.ndarray) -> np.ndarray:
    """Generate a random vector which is orthogonal to `vec`."""
    rand_vec = random_vector(vec.shape[0])
    return orth(np.column_stack((vec, rand_vec)))[:, 1]


def init_c(
    heaters: int, diag_mean: float, diag_sd: float, corr_low: float, corr_high: float
) -> np.ndarray:
    """Initialise a random C matrix where the diagonal is sampled from `N(diag_mean, diag_sd)`
    and the off-diagonal is sampled from `U(corr_low, corr_high)`."""
    c_matrix = rng.uniform(corr_low, corr_high, (heaters, heaters))
    c_matrix[np.diag_indices_from(c_matrix)] = rng.normal(diag_mean, diag_sd, heaters)
    return c_matrix


def mean_cov(arr: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the mean and co-variance of `arr`, weighted by `weights`."""
    mean = np.sum(weights * arr, axis=0)
    cov = np.sum(weights[:, :, np.newaxis] @ (arr[:, :, np.newaxis] @ arr[:, np.newaxis, :]), axis=0) - (mean[:, np.newaxis] * mean)
    return mean, cov


def likelihood(
    input_state: np.ndarray, output_state: np.ndarray, powers: np.ndarray, c_mat: np.ndarray,
    phi0: np.ndarray
) -> float:
    """Calculate the likelihood |⟨D|U(Cx)|Ψ⟩|², where U(Cx) is the unitary operator associated with
    `powers`, `c_mat`, and `phi0`."""
    unitary = get_unitary(powers, c_mat, phi0)
    return np.abs(np.conjugate(output_state) @ unitary @ input_state) ** 2


def liu_west_resample(particles: np.ndarray, weights: np.ndarray, a: float) -> np.ndarray:
    """Resample `particles` using the Liu and West (2001) resampler."""
    flat_particles = particles.reshape((particles.shape[0], -1))
    mean, cov = mean_cov(flat_particles, weights)
    cov *= (1 - (a ** 2))

    resample_means = (a * flat_particles) + ((1 - a) * mean)
    new_particles = np.zeros_like(flat_particles)
    for i in range(weights.shape[0]):
        new_particles[i] = rng.multivariate_normal(resample_means[i], cov)

    return new_particles.reshape(particles.shape)


def update_estimate(
    particles_c: np.ndarray, particles_phi: np.ndarray, weights: np.ndarray, input_state: np.ndarray,
    output_state: np.ndarray, powers: np.ndarray, resample_threshold: float, resample_a: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update the weightings for each particle estimate. If necessary, the particles are also
    resampled.

    Args:
        particles_c: array of stacked matrices, where each matrix is a particle estimate for the
        C matrix.
        particles_phi: array of stacked vectors, where each vector is a particle estimate for the
        phi offset for each heater.
        weights: vector of weights for each particle.
        input_state: input state data |Ψ⟩.
        output_state: output state data ⟨D|.
        powers: vector of powers for each heater.
        resample_threshold: resampling threshold for the Liu-West resampler.
        resample_a: `a` parameter for the Liu-West resampler.

    Returns:
        Updated particles for C and phi, and the new particle weights.
    """
    new_weights = np.zeros_like(weights)
    for i in range(weights.shape[0]):
        new_weights[i, :] = weights[i] * likelihood(
            input_state, output_state, powers, particles_c[i], particles_phi[i]
        )
    new_weights /= np.sum(new_weights)

    if np.sum(new_weights ** 2) > 1 / (resample_threshold * weights.shape[0]):
        particles_c = liu_west_resample(particles_c, weights, resample_a)
        particles_phi = liu_west_resample(particles_phi, weights, resample_a)
        new_weights.fill(1.0 / new_weights.shape[0])

    return particles_c, particles_phi, new_weights


# TODO: put data loading + preparation into functions or other file
data_path = r'BayesianXTalk5heaters2018-03-10 17-27-02.h5'
with h5py.File(data_path, 'r') as h5_data:
    input_states = h5_data['InputStates'][:]
    output_states = h5_data['OutputStates'][:]
    all_powers = h5_data['Powers'][:, 2:-2]
    exp_counts = h5_data['ExpCounts'][:]

prob_0 = exp_counts[:, 0] / np.sum(exp_counts, axis=1)

out_states = np.zeros_like(output_states)
for i, state in enumerate(out_states):
    if prob_0[i] > rng.random():
        out_states[i] = output_states[i]
    else:
        out_states[i] = orthogonal_vector(output_states[i])

# TODO: put prior generation in function too? probably should move all of the experiment constants
#   up to the top aswell
data_length, heaters = all_powers.shape
particle_count = 300
resample_threshold = 0.4
resample_a = 0.9  # a is a parameter for the Liu-West resampler

power_interval = (0., 50.)
c_diag_mean = 2 * np.pi / max(power_interval)
c_diag_sd = 0.3 * c_diag_mean
corr_interval = (-0.2 * c_diag_mean, 0.2 * c_diag_mean)

# TODO: experiment with the data layout for the particles
# right now they're stored as 3d stacked matrices, where each matrix is a single particle
# it requires some funky reshaping in the mean_cov function to get it to work properly
# should probably just simplify it and reshape as necessary :\
particles_c = np.zeros((particle_count, heaters, heaters, ))
particles_phi = np.zeros((particle_count, heaters))
weights = np.full(particle_count, 1.0 / particle_count).reshape((-1, 1))
for i in range(particle_count):
    particles_c[i, :, :] = init_c(heaters, c_diag_mean, c_diag_sd, *corr_interval)
    particles_phi[i, :] = rng.uniform(0, 2 * np.pi, heaters)

# update all parameters, and construct a list of the weight + particles at each stage
particles_evolution = []
for i in tqdm(range(data_length)):
    particles_c, particles_phi, weights = update_estimate(
        particles_c, particles_phi, weights, input_states[i], output_states[i], all_powers[i],
        resample_threshold, resample_a
    )
    particles_evolution.append((particles_c, particles_phi, weights))
