import h5py
import numpy as np
from numpy.random import default_rng
from scipy.linalg import orth


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
    and the off-diagonal is sampled from `U(corr_interval)`."""

    c_matrix = rng.uniform(corr_low, corr_high, (heaters, heaters))
    c_matrix[np.diag_indices_from(c_matrix)] = rng.normal(diag_mean, diag_sd, heaters)
    return c_matrix


# TODO: put data loading + preparation into functions or other file
data_path = r'BayesianXTalk5heaters2018-03-10 17-27-02.h5'
with h5py.File(data_path, 'r') as h5_data:
    input_states = h5_data['InputStates'][:]
    output_states = h5_data['OutputStates'][:]
    power_h5 = h5_data['Powers'][:, 2:-2]
    exp_counts = h5_data['ExpCounts'][:]

prob_0 = exp_counts[:, 0] / np.sum(exp_counts, axis=1)

# TODO: look into why we do this
out_states = np.zeros_like(output_states)
for i, state in enumerate(out_states):
    if prob_0[i] > rng.random():
        out_states[i] = output_states[i]
    else:
        out_states[i] = orthogonal_vector(output_states[i])


# TODO: put prior generation in function too? probably should move all of the experiment constants
#   up to the top aswell
heaters = power_h5.shape[1]
particle_count = 300
resample_threshold = 0.4
a = 0.9  # a is a parameter for the Liu-West resampler

power_interval = (0., 50.)
c_diag_mean = 2 * np.pi / max(power_interval)
c_diag_sd = 0.3 * c_diag_mean
corr_interval = (-0.2 * c_diag_mean, 0.2 * c_diag_mean)

# generate a list of particles, where each particle is (C, ϕ)
particles = [
    (init_c(heaters, c_diag_mean, c_diag_sd, *corr_interval), rng.uniform(0, 2 * np.pi, heaters))
    for _ in range(particle_count)
]
weights = np.full((particle_count,), 1.0 / particle_count)
