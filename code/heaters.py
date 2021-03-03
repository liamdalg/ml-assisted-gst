import h5py
import numpy as np
from numpy.random import default_rng
from scipy.linalg import orth


def random_vector(dim: int) -> np.ndarray:
    """Generate a complex random vector of dimension `dim`."""
    real = rng.standard_normal((dim,))
    imag = rng.standard_normal((dim,))
    return real + (imag * 1j)


DATA_PATH = r'BayesianXTalk5heaters2018-03-10 17-27-02.h5'


# h5py's default label for complex numbers is ('r', 'i')
# change to ('Re', 'Im') to match Mathematica's format
conf = h5py.get_config()
conf.complex_names = ('Re', 'Im')
rng = default_rng()

with h5py.File(DATA_PATH, 'r') as h5_data:
    input_states = h5_data['InputStates'][:]
    output_states = h5_data['OutputStates'][:]
    power_h5 = h5_data['Powers'][:, 2:-2].flatten()
    exp_counts = h5_data['ExpCounts'][:]

prob_0 = exp_counts[:, 0] / np.sum(exp_counts, axis=1)

# TODO: look into why we do this
out_states = np.zeros_like(output_states)
for i, state in enumerate(out_states):
    if prob_0[i] > rng.random():
        vec = random_vector(2)
        out_states[i] = orth(np.column_stack((output_states[i], vec)))[:, 1]
    else:
        out_states[i] = output_states[i]
