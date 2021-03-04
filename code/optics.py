import numpy as np


def beam_splitter(mode1: int, mode2: int, modes: int) -> np.ndarray:
    mat = np.identity(modes, dtype=np.complex128)
    mat[mode1, mode1] = mat[mode2, mode2] = 1.0 / np.sqrt(2)
    mat[mode1, mode2] = mat[mode2, mode1] = 1.0j / np.sqrt(2)
    return mat


def beam_splitter_inv(mode1: int, mode2: int, modes: int) -> np.ndarray:
    mat = np.identity(modes, dtype=np.complex128)
    mat[mode1, mode1] = mat[mode2, mode2] = 1.0j / np.sqrt(2)
    mat[mode1, mode2] = mat[mode2, mode1] = 1.0 / np.sqrt(2)
    return mat


def phase_shifter(mode: int, modes: int, angle: float) -> np.ndarray:
    mat = np.identity(modes, dtype=np.complex128)
    mat[mode, mode] = np.exp(1j * angle)
    return mat


def swap(mode1: int, mode2: int, modes: int) -> np.ndarray:
    mat = np.identity(modes, dtype=np.complex128)
    mat[:, [mode1, mode2]] = mat[:, [mode2, mode1]]
    return mat


def mach_zehnder(mode_con: int, mode_sen: int, modes: int, angle: float) -> np.ndarray:
    bs = beam_splitter_inv(mode_con, mode_sen, modes)
    phase = phase_shifter(mode_con, modes, angle)
    return bs @ phase @ bs


# TODO: write tests for this!
