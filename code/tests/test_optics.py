import pytest

import numpy as np

from mlgst import optics

SQRT_2 = np.sqrt(2)
INV_SQRT_2 = 1 / np.sqrt(2)


def test_bs():
    bs_012 = optics.beam_splitter(0, 1, 2)
    bs_012_expected = INV_SQRT_2 * np.array([[1, 1j], [1j, 1]])
    assert np.allclose(bs_012, bs_012_expected)

    bs_102 = optics.beam_splitter(1, 0, 2)
    bs_102_expected = INV_SQRT_2 * np.array([[1, 1j], [1j, 1]])
    assert np.allclose(bs_102, bs_102_expected)

    bs_134 = optics.beam_splitter(1, 3, 4)
    bs_134_expected = np.array([
        [1, 0, 0, 0],
        [0, INV_SQRT_2, 0, 1j * INV_SQRT_2],
        [0, 0, 1, 0],
        [0, 1j * INV_SQRT_2, 0, INV_SQRT_2]
    ])
    assert np.allclose(bs_134, bs_134_expected)


def test_bsi():
    bsi_012 = optics.beam_splitter_inv(0, 1, 2)
    bsi_012_expected = INV_SQRT_2 * np.array([[1j, 1], [1, 1j]])
    assert np.allclose(bsi_012, bsi_012_expected)

    bsi_102 = optics.beam_splitter_inv(1, 0, 2)
    bsi_102_expected = INV_SQRT_2 * np.array([[1j, 1], [1, 1j]])
    assert np.allclose(bsi_102, bsi_102_expected)

    bsi_134 = optics.beam_splitter_inv(1, 3, 4)
    bsi_134_expected = np.array([
        [1, 0, 0, 0],
        [0, 1j * INV_SQRT_2, 0, INV_SQRT_2],
        [0, 0, 1, 0],
        [0, INV_SQRT_2, 0, 1j * INV_SQRT_2]
    ])
    assert np.allclose(bsi_134, bsi_134_expected)


def test_phase():
    phase_02pi = optics.phase_shifter(0, 2, np.pi)
    phase_02pi_expected = np.array([[np.exp(1j * np.pi), 0], [0, 1]])
    assert np.allclose(phase_02pi, phase_02pi_expected)

    phase_12pi = optics.phase_shifter(1, 2, np.pi)
    phase_12pi_expected = np.array([[1, 0], [0, np.exp(1j * np.pi)]])
    assert np.allclose(phase_12pi, phase_12pi_expected)


def test_mzi():
    mzi_012pi = optics.mach_zehnder(0, 1, 2, np.pi)
    mzi_012pi_expected = 0.5 * np.array([
        [1 - np.exp(1j * np.pi), 1j + 1j * np.exp(1j * np.pi)],
        [1j + 1j * np.exp(1j * np.pi), -1 + np.exp(1j * np.pi)]
    ])
    assert np.allclose(mzi_012pi, mzi_012pi_expected)

    mzi_102 = optics.mach_zehnder(1, 0, 2, np.pi)
    mzi_102_expected = 0.5 * np.array([
        [-1 + np.exp(1j * np.pi), 1j + 1j * np.exp(1j * np.pi)],
        [1j + 1j * np.exp(1j * np.pi), 1 - np.exp(1j * np.pi)]
    ])
    assert np.allclose(mzi_102, mzi_102_expected)

    mzi_134 = optics.mach_zehnder(1, 3, 4, np.pi)
    mzi_134_expected = 0.5 * np.array([
        [2, 0, 0, 0],
        [0, 1 - np.exp(1j * np.pi), 0, 1j + 1j * np.exp(1j * np.pi)],
        [0, 0, 2, 0],
        [0, 1j + 1j * np.exp(1j * np.pi), 0, -1 + np.exp(1j * np.pi)]
    ])
    assert np.allclose(mzi_134, mzi_134_expected)


def test_unitary():
    powers_5 = np.array([0.130697, 0.801256, 0.365318, 0.427437, 0.0590873])
    c_5 = np.array([
        [0.721182, 0.970473, 0.57333, 0.617351, 0.0879615],
        [0.859966, 0.534076, 0.108389, 0.921755, 0.0301802],
        [0.38583, 0.617491, 0.0828841, 0.459525, 0.238587],
        [0.656543, 0.597688, 0.676801, 0.889742, 0.203281],
        [0.442914, 0.914441, 0.732551, 0.261567, 0.866731]
    ])
    phi_5 = np.array([0.0462296, 0.115686, 0.179755, 0.19348, 0.0585595])
    unitary_5 = optics.get_unitary(powers_5, c_5, phi_5)
    unitary_5_expected = np.array([
        [0.491983 + 0.782342j, -0.133828 + 0.357748j],
        [0.186425 + 0.333375j, 0.367922 - 0.847785j]
    ])
    assert np.allclose(unitary_5, unitary_5_expected)

    powers_7 = np.array([0.694631, 0.646707, 0.56385, 0.791936, 0.448527, 0.740797, 0.422611])
    c_7 = np.array([
        [0.0476046, 0.504953, 0.397004, 0.307984, 0.0711502, 0.322256, 0.476771],
        [0.497568, 0.0672892, 0.192281, 0.513033, 0.608238, 0.830069, 0.796176],
        [0.214502, 0.308845, 0.836401, 0.423166, 0.0100205, 0.719198, 0.117782],
        [0.451509, 0.894522, 0.392177, 0.551481, 0.538789, 0.854321, 0.155581],
        [0.983404, 0.01756, 0.495607, 0.723889, 0.477488, 0.778591, 0.497808],
        [0.48512, 0.020831, 0.0166447, 0.698334, 0.0846799, 0.399288, 0.987171],
        [0.990478, 0.544924, 0.211428, 0.477875, 0.939163, 0.069104, 0.921353]
    ])
    phi_7 = np.array([0.0150283, 0.0532362, 0.182773, 0.0121632, 0.114404, 0.154429, 0.167751])
    unitary_7 = optics.get_unitary(powers_7, c_7, phi_7)
    unitary_7_expected = np.array([
        [0.441478 - 0.621257j, -0.441045 - 0.473937j],
        [-0.00896864 + 0.647346j, -0.74421 - 0.164363j]
    ])
    assert np.allclose(unitary_7, unitary_7_expected)
