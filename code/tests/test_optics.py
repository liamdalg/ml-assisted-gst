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
