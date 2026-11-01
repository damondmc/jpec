"""Estimating the floor from data that may contain a signal."""

import numpy as np
import pytest

from jpec.noise.floor import median_bias, noise_floor

N_SFTS, N_BIN = 40, 512
WIDTH, BATCH_SIZE = 101, 512


@pytest.fixture(scope="module")
def power():
    """Scaled chi2_2 power with a floor that varies over the band."""
    rng = np.random.default_rng(5)
    shape = 1.0 + 2.0 * np.linspace(0.0, 1.0, N_BIN) ** 2
    return rng.exponential(1.0, size=(N_SFTS, N_BIN)) * shape, shape


def test_median_bias_tends_to_ln2():
    assert median_bias(10001) == pytest.approx(np.log(2.0), abs=1e-4)


def test_noise_floor_recovers_the_mean(power):
    """The bias correction is what makes this an estimate of the mean."""
    observed, shape = power
    estimated = noise_floor(observed, width=WIDTH, batch_size=BATCH_SIZE).mean(axis=0)
    middle_bin_idxs = slice(100, N_BIN - 100)
    assert np.abs(estimated[middle_bin_idxs] / shape[middle_bin_idxs] - 1.0).max() < 0.1


def test_noise_floor_resists_a_line_far_better_than_a_mean(power):
    """The reason for a median, measured rather than asserted.

    A line 1000x the floor still moves the median estimate, by shifting it one
    rank within each window it falls in: about 0.2 at width 101, taken as a max
    over every affected bin. The same line moves a mean filter by 9.2, so the
    median is the difference between a usable floor and a useless one.
    """
    observed, _ = power
    with_line = observed.copy()
    with_line[:, N_BIN // 2] *= 1000.0

    median_shift = np.abs(
        noise_floor(with_line, width=WIDTH, batch_size=BATCH_SIZE)
        / noise_floor(observed, width=WIDTH, batch_size=BATCH_SIZE)
        - 1.0
    ).max()

    kernel = np.ones(WIDTH) / WIDTH
    mean_shift = np.abs(
        np.convolve(with_line.mean(axis=0), kernel, "same")
        / np.convolve(observed.mean(axis=0), kernel, "same")
        - 1.0
    ).max()

    assert median_shift < 0.5
    assert mean_shift > 20.0 * median_shift


def test_normalized_power_unit_mean(power):
    observed, _ = power
    middle_bin_idxs = slice(100, N_BIN - 100)
    normalized = observed / noise_floor(observed, width=WIDTH, batch_size=BATCH_SIZE)
    assert normalized[:, middle_bin_idxs].mean() == pytest.approx(1.0, rel=0.05)
