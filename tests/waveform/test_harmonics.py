import math

import jax.numpy as jnp
import numpy as np
import pytest
from signals import Case, search_template_harmonics

from jpec.constant import ARC_PAD_BINS, THETA_OVERSAMPLE
from jpec.waveform.harmonics import (
    harmonic_count,
    phase_count,
    template_harmonics,
    theta_sample_count,
)
from jpec.waveform.plane import arc_bin_offsets, template_plane
from jpec.waveform.window import Window

# share of each modulation depth's harmonics the phase grid is checked over
HARMONIC_FRACTION = 0.6


def test_harmonic_count_is_at_least_one():
    assert harmonic_count(0.0) == 1


@pytest.mark.parametrize("j_max", [1, 7, 28, 91, 351])
def test_theta_samples_is_a_power_of_two_above_nyquist(j_max):
    n = theta_sample_count(j_max, oversample=THETA_OVERSAMPLE)
    assert n > 2 * j_max
    assert n & (n - 1) == 0


@pytest.mark.parametrize("phase_mismatch", [0.005, 0.01, 0.05])
@pytest.mark.parametrize("modulation_depth_bins", [2.6, 5.0, 15.0, 30.0, 64.0])
def test_phase_count_holds_the_phase_mismatch(modulation_depth_bins, phase_mismatch):
    """A phase half a grid step from the signal's loses at most the phase mismatch."""
    window = Window(kind="tukey", beta=0.5)
    j_max = harmonic_count(modulation_depth_bins)
    n_harmonics = math.ceil(HARMONIC_FRACTION * j_max)
    tau = template_harmonics(
        arc_bin_offsets(modulation_depth_bins, pad_bins=ARC_PAD_BINS),
        modulation_depth_bins=modulation_depth_bins,
        n_theta=theta_sample_count(j_max, oversample=THETA_OVERSAMPLE),
        j_max=j_max,
        window=window,
    )[:n_harmonics]
    harmonic_power = np.asarray(jnp.sum(jnp.abs(tau) ** 2, axis=1))
    n_phi = phase_count(harmonic_power, phase_mismatch=phase_mismatch)

    # S at the grid point furthest from the signal's phase, pi / n_phi away
    harmonics = np.arange(1, n_harmonics + 1)
    worst_loss = 1.0 - float(
        (harmonic_power * np.cos(harmonics * np.pi / n_phi)).sum()
        / harmonic_power.sum()
    )
    assert worst_loss <= phase_mismatch
    assert n_phi >= 2 * n_harmonics + 2


@pytest.mark.parametrize("n_harmonics", [1, 17, 64, 140, 256, 403])
def test_phase_count_is_an_even_product_of_small_primes(n_harmonics):
    """The grid size factors into 2, 3, 5 and 7 and holds at least one 2."""
    harmonic_power = 1.0 / np.arange(1, n_harmonics + 1) ** 2
    n_phi = phase_count(harmonic_power, phase_mismatch=0.01)
    assert n_phi % 2 == 0
    remainder = n_phi
    for prime in (2, 3, 5, 7):
        while remainder % prime == 0:
            remainder //= prime
    assert remainder == 1


def test_template_harmonics_are_the_mean_normalized_transform(case: Case, theta):
    """The /n in the transform, checked against the plane's rfft, with j = 0 dropped."""
    offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    plane = template_plane(
        offsets,
        theta=theta,
        modulation_depth_bins=case.modulation_depth_bins,
        window=case.window,
    )
    j_max = harmonic_count(case.modulation_depth_bins)
    case_template_harmonics = template_harmonics(
        offsets,
        modulation_depth_bins=case.modulation_depth_bins,
        n_theta=theta.size,
        j_max=j_max,
        window=case.window,
    )
    assert case_template_harmonics.shape == (j_max, plane.shape[1])
    expected = jnp.fft.rfft(plane, axis=0)[1 : j_max + 1] / theta.size
    np.testing.assert_allclose(case_template_harmonics, expected, rtol=1e-14)


def test_template_harmonics_normalization(case: Case):
    """The statistic's own phase grid already resolves every harmonic kept."""
    coarse = search_template_harmonics(case.modulation_depth_bins, case.window)
    j_max = harmonic_count(case.modulation_depth_bins)
    fine = template_harmonics(
        arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS),
        modulation_depth_bins=case.modulation_depth_bins,
        n_theta=8192,
        j_max=j_max,
        window=case.window,
    )
    assert coarse.shape == fine.shape
    assert float(jnp.max(jnp.abs(coarse - fine)) / jnp.max(jnp.abs(fine))) < 1e-12


def test_template_harmonics_are_real(case: Case):
    template = search_template_harmonics(case.modulation_depth_bins, case.window)
    imaginary = jnp.max(jnp.abs(jnp.imag(template)))
    assert float(imaginary / jnp.max(jnp.abs(jnp.real(template)))) < 1e-12
