import jax.numpy as jnp
import numpy as np
import pytest
from signals import Case, search_template_harmonics

from jpec.constant import ARC_PAD_BINS, THETA_OVERSAMPLE
from jpec.waveform.harmonics import (
    harmonic_count,
    template_harmonics,
    theta_sample_count,
)
from jpec.waveform.plane import arc_bin_offsets, template_plane


def test_harmonic_count_is_at_least_one():
    assert harmonic_count(0.0) == 1


@pytest.mark.parametrize("j_max", [1, 7, 28, 91, 351])
def test_phase_samples_is_a_power_of_two_above_nyquist(j_max):
    n = theta_sample_count(j_max, oversample=THETA_OVERSAMPLE)
    assert n > 2 * j_max
    assert n & (n - 1) == 0


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
