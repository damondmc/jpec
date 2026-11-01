import jax.numpy as jnp
import numpy as np
import pytest

from jpec.detector.doppler import (
    doppler_corrected_power,
    shift_spectrum,
    sky_direction,
)
from jpec.detector.sft import SFTData
from jpec.detector.site import Detector
from jpec.waveform.window import Window

T0 = 1238166018.0
TSFT = 1800.0
ALPHA, DELTA = 1.1, 0.3
N_SFTS = 30


def sim_segment(sinusoid_bin: float, n_samples: int = 2**14, beta: float = 0.001):
    """One SFT on the time axis: a Tukey-windowed sinusoid at ``sinusoid_bin``.

    Returns:
        The time-sample index and the strain at each sample.
    """
    sample_idxs = np.arange(n_samples)
    samples_from_edge = np.minimum(sample_idxs, n_samples - 1 - sample_idxs)
    taper = round(beta * n_samples)
    taper_position = (2.0 * samples_from_edge - (taper - 1)) / max(taper - 1, 1)
    window = np.where(
        samples_from_edge < (taper + 1) // 2,
        np.cos(0.5 * np.pi * taper_position) ** 2,
        1.0,
    )
    strain = window * np.cos(2.0 * np.pi * sinusoid_bin * sample_idxs / n_samples)
    return sample_idxs, strain


@pytest.fixture(scope="module")
def data():
    """``N_SFTS`` SFTs on the time axis, each the same sinusoid, one per day."""
    _, strain = sim_segment(4000.37)
    sft_data = np.tile(np.fft.fft(strain), (N_SFTS, 1)).astype(np.complex64)
    return SFTData(
        detector=Detector("H1"),
        sft_data=sft_data,
        timestamps=T0 + 86400.0 * np.arange(N_SFTS),
        freq_low=100.0,
        tsft=TSFT,
        window=Window(),
    )


@pytest.fixture(scope="module")
def velocity(data):
    return data.detector.velocity(data.timestamps_midpoints)


def test_sky_direction_is_a_unit_vector_in_the_equatorial_frame():
    """Right ascension runs around the equator from x, declination up to the pole."""
    np.testing.assert_allclose(sky_direction(0.0, 0.0), [1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(
        sky_direction(0.5 * np.pi, 0.0), [0.0, 1.0, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        sky_direction(ALPHA, 0.5 * np.pi), [0.0, 0.0, 1.0], atol=1e-15
    )
    assert float(jnp.linalg.norm(sky_direction(ALPHA, DELTA))) == pytest.approx(1.0)


def test_correction_is_the_projected_shift(data, velocity):
    """shift = f0 * v.n/c * tsft, shift_spectrum on amplitudes, then 2|.|^2 / floor."""
    direction = np.array(
        [np.cos(DELTA) * np.cos(ALPHA), np.cos(DELTA) * np.sin(ALPHA), np.sin(DELTA)]
    )
    shift_bins = 150.0 * (velocity @ direction) * TSFT
    assert 1.0 < np.abs(shift_bins).max() < 30.0
    sft_time_series = jnp.fft.ifft(jnp.asarray(data.sft_data), axis=-1)
    moved = shift_spectrum(sft_time_series, shift=jnp.asarray(shift_bins))
    assert moved.dtype == data.sft_data.dtype
    floor = jnp.full(data.sft_data.shape, 2.0)
    corrected_sft_power = doppler_corrected_power(
        [sft_time_series],
        floor=[floor],
        velocity_per_detector=[jnp.asarray(velocity)],
        tsft=data.tsft,
        alpha=ALPHA,
        delta=DELTA,
        doppler_reference=150.0,
    )
    np.testing.assert_allclose(
        corrected_sft_power, 2.0 * np.abs(moved) ** 2 / floor, rtol=1e-6
    )


def test_whole_bin_shift_is_an_exact_roll():
    """An integer shift is a circular roll along the frequency axis."""
    _, strain = sim_segment(4000.0)
    spectrum = jnp.asarray(np.fft.fft(strain))[None, :]
    moved = shift_spectrum(jnp.fft.ifft(spectrum, axis=-1), shift=jnp.asarray([7.0]))
    np.testing.assert_allclose(moved, np.roll(spectrum, -7, axis=-1), atol=1e-9)


def band_shift_error(margin_bins: int, shift_bins: float = 0.37) -> float:
    """Error of shifting a band extending ``margin_bins`` either side of a sinusoid."""
    sinusoid_bin = 4000.37
    sample_idxs, strain = sim_segment(sinusoid_bin)
    n_samples = strain.size
    spectrum = np.fft.fft(strain)
    truth = np.fft.fft(
        strain * np.exp(-2j * np.pi * shift_bins * sample_idxs / n_samples)
    )

    bin_low, bin_high = int(sinusoid_bin) - margin_bins, int(sinusoid_bin) + margin_bins
    band = jnp.asarray(spectrum[bin_low:bin_high])[None, :]
    moved = shift_spectrum(jnp.fft.ifft(band, axis=-1), shift=jnp.asarray([shift_bins]))
    center_bin_idxs = slice(margin_bins - 8, margin_bins + 8)
    return float(
        np.abs(
            np.asarray(moved)[0, center_bin_idxs]
            - truth[bin_low:bin_high][center_bin_idxs]
        ).max()
        / np.abs(truth[bin_low:bin_high]).max()
    )


def test_fractional_shift_needs_band_margin():
    """A fractional shift is circular in the band, so truncated leakage limits it.

    The Dirichlet tail is 1/(pi*u) at u bins from the line, so the leakage cut
    at the band edge leaves an error of about 1/(pi*margin_bins)."""
    narrow_margin_bins, wide_margin_bins = 32, 256
    err_narrow, err_wide = (
        band_shift_error(narrow_margin_bins),
        band_shift_error(wide_margin_bins),
    )
    buffer = 1.6
    assert err_wide < buffer / (np.pi * wide_margin_bins)
    assert err_wide < err_narrow * (narrow_margin_bins / wide_margin_bins)
