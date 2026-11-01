import jax
import jax.numpy as jnp
import numpy as np
import pytest
from signals import (
    DAY,
    Case,
    harmonic_transform,
    orbit_offset,
    search_data_harmonics,
    search_template_harmonics,
)

from jpec.constant import ARC_PAD_BINS, PHASE_OVERSAMPLE
from jpec.detector.sft import SFTData
from jpec.detector.site import Detector
from jpec.noise.floor import noise_floor
from jpec.noise.gaussian import gaussian_noise_sft
from jpec.stat.matched_filter import (
    data_harmonics,
    harmonic_overlap,
    orbital_basis,
    statistic_over_phase,
)
from jpec.waveform.harmonics import harmonic_count, phase_sample_count
from jpec.waveform.plane import arc_bin_offsets, plane_from_arc, template_plane
from jpec.waveform.track import observed_frequency
from jpec.waveform.window import Window, tukey_window

T0 = 1238166018.0
ALPHA, DELTA = 1.1, 0.3
N_SAMPLES = 1024  # per SFT: frequency bins of width 1/tsft, f0 at the middle one
TUKEY_BETA = 0.001
NOISE_SIGMA = (
    256.0  # per quadrature, against a signal of ~N_SAMPLES in its frequency bin
)


def plane_harmonics(plane, theta, j_max):
    """Data harmonics of a plane at these orbital phases, as times at unit omega."""
    return data_harmonics(
        plane, orbital_basis(theta, omega=jnp.asarray(1.0), j_max=j_max)
    )


def test_data_harmonics_match_the_fft_on_a_uniform_grid(case: Case, theta):
    plane = template_plane(
        arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS),
        theta=theta,
        modulation_depth_bins=case.modulation_depth_bins,
        window=case.window,
    )
    j_max = harmonic_count(case.modulation_depth_bins)
    direct = plane_harmonics(plane, theta, j_max)
    fast = harmonic_transform(plane)[1 : j_max + 1]
    np.testing.assert_allclose(direct, fast, atol=1e-12)


def test_gappy_harmonics_converge_with_more_sfts(case: Case):
    parameters = case.parameters
    j_max = harmonic_count(case.modulation_depth_bins)
    offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    n = phase_sample_count(j_max, oversample=PHASE_OVERSAMPLE)

    grid = jnp.arange(n) * (parameters.period / n)
    clean = plane_harmonics(
        plane_from_arc(
            orbit_offset(parameters, grid) * case.tsft, offsets, window=case.window
        ),
        parameters.omega * grid,
        j_max,
    )

    def gappy(density: int) -> float:
        rng = np.random.default_rng(0)
        kept = np.sort(rng.choice(6 * n * density, size=4 * n * density, replace=False))
        t = jnp.asarray(kept * (parameters.period / (n * density)))
        ragged = plane_harmonics(
            plane_from_arc(
                orbit_offset(parameters, t) * case.tsft, offsets, window=case.window
            ),
            parameters.omega * t,
            j_max,
        )
        return float(jnp.max(jnp.abs(ragged - clean)) / jnp.max(jnp.abs(clean)))

    # sixteen times the SFTs should buy a factor of four, as 1/sqrt(N) predicts;
    # two leaves margin for the random subset
    sparse, dense = gappy(1), gappy(16)
    assert sparse > 2.0 * dense


@pytest.fixture
def sfts(case: Case) -> SFTData:
    """Twenty days of complex SFTs of the signal in Gaussian noise.

    The signal is Doppler-shifted by H1's motion.

    Each SFT is the FFT of a Tukey-windowed complex sinusoid at the frequency
    the detector sees at that SFT's midpoint, so the amplitudes carry the
    window's leakage and the phase a real SFT would, plus noise for the floor
    to measure.
    """
    parameters = case.parameters
    starts = T0 + np.arange(0.0, 20.0 * DAY, case.tsft)
    midpoints = starts + case.tsft / 2.0
    velocity = Detector("H1").velocity(midpoints)
    doppler = velocity @ np.asarray(
        [np.cos(DELTA) * np.cos(ALPHA), np.cos(DELTA) * np.sin(ALPHA), np.sin(DELTA)]
    )
    observed = np.asarray(
        observed_frequency(
            jnp.asarray(midpoints - T0),
            f0=parameters.f0,
            ap=parameters.ap,
            omega=parameters.omega,
            phi_b=parameters.phi_b,
            doppler=jnp.asarray(doppler),
        )
    )

    freq_low = float(parameters.f0) - (N_SAMPLES // 2) / case.tsft
    sample = np.arange(N_SAMPLES) / N_SAMPLES
    window = np.asarray(tukey_window(N_SAMPLES, TUKEY_BETA))
    signal_samples = np.exp(
        2j * np.pi * (observed - freq_low)[:, None] * case.tsft * sample[None, :]
    )
    noise = gaussian_noise_sft(
        starts,
        Detector("H1"),
        key=jax.random.key(0),
        freq_low=freq_low,
        n_band_bins=N_SAMPLES,
        tsft=case.tsft,
        sqrt_sn=NOISE_SIGMA / np.sqrt(case.tsft / 4.0),
        window=Window(kind="tukey", beta=TUKEY_BETA),
    )
    signal = np.fft.fft(window[None, :] * signal_samples, axis=1)
    sft_data = (signal + noise.sft_data).astype(np.complex64)
    return SFTData(
        detector=noise.detector,
        sft_data=sft_data,
        timestamps=starts,
        freq_low=freq_low,
        tsft=case.tsft,
        window=noise.window,
    )


def statistic_from_sft(sft: SFTData, case: Case, *, velocity: np.ndarray):
    """load, correct, square, slice, transform, match: one template's search path."""
    modulation_depth_bins = case.modulation_depth_bins
    j_max = harmonic_count(modulation_depth_bins)
    half_width = int(np.ceil(modulation_depth_bins)) + 8
    window_data_harmonics, offsets = search_data_harmonics(
        sft,
        case.parameters._replace(alpha=ALPHA, delta=DELTA),
        floor=noise_floor(jnp.asarray(sft.sft_power), width=101, batch_size=512),
        velocity=velocity,
        times=jnp.asarray(sft.timestamps_midpoints - T0),
        half_width_bins=half_width,
        j_max=j_max,
    )
    assert window_data_harmonics.shape == (j_max, 2 * half_width + 1)
    assert offsets[half_width] == pytest.approx(0.0)
    template_harmonics = search_template_harmonics(
        modulation_depth_bins, sft.window, bin_offsets=jnp.asarray(offsets)
    )
    coefficients = harmonic_overlap(
        data_harmonics=window_data_harmonics[: template_harmonics.shape[0]],
        template_harmonics=template_harmonics,
    )
    return statistic_over_phase(
        coefficients, n_phi=phase_sample_count(j_max, oversample=PHASE_OVERSAMPLE)
    )


def test_search_path_recovers_the_orbital_phase(case: Case, sfts):
    statistic = statistic_from_sft(sfts, case, velocity=sfts.velocity)
    n_phi = statistic.shape[0]
    peak = 2.0 * np.pi * int(jnp.argmax(statistic)) / n_phi
    error = (peak - case.parameters.phi_b + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(error) <= 2.0 * np.pi / n_phi
    assert float(jnp.max(statistic) / jnp.sqrt(jnp.mean(statistic**2))) > 3.0


def test_leaving_the_earth_in_loses_the_signal(case: Case, sfts):
    """Without the correction the harmonics smear and the peak collapses."""
    corrected = statistic_from_sft(sfts, case, velocity=sfts.velocity)
    uncorrected = statistic_from_sft(sfts, case, velocity=np.zeros_like(sfts.velocity))
    assert float(jnp.max(uncorrected)) < 0.2 * float(jnp.max(corrected))
