import jax
import jax.numpy as jnp
import numpy as np
import pytest
from signals import Case, search_data_harmonics, search_template_harmonics

from jpec.constant import ARC_PAD_BINS, DAY, PHASE_OVERSAMPLE
from jpec.detector.sft import SFTData
from jpec.detector.site import Detector
from jpec.injection.signal import (
    Injection,
    Orientation,
    sft_with_simulated_signal,
    simulated_signal_sft_data,
)
from jpec.noise.floor import noise_floor
from jpec.noise.gaussian import gaussian_noise_sft
from jpec.params import Parameters
from jpec.stat.matched_filter import max_over_phase
from jpec.waveform.harmonics import harmonic_count, phase_sample_count
from jpec.waveform.plane import amplitude_from_arc, arc_bin_offsets, plane_from_arc


def test_amplitude_squares_to_the_plane(case: Case, theta):
    """The complex amplitude of a sinusoid and the power plane are one kernel."""
    offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    arc = -case.modulation_depth_bins * jnp.cos(theta)
    amplitude = amplitude_from_arc(arc, offsets, window=case.window)
    np.testing.assert_allclose(
        jnp.abs(amplitude) ** 2,
        plane_from_arc(arc, offsets, window=case.window),
        rtol=1e-12,
    )


T0 = 1368975618.0
TSFT = 1024.0
DAYS = 20.0
FREQ_LOW, N_BAND_BINS = 249.9, 256
SQRT_SN = 1.0
SIGNAL_PARAMETERS = Parameters(
    f0=250.0, ap=6.6, period=9.0 * DAY, phi_b=2.3, alpha=1.2, delta=0.3
)
FACE_ON = Injection(SIGNAL_PARAMETERS, Orientation(cos_iota=1.0, psi=0.0))


@pytest.fixture(scope="module")
def timestamps():
    return T0 + np.arange(0.0, DAYS * DAY, TSFT)


@pytest.fixture(scope="module")
def t_ref(timestamps):
    return float((timestamps + TSFT / 2.0).mean())


def noise(timestamps, *, seed: int, sqrt_sn: float) -> SFTData:
    """Gaussian noise SFTs on the test band."""
    return gaussian_noise_sft(
        timestamps,
        Detector("H1"),
        key=jax.random.key(seed),
        freq_low=FREQ_LOW,
        n_band_bins=N_BAND_BINS,
        tsft=TSFT,
        sqrt_sn=sqrt_sn,
    )


def inject(sft: SFTData, *, t_ref: float, h0: float) -> SFTData:
    """The test signal added to some SFTs."""
    unit_sft_data = simulated_signal_sft_data(sft, FACE_ON, t_ref=t_ref)
    return sft_with_simulated_signal(sft, np.asarray(unit_sft_data), h0=h0)


def test_gaussian_sfts_have_the_asked_for_floor(timestamps):
    """2|x|^2 has mean sqrt_sn^2 tsft, the normalization null_sigma assumes."""
    data = noise(timestamps, seed=0, sqrt_sn=SQRT_SN)
    assert data.sft_data.shape == (timestamps.size, N_BAND_BINS)
    assert data.sft_power.mean() == pytest.approx(SQRT_SN**2 * TSFT, rel=0.02)


def test_injection_lands_on_the_signal_track(timestamps, t_ref):
    """The loudest bin of each SFT follows the Doppler-shifted binary track."""
    quiet = noise(timestamps, seed=1, sqrt_sn=0.0)
    loud = inject(quiet, t_ref=t_ref, h0=1.0)

    omega = 2.0 * np.pi / float(SIGNAL_PARAMETERS.period)
    times = timestamps + TSFT / 2.0 - t_ref
    doppler = quiet.velocity @ np.array(
        [
            np.cos(SIGNAL_PARAMETERS.delta) * np.cos(SIGNAL_PARAMETERS.alpha),
            np.cos(SIGNAL_PARAMETERS.delta) * np.sin(SIGNAL_PARAMETERS.alpha),
            np.sin(SIGNAL_PARAMETERS.delta),
        ]
    )
    modulation = float(SIGNAL_PARAMETERS.f0) * float(SIGNAL_PARAMETERS.ap) * omega
    track = float(SIGNAL_PARAMETERS.f0) * (1.0 + doppler) - modulation * np.cos(
        omega * times - float(SIGNAL_PARAMETERS.phi_b)
    )
    expected = (track - FREQ_LOW) * TSFT
    assert np.abs(np.argmax(loud.sft_power, axis=1) - expected).max() < 1.0


def test_injection_adds_and_scales_with_h0(timestamps, t_ref):
    """Injecting into noise adds the same amplitudes; signal power grows as h0^2."""
    noisy = noise(timestamps, seed=2, sqrt_sn=SQRT_SN)
    quiet = noise(timestamps, seed=2, sqrt_sn=0.0)
    h0 = 1e3 * SQRT_SN
    added = inject(noisy, t_ref=t_ref, h0=h0)
    alone = inject(quiet, t_ref=t_ref, h0=h0)
    np.testing.assert_allclose(
        added.sft_data - noisy.sft_data, alone.sft_data, rtol=1e-5
    )

    louder = inject(quiet, t_ref=t_ref, h0=2.0 * h0)
    assert louder.sft_power.sum() / alone.sft_power.sum() == pytest.approx(
        4.0, rel=1e-4
    )


def test_injection_is_in_strain_whatever_the_strain_unit(timestamps, t_ref):
    """SFTs in a strain unit take the signal in strain x s, divided down to theirs."""
    quiet = noise(timestamps, seed=4, sqrt_sn=0.0)
    unscaled = inject(quiet, t_ref=t_ref, h0=1.0)
    scaled = inject(quiet.rescaled(4.0), t_ref=t_ref, h0=1.0)
    np.testing.assert_allclose(
        scaled.sft_data * scaled.strain_unit, unscaled.sft_data, rtol=1e-6
    )


def test_the_search_recovers_an_injection(timestamps, t_ref):
    """One template scored on simulated SFTs: the statistic peaks at the true period."""
    data = inject(
        noise(timestamps, seed=3, sqrt_sn=SQRT_SN), t_ref=t_ref, h0=0.5 * SQRT_SN
    )
    floor = noise_floor(jnp.asarray(data.sft_power), width=101, batch_size=512)
    times = jnp.asarray(data.timestamps_midpoints - t_ref)
    modulation_depth_bins = float(SIGNAL_PARAMETERS.modulation_depth_bins(TSFT))
    j_max = harmonic_count(modulation_depth_bins)

    def max_at(period: float) -> float:
        data_harmonics, offsets = search_data_harmonics(
            data,
            SIGNAL_PARAMETERS._replace(period=period),
            floor=floor,
            velocity=data.velocity,
            times=times,
            half_width_bins=int(np.ceil(modulation_depth_bins)) + 8,
            j_max=j_max,
        )
        template_harmonics = search_template_harmonics(
            modulation_depth_bins, data.window, bin_offsets=jnp.asarray(offsets)
        )
        statistic, _ = max_over_phase(
            data_harmonics=data_harmonics[: template_harmonics.shape[0]],
            template_harmonics=template_harmonics,
            n_phi=phase_sample_count(j_max, oversample=PHASE_OVERSAMPLE),
        )
        return float(statistic)

    assert max_at(float(SIGNAL_PARAMETERS.period)) > 10.0 * max_at(
        1.5 * float(SIGNAL_PARAMETERS.period)
    )
