"""Gaussian noise SFTs: independent per bin, or cut from one continuous series."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int, PRNGKeyArray

from jpec.detector.sft import STRAIN_UNIT, SFTData
from jpec.detector.site import Detector
from jpec.waveform.window import Window

# SFTs transformed at once
SFT_BATCH_SIZE = 1024


def gaussian_noise_series_sft(
    timestamps: Float[np.ndarray, " n_sft"],
    detector: Detector,
    *,
    key: PRNGKeyArray,
    freq_low: float,
    n_band_bins: int,
    tsft: float,
    sqrt_sn: float,
    window: Window,
    rate: float,
) -> SFTData:
    """SFTs cut from one continuous Gaussian series, so overlapping SFTs share samples.

        E|x|^2 = sqrt_sn^2 * tsft / 2,  as in gaussian_noise_sft()

    Args:
        timestamps: SFT start times, absolute GPS [s].
        detector: The interferometer the SFTs belong to.
        key: PRNG key.
        freq_low: Start frequency of the first frequency bin [Hz].
        n_band_bins: Frequency bins per SFT.
        tsft: SFT length [s].
        sqrt_sn: Amplitude spectral density of the strain, flat over the band
            [1/sqrt(Hz)].
        window: The window each SFT is made with.
        rate: Sampling rate of the complex baseband series [Hz], above the band's
            width.

    Returns:
        SFTs on the timestamps in ascending order, at `STRAIN_UNIT`.
    """
    starts = np.sort(np.asarray(timestamps, dtype=np.float64))
    # the band as a complex baseband about its middle bin, sampled at rate: the SFT
    # bin m from the middle is bin m of the DFT of the windowed baseband segment
    n_samples_per_sft = int(round(tsft * rate))
    middle_bin_offsets = np.arange(n_band_bins) - n_band_bins // 2
    window_samples = np.asarray(window.samples(n_samples_per_sft))
    # E|X|^2 = Sn tsft / 2 for the unit-variance series after the window and the
    # 1 / rate of the DFT; in float64 on the host, since Sn ~ 1e-46 underflows float32
    noise_amplitude = float(
        sqrt_sn * np.sqrt(0.5 * tsft) * rate / np.sqrt(np.sum(window_samples**2))
    )
    # in units of STRAIN_UNIT, as load_sft_data() returns SFTs, so the power is ~1 in
    # float32
    noise_amplitude /= STRAIN_UNIT
    window_samples = jnp.asarray(window_samples)

    start_sample_idxs = jnp.asarray(np.round((starts - starts[0]) * rate).astype(int))
    n_samples = int(start_sample_idxs[-1]) + n_samples_per_sft
    real, imag = jax.random.normal(key, (2, n_samples))
    noise_series = (real + 1j * imag) / jnp.sqrt(2.0)

    def sft_at(start_sample_idx: Int[Array, ""]) -> Complex[Array, " n_band"]:
        segment = jax.lax.dynamic_slice(
            noise_series, (start_sample_idx,), (n_samples_per_sft,)
        )
        spectrum = jnp.fft.fftshift(jnp.fft.fft(segment * window_samples)) / rate
        return noise_amplitude * spectrum[n_samples_per_sft // 2 + middle_bin_offsets]

    # lax.map, not vmap: SFT_BATCH_SIZE segments in memory at a time
    sft_data = jax.lax.map(sft_at, start_sample_idxs, batch_size=SFT_BATCH_SIZE)
    return SFTData(
        detector=detector,
        sft_data=np.asarray(sft_data),
        timestamps=starts,
        freq_low=freq_low,
        tsft=tsft,
        window=window,
        strain_unit=STRAIN_UNIT,
    )


def gaussian_noise_sft(
    timestamps: Float[np.ndarray, " n_sft"],
    detector: Detector,
    *,
    key: PRNGKeyArray,
    freq_low: float,
    n_band_bins: int,
    tsft: float,
    sqrt_sn: float,
    window: Window = Window(),
) -> SFTData:
    """SFTs of stationary Gaussian strain, no signal in them.

        E|x|^2 = sqrt_sn^2 * tsft / 2,  so the power 2|x|^2 has mean sqrt_sn^2 * tsft

    Args:
        timestamps: SFT start times, absolute GPS [s].
        detector: The interferometer the SFTs belong to.
        key: PRNG key.
        freq_low: Start frequency of the first frequency bin [Hz].
        n_band_bins: Frequency bins per SFT.
        tsft: SFT length [s].
        sqrt_sn: Amplitude spectral density of the strain, flat over the band
            [1/sqrt(Hz)].
        window: The window the SFTs are taken to have been made with.

    Returns:
        SFTs whose amplitudes are independent circular Gaussians.
    """
    sigma = sqrt_sn * jnp.sqrt(tsft / 4.0)
    real, imag = jax.random.normal(key, (2, timestamps.size, n_band_bins))
    sft_data = sigma * (real + 1j * imag)
    return SFTData(
        detector=detector,
        sft_data=np.asarray(sft_data),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        freq_low=freq_low,
        tsft=tsft,
        window=window,
    )
