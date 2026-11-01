"""Doppler correction of SFT amplitudes toward a sky position, before squaring."""

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from jpec.typing import Scalar


def sky_direction(alpha: Scalar, delta: Scalar) -> Float[Array, " 3"]:
    """Unit vector toward a sky position, in the frame `Detector.velocity` uses.

    Args:
        alpha: Right ascension [rad].
        delta: Declination [rad].

    Returns:
        `(cos(delta) cos(alpha), cos(delta) sin(alpha), sin(delta))`.
    """
    return jnp.stack(
        [
            jnp.cos(delta) * jnp.cos(alpha),
            jnp.cos(delta) * jnp.sin(alpha),
            jnp.sin(delta),
        ]
    )


def shift_spectrum(
    sft_time_series: Complex[Array, "n_sft n_band"],
    *,
    shift: Float[Array, " n_sft"],
) -> Complex[Array, "n_sft n_band"]:
    """Correct the spectrum for Doppler shift, moving bin `k + dk` to bin `k`.

        Y[k] = X[k + dk],  by  y[m] = x[m] exp(-2 pi i dk m / n_band_bins)

    The phase factor multiplies the inverse transform of the band, not the strain, so
    the shift is circular within the band: anything pushed past the top edge
    reappears at the bottom edge, and vice versa. It is exact only for a whole
    number of bins. A fractional one truncates the leakage that falls off the
    ends, which is why the band wants margin beyond the template's.

    Args:
        sft_time_series: `x[m]`, the inverse transform of each SFT's band; it does not
            depend on the shift, so it is taken once.
        shift: `dk` at each SFT [bins].

    Returns:
        The shifted spectrum, in `sft_time_series.dtype`.
    """
    n_band_bins = sft_time_series.shape[-1]
    sample_idxs = jnp.arange(n_band_bins)
    phase = jnp.exp(-2j * jnp.pi * shift[:, None] * sample_idxs[None, :] / n_band_bins)
    shifted_x_n = sft_time_series * phase.astype(sft_time_series.dtype)
    return jnp.fft.fft(shifted_x_n, axis=-1).astype(sft_time_series.dtype)


def doppler_corrected_power(
    sft_time_series: list[Complex[Array, "n_sft n_band"]],
    *,
    floor: list[Float[Array, "n_sft n_band"]],
    velocity_per_detector: list[Float[Array, "n_sft 3"]],
    tsft: float,
    alpha: Scalar,
    delta: Scalar,
    doppler_reference: Scalar,
) -> Float[Array, "n_sft n_band"]:
    """A band Doppler-corrected toward one sky position, squared and normalized.

        shift(t) = doppler_reference (v(t) . n_hat) tsft   [bins]

    with `v` the detector velocity over c and `n_hat` the direction to the sky
    position.

    Args:
        sft_time_series: Inverse transform of the scaled SFT amplitudes' band, per
            detector.
        floor: Mean noise power in each SFT and bin, per detector.
        velocity_per_detector: Detector velocity `v/c` at each SFT, per detector.
        tsft: SFT length [s].
        alpha: Right ascension [rad].
        delta: Declination [rad].
        doppler_reference: Frequency the Doppler shift is taken at [Hz].

    Returns:
        Normalized power at each SFT of every detector, concatenated, the full band
        wide.
    """
    direction = sky_direction(alpha, delta)
    planes = []
    for detector_time_series, detector_floor, detector_velocity in zip(
        sft_time_series, floor, velocity_per_detector
    ):
        shift = doppler_reference * (detector_velocity @ direction) * tsft
        corrected = shift_spectrum(detector_time_series, shift=shift)
        planes.append(2.0 * jnp.abs(corrected) ** 2 / detector_floor)
    return jnp.concatenate(planes)
