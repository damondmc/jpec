"""The transform: a sky point's Doppler-corrected band and its data harmonics."""

from functools import partial

import jax
from jaxtyping import Array, Complex, Float, Int

from jpec.detector.doppler import doppler_corrected_mean_power, doppler_corrected_power
from jpec.stat.matched_filter import data_harmonics


@partial(jax.jit, static_argnames="n_transformed_bins")
def weighted_excess_power_per_reference(
    sft_time_series: list[Complex[Array, "n_sft n_band"]],
    floor: list[Float[Array, "n_sft n_band"]],
    velocity_per_detector: list[Float[Array, "n_sft 3"]],
    doppler_references: Float[Array, " n_doppler_reference"],
    transformed_start_idxs: Int[Array, " n_doppler_reference"],
    *,
    n_transformed_bins: int,
    tsft: float,
    alpha: float,
    delta: float,
    sft_weights: Float[Array, " n_sft"],
    mean_power: list[Float[Array, " n_band"]],
) -> Float[Array, "n_doppler_reference n_sft n_transformed_bin"]:
    """The band corrected toward one sky position at every Doppler reference in turn.

    Each SFT's power less the power expected from its detector's time-averaged
    spectrum is multiplied by the SFT's weight:

        sft_weights[n] (power[n, k] - expected[n, k])

    Args:
        sft_time_series: Inverse transform of the band, per detector.
        floor: Mean noise power in each SFT and bin, per detector.
        velocity_per_detector: Detector velocity `v/c` at each SFT, per detector.
        doppler_references: Frequencies the Doppler shift is taken at [Hz].
        transformed_start_idxs: First band bin each reference keeps.
        n_transformed_bins: Number of bins each reference keeps from its first.
        tsft: SFT length [s].
        alpha: Right ascension [rad].
        delta: Declination [rad].
        sft_weights: Weight of each SFT toward the sky position, the detectors
            concatenated.
        mean_power: Normalized power averaged over the SFTs, per detector, in the
            detector frame.

    Returns:
        The weighted power at each Doppler reference, over the bins it keeps.
    """

    def at_reference(
        reference_and_start_idx: tuple[Float[Array, ""], Int[Array, ""]],
    ) -> Float[Array, "n_sft n_transformed_bin"]:
        doppler_reference, transformed_start_idx = reference_and_start_idx
        corrected_power = doppler_corrected_power(
            sft_time_series,
            floor=floor,
            velocity_per_detector=velocity_per_detector,
            tsft=tsft,
            alpha=alpha,
            delta=delta,
            doppler_reference=doppler_reference,
        )
        corrected_power = jax.lax.dynamic_slice_in_dim(
            corrected_power, transformed_start_idx, n_transformed_bins, axis=-1
        )
        expected_power = doppler_corrected_mean_power(
            mean_power,
            velocity_per_detector=velocity_per_detector,
            tsft=tsft,
            alpha=alpha,
            delta=delta,
            doppler_reference=doppler_reference,
            start_idx=transformed_start_idx,
            n_bins=n_transformed_bins,
        )
        # the expected power is subtracted so the weights do not spread the floor and
        # steady lines from j = 0 into the harmonics matched
        return sft_weights[:, None] * (corrected_power - expected_power)

    # one reference's transforms in memory at a time
    return jax.lax.map(at_reference, (doppler_references, transformed_start_idxs))


@partial(jax.jit, static_argnames="j_max")
def data_harmonics_per_sky_point(
    weighted_excess_power: Float[Array, "n_sky n_doppler_reference n_sft n_freq"],
    orbital_phase: Float[Array, "n_sky n_period n_sft"],
    *,
    j_max: int,
) -> Complex[Array, "n_sky n_doppler_reference n_period n_harmonic n_freq"]:
    """data_harmonics() of the weighted excess power at each sky point and reference.

    Args:
        weighted_excess_power: weighted_excess_power_per_reference() at each sky
            point.
        orbital_phase: barycenter_orbital_phase() at each sky point, SFT and trial
            period of a batch [rad].
        j_max: Highest harmonic to keep.

    Returns:
        `F[j, k]` at each sky point, reference and trial period.
    """
    per_reference = jax.vmap(partial(data_harmonics, j_max=j_max), in_axes=(0, None))
    return jax.vmap(per_reference)(weighted_excess_power, orbital_phase)
