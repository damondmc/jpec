"""The transform: a sky batch's Doppler-corrected band and its data harmonics."""

from functools import partial

import jax
from jaxtyping import Array, Complex, Float, Int

from jpec.detector.doppler import doppler_corrected_mean_power, doppler_corrected_power
from jpec.stat.matched_filter import barycenter_orbital_phase, data_harmonics


@partial(jax.jit, static_argnames="n_transformed_bins")
def weighted_excess_power_per_reference(
    sft_time_series: list[Complex[Array, "n_sft n_band"]],
    floor: list[Float[Array, "n_sft n_band"]],
    velocity_per_detector: list[Float[Array, "n_sft 3"]],
    doppler_references: Float[Array, " n_ref"],
    transformed_start_idxs: Int[Array, " n_ref"],
    *,
    n_transformed_bins: int,
    tsft: float,
    alpha: float,
    delta: float,
    sft_weights: Float[Array, " n_sft"],
    mean_power: list[Float[Array, " n_band"]],
) -> Float[Array, "n_ref n_sft n_transformed_bin"]:
    """The weighted excess power `P[n, k]` of one sky position, per Doppler reference.

        P[n, k] = w[n] (power[n, k] - mean power[n, k])

    with `w[n]` the SFT's weight, and the power and its detector's time-averaged
    spectrum both Doppler-corrected.

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
        The weighted excess power at each Doppler reference, over the bins it keeps.
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


@partial(jax.jit, static_argnames=("period_batch_size", "j_max"))
def data_harmonics_per_sky_point(
    weighted_excess_power: Float[
        Array, "n_sky n_ref n_sft n_transformed_bin"
    ],
    orbital_phase: Float[Array, "n_class n_period n_sft"],
    omega: Float[Array, "n_class n_period"],
    solar_roemer_delays: Float[Array, "n_sky n_sft"],
    omega2_class: Int[Array, ""],
    period_start_idx: Int[Array, ""],
    *,
    period_batch_size: int,
    j_max: int,
) -> Complex[
    Array, "n_sky n_ref n_batch_period n_harmonic n_transformed_bin"
]:
    """`F[j, k]` of a sky batch at a run of trial periods.

        x_n = omega t_n + omega R_n,   F[j, k] = <exp(-i j x_n) P[n, k]>_n

    Args:
        weighted_excess_power: The Doppler-corrected band `P[n, k]` of each sky
            point, per Doppler reference.
        orbital_phase: Orbital phase `omega t_n mod 2 pi` in detector time at each
            trial period and SFT, per omega^2 class [rad].
        omega: Orbital angular frequency of each trial period, per omega^2 class
            [rad/s].
        solar_roemer_delays: Solar System Roemer delay `R_n` at each sky point
            and SFT [s].
        omega2_class: Omega^2 class of the sky batch.
        period_start_idx: First trial period of the run.
        period_batch_size: Number of trial periods in the run.
        j_max: Highest harmonic to keep.

    Returns:
        `F[j, k]` at each sky point, Doppler reference and trial period of the run.
    """
    n_sfts = orbital_phase.shape[-1]
    run_phase = jax.lax.dynamic_slice(
        orbital_phase,
        (omega2_class, period_start_idx, 0),
        (1, period_batch_size, n_sfts),
    )[0]
    run_omega = jax.lax.dynamic_slice(
        omega, (omega2_class, period_start_idx), (1, period_batch_size)
    )[0]
    phase = barycenter_orbital_phase(
        orbital_phase=run_phase,
        omega=run_omega,
        solar_roemer_delays=solar_roemer_delays,
    )
    per_reference = jax.vmap(partial(data_harmonics, j_max=j_max), in_axes=(0, None))
    return jax.vmap(per_reference)(weighted_excess_power, phase)
