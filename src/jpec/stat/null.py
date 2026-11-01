"""Statistic under noise alone: its sigma, false-alarm probability and thresholds."""

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from jpec.typing import Scalar


def null_sigma(n_sfts: int, *, power_variance: float) -> Scalar:
    """Standard deviation of `S(phi_b)` under noise, on floor-normalized power.

        Var(S) = Var(P) / (2 * n_sfts),  P = P_ik / floor_ik

    Gaussian noise divided by its floor is scaled chi2_2 with unit mean, so
    `Var(P) = 1`; real data can be wider, and then takes its fitted value.

    Args:
        n_sfts: SFTs the harmonics were taken over.
        power_variance: `Var(P)` in one frequency bin of one SFT: 1 for Gaussian
            noise, the fitted null width squared on real data.

    Returns:
        Standard deviation of `S(phi_b)`.
    """
    return jnp.sqrt(power_variance / (2.0 * n_sfts))


def rms_harmonic_number(
    template_harmonics: Complex[Array, "n_harmonic n_freq"],
) -> Scalar:
    """How fast `S(phi_b)` wiggles as `phi_b` turns.

        j_rms = sqrt(sum_j j^2 |tau_j|^2 / sum_j |tau_j|^2)

    Args:
        template_harmonics: `tau[j, k]` for harmonics `j=1..j_max`.

    Returns:
        Power-weighted RMS harmonic number.
    """
    power = jnp.sum(jnp.abs(template_harmonics) ** 2, axis=1)
    j = jnp.arange(1, power.shape[0] + 1)
    return jnp.sqrt(jnp.sum(j**2 * power) / jnp.sum(power))


def false_alarm_probability(
    threshold: Scalar,
    *,
    n_sfts: int,
    j_rms: Scalar,
    power_variance: float,
) -> Scalar:
    """Chance noise alone puts the phase maximum of `S` at or above `threshold`.

        FAP = 1 - exp(-j_rms * exp(-threshold^2 / (2 sigma^2)))

    Assuming a Gaussian noise model on floor-normalized power.

    Args:
        threshold: Level on `S` the maximum over phase is compared against.
        n_sfts: SFTs the harmonics were taken over.
        j_rms: Template's RMS harmonic number, from rms_harmonic_number().
        power_variance: `Var(P)` of the floor-normalized power, as in null_sigma().

    Returns:
        False-alarm probability in `[0, 1]`.
    """
    scaled = threshold / null_sigma(n_sfts, power_variance=power_variance)
    return 1.0 - jnp.exp(-j_rms * jnp.exp(-0.5 * scaled**2))


def detection_threshold(
    fap: float,
    *,
    n_sfts: int,
    j_rms: Scalar,
    power_variance: float,
) -> Scalar:
    """The threshold on `S(phi)` needed for the given false-alarm probability.

    Taken on floor-normalized power.

    Args:
        fap: Target false-alarm probability, in `(0, 1)`.
        n_sfts: SFTs the harmonics were taken over.
        j_rms: Template's RMS harmonic number, from rms_harmonic_number().
        power_variance: `Var(P)` of the floor-normalized power, as in null_sigma().

    Returns:
        Threshold on the detection statistic `S(phi)`.
    """
    if not 0.0 < fap < 1.0:
        raise ValueError(f"fap must be in (0, 1), got {fap}")
    crossings = -jnp.log(1.0 - fap) / j_rms
    if crossings >= 1.0:
        raise ValueError(
            f"FAP={fap} is above what j_rms={float(j_rms):.1f} crossings can deliver; "
            "the maximum over phase is below the mean there and the formula "
            "does not apply"
        )
    scaled = jnp.sqrt(-2.0 * jnp.log(crossings))
    return scaled * null_sigma(n_sfts, power_variance=power_variance)


def toplist_threshold(
    n_candidates: float,
    *,
    n_templates: Float[Array, " n_modulation_depth"],
    j_rms: Float[Array, " n_modulation_depth"],
    n_sfts: int,
    power_variance: float,
) -> Scalar:
    """Threshold on `S(phi)` giving `n_candidates` templates on Gaussian noise.

        n_candidates = sum_d n_templates[d] * j_rms[d] * exp(-threshold^2 / (2 sigma^2))

    Taken on floor-normalized power.

    Args:
        n_candidates: Templates noise alone puts above the threshold, on average.
        n_templates: Templates at each modulation depth, over sky positions, f0 and
            periods.
        j_rms: RMS harmonic number at each modulation depth, from rms_harmonic_number().
        n_sfts: SFTs the harmonics were taken over.
        power_variance: `Var(P)` of the floor-normalized power, as in null_sigma().

    Returns:
        Threshold on the detection statistic `S(phi)`.
    """
    n_crossings = jnp.sum(n_templates * j_rms)
    crossings = n_candidates / n_crossings
    if crossings >= 1.0:
        raise ValueError(
            f"{n_candidates} candidates is more than {float(n_crossings):.0f} "
            "independent maxima the templates can deliver; the formula needs a small "
            "false-alarm probability"
        )
    scaled = jnp.sqrt(-2.0 * jnp.log(crossings))
    return scaled * null_sigma(n_sfts, power_variance=power_variance)
