"""Matched filter: orbital phase, data harmonics, statistic over orbital phase."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int
from numpy.typing import ArrayLike


def orbital_phase_mod_2pi(
    times: ArrayLike, omega: ArrayLike
) -> Float[np.ndarray, "... n_sft"]:
    """Orbital phase `omega t mod 2 pi` at each SFT, formed in float64.

    Args:
        times: SFT midpoints from the epoch `phi_b` is quoted at [s].
        omega: Orbital angular frequency [rad/s]; an array of them, one per trial
            period, gives one row each.

    Returns:
        The phase in `[0, 2 pi)` [rad].
    """
    phase = np.asarray(omega, dtype=np.float64)[..., None] * np.asarray(
        times, dtype=np.float64
    )
    return np.mod(phase, 2.0 * np.pi)


def barycenter_orbital_phase(
    orbital_phase: Float[Array, "n_period n_sft"],
    omega: Float[Array, " n_period"],
    solar_roemer_delays: Float[Array, "n_sky n_sft"],
) -> Float[Array, "n_sky n_period n_sft"]:
    """The orbital phase in barycenter time at each sky point.

        x_n = omega t_n + omega delay_n

    with `omega t_n` the phase in detector time and `delay_n` the sky point's Solar
    System Roemer delay at the n-th SFT.

    Args:
        orbital_phase: orbital_phase_mod_2pi() at the SFT midpoints, detector time
            [rad].
        omega: Orbital angular frequency of each trial period [rad/s].
        solar_roemer_delays: The delay at each sky point and SFT [s].

    Returns:
        The phase at each sky point, trial period and SFT [rad].
    """
    return orbital_phase[None] + omega[None, :, None] * solar_roemer_delays[:, None, :]


def data_harmonics(
    sft_power: Float[Array, "n_sft n_freq"],
    orbital_phase: Float[Array, "... n_sft"],
    *,
    j_max: int,
) -> Complex[Array, "... n_harmonic n_freq"]:
    """The SFT power expressed in the harmonic basis on orbital phase.

        F[j, k] = <exp(-i j x_n) sft_power[n, k]>_n,  j = 1..j_max

    Args:
        sft_power: Normalized power, SFT n by frequency bin k.
        orbital_phase: `x_n`, orbital_phase_mod_2pi() at the SFT midpoints [rad]; a
            leading axis of trial periods gives one `F[j, k]` each.
        j_max: Highest harmonic to keep.

    Returns:
        `F[j, k]` for each phase given.
    """
    j = jnp.arange(1, j_max + 1, dtype=orbital_phase.dtype)
    harmonic_phase = j[:, None] * orbital_phase[..., None, :]
    real = jnp.cos(harmonic_phase) @ sft_power
    imag = -jnp.sin(harmonic_phase) @ sft_power
    return jax.lax.complex(real, imag) / sft_power.shape[0]


def harmonic_overlap(
    data_harmonics: Complex[Array, "n_harmonic n_freq"],
    template_harmonics: Complex[Array, "n_harmonic n_freq"],
) -> Complex[Array, " n_harmonic"]:
    """Overlap of data and template at each harmonic, summed over frequency.

    Args:
        data_harmonics: `F[j, k]`, starting with 1st harmonic.
        template_harmonics: `tau[j, k]` on the same `[j, k]` grid.

    Returns:
        `c[j] = sum_k conj(tau[j, k]) * F[j, k] / ||tau||`.
    """
    if data_harmonics.shape != template_harmonics.shape:
        raise ValueError(
            f"shape mismatch: data_harmonics {data_harmonics.shape}, "
            f"template_harmonics {template_harmonics.shape}"
        )
    norm = jnp.sqrt(jnp.sum(jnp.abs(template_harmonics) ** 2))
    # einsum, so a vmap over templates or periods lowers to one matmul
    return jnp.einsum("jk,jk->j", jnp.conj(template_harmonics), data_harmonics) / norm


def statistic_over_phase(
    coefficients: Complex[Array, " n_harmonic"], *, n_phi: int
) -> Float[Array, " n_phi"]:
    """The statistic at every orbital phase at once.

        S(phi_b) = Re{ sum_j c[j] exp(+i j phi_b) }

    which is exactly the real inverse FFT with `phi_b = 2 pi m / n_phi`, since for
    `x[0] = 0` and `x[j] = 0` from `j = n_phi/2` up

        irfft(x)[m] = (2 / n_phi) Re{ sum_j x[j] exp(+2 pi i j m / n_phi) }

    and multiplying by `n_phi/2` cancels its 2 / n_phi. Every phase therefore costs
    one transform rather than one template each.

    Args:
        coefficients: `c[j]` starting with 1st harmonic.
        n_phi: Orbital phases to evaluate, more than `2*len(coefficients)`.

    Returns:
        Statistic at `phi_b = 2 pi m / n_phi`, `m = 0..n_phi-1`.
    """
    # a zero for the dropped j=0 slot, so harmonic j lands on index j
    padded = jnp.concatenate([jnp.zeros((1,), coefficients.dtype), coefficients])
    return jnp.fft.irfft(padded, n=n_phi) * (n_phi / 2)


def max_over_phase(
    data_harmonics: Complex[Array, "n_harmonic n_freq"],
    template_harmonics: Complex[Array, "n_harmonic n_freq"],
    *,
    n_phi: int,
) -> tuple[Float[Array, ""], Int[Array, ""]]:
    """One template's statistic maximized over the orbital phase, and the phase index.

    Args:
        data_harmonics: `F[j, k]`, starting with 1st harmonic.
        template_harmonics: `tau[j, k]` on the same grid.
        n_phi: Orbital phases to evaluate, more than twice the harmonics kept.

    Returns:
        The maximum of `S(phi_b)` and the index `m` of `phi_b = 2 pi m / n_phi` it
        sits at.
    """
    coefficients = harmonic_overlap(
        data_harmonics=data_harmonics, template_harmonics=template_harmonics
    )
    statistic = statistic_over_phase(coefficients, n_phi=n_phi)
    phase_idx = jnp.argmax(statistic)
    return statistic[phase_idx], phase_idx
