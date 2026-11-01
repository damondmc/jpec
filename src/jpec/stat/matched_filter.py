"""Matched filter: orbital basis, data harmonics, statistic over orbital phase."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int


def orbital_basis(
    times: Float[Array, " n_sft"],
    *,
    omega: Float[Array, "..."],
    j_max: int,
) -> Complex[Array, "... n_harmonic n_sft"]:
    """The harmonic basis on orbital phase.

        basis[j, n] = exp(-i j omega t_n),  j = 1..j_max

    The phase is formed here, so under jit it is never held as an array of its own.

    Args:
        times: SFT midpoints from the epoch `phi_b` is quoted at [s].
        omega: Orbital angular frequency [rad/s]; an array of them, one per trial
            period, gives one basis each.
        j_max: Highest harmonic to keep.

    Returns:
        `basis[..., j, n]`, j-th harmonic at the n-th SFT.
    """
    theta = jnp.asarray(omega)[..., None] * times
    j = jnp.arange(1, j_max + 1)
    return jnp.exp(-1j * j[:, None] * theta[..., None, :])


def data_harmonics(
    sft_power: Float[Array, "n_sft n_freq"],
    orbital_basis: Complex[Array, "... n_harmonic n_sft"],
) -> Complex[Array, "... n_harmonic n_freq"]:
    """The SFT power expressed in the orbital basis.

        F[j, k] = <orbital_basis[j, n] sft_power[n, k]>_n

    Args:
        sft_power: Normalized power, SFT n by frequency bin k.
        orbital_basis: orbital_basis() at the SFTs' midpoints.

    Returns:
        `F[j, k]` for each basis given.
    """
    # the power is real, so two real matmuls instead of one complex one
    real = jnp.real(orbital_basis) @ sft_power
    imag = jnp.imag(orbital_basis) @ sft_power
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
