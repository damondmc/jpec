"""Template harmonics tau[j, k], and how many harmonics and phases a template needs."""

import math

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from jpec.typing import Scalar
from jpec.waveform.plane import template_plane
from jpec.waveform.window import Window


def harmonic_count(modulation_depth_bins: float) -> int:
    """The highest orbital harmonic `j_max` that carries signal.

    Args:
        modulation_depth_bins: Half-width of the frequency arc [bins].

    Returns:
        Highest harmonic carrying signal, at least 1.
    """
    return max(math.ceil(2.0 * jnp.pi * modulation_depth_bins), 1)


def theta_sample_count(j_max: int, *, oversample: int) -> int:
    """How many orbital phases `theta` to sample a template at for `j_max` harmonics.

    Nyquist needs more than `2*j_max`, but the cutoff is soft, so bare Nyquist
    aliases. Rounding up to a power of two is what the FFT wants.

    Args:
        j_max: Highest harmonic to be kept.
        oversample: Samples per Nyquist pair.

    Returns:
        Number of phase samples.
    """
    return 1 << math.ceil(math.log2(oversample * j_max))


def phase_count(
    harmonic_power: Float[np.ndarray, " n_harmonic"], *, phase_mismatch: float
) -> int:
    """Orbital phases the statistic is read at, to hold the phase mismatch.

    A template read `delta` from the signal's phase loses

        mu_phi = 1 - sum_j P[j] cos(j delta) / sum_j P[j]
        ~  j_rms^2 delta^2 / 2, j_rms^2 = sum_j j^2 P[j] / sum_j P[j]

    and a grid of n points leaves `delta` at most `pi / n`, so

        n >= pi j_rms / sqrt(2 mu_phi),  and n >= 2 n_harmonic + 2

    for the inverse transform to hold every harmonic. It is rounded up to the
    smallest even 2^a 3^b 5^c 7^d, or to the next power of 2 when that size is above
    1024 and above 0.8 of the power of 2, the sizes cuFFT is optimized for.

    Args:
        harmonic_power: `P[j] = sum_k |tau[j, k]|^2` of the template, from harmonic 1.
        phase_mismatch: Largest mismatch the phase grid may leave, `mu_phi`.

    Returns:
        Number of orbital phases.
    """
    j = np.arange(1, harmonic_power.size + 1)
    j_rms = math.sqrt(float((j**2 * harmonic_power).sum() / harmonic_power.sum()))
    # the phases the mismatch asks for, and the inverse transform's floor
    n_phi_needed = max(
        math.ceil(math.pi * j_rms / math.sqrt(2.0 * phase_mismatch)),
        2 * harmonic_power.size + 2,
    )
    # the smallest even product of 2, 3, 5 and 7 not below it; a power of 2 always
    # reaches it
    n_exponents = n_phi_needed.bit_length() + 1
    sizes = (
        2**twos * 3**threes * 5**fives * 7**sevens
        for twos in range(1, n_exponents)
        for threes in range(n_exponents)
        for fives in range(n_exponents)
        for sevens in range(n_exponents)
    )
    n_phi = min(size for size in sizes if size >= n_phi_needed)
    next_power_2 = 1 << (n_phi_needed - 1).bit_length()
    if n_phi > 1024 and n_phi > 0.8 * next_power_2:
        return next_power_2
    return n_phi


def template_harmonics(
    bin_offsets: Float[Array, " n_freq"],
    *,
    modulation_depth_bins: Scalar,
    n_theta: int,
    j_max: int,
    window: Window,
) -> Complex[Array, "n_harmonic n_freq"]:
    """`tau[j, k]`, the template the search matches against.

    The phase grid and harmonic count are the caller's, so the templates of a
    bank stack into one array and the modulation depth may be traced.

    Args:
        bin_offsets: Bin centres, as offsets from f0 [bins].
        modulation_depth_bins: Half-width of the frequency arc [bins].
        n_theta: Orbital phases sampled, uniform over `[0, 2*pi)`; more than
            `2*j_max`.
        j_max: Highest harmonic kept.
        window: The window the SFTs were made with.

    Returns:
        `tau[j, k]` from harmonic 1 to `j_max`.
    """
    theta = jnp.arange(n_theta) * (2.0 * jnp.pi / n_theta)
    plane = template_plane(
        bin_offsets,
        theta=theta,
        modulation_depth_bins=modulation_depth_bins,
        window=window,
    )
    plane_harmonics = jnp.fft.rfft(plane, axis=0) / n_theta
    # j = 0 dropped, the mean power in each bin, which knows nothing about
    # the orbit and is where instrumental lines sit
    return plane_harmonics[1 : j_max + 1]
