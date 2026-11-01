"""Template harmonics tau[j, k], and how many harmonics and phases a template needs."""

import math

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from jpec.typing import Scalar
from jpec.waveform.plane import template_plane
from jpec.waveform.window import Window


def harmonic_count(modulation_depth_bins: float) -> int:
    """The highest orbital harmonic jmax that carries signal.

    Args:
        modulation_depth_bins: Half-width of the arc [bins].

    Returns:
        Highest harmonic carrying signal, at least 1.
    """
    return max(math.ceil(2.0 * jnp.pi * modulation_depth_bins), 1)


def phase_sample_count(j_max: int, *, oversample: int) -> int:
    """How many orbital phases `theta` to sample to resolve `j_max` harmonics.

    Nyquist needs more than `2*j_max`, but the cutoff is soft, so bare Nyquist
    aliases. Rounding up to a power of two is what the FFT wants.

    Args:
        j_max: Highest harmonic to be kept.
        oversample: Samples per Nyquist pair.

    Returns:
        Number of phase samples.
    """
    return 1 << math.ceil(math.log2(oversample * j_max))


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
        modulation_depth_bins: Half-width of the arc [bins].
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
