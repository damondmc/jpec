"""The template plane: a track's amplitude spread over frequency bins at each SFT."""

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from jpec.typing import Scalar
from jpec.waveform.window import Window


def arc_bin_offsets(
    modulation_depth_bins: float, *, pad_bins: float
) -> Float[Array, " n_freq"]:
    """Bin offsets from f0 wide enough to hold the arc and its leakage.

    Args:
        modulation_depth_bins: Half-width of the frequency arc [bins].
        pad_bins: Extra bins beyond the arc on each side.

    Returns:
        Integer bin offsets from f0, ascending.
    """
    half = jnp.ceil(modulation_depth_bins) + pad_bins
    return jnp.arange(-half, half + 1.0)  # [-half, ..., half]


def plane_from_arc(
    arc: Float[Array, " n_sft"],
    bin_offsets: Float[Array, " n_freq"],
    *,
    window: Window,
) -> Float[Array, "n_sft n_freq"]:
    """Power in each frequency bin at each SFT, given where the arc sits.

    Args:
        arc: Signal offset from f0 at each SFT [bins].
        bin_offsets: Frequency bin centres, as offsets from f0 [bins].
        window: The window the SFTs were made with.

    Returns:
        Power in each frequency bin at each SFT, in `arc.dtype`.
    """
    return window.power(bin_offsets.astype(arc.dtype)[None, :] - arc[:, None])


def amplitude_from_arc(
    arc: Float[Array, " n_sft"],
    bin_offsets: Float[Array, " n_freq"],
    *,
    window: Window,
) -> Complex[Array, "n_sft n_freq"]:
    """Complex amplitude in each bin at each SFT for a unit sinusoid at the arc.

    Args:
        arc: Signal position at each SFT [bins], from the origin of `bin_offsets`.
        bin_offsets: Frequency bin centres [bins], from f0 for a template or from
            the band's first bin for an injection.
        window: The window the SFTs were made with.

    Returns:
        Amplitude in each frequency bin at each SFT, 1 on a frequency bin the
        sinusoid sits on.
    """
    offset = bin_offsets.astype(arc.dtype)[None, :] - arc[:, None]
    response = window.amplitude(offset)
    return response * jnp.exp(-1j * jnp.pi * offset)


def template_plane(
    bin_offsets: Float[Array, " n_freq"],
    *,
    theta: Float[Array, " n_theta"],
    modulation_depth_bins: Scalar,
    window: Window,
) -> Float[Array, "n_theta n_freq"]:
    """The plane drawn against orbital phase.

    Args:
        bin_offsets: Frequency bin centres, as offsets from f0 [bins].
        theta: Orbital phases the plane is sampled at, `omega t - phi_b` [rad].
        modulation_depth_bins: Half-width of the frequency arc [bins].
        window: The window the SFTs were made with.

    Returns:
        Power in each bin at each orbital phase.
    """
    arc = -modulation_depth_bins * jnp.cos(theta)
    return plane_from_arc(arc, bin_offsets, window=window)
