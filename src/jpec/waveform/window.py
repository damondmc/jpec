"""SFT windows and their spectra: rectangular, Tukey and Hann."""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


def rectangular_window(length: int) -> Float[Array, " length"]:
    """No window at all.

    Args:
        length: Number of samples.

    Returns:
        Ones.
    """
    return jnp.ones(length)


def rectangular_spectrum(offset: Float[Array, "..."]) -> Float[Array, "..."]:
    """Fourier transform of `rectangular_window()`.

    Args:
        offset: Distance from the sinusoid to the frequency bin's centre [bins].

    Returns:
        Fourier transform window.
    """
    return jnp.sinc(offset)


def tukey_window(length: int, beta: float) -> Float[Array, " length"]:
    """The Tukey window sampled over ``length`` points.

    With ``d = min(i, length - 1 - i)`` the distance to the nearer end and
    ``T = round(beta * length)`` the taper width in samples::

        Y    = (2d - (T - 1)) / (T - 1)
        w[i] = cos(pi/2 * Y)**2            , 0 <= d < (T + 1) // 2
             = 1                           , otherwise

    Args:
        length: Number of samples.
        beta: Fraction of the window that is tapered.
            beta = 0.0: rectangular, beta = 1.0: Hann.

    Returns:
        Window values
    """
    transition = round(beta * length)
    from_edge = jnp.minimum(jnp.arange(length), length - 1 - jnp.arange(length))
    span = max(transition - 1, 1)
    y = (2.0 * from_edge - (transition - 1)) / span
    return jnp.where(
        from_edge < (transition + 1) // 2, jnp.cos(0.5 * jnp.pi * y) ** 2, 1.0
    )


def tukey_spectrum(offset: Float[Array, "..."], *, beta: float) -> Float[Array, "..."]:
    """Fourier transform of `tukey_window()`, at ``offset`` bins in closed form.

    The transform is given by:
        W(f) = 2a sinc(2a f) + 2 Re[ (1/2) S(f)
                + (1/4) exp(-2 pi i a / beta) S(f - 1/beta)
                + (1/4) exp(+2 pi i a / beta) S(f + 1/beta) ],

        S(u) = L sinc(u L) exp(-2 pi i u m)

    with ``a = (1 - beta)/2`` the flat half-width, ``L = beta/2`` the width of the
    taper at each end, and ``m = 1/2 - L/2`` its midpoint.

    Args:
        offset: Distance from the sinusoid to the frequency bin's centre [bins].
        beta: Fraction of the window that is tapered, beta = 0: rectangular, beta = 1: Hann.

    Returns:
        Fourier transform window.
    """
    flat_halfwidth = 0.5 * (1.0 - beta)
    taper_width = 0.5 * beta
    middle = 0.5 - 0.5 * taper_width
    inv_beta = 1.0 / beta

    def s_integral(u: Float[Array, "..."]) -> Complex[Array, "..."]:
        return (
            taper_width * jnp.sinc(u * taper_width) * jnp.exp(-2j * jnp.pi * u * middle)
        )

    phase = jnp.exp(-2j * jnp.pi * flat_halfwidth * inv_beta)
    tapered = (
        0.5 * s_integral(offset)
        + 0.25 * phase * s_integral(offset - inv_beta)
        + 0.25 * jnp.conj(phase) * s_integral(offset + inv_beta)
    )
    return 2.0 * flat_halfwidth * jnp.sinc(
        2.0 * flat_halfwidth * offset
    ) + 2.0 * jnp.real(tapered)


@dataclass(frozen=True)
class Window:
    """The window an SFT was made with, and the response it gives a frequency bin.

    Args:
        kind: ``"rectangular"``, ``"tukey"`` or ``"hann"``, the Tukey window at beta = 1.
        beta: Fraction of the window that is not flat, in ``(0, 1]``; used only by ``"tukey"``.
    """

    kind: str = "tukey"
    beta: float = 0.001

    def __post_init__(self) -> None:
        """Check the kind is a known one and a Tukey beta lies in (0, 1]."""
        if self.kind not in ("rectangular", "tukey", "hann"):
            raise ValueError(
                f"window must be 'rectangular', 'tukey' or 'hann', got {self.kind!r}"
            )
        if self.kind == "tukey" and not 0.0 < self.beta <= 1.0:
            raise ValueError(
                f"beta must be in (0, 1] for 'tukey' window, got {self.beta}"
            )

    def samples(self, length: int) -> Float[Array, " length"]:
        """The window itself, sampled over ``length`` points.

        Args:
            length: Number of samples.

        Returns:
            Window values.
        """
        if self.kind == "rectangular":
            return rectangular_window(length)
        beta = 1.0 if self.kind == "hann" else self.beta
        return tukey_window(length, beta)

    def amplitude(self, offset: Float[Array, "..."]) -> Float[Array, "..."]:
        """Amplitude a frequency bin receives from a unit sinusoid ``offset`` bins from its centre.

        Real, because the window is symmetric about the SFT's middle; the
        linear phase from that middle is left to the caller.

        Args:
            offset: Distance from the sinusoid to the frequency bin's centre [bins].

        Returns:
            Response, same shape as ``offset``, 1 on the centre.
        """
        if self.kind == "rectangular":
            return rectangular_spectrum(offset)
        beta = 1.0 if self.kind == "hann" else self.beta
        return tukey_spectrum(offset, beta=beta) / (1.0 - 0.5 * beta)

    def power(self, offset: Float[Array, "..."]) -> Float[Array, "..."]:
        """Power a frequency bin receives from a sinusoid ``offset`` bins from its centre.

        Args:
            offset: Distance from the sinusoid to the frequency bin's centre [bins].

        Returns:
            Response, same shape as ``offset``, 1 on the centre.
        """
        return self.amplitude(offset) ** 2
