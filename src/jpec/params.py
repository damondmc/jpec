"""One template's parameters, the box over them, and the conversions between them."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jpec.typing import Scalar


class Parameters(NamedTuple):
    """One point of the search space.

    A pytree, so it passes through `jax` transformations, and a tuple, so
    `jnp.asarray(parameters)` is the vector in field order.

    Attributes:
        f0: Signal frequency at the Solar System Barycenter [Hz].
        ap: Projected semi-major axis [light-seconds].
        period: Orbital period [s].
        phi_b: Orbital phase at the reference epoch [rad].
        alpha: Right ascension [rad].
        delta: Declination [rad].
    """

    f0: Scalar
    ap: Scalar
    period: Scalar
    phi_b: Scalar
    alpha: Scalar
    delta: Scalar

    @property
    def omega(self) -> Float[Array, ""]:
        """Orbital angular frequency `2*pi/P` [rad/s]."""
        return omega_from_period(jnp.asarray(self.period))

    @property
    def modulation_depth(self) -> Scalar:
        """Half-width of the frequency arc [Hz]."""
        return self.f0 * self.ap * self.omega

    def modulation_depth_bins(self, tsft: float) -> Scalar:
        """Half-width of the frequency arc [bins].

        Args:
            tsft: SFT length [s].

        Returns:
            The modulation depth in frequency bins.
        """
        return self.modulation_depth * tsft


def omega_from_period(period: Float[Array, "..."]) -> Float[Array, "..."]:
    """Orbital angular frequency `2*pi/P` [rad/s].

    Args:
        period: Orbital period [s].

    Returns:
        `omega`, elementwise.
    """
    return 2.0 * jnp.pi / period


def ap_from_modulation_depth(
    modulation_depth: Float[np.ndarray, "..."],
    *,
    f0: float,
    omega: Float[np.ndarray, "..."],
) -> Float[np.ndarray, "..."]:
    """The projected semi-major axis a modulation depth implies [light-seconds].

    Args:
        modulation_depth: Half-width of the frequency arc [Hz].
        f0: Signal frequency [Hz].
        omega: Orbital angular frequency [rad/s].

    Returns:
        `ap`, elementwise over whatever shapes broadcast.
    """
    return modulation_depth / (f0 * omega)


class Prior(NamedTuple):
    """The region searched: a closed `(low, high)` interval per parameter.

    Attributes:
        f0: Signal frequency [Hz].
        ap: Projected semi-major axis [light-seconds].
        period: Orbital period [s].
        phi_b: Orbital phase at the reference epoch [rad].
        alpha: Right ascension [rad].
        delta: Declination [rad].
    """

    f0: tuple[float, float]
    ap: tuple[float, float]
    period: tuple[float, float]
    phi_b: tuple[float, float]
    alpha: tuple[float, float]
    delta: tuple[float, float]
