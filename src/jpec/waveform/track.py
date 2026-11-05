"""The orbit's track: orbital phase, binary Roemer delay, and the observed frequency."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jpec.typing import Scalar


def orbital_phase(
    t: Float[Array, " n_time"],
    *,
    omega: Scalar,
    phi_b: Scalar,
) -> Float[Array, " n_time"]:
    """Orbital phase `theta = omega*t - phi_b`, unwrapped.

    Args:
        t: Barycenter times from the epoch of `phi_b` [s].
        omega: Orbital angular frequency [rad/s].
        phi_b: Orbital phase at the epoch of `t` [rad].

    Returns:
        Orbital phase [rad].
    """
    return omega * t - phi_b


def binary_roemer_delay(
    t: Float[Array, " n_time"],
    *,
    ap: Scalar,
    omega: Scalar,
    phi_b: Scalar,
) -> Float[Array, " n_time"]:
    """Binary Roemer delay, `ap * sin(omega*t - phi_b)`.

    Args:
        t: Barycenter times from the epoch of `phi_b` [s].
        ap: Projected semi-major axis [light-seconds].
        omega: Orbital angular frequency [rad/s].
        phi_b: Orbital phase at the epoch of `t` [rad].

    Returns:
        Delay [s], amplitude `ap`.
    """
    return ap * jnp.sin(orbital_phase(t, omega=omega, phi_b=phi_b))


def frequency_offset(
    t: Float[Array, " n_time"],
    *,
    modulation_depth: Scalar,
    omega: Scalar,
    phi_b: Scalar,
) -> Float[Array, " n_time"]:
    """Signal frequency offset f(t) - f0.

        f(t)       = f0 * (1 - ap*omega*cos(omega*t - phi_b))
        f(t) - f0  = -modulation_depth * cos(omega*t - phi_b)

    Args:
        t: Barycenter times from the epoch of `phi_b` [s].
        modulation_depth: Modulation depth [Hz].
        omega: Orbital angular frequency [rad/s].
        phi_b: Orbital phase at the epoch of `t` [rad].

    Returns:
        Frequency offset [Hz], within `+/- modulation_depth`.
    """
    return -modulation_depth * jnp.cos(orbital_phase(t, omega=omega, phi_b=phi_b))


def observed_frequency(
    t: Float[Array, " n_time"],
    *,
    f0: Scalar,
    ap: Scalar,
    omega: Scalar,
    phi_b: Scalar,
    doppler: Float[Array, " n_time"],
) -> Float[Array, " n_time"]:
    """Signal frequency at the detector.

        f(t) = f0 * (1 + doppler) - f0*ap*omega * cos(omega*t - phi_b)

    Args:
        t: Barycenter times from the epoch of `phi_b` [s].
        f0: Signal frequency at the Solar System Barycenter [Hz].
        ap: Projected semi-major axis [light-seconds].
        omega: Orbital angular frequency [rad/s].
        phi_b: Orbital phase at the epoch of `t` [rad].
        doppler: Observer's line-of-sight velocity over c, per time.

    Returns:
        Signal frequency [Hz].
    """
    return f0 * (1.0 + doppler) + frequency_offset(
        t, modulation_depth=f0 * ap * omega, omega=omega, phi_b=phi_b
    )
