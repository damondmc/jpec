"""The injection population: sources drawn over the box and over their orientation."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from jpec.injection.signal import Injection, Orientation
from jpec.params import Parameters, Prior


def draw_injections(
    prior: Prior,
    *,
    cos_iota: tuple[float, float],
    psi: tuple[float, float],
    key: PRNGKeyArray,
    n_injections: int,
) -> list[Injection]:
    """Sources drawn uniform in every parameter, the sky uniform on the sphere.

    Args:
        prior: The box drawn from, its period in seconds and its angles in radians.
        cos_iota: Interval of the cosine of the inclination.
        psi: Interval of the polarization angle [rad].
        key: PRNG key.
        n_injections: Number of sources drawn.

    Returns:
        The sources, in the order they were drawn.
    """
    sin_delta = (float(np.sin(prior.delta[0])), float(np.sin(prior.delta[1])))
    intervals = (
        prior.f0,
        prior.ap,
        prior.period,
        prior.phi_b,
        prior.alpha,
        sin_delta,
        cos_iota,
        psi,
    )
    low, high = (jnp.array(edge) for edge in zip(*intervals))
    uniform = jax.random.uniform(
        key, (n_injections, len(intervals)), minval=low, maxval=high
    )
    # one copy to the host, so every field is a float and not a 0-d device array
    draws = np.asarray(uniform, dtype=np.float64)
    return [
        Injection(
            parameters=Parameters(
                f0=float(f0),
                ap=float(ap),
                period=float(period),
                phi_b=float(phi_b),
                alpha=float(alpha),
                delta=float(np.arcsin(sin_delta_drawn)),
            ),
            orientation=Orientation(
                cos_iota=float(cos_iota_drawn), psi=float(psi_drawn)
            ),
        )
        for (
            f0,
            ap,
            period,
            phi_b,
            alpha,
            sin_delta_drawn,
            cos_iota_drawn,
            psi_drawn,
        ) in draws
    ]
