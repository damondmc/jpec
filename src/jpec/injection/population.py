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
    intervals = {
        "f0": prior.f0,
        "ap": prior.ap,
        "period": prior.period,
        "phi_b": prior.phi_b,
        "alpha": prior.alpha,
        "sin_delta": sin_delta,
        "cos_iota": cos_iota,
        "psi": psi,
    }
    low = jnp.array([interval[0] for interval in intervals.values()])
    high = jnp.array([interval[1] for interval in intervals.values()])
    uniform = jax.random.uniform(
        key, (n_injections, len(intervals)), minval=low, maxval=high
    )
    # one copy to the host, so every field is a float and not a 0-d device array
    draws = np.asarray(uniform, dtype=np.float64)
    injections = []
    for row in draws:
        draw = dict(zip(intervals, row.tolist(), strict=True))
        parameters = Parameters(
            f0=draw["f0"],
            ap=draw["ap"],
            period=draw["period"],
            phi_b=draw["phi_b"],
            alpha=draw["alpha"],
            delta=float(np.arcsin(draw["sin_delta"])),
        )
        orientation = Orientation(cos_iota=draw["cos_iota"], psi=draw["psi"])
        injections.append(Injection(parameters=parameters, orientation=orientation))
    return injections
