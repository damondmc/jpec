"""Detection efficiency against amplitude, and its sigmoid fit."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from jpec.typing import Scalar

# the likelihood is convex in (intercept, slope), so this is ample
NEWTON_STEPS = 64
# keeps the fit finite where every injection above one amplitude was recovered
RIDGE = 1e-6


def detection_efficiency(
    log_h0: Float[Array, " n_h0"], *, intercept: Scalar, slope: Scalar
) -> Float[Array, " n_h0"]:
    """Fraction of injections recovered at each amplitude.

        p(h0) = 1 / (1 + exp(-(intercept + slope * log h0)))

    Args:
        log_h0: Natural log of the strain amplitudes.
        intercept: Log odds of recovery at `h0 = 1`.
        slope: Steepness in log amplitude.

    Returns:
        Recovered fraction at each amplitude, in `(0, 1)`.
    """
    return jax.nn.sigmoid(intercept + slope * log_h0)


def _penalized_negative_log_likelihood(
    theta: Float[Array, " 2"],
    log_h0: Float[Array, " n_injection"],
    recovered: Bool[Array, " n_injection"],
) -> Scalar:
    """Bernoulli negative log-likelihood with a weak ridge, convex in `theta`."""
    z = theta[0] + theta[1] * log_h0
    likelihood = -jnp.sum(
        jnp.where(recovered, jax.nn.log_sigmoid(z), jax.nn.log_sigmoid(-z))
    )
    return likelihood + RIDGE * jnp.sum(theta**2)


class EfficiencyFit(NamedTuple):
    """The efficiency curve fitted to a set of injections.

    Attributes:
        intercept: Log odds of recovery at `h0 = 1`.
        slope: Steepness in log amplitude.
        covariance: Covariance of `(intercept, slope)` from the likelihood's
            curvature.
    """

    intercept: Scalar
    slope: Scalar
    covariance: Float[Array, "2 2"]

    def log_h0_at(self, efficiency: float) -> tuple[Scalar, Scalar]:
        """The log amplitude recovered a given fraction of the time, and its error.

            log h0 = (logit(efficiency) - intercept) / slope

        Args:
            efficiency: Recovered fraction to solve for, in `(0, 1)`.

        Returns:
            The log amplitude and its standard error, the covariance carried through by
            the derivative of that expression rather than by hand.
        """

        def quantile(theta: Float[Array, " 2"]) -> Scalar:
            return (jnp.log(efficiency / (1.0 - efficiency)) - theta[0]) / theta[1]

        theta = jnp.stack([self.intercept, self.slope])
        gradient = jax.grad(quantile)(theta)
        return quantile(theta), jnp.sqrt(gradient @ self.covariance @ gradient)

    def h0_at(self, efficiency: float) -> tuple[Scalar, Scalar]:
        """The amplitude recovered a given fraction of the time, and its error.

        Args:
            efficiency: Recovered fraction to solve for, in `(0, 1)`.

        Returns:
            `h0` and its standard error, which is `h0` times the log amplitude's
            because the fit is in the log.
        """
        log_h0, log_error = self.log_h0_at(efficiency)
        h0 = jnp.exp(log_h0)
        return h0, h0 * log_error


@jax.jit
def _maximize(
    log_h0: Float[Array, " n_injection"],
    recovered: Bool[Array, " n_injection"],
) -> tuple[Float[Array, " 2"], Float[Array, "2 2"]]:
    """Newton on the penalized likelihood, and its curvature at the maximum."""

    def step(_: int, theta: Float[Array, " 2"]) -> Float[Array, " 2"]:
        gradient = jax.grad(_penalized_negative_log_likelihood)(
            theta, log_h0, recovered
        )
        curvature = jax.hessian(_penalized_negative_log_likelihood)(
            theta, log_h0, recovered
        )
        return theta - jnp.linalg.solve(curvature, gradient)

    theta = jax.lax.fori_loop(0, NEWTON_STEPS, step, jnp.zeros(2, log_h0.dtype))
    curvature = jax.hessian(_penalized_negative_log_likelihood)(
        theta, log_h0, recovered
    )
    return theta, jnp.linalg.inv(curvature)


def fit_efficiency(
    h0: Float[np.ndarray, " n_injection"],
    recovered: Bool[np.ndarray, " n_injection"],
) -> EfficiencyFit:
    """The efficiency curve the recoveries imply, by maximum likelihood.

    Args:
        h0: Strain amplitude each injection was made at.
        recovered: Whether each injection was recovered.

    Returns:
        The fit, its covariance from the likelihood's curvature at the maximum.
    """
    theta, covariance = _maximize(jnp.log(jnp.asarray(h0)), jnp.asarray(recovered))
    return EfficiencyFit(intercept=theta[0], slope=theta[1], covariance=covariance)
