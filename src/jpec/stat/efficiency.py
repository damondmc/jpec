"""The detection efficiency against amplitude: its sigmoid fit, and the recovery curve that refines it."""

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
        intercept: Log odds of recovery at ``h0 = 1``.
        slope: Steepness in log amplitude.

    Returns:
        Recovered fraction at each amplitude, in ``(0, 1)``.
    """
    return jax.nn.sigmoid(intercept + slope * log_h0)


def _penalized_negative_log_likelihood(
    theta: Float[Array, " 2"],
    log_h0: Float[Array, " n_h0"],
    n_recovered: Float[Array, " n_h0"],
    n_injections: float,
) -> Scalar:
    """Binomial negative log-likelihood with a weak ridge, convex in ``theta``."""
    z = theta[0] + theta[1] * log_h0
    likelihood = -jnp.sum(
        n_recovered * jax.nn.log_sigmoid(z)
        + (n_injections - n_recovered) * jax.nn.log_sigmoid(-z)
    )
    return likelihood + RIDGE * jnp.sum(theta**2)


class EfficiencyFit(NamedTuple):
    """The efficiency curve fitted to a set of injections.

    Args:
        intercept: Log odds of recovery at ``h0 = 1``.
        slope: Steepness in log amplitude.
        covariance: Covariance of ``(intercept, slope)`` from the likelihood's curvature.
    """

    intercept: Scalar
    slope: Scalar
    covariance: Float[Array, "2 2"]

    def log_h0_at(self, efficiency: float) -> tuple[Scalar, Scalar]:
        """The log amplitude recovered a given fraction of the time, and its error.

            log h0 = (logit(efficiency) - intercept) / slope

        Args:
            efficiency: Recovered fraction to solve for, in ``(0, 1)``.

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
            efficiency: Recovered fraction to solve for, in ``(0, 1)``.

        Returns:
            ``h0`` and its standard error, which is ``h0`` times the log amplitude's
            because the fit is in the log.
        """
        log_h0, log_error = self.log_h0_at(efficiency)
        h0 = jnp.exp(log_h0)
        return h0, h0 * log_error


@jax.jit
def _maximize(
    log_h0: Float[Array, " n_h0"],
    n_recovered: Float[Array, " n_h0"],
    n_injections: float,
    start: Float[Array, " 2"],
) -> tuple[Float[Array, " 2"], Float[Array, "2 2"]]:
    """Newton on the penalized likelihood, and its curvature at the maximum."""

    def step(_: int, theta: Float[Array, " 2"]) -> Float[Array, " 2"]:
        gradient = jax.grad(_penalized_negative_log_likelihood)(
            theta, log_h0, n_recovered, n_injections
        )
        curvature = jax.hessian(_penalized_negative_log_likelihood)(
            theta, log_h0, n_recovered, n_injections
        )
        return theta - jnp.linalg.solve(curvature, gradient)

    theta = jax.lax.fori_loop(0, NEWTON_STEPS, step, start)
    curvature = jax.hessian(_penalized_negative_log_likelihood)(
        theta, log_h0, n_recovered, n_injections
    )
    return theta, jnp.linalg.inv(curvature)


def fit_efficiency(
    h0: Float[np.ndarray, " n_h0"],
    recovered: Bool[np.ndarray, "n_draw n_h0"],
) -> EfficiencyFit:
    """The efficiency curve the recoveries imply, by maximum likelihood.

    Args:
        h0: Strain amplitudes injected.
        recovered: Whether each injection was recovered at each amplitude.

    Returns:
        The fit, its covariance from the likelihood's curvature at the maximum.
    """
    log_h0 = jnp.log(jnp.asarray(h0))
    n_recovered = jnp.sum(jnp.asarray(recovered), axis=0).astype(float)
    n_injections = float(recovered.shape[0])

    # a straight line through the two ends of the measured log odds, which Newton then sharpens
    fraction = jnp.clip(
        n_recovered / n_injections, 0.5 / n_injections, 1.0 - 0.5 / n_injections
    )
    odds = jnp.log(fraction / (1.0 - fraction))
    slope = (odds[-1] - odds[0]) / (log_h0[-1] - log_h0[0])
    start = jnp.stack([odds[0] - slope * log_h0[0], slope])

    theta, covariance = _maximize(log_h0, n_recovered, n_injections, start)
    return EfficiencyFit(intercept=theta[0], slope=theta[1], covariance=covariance)


class RecoveryCurve:
    """The fraction of sources recovered against the amplitude they were injected at.

    Args:
        efficiency: The recovered fraction the quoted amplitude is taken at.
    """

    def __init__(self, *, efficiency: float) -> None:
        """Start the curve with no amplitude tried."""
        self.efficiency = efficiency
        self.h0_values: Float[np.ndarray, " n_h0"] = np.empty(0)
        self.recovered: Bool[np.ndarray, "n_source n_h0"] = np.empty((0, 0), dtype=bool)

    def add(self, h0: float, recovered: Bool[np.ndarray, " n_source"]) -> None:
        """Record whether each source was recovered at one amplitude, keeping the h0 values sorted.

        Args:
            h0: Strain amplitude the sources were injected at.
            recovered: Whether each source's loudest statistic beat the threshold.
        """
        if self.h0_values.size == 0:
            self.recovered = np.empty((recovered.size, 0), dtype=bool)
        insert_idx = int(np.searchsorted(self.h0_values, h0))
        self.h0_values = np.insert(self.h0_values, insert_idx, h0)
        self.recovered = np.insert(self.recovered, insert_idx, recovered, axis=1)

    @property
    def fraction(self) -> Float[np.ndarray, " n_h0"]:
        """The fraction of sources recovered at each amplitude."""
        return self.recovered.mean(axis=0)

    def fit(self) -> tuple[float, float]:
        """The amplitude recovered the wanted fraction of the time, and its standard error.

        Returns:
            ``h0`` at the efficiency and its error, both NaN while the amplitudes tried do not
            bracket the efficiency.
        """
        fraction = self.fraction
        if fraction.max() < self.efficiency or fraction.min() > self.efficiency:
            return float("nan"), float("nan")
        fit = fit_efficiency(self.h0_values, self.recovered)
        h0_at_efficiency, h0_error = fit.h0_at(self.efficiency)
        return float(h0_at_efficiency), float(h0_error)

    def next_h0(self, h0_at_efficiency: float) -> float:
        """The one amplitude to add next: the fit itself, or a step out to bracket it.

        Args:
            h0_at_efficiency: The fit from `fit`, NaN while unbracketed.

        Returns:
            Twice the loudest amplitude tried while every one falls short of the efficiency,
            half the weakest while every one exceeds it, and the fit otherwise.
        """
        fraction = self.fraction
        if fraction.max() < self.efficiency:
            return 2.0 * float(self.h0_values[-1])
        if fraction.min() > self.efficiency:
            return 0.5 * float(self.h0_values[0])
        return h0_at_efficiency
