"""The metric of the statistic: physical, projected, orbital and on the sky."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float

from jpec.detector.doppler import line_of_sight_velocity, solar_roemer_delay
from jpec.detector.site import Observation
from jpec.params import Parameters
from jpec.waveform.track import observed_frequency


def physical_metric(
    point: Parameters,
    observation: Observation,
    *,
    g_freq_bin: float,
) -> Float[np.ndarray, "6 6"]:
    """Mismatch metric on the physical parameters at one point, closed form.

        g_mu_nu = g_freq_bin tsft^2 <d_mu f(t_n) d_nu f(t_n)>_n

    with `f(t_n)` the frequency track at the n-th SFT midpoint, its orbit in
    barycenter time. The derivatives are taken by JAX, which must run in float64.

    Args:
        point: Point to evaluate at.
        observation: The run's SFT length, midpoints and detector motion.
        g_freq_bin: Mismatch per squared frequency bin of track error, for the
            window the SFTs were made with [1 / bins^2].

    Returns:
        `g[mu, nu]` in the field order of Parameters.

    Raises:
        RuntimeError: If JAX runs in float32.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "physical_metric needs JAX in float64; "
            'set jax.config.update("jax_enable_x64", True)'
        )
    flat, unravel = ravel_pytree(
        Parameters(*(jnp.asarray(x, dtype=float) for x in point))
    )

    times = jnp.asarray(observation.times)
    velocity = jnp.asarray(observation.velocity)
    position = jnp.asarray(observation.position)

    def track(vector: Float[Array, " 6"]) -> Float[Array, " n_sft"]:
        p = unravel(vector)
        alpha, delta = p.alpha[None], p.delta[None]
        delay = solar_roemer_delay(position, alpha=alpha, delta=delta)[0]
        return observed_frequency(
            times + delay,
            f0=p.f0,
            ap=p.ap,
            omega=p.omega,
            phi_b=p.phi_b,
            doppler=line_of_sight_velocity(velocity, alpha=alpha, delta=delta)[0],
        )

    jacobian = np.asarray(jax.jacfwd(track)(flat)) * observation.tsft
    return g_freq_bin * (jacobian.T @ jacobian) / jacobian.shape[0]


def project_out(
    metric: Float[np.ndarray, "6 6"], name: str = "phi_b"
) -> Float[np.ndarray, "5 5"]:
    """Metric on the remaining parameters when one is maximized over.

        g'_mu_nu = g_mu_nu - g_mu_x g_xx^-1 g_x_nu

    Args:
        metric: `g_mu_nu` in the field order of Parameters.
        name: The field maximized over, `phi_b` by default.

    Returns:
        Metric on the other five fields, in their original order.
    """
    metric = np.asarray(metric)
    projected_idx = Parameters._fields.index(name)
    kept_idxs = [i for i in range(len(Parameters._fields)) if i != projected_idx]
    g_keep = metric[np.ix_(kept_idxs, kept_idxs)]
    g_cross = metric[kept_idxs, projected_idx]
    return g_keep - np.outer(g_cross, g_cross) / metric[projected_idx, projected_idx]


def orbital_metric(
    metric: Float[np.ndarray, "5 5"], point: Parameters
) -> Float[np.ndarray, "3 3"]:
    """The physical metric moved onto the orbital lattice's axes.

    Args:
        metric: physical_metric() with `phi_b` projected out, at `point`.
        point: Where the metric was taken.

    Returns:
        `g` on `(f0, modulation_depth, omega^2)`, the depth in Hz.
    """
    f0, ap, period = float(point.f0), float(point.ap), float(point.period)
    modulation_depth = float(point.modulation_depth)
    omega2 = float(point.omega) ** 2

    # Jacobian = d(f0, ap, P) / d(f0, modulation_depth, omega^2)
    jacobian = np.array(
        [
            [1.0, 0.0, 0.0],
            [-ap / f0, ap / modulation_depth, -ap / (2.0 * omega2)],
            [0.0, 0.0, -period / (2.0 * omega2)],
        ]
    )
    return jacobian.T @ np.asarray(metric)[:3, :3] @ jacobian


def sky_metric(
    velocity: Float[np.ndarray, "n_sft 3"], *, f0: float, g_f0f0: float
) -> tuple[Float[np.ndarray, " 3"], Float[np.ndarray, "3 3"]]:
    """The sky metric in the frame that diagonalizes it.

        g_sky = g_f0f0 f0^2 Cov(v)

    Args:
        velocity: Detector velocity `v/c` at each SFT, every detector.
        f0: Frequency the metric is for [Hz].
        g_f0f0: The f0 metric [1/Hz^2].

    Returns:
        The mismatch per unit offset squared along each eigenvector, largest first,
            and the eigenvectors as columns, so that `direction @ eigenvectors`
            gives `(x1, x2, x3)`.
    """
    covariance = np.cov(np.asarray(velocity).T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    return g_f0f0 * f0**2 * eigenvalues, eigenvectors
