"""The metric of the statistic: physical, projected, orbital and on the sky."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float

from jpec.detector.doppler import sky_direction
from jpec.params import Parameters
from jpec.waveform.track import observed_frequency
from jpec.waveform.window import Window


def freq_bin_curvature(window: Window, *, n_positions: int) -> float:
    """Mismatch per squared frequency bin of track error, `g_freq_bin`, for a window.

        1 - <sum_k W(k - x) W(k - x - d)>_x / <sum_k W(k - x)^2>_x  ~  g_freq_bin d^2

    with `d` the track's error in frequency bins and `x` the track's fractional
    position in its bin.

    Args:
        window: The window the SFTs were made with.
        n_positions: Fractional positions averaged over.

    Returns:
        `g_freq_bin` [1 / frequency bins^2].
    """
    # x: the track's position in its bin, at the centres of n_positions equal slices
    # of [0, 1)
    position = (jnp.arange(n_positions) + 0.5) / n_positions
    # k: each bin counted from the bin the track is in
    bin_offsets = jnp.arange(-40, 41)
    # k - x: each bin's centre from the track [bins], one row per position
    offsets = bin_offsets[None, :] - position[:, None]
    # W(k - x): the power each bin takes from a signal on the track
    centred = window.power(offsets)

    def overlap(shift: Float[Array, ""]) -> Float[Array, ""]:
        # W(k - x - d): the same for a track off by d = shift bins
        shifted = window.power(offsets - shift)
        return jnp.sum(centred * shifted) / jnp.sum(centred * centred)

    # the second derivative at d = 0 by autodiff, which a float32 difference cannot
    # resolve
    return -0.5 * float(jax.grad(jax.grad(overlap))(0.0))


def physical_metric(
    point: Parameters,
    *,
    times: Float[Array, " n_sft"],
    velocity: Float[Array, "n_sft 3"],
    tsft: float,
    g_freq_bin: float,
) -> Float[Array, "6 6"]:
    """Mismatch metric on the physical parameters at one point, closed form.

        g_mu_nu = g_freq_bin tsft^2 <d_mu f(t_n) d_nu f(t_n)>_n

    with `f(t_n)` the frequency track at the n-th SFT midpoint.

    Args:
        point: Point to evaluate at.
        times: SFT midpoints from the reference epoch [s].
        velocity: Detector velocity `v/c` at each SFT midpoint.
        tsft: SFT length [s].
        g_freq_bin: freq_bin_curvature() of the window the SFTs were made with.

    Returns:
        `g[mu, nu]` in the field order of Parameters.
    """
    flat, unravel = ravel_pytree(
        Parameters(*(jnp.asarray(x, dtype=float) for x in point))
    )

    def track(vector: Float[Array, " 6"]) -> Float[Array, " n_sft"]:
        p = unravel(vector)
        return observed_frequency(
            times,
            f0=p.f0,
            ap=p.ap,
            omega=p.omega,
            phi_b=p.phi_b,
            doppler=velocity @ sky_direction(p.alpha, p.delta),
        )

    jacobian = jax.jacfwd(track)(flat) * tsft
    return g_freq_bin * (jacobian.T @ jacobian) / jacobian.shape[0]


def project_out(
    metric: Float[Array, "6 6"], name: str = "phi_b"
) -> Float[Array, "5 5"]:
    """Metric on the remaining parameters when one is maximized over.

        g'_mu_nu = g_mu_nu - g_mu_x g_xx^-1 g_x_nu

    Args:
        metric: `g_mu_nu` in the field order of Parameters.
        name: The field maximized over, `phi_b` by default.

    Returns:
        Metric on the other five fields, in their original order.
    """
    projected_idx = Parameters._fields.index(name)
    kept_idxs = jnp.array(
        [i for i in range(len(Parameters._fields)) if i != projected_idx]
    )
    g_keep = metric[jnp.ix_(kept_idxs, kept_idxs)]
    g_cross = metric[kept_idxs, projected_idx]
    return g_keep - jnp.outer(g_cross, g_cross) / metric[projected_idx, projected_idx]


def orbital_lattice_metric(
    metric: Float[Array, "5 5"], point: Parameters, *, tsft: float
) -> Float[Array, "3 3"]:
    """The physical metric moved onto the orbital lattice's axes.

    Args:
        metric: physical_metric() with `phi_b` projected out, at `point`.
        point: Where the metric was taken.
        tsft: SFT length [s].

    Returns:
        `g` on `(f0, modulation_depth_bins, omega^2)`.
    """
    modulation_depth_bins = point.modulation_depth_bins(tsft)
    omega2 = point.omega**2

    # Jacobian = d(f0, ap, P) / d(f0, modulation_depth_bins, omega^2)
    jacobian = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [
                -point.ap / point.f0,
                point.ap / modulation_depth_bins,
                -point.ap / (2.0 * omega2),
            ],
            [0.0, 0.0, -point.period / (2.0 * omega2)],
        ]
    )
    return jacobian.T @ metric[:3, :3] @ jacobian


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
