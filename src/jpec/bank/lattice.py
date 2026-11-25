"""The A5* generator under a metric, its row shifts, and the sky lattice."""

import math
from fractions import Fraction

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import KDTree

# a row ratio within this of a fraction is that fraction
SHIFT_FRACTION_TOLERANCE = 1e-9
# most distinct shift fractions one generator row may give
MAX_SHIFT_FRACTIONS = 1000
# rim ring spacing in cubic steps along the rim
RIM_STEP = 0.5
# angles the rim's length in cubic steps is integrated over
RIM_SAMPLES = 200000


def cubic_step(
    curvature: Float[np.ndarray, "..."], mismatch: float
) -> Float[np.ndarray, "..."]:
    """The step of a cubic lattice along an axis of curvature `g`: `2 sqrt(mu / g)`.

    Args:
        curvature: `g_ii` of the axis, or of several.
        mismatch: Mismatch `mu` of one axis.

    Returns:
        The step, in the axis's own units.
    """
    return 2.0 * (mismatch / curvature) ** 0.5


def a5star_generator() -> Float[np.ndarray, "5 5"]:
    """The A5* generator matrix in triangular form, at covering radius 1.

    Column i is basis vector i, so coordinate k of a lattice point depends only on the
    indices from k on.

    Returns:
        `R`, upper triangular with a positive diagonal.
    """
    n_axes = 5
    a5_gram = 2.0 * np.eye(n_axes) - np.eye(n_axes, k=1) - np.eye(n_axes, k=-1)
    # lower triangular, rows are A5* basis vectors: its transpose is R before scaling
    basis = np.linalg.cholesky(np.linalg.inv(a5_gram))
    # covering radius of An* at this Gram matrix, sqrt(n (n + 2) / (12 (n + 1)))
    # (Conway and Sloane, Sphere Packings, Lattices and Groups, ch. 4)
    covering_radius = np.sqrt(n_axes * (n_axes + 2) / (12.0 * (n_axes + 1)))
    return basis.T / covering_radius


def triangular_generator(
    metric: Float[np.ndarray, "n_axis n_axis"],
    mismatch: float,
    generator: Float[np.ndarray, "n_axis n_axis"],
) -> Float[np.ndarray, "n_axis n_axis"]:
    """The generator matrix of a lattice that covers a metric to a mismatch.

        T = sqrt(mu) L^-T R,   g = L L^T

    with `L` lower triangular. Lattice point `n` sits at `origin + T n`, and every
    point lies within mismatch `mu` of one.

    Args:
        metric: `g`, an offset `dx` costing mismatch `dx^T g dx`.
        mismatch: `mu`, the covering mismatch.
        generator: `R`, upper triangular at covering radius 1.

    Returns:
        `T`, upper triangular, in the metric's coordinates.
    """
    cholesky = np.linalg.cholesky(metric)
    return np.sqrt(mismatch) * np.linalg.solve(cholesky.T, generator)


def shift_fractions(
    ratios: Float[np.ndarray, " n_outer"],
) -> Float[np.ndarray, " n_shift"]:
    """Every value of `sum_j r_j n_j mod 1` over integer `n_j`.

    For row `k` of an upper triangular generator `T`,

        r = T[k, k+1:] / T[k, k]

    With `r_j = a_j / b_j` in lowest terms, the values are the multiples of `1 / q`,
    `q = lcm(b_j)`.

    Args:
        ratios: `r`, a generator row's entries past the diagonal over its diagonal.

    Returns:
        `0, 1/q, ..., (q - 1)/q`.

    Raises:
        ValueError: If a ratio is not a fraction, or `q` is more than
            `MAX_SHIFT_FRACTIONS`.
    """
    denominators = []
    for ratio in np.asarray(ratios, dtype=float):
        fraction = Fraction(ratio).limit_denominator(MAX_SHIFT_FRACTIONS)
        if abs(float(fraction) - ratio) > SHIFT_FRACTION_TOLERANCE:
            raise ValueError(f"the ratio {ratio} is not a fraction")
        denominators.append(fraction.denominator)
    n_shifts = math.lcm(*denominators)
    if n_shifts > MAX_SHIFT_FRACTIONS:
        raise ValueError(
            f"the ratios {ratios} give more than {MAX_SHIFT_FRACTIONS} shifts"
        )
    return np.arange(n_shifts) / n_shifts


def split_shift(
    shift: Float[np.ndarray, "..."], fractions: Float[np.ndarray, " n_shift"]
) -> tuple[Int[np.ndarray, "..."], Int[np.ndarray, "..."]]:
    """Each shift as one of a row's shift fractions plus whole steps.

        shift = whole_steps + fractions[idx],   idx = rint(q shift) mod q

    with `q` the number of fractions.

    Args:
        shift: Shifts, in steps of the axis.
        fractions: The row's shift fractions, `0, 1/q, ..., (q - 1)/q`.

    Returns:
        The index of each shift's fraction, and its whole steps.
    """
    n_shifts = fractions.size
    # the shift in units of 1 / q
    units = np.rint(np.asarray(shift) * n_shifts).astype(int)
    return units % n_shifts, units // n_shifts


def sphere_points(
    x1: Float[np.ndarray, " n_disc_point"],
    x2: Float[np.ndarray, " n_disc_point"],
    *,
    s1: float,
    s2: float,
) -> Float[np.ndarray, "n_point 3"]:
    """Disc points lifted to both hemispheres, with a ring on the rim `x3 = 0`.

    Args:
        x1: x1 of each disc point.
        x2: x2 of each disc point.
        s1: Cubic step along x1.
        s2: Cubic step along x2.

    Returns:
        `(x1, x2, x3)` of each point.
    """
    x3 = np.sqrt(1.0 - x1**2 - x2**2)
    # the rim at angle theta runs along (-sin, cos), whose cubic step is
    # 1 / sqrt(sin^2 / s1^2 + cos^2 / s2^2)
    theta = np.linspace(0.0, 2.0 * np.pi, RIM_SAMPLES + 1)
    steps_per_radian = np.sqrt(np.sin(theta) ** 2 / s1**2 + np.cos(theta) ** 2 / s2**2)
    # cubic steps along the rim from theta = 0
    rim_length = cumulative_trapezoid(steps_per_radian, theta, initial=0.0)
    n_rim_points = int(np.ceil(rim_length[-1] / RIM_STEP))
    # theta from the CDF
    rim_angle = np.interp(
        np.arange(n_rim_points) * rim_length[-1] / n_rim_points, rim_length, theta
    )
    ring = np.stack(
        [np.cos(rim_angle), np.sin(rim_angle), np.zeros_like(rim_angle)], axis=1
    )
    # return upper hemi + lower hemi + rim
    return np.concatenate(
        [np.stack([x1, x2, x3], axis=1), np.stack([x1, x2, -x3], axis=1), ring]
    )


def in_sky_box(
    alpha: Float[np.ndarray, " n_sky"],
    delta: Float[np.ndarray, " n_sky"],
    *,
    box: tuple[float, float, float, float],
    margin: float,
) -> Bool[np.ndarray, " n_sky"]:
    """Whether each sky position lies within `margin` of a box in `(alpha, delta)`.

    Args:
        alpha: Right ascension of each position [rad].
        delta: Declination of each position [rad].
        box: `(alpha_low, alpha_high, delta_low, delta_high)` [rad].
        margin: Distance on the sky past the box's edges still kept [rad].

    Returns:
        The mask.
    """
    alpha_low, alpha_high, delta_low, delta_high = box
    alpha_centre = 0.5 * (alpha_low + alpha_high)
    alpha_half_width = 0.5 * (alpha_high - alpha_low)
    alpha_offset = np.abs((alpha - alpha_centre + np.pi) % (2.0 * np.pi) - np.pi)
    in_alpha = alpha_offset <= alpha_half_width + margin / np.maximum(
        np.cos(delta), 1e-3
    )
    in_delta = (delta >= delta_low - margin) & (delta <= delta_high + margin)
    return in_alpha & in_delta


def sky_lattice(
    sky_generator: Float[np.ndarray, "2 2"],
    *,
    sky_eigenvectors: Float[np.ndarray, "3 3"],
    cubic_steps: Float[np.ndarray, " 2"],
    box: tuple[float, float, float, float],
    margin: float,
) -> tuple[
    Float[np.ndarray, " n_sky"], Float[np.ndarray, " n_sky"], Int[np.ndarray, "n_sky 2"]
]:
    """The sky positions of a 2D lattice laid on the sky metric's resolved directions.

    The lattice's points in the unit disc of `(x1, x2)` go on both hemispheres, and a
    ring on the rim closes the strip between them. Each rim point takes the indices of
    the disc point nearest it in cubic steps.

    Args:
        sky_generator: `T` on `(x1, x2)`, upper triangular.
        sky_eigenvectors: The sky metric's eigenvectors as columns, largest first.
        cubic_steps: Cubic steps along x1 and x2, which space the rim ring.
        box: `(alpha_low, alpha_high, delta_low, delta_high)` of the sky kept [rad].
        margin: Distance on the sky past the box's edges still kept [rad].

    Returns:
        Right ascension and declination of each position [rad], and its lattice
            indices.
    """
    x1_step = sky_generator[0, 0]
    x2_step = sky_generator[1, 1]
    # how far one step in n2 shifts x1
    x1_shift = sky_generator[0, 1]
    # the points T n inside the unit disc
    n2_max = int(np.ceil(1.0 / x2_step))
    n1_max = int(np.ceil((1.0 + abs(x1_shift) * n2_max) / x1_step))
    candidate_lattice_idxs = np.stack(
        np.meshgrid(
            np.arange(-n1_max, n1_max + 1),
            np.arange(-n2_max, n2_max + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    candidate_coords = candidate_lattice_idxs @ sky_generator.T
    inside = (candidate_coords**2).sum(axis=1) < 1.0
    disc_coords = candidate_coords[inside]
    disc_lattice_idxs = candidate_lattice_idxs[inside]
    lattice_coords = sphere_points(
        disc_coords[:, 0],
        disc_coords[:, 1],
        s1=float(cubic_steps[0]),
        s2=float(cubic_steps[1]),
    )
    rim_coords = lattice_coords[2 * disc_coords.shape[0] :, :2]
    # a rim point's shifts are approximated by those of the nearest disc point
    _, nearest_disc_idxs = KDTree(disc_coords / cubic_steps).query(
        rim_coords / cubic_steps
    )
    lattice_idxs = np.concatenate(
        [disc_lattice_idxs, disc_lattice_idxs, disc_lattice_idxs[nearest_disc_idxs]]
    )

    directions = lattice_coords @ sky_eigenvectors.T
    alpha = np.arctan2(directions[:, 1], directions[:, 0]) % (2.0 * np.pi)
    delta = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    keep = in_sky_box(alpha, delta, box=box, margin=margin)
    return alpha[keep], delta[keep], lattice_idxs[keep]
