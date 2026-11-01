"""The cubic step, bcc orbital subgrids per ap block, and the hexagonal sky lattice."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
from scipy.integrate import cumulative_trapezoid

from jpec.bank.metric import sky_metric
from jpec.params import ap_from_modulation_depth_bins, omega_from_period

# the body-centred cubic (A3*) lattice: two cubic subgrids offset by half a step on
# every axis
# side of a bcc subgrid in cubic steps, for the cubic grid's covering radius
BCC_SUBGRID_STEP = float(np.sqrt(2.4))
# where each subgrid sits within its step, on all three axes
BCC_SUBGRID_OFFSETS = (0.25, 0.75)
# along x1 and x2, in square steps, for the square grid's covering radius
HEXAGONAL_STEPS = (float(np.sqrt(1.5)), float(np.sqrt(4.5)))
# staggered rows, which is what makes the sky lattice hexagonal
HEXAGONAL_ROW_SHIFTS = (0.0, 0.5)
# rim ring spacing in cubic steps along the rim: s1/2 where it runs along x1, s2/2
# along x2
RIM_STEP = 0.5
# angles the rim's length in cubic steps is integrated over
RIM_SAMPLES = 200000


def cubic_step(curvature: Float[Array, "..."], mismatch: float) -> Float[Array, "..."]:
    """The step of a cubic lattice along an axis of curvature `g`: `2 sqrt(mu / g)`.

    Args:
        curvature: `g_ii` of the axis, or of several.
        mismatch: Mismatch `mu` of one axis.

    Returns:
        The step, in the axis's own units.
    """
    return 2.0 * (mismatch / curvature) ** 0.5


class OrbitalSubgrid(NamedTuple):
    """One subgrid of the orbital lattice over an f0 band.

    Attributes:
        f0: The f0 points [Hz].
        modulation_depth_bins: The modulation depths [bins], one grid for every period.
        period: Every trial period [s], ap block after ap block.
        ap_block_idxs: The ap block each period belongs to.
        in_box: Whether the template's ap lies in its ap block's range.
    """

    f0: Float[np.ndarray, " n_f0"]
    modulation_depth_bins: Float[np.ndarray, " n_modulation_depth"]
    period: Float[np.ndarray, " n_period"]
    ap_block_idxs: Int[np.ndarray, " n_period"]
    in_box: Bool[np.ndarray, "n_period n_modulation_depth"]

    @property
    def n_templates_per_sky_point(self) -> int:
        """Templates in the box at one sky point."""
        return int(self.f0.size * self.in_box.sum())


class ApBlock(NamedTuple):
    """Cubic steps the metric gives at one point, and the ap range they are used over.

    Attributes:
        ap_low: Low edge of the ap block [light-seconds].
        ap_high: High edge [light-seconds], the ap its omega^2 step is set for.
        modulation_depth_step: Cubic modulation-depth step [bins].
        omega2_step: Cubic omega^2 step [rad^2/s^2].
        g_f0f0: The f0 entry of the projected `(f0, ap, P)` metric, which the f0
            grid and the sky lattice use.
    """

    ap_low: float
    ap_high: float
    modulation_depth_step: float
    omega2_step: float
    g_f0f0: float


def axis_points_covering(
    low: float, high: float, step: float, *, offset: float
) -> Float[np.ndarray, " n_point"]:
    """Points of one lattice axis, from one step below `low` to one above `high`.

    The points are `low + step (index + offset)`.

    Args:
        low: Where the axis is anchored and the range starts, in its own units.
        high: Where the range ends.
        step: The spacing.
        offset: Where the subgrid sits within a step.

    Returns:
        The points, increasing.
    """
    n_steps = max(int(np.ceil((high - low) / step)), 1)
    return low + step * (np.arange(-1, n_steps + 1) + offset)


def orbital_subgrid(
    f0_low: float,
    f0_high: float,
    *,
    ap_low: float,
    ap_high: float,
    period_low: float,
    period_high: float,
    f0_step: float,
    modulation_depth_step: float,
    ap_blocks: list[ApBlock],
    f0_reference: float,
    tsft: float,
    offset: float,
) -> OrbitalSubgrid:
    """One bcc subgrid in `(f0, modulation_depth_bins, omega^2)` over one f0 band.

    It runs one step past every edge.

    Args:
        f0_low: Low edge of the f0 band [Hz].
        f0_high: High edge [Hz].
        ap_low: Low edge of the box in ap [light-seconds].
        ap_high: High edge [light-seconds].
        period_low: Low edge of the box in the orbital period [s].
        period_high: High edge [s].
        f0_step: Cubic f0 step [Hz].
        modulation_depth_step: Cubic modulation-depth step [bins].
        ap_blocks: Every ap block, its cubic omega^2 step taken at `f0_reference`.
        f0_reference: Frequency the omega^2 steps were taken at [Hz]; they scale as
            `1 / f0`.
        tsft: SFT length [s].
        offset: Where the subgrid sits within its step, one of `BCC_SUBGRID_OFFSETS`.

    Returns:
        The subgrid.
    """
    # the modulation-depth and omega^2 axes are laid at the band's centre frequency
    f0_centre = 0.5 * (f0_low + f0_high)
    omega_low = float(omega_from_period(jnp.asarray(period_high)))
    omega_high = float(omega_from_period(jnp.asarray(period_low)))

    f0_step = BCC_SUBGRID_STEP * f0_step
    f0 = axis_points_covering(f0_low, f0_high, f0_step, offset=offset)

    subgrid_modulation_depth_step = BCC_SUBGRID_STEP * modulation_depth_step
    modulation_depth_bins = axis_points_covering(
        f0_centre * ap_low * omega_low * tsft,
        f0_centre * ap_high * omega_high * tsft,
        subgrid_modulation_depth_step,
        offset=offset,
    )

    periods, ap_block_idxs, in_box = [], [], []
    for ap_block_idx, ap_block in enumerate(ap_blocks):
        # the cubic omega^2 step scales as 1 / f0 from where it was measured
        omega2_step = BCC_SUBGRID_STEP * ap_block.omega2_step * f0_reference / f0_centre
        omega2 = axis_points_covering(
            omega_low**2, omega_high**2, omega2_step, offset=offset
        )
        omega = np.sqrt(omega2)
        # [n_period, n_modulation_depth]
        ap = np.asarray(
            ap_from_modulation_depth_bins(
                jnp.asarray(modulation_depth_bins[None, :]),
                f0=jnp.asarray(f0_centre),
                omega=jnp.asarray(omega[:, None]),
                tsft=tsft,
            )
        )
        # one modulation-depth step, in ap
        ap_margin = np.asarray(
            ap_from_modulation_depth_bins(
                jnp.asarray(subgrid_modulation_depth_step),
                f0=jnp.asarray(f0_centre),
                omega=jnp.asarray(omega[:, None]),
                tsft=tsft,
            )
        )
        periods.append(2.0 * np.pi / omega)
        ap_block_idxs.append(np.full(omega.size, ap_block_idx))
        in_box.append(
            (ap >= ap_block.ap_low - ap_margin) & (ap <= ap_block.ap_high + ap_margin)
        )
    return OrbitalSubgrid(
        f0=f0,
        modulation_depth_bins=modulation_depth_bins,
        period=np.concatenate(periods),
        ap_block_idxs=np.concatenate(ap_block_idxs),
        in_box=np.concatenate(in_box),
    )


def hexagonal_disc_points(
    h1: float, h2: float
) -> tuple[Float[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
    """The hexagonal lattice in `(x1, x2)` inside the unit disc.

    Args:
        h1: Hexagonal step along x1.
        h2: Hexagonal step along x2.

    Returns:
        x1 and x2 of each point.
    """

    def direction_cosine_points(
        step: float, shift: float
    ) -> Float[np.ndarray, " n_point"]:
        """Points `(k + shift) step` over `[-1, 1]`; one at 0 if a step spans it."""
        if step >= 2.0:
            return np.zeros(1)
        k_max = int(np.ceil(1.0 / step))
        return (np.arange(-k_max, k_max + 1) + shift) * step

    grids = [
        np.meshgrid(
            direction_cosine_points(h1, shift),
            direction_cosine_points(h2, shift),
            indexing="ij",
        )
        for shift in HEXAGONAL_ROW_SHIFTS
    ]
    x1 = np.concatenate([grid[0].ravel() for grid in grids])
    x2 = np.concatenate([grid[1].ravel() for grid in grids])
    inside = x1**2 + x2**2 < 1.0
    return x1[inside], x2[inside]


def sphere_points(
    x1: Float[np.ndarray, " n_point"],
    x2: Float[np.ndarray, " n_point"],
    *,
    s1: float,
    s2: float,
) -> Float[np.ndarray, "n_lattice_point 3"]:
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
    velocity: Float[np.ndarray, "n_sft 3"],
    *,
    f0: float,
    g_f0f0: float,
    mismatch: float,
    box: tuple[float, float, float, float],
) -> tuple[Float[np.ndarray, " n_sky"], Float[np.ndarray, " n_sky"]]:
    """The sky positions of a lattice laid on the sky metric's resolved directions.

    A template `u = n_hat_true - n_hat` off costs

        mu = g_f0f0 f0^2 u^T Cov(v) u

    Args:
        velocity: Detector velocity `v/c` at each SFT, every detector.
        f0: Frequency the lattice is for [Hz].
        g_f0f0: The f0 metric [1/Hz^2].
        mismatch: Mismatch budget of one axis.
        box: `(alpha_low, alpha_high, delta_low, delta_high)` of the sky kept
            [rad]; the points within one lattice step of it are kept too, so its
            edges are covered.

    Returns:
        Right ascension and declination of each position [rad].
    """
    curvature, sky_eigenvectors = sky_metric(velocity, f0=f0, g_f0f0=g_f0f0)
    s1, s2, _ = np.asarray(cubic_step(jnp.asarray(curvature), mismatch))
    # hexagonal in (x1, x2): its covering radius is the square cell's corner, 2 mu,
    # with 1.30x fewer points
    h1, h2 = HEXAGONAL_STEPS[0] * s1, HEXAGONAL_STEPS[1] * s2

    x1, x2 = hexagonal_disc_points(h1, h2)
    lattice_x = sphere_points(x1, x2, s1=s1, s2=s2)
    directions = lattice_x @ sky_eigenvectors.T
    alpha = np.arctan2(directions[:, 1], directions[:, 0]) % (2.0 * np.pi)
    delta = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    # one lattice step, in direction cosines, which is radians on the sky
    keep = in_sky_box(alpha, delta, box=box, margin=max(h1, h2))
    return alpha[keep], delta[keep]
