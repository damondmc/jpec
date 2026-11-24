"""Template bank: an A5* lattice over the sky, over an f0 interval or around a point."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float, Int

from jpec.bank.lattice import (
    a5star_generator,
    cubic_step,
    shift_fractions,
    sky_lattice,
    split_shift,
    triangular_generator,
)
from jpec.bank.metric import sky_metric
from jpec.bank.tiling import ApBlock, LatticeTiling
from jpec.detector.doppler import sky_direction
from jpec.params import Parameters, ap_from_modulation_depth, omega_from_period
from jpec.waveform.window import Window


def padded_axis_idxs(
    span: float, *, step: float, margin: float
) -> Int[np.ndarray, " n_point"]:
    """Lattice indices of one axis's grid over a range, anchored at its low edge.

    Point `n` sits at `low + n step`, before a shift in `[0, step)`. The grid runs past
    each edge by `margin` rounded up to whole steps, and one step more below for the
    shift.

    Args:
        span: Width of the range, in the axis's units.
        step: The spacing.
        margin: How far past each edge a point is still needed.

    Returns:
        The indices, increasing.
    """
    n_pad = int(np.ceil(margin / step))
    return np.arange(-n_pad - 1, int(np.ceil(span / step)) + n_pad + 1)


def idxs_around(
    grid: Float[np.ndarray, " n_axis_point"], value: float, *, n_steps: int
) -> Int[np.ndarray, " n_kept"]:
    """Indices of the `n_steps` points each side of the one nearest `value`.

    They are clipped to the axis.

    Args:
        grid: One base grid, increasing.
        value: The point searched around.
        n_steps: Number of points kept each side.

    Returns:
        The indices, increasing.
    """
    nearest_idx = int(np.argmin(np.abs(grid - value)))
    return np.arange(
        max(nearest_idx - n_steps, 0), min(nearest_idx + n_steps + 1, grid.size)
    )


class TemplateBank(NamedTuple):
    """One job's templates: an A5* lattice on `(f0, depth, omega^2, x_a, x_b)`.

    Lattice point `n` sits at `origin + T n`, with `T` upper triangular and the origin
    at the low edge of f0, depth and omega^2. Base grids are shared by every sky point,
    and sky point `i` moves the template at `(period p, depth q, f0 k)` to

        f0       f0[k] + f0_shift(i)[p, q]
        depth    modulation_depth[q]
                 + depth_shift_fractions[depth_class(i)[p]] * modulation_depth_step
        omega^2  omega2[p] + omega2_shift_fractions[omega2_class[i]] * omega2_step[p]

    Attributes:
        tsft: SFT length [s].
        window: The window the SFTs were made with.
        f0_centre: The band centre the ap ranges and depths convert at [Hz].
        generator: `T` of the first ap block.
        f0: The base f0 grid [Hz].
        f0_idxs: Lattice index of each f0 point.
        modulation_depth: The base modulation-depth grid [Hz].
        modulation_depth_idxs: Lattice index of each modulation depth.
        omega2: The base omega^2 grid [rad^2/s^2], ap block after ap block.
        omega2_idxs: Lattice index of each omega^2 point within its ap block.
        omega2_step: The omega^2 step of each point's ap block [rad^2/s^2].
        ap_block_idxs: The ap block of each omega^2 point.
        ap_blocks: Every ap block.
        alpha: Right ascension of each sky point [rad].
        delta: Declination of each sky point [rad].
        sky_idxs: Lattice indices `(n_3, n_4)` of each sky point.
    """

    tsft: float
    window: Window
    f0_centre: float
    generator: Float[np.ndarray, "5 5"]
    f0: Float[np.ndarray, " n_f0"]
    f0_idxs: Int[np.ndarray, " n_f0"]
    modulation_depth: Float[np.ndarray, " n_modulation_depth"]
    modulation_depth_idxs: Int[np.ndarray, " n_modulation_depth"]
    omega2: Float[np.ndarray, " n_period"]
    omega2_idxs: Int[np.ndarray, " n_period"]
    omega2_step: Float[np.ndarray, " n_period"]
    ap_block_idxs: Int[np.ndarray, " n_period"]
    ap_blocks: list[ApBlock]
    alpha: Float[np.ndarray, " n_sky"]
    delta: Float[np.ndarray, " n_sky"]
    sky_idxs: Int[np.ndarray, "n_sky 2"]

    @property
    def f0_step(self) -> float:
        """The f0 step [Hz]."""
        return float(self.generator[0, 0])

    @property
    def modulation_depth_step(self) -> float:
        """The modulation-depth step [Hz]."""
        return float(self.generator[1, 1])

    @property
    def f0_high(self) -> float:
        """Highest f0 a template reaches with its shift [Hz]."""
        return float(self.f0[-1] + self.f0_step)

    @property
    def max_modulation_depth(self) -> float:
        """Deepest modulation depth a template reaches with its shift [Hz]."""
        return float(self.modulation_depth[-1] + self.modulation_depth_step)

    @property
    def omega2_shift_fractions(self) -> Float[np.ndarray, " n_omega2_class"]:
        """Every omega^2 shift a sky point gives, in omega^2 steps, increasing."""
        return shift_fractions(self.generator[2, 3:] / self.generator[2, 2])

    @property
    def depth_shift_fractions(self) -> Float[np.ndarray, " n_depth_class"]:
        """Every depth shift a (sky point, period) gives, in depth steps, increasing."""
        return shift_fractions(self.generator[1, 2:] / self.generator[1, 1])

    @property
    def omega2_class(self) -> Int[np.ndarray, " n_sky"]:
        """Index of each sky point's omega^2 shift fraction."""
        omega2_class, _ = self._split_omega2_shift(self.sky_idxs)
        return omega2_class

    @property
    def n_templates(self) -> int:
        """Number of templates in range, over the bank's sky points."""
        classes = np.stack(
            [
                np.concatenate([[omega2_class], self.depth_class(sky_idx)])
                for sky_idx, omega2_class in enumerate(self.omega2_class)
            ]
        )
        unique_classes, n_sky_points = np.unique(classes, axis=0, return_counts=True)
        n_templates_per_sky_point = np.array(
            [
                self.ap_in_range(int(sky_classes[0]), sky_classes[1:]).sum()
                for sky_classes in unique_classes
            ]
        )
        return int(self.f0.size * (n_templates_per_sky_point * n_sky_points).sum())

    def _split_omega2_shift(
        self, sky_lattice_idxs: Int[np.ndarray, "... 2"]
    ) -> tuple[Int[np.ndarray, "..."], Int[np.ndarray, "..."]]:
        """The omega^2 shift of sky points, in omega^2 steps.

            shift = (T[2, 3] n_3 + T[2, 4] n_4) / T[2, 2] = 2/3 n_3 + 1/3 n_4

        Args:
            sky_lattice_idxs: Lattice indices `(n_3, n_4)` of each sky point.

        Returns:
            The index of each one's omega^2 shift fraction, and the whole omega^2 steps
                its shift holds.
        """
        shift = sky_lattice_idxs @ self.generator[2, 3:] / self.generator[2, 2]
        class_idx, whole_steps = split_shift(shift, self.omega2_shift_fractions)
        return class_idx, whole_steps

    def _split_depth_shift(
        self,
        sky_lattice_idxs: Int[np.ndarray, "... 2"],
        period_lattice_idxs: Int[np.ndarray, "..."],
    ) -> tuple[Int[np.ndarray, "..."], Int[np.ndarray, "..."]]:
        """The depth shift of each (sky point, period), in depth steps.

            shift = (T[1, 2] n_2 + T[1, 3] n_3 + T[1, 4] n_4) / T[1, 1]
                  = 3/4 n_2 + 1/2 n_3 + 1/4 n_4

        Args:
            sky_lattice_idxs: Lattice indices `(n_3, n_4)` of each sky point.
            period_lattice_idxs: Lattice index `n_2` of each period.

        Returns:
            The index of each period's depth shift fraction, and the whole depth steps
                its shift holds.
        """
        shift = (
            self.generator[1, 2] * period_lattice_idxs
            + sky_lattice_idxs @ self.generator[1, 3:]
        ) / self.generator[1, 1]
        class_idx, whole_steps = split_shift(shift, self.depth_shift_fractions)
        return class_idx, whole_steps

    def _f0_shift(
        self,
        sky_lattice_idxs: Int[np.ndarray, "... 2"],
        period_lattice_idxs: Int[np.ndarray, "..."],
        depth_lattice_idxs: Int[np.ndarray, "..."],
    ) -> Float[np.ndarray, "..."]:
        """How far each sky point moves the f0 grid of its (period, depth) points [Hz].

            shift = (T[0, 1] n_1 + T[0, 2] n_2 + T[0, 3] n_3 + T[0, 4] n_4) mod T[0, 0]

        Args:
            sky_lattice_idxs: Lattice indices `(n_3, n_4)` of each sky point.
            period_lattice_idxs: Lattice index `n_2` of each period.
            depth_lattice_idxs: Lattice index `n_1` of each depth.

        Returns:
            The shift, in `[0, f0_step)`.
        """
        return (
            self.generator[0, 1] * depth_lattice_idxs
            + self.generator[0, 2] * period_lattice_idxs
            + sky_lattice_idxs @ self.generator[0, 3:]
        ) % self.f0_step

    def period_lattice_idxs(self, sky_idx: int) -> Int[np.ndarray, " n_period"]:
        """Lattice index `n_2` of each base omega^2 point at one sky point.

        Args:
            sky_idx: Index of the sky point.

        Returns:
            The indices, one per base omega^2 point.
        """
        _, omega2_whole_steps = self._split_omega2_shift(self.sky_idxs[sky_idx])
        return self.omega2_idxs - omega2_whole_steps

    def depth_class(self, sky_idx: int) -> Int[np.ndarray, " n_period"]:
        """Index of each period's depth shift fraction at one sky point.

        Args:
            sky_idx: Index of the sky point.

        Returns:
            The depth class per period.
        """
        depth_class, _ = self._split_depth_shift(
            self.sky_idxs[sky_idx], self.period_lattice_idxs(sky_idx)
        )
        return depth_class

    def f0_shift(
        self, sky_idx: int
    ) -> Float[np.ndarray, "n_period n_modulation_depth"]:
        """How far one sky point moves each (period, depth)'s f0 grid [Hz].

        Args:
            sky_idx: Index of the sky point.

        Returns:
            The shift, in `[0, f0_step)`.
        """
        sky_lattice_idxs = self.sky_idxs[sky_idx]
        period_lattice_idxs = self.period_lattice_idxs(sky_idx)
        _, depth_whole_steps = self._split_depth_shift(
            sky_lattice_idxs, period_lattice_idxs
        )
        return self._f0_shift(
            sky_lattice_idxs,
            period_lattice_idxs[:, None],
            self.modulation_depth_idxs[None, :] - depth_whole_steps[:, None],
        )

    def modulation_depth_at(
        self, depth_class: int | Int[np.ndarray, " n_period"]
    ) -> (
        Float[np.ndarray, " n_modulation_depth"]
        | Float[np.ndarray, "n_period n_modulation_depth"]
    ):
        """The base modulation-depth grid moved by a depth class's shift [Hz].

        Args:
            depth_class: Index of a depth shift fraction, or one per period.

        Returns:
            The moved grid, one per depth class given.
        """
        return (
            self.modulation_depth
            + self.depth_shift_fractions[depth_class][..., None]
            * self.modulation_depth_step
        )

    def omega_at(self, omega2_class: int) -> Float[np.ndarray, " n_period"]:
        """Orbital angular frequency of the base omega^2 grid moved by a class's shift.

        Args:
            omega2_class: Index of an omega^2 shift fraction.

        Returns:
            `sqrt(omega2 + omega2_shift_fractions[omega2_class] * omega2_step)`
                [rad/s].
        """
        return np.sqrt(
            self.omega2 + self.omega2_shift_fractions[omega2_class] * self.omega2_step
        )

    def ap_in_range(
        self, omega2_class: int, depth_class: Int[np.ndarray, " n_period"]
    ) -> Bool[np.ndarray, "n_period n_modulation_depth"]:
        """Whether each template's ap lies in its ap block, within one depth step.

        Args:
            omega2_class: Index of the sky point's omega^2 shift fraction.
            depth_class: Index of each period's depth shift fraction.

        Returns:
            The mask.
        """
        omega = self.omega_at(omega2_class)[:, None]
        ap = ap_from_modulation_depth(
            self.modulation_depth_at(depth_class), f0=self.f0_centre, omega=omega
        )
        # one modulation-depth step, in ap
        ap_margin = ap_from_modulation_depth(
            self.modulation_depth_step, f0=self.f0_centre, omega=omega
        )
        ap_low = np.array([ap_block.ap_low for ap_block in self.ap_blocks])
        ap_high = np.array([ap_block.ap_high for ap_block in self.ap_blocks])
        return (ap >= ap_low[self.ap_block_idxs][:, None] - ap_margin) & (
            ap <= ap_high[self.ap_block_idxs][:, None] + ap_margin
        )

    def parameters(
        self,
        *,
        sky_idx: int,
        f0_idx: int,
        period_idx: int,
        modulation_depth_idx: int,
        phase_idx: int,
        n_phi: int,
    ) -> Parameters:
        """The template the bank's indices point at, its f0 the lattice point's.

        Args:
            sky_idx: Index of the sky point.
            f0_idx: Index of the base f0 point.
            period_idx: Index of the base omega^2 point.
            modulation_depth_idx: Index of the base modulation depth.
            phase_idx: Index of the orbital phase, of `n_phi`.
            n_phi: Number of orbital phases the statistic was read at.

        Returns:
            The template's parameters.
        """
        # the lattice indices and shifts of this one template, not of every
        # (period, depth) of the sky point
        sky_lattice_idxs = self.sky_idxs[sky_idx]
        omega2_class, omega2_whole_steps = self._split_omega2_shift(sky_lattice_idxs)
        period_lattice_idx = self.omega2_idxs[period_idx] - omega2_whole_steps
        depth_class, depth_whole_steps = self._split_depth_shift(
            sky_lattice_idxs, period_lattice_idx
        )
        f0 = float(
            self.f0[f0_idx]
            + self._f0_shift(
                sky_lattice_idxs,
                period_lattice_idx,
                self.modulation_depth_idxs[modulation_depth_idx] - depth_whole_steps,
            )
        )
        modulation_depth = float(
            self.modulation_depth_at(depth_class)[modulation_depth_idx]
        )
        omega = float(self.omega_at(int(omega2_class))[period_idx])
        ap = ap_from_modulation_depth(modulation_depth, f0=f0, omega=omega)
        return Parameters(
            f0=f0,
            ap=float(ap),
            period=2.0 * np.pi / omega,
            phi_b=2.0 * np.pi * phase_idx / n_phi,
            alpha=float(self.alpha[sky_idx]),
            delta=float(self.delta[sky_idx]),
        )

    @classmethod
    def over_f0_interval(
        cls,
        tiling: LatticeTiling,
        f0_low: float,
        f0_high: float,
        *,
        velocity: Float[np.ndarray, "n_sft 3"],
        window: Window,
    ) -> "TemplateBank":
        """The lattice over an f0 interval and the tiling's box.

        Args:
            tiling: The setup's tiling.
            f0_low: Low edge of the interval [Hz].
            f0_high: High edge [Hz].
            velocity: Detector velocity `v/c` at each SFT, every detector.
            window: The window the SFTs were made with.

        Returns:
            The bank over the whole sky lattice.
        """
        prior = tiling.prior
        tsft = tiling.tsft
        f0_centre = 0.5 * (f0_low + f0_high)
        g_sky_diag, sky_eigenvectors = sky_metric(
            velocity, f0=f0_centre, g_f0f0=tiling.g_f0f0
        )
        metrics = tiling.lattice_metric(
            velocity, f0=f0_centre, sky_eigenvectors=sky_eigenvectors
        )
        generators = np.stack(
            [
                triangular_generator(metric, tiling.mismatch, a5star_generator())
                for metric in metrics
            ]
        )
        alpha_low, alpha_high = prior.alpha
        delta_low, delta_high = prior.delta
        is_point_sky = alpha_low == alpha_high and delta_low == delta_high
        # a point sky holds the two sky axes fixed
        n_free_axes = 3 if is_point_sky else 5
        free_metrics = metrics[:, :n_free_axes, :n_free_axes]
        # covering ellipsoid's half-width on each axis, sqrt(mu (g^-1)_kk), per ap block
        halfwidths = np.sqrt(
            tiling.mismatch
            * np.diagonal(np.linalg.inv(free_metrics), axis1=-2, axis2=-1)
        )
        # the ap blocks differ only in T[2, 2:], which scales with their omega^2 step,
        # so the first block's T gives every block's shift fractions
        generator = generators[0]
        f0_step = generator[0, 0]
        depth_step = generator[1, 1]
        # one omega^2 step per ap block
        omega2_steps = generators[:, 2, 2]
        
        f0_halfwidth = halfwidths[0, 0]
        depth_halfwidth = halfwidths[0, 1]
        omega2_halfwidths = halfwidths[:, 2]

        f0_idxs = padded_axis_idxs(
            f0_high - f0_low, step=f0_step, margin=f0_halfwidth
        )
        omega_low = float(omega_from_period(jnp.asarray(prior.period[1])))
        omega_high = float(omega_from_period(jnp.asarray(prior.period[0])))
        depth_low = f0_centre * prior.ap[0] * omega_low
        depth_high = f0_centre * prior.ap[1] * omega_high
        depth_idxs = padded_axis_idxs(
            depth_high - depth_low,
            step=depth_step,
            margin=depth_halfwidth,
        )
        omega2_idxs_per_block = [
            padded_axis_idxs(
                omega_high**2 - omega_low**2,
                step=omega2_step,
                margin=omega2_halfwidth,
            )
            for omega2_step, omega2_halfwidth in zip(
                omega2_steps, omega2_halfwidths, strict=True
            )
        ]
        n_periods_per_block = [block_idxs.size for block_idxs in omega2_idxs_per_block]

        if is_point_sky:
            alpha, delta = np.array([alpha_low]), np.array([delta_low])
            sky_idxs = np.zeros((1, 2), dtype=int)
        else:
            sky_halfwidth = float(halfwidths[0, 3:].max())
            alpha, delta, sky_idxs = sky_lattice(
                generator[3:, 3:],
                sky_eigenvectors=sky_eigenvectors,
                cubic_steps=cubic_step(g_sky_diag[:2], tiling.mismatch_per_axis),
                box=(alpha_low, alpha_high, delta_low, delta_high),
                margin=sky_halfwidth,
            )

        return cls(
            tsft=tsft,
            window=window,
            f0_centre=f0_centre,
            generator=generator,
            f0=f0_low + f0_step * f0_idxs,
            f0_idxs=f0_idxs,
            modulation_depth=(
                depth_low + depth_step * depth_idxs
            ),
            modulation_depth_idxs=depth_idxs,
            omega2=np.concatenate(
                [
                    omega_low**2 + omega2_step * block_idxs
                    for omega2_step, block_idxs in zip(
                        omega2_steps, omega2_idxs_per_block, strict=True
                    )
                ]
            ),
            omega2_idxs=np.concatenate(omega2_idxs_per_block),
            omega2_step=np.repeat(omega2_steps, n_periods_per_block),
            ap_blocks=tiling.ap_blocks,
            ap_block_idxs=np.repeat(
                np.arange(len(tiling.ap_blocks)), n_periods_per_block
            ),
            alpha=alpha,
            delta=delta,
            sky_idxs=sky_idxs,
        )

    @classmethod
    def around_point(
        cls,
        tiling: LatticeTiling,
        point: Parameters,
        *,
        velocity: Float[np.ndarray, "n_sft 3"],
        window: Window,
        n_sky_neighbours: int,
        n_steps: int,
    ) -> "TemplateBank":
        """The search's own lattice points around one source, anchored on the box.

        Args:
            tiling: The lattice, over the box the sources were drawn from.
            point: The parameters searched around.
            velocity: Detector velocity `v/c` at each SFT, every detector.
            window: The window the SFTs were made with.
            n_sky_neighbours: Number of sky lattice points searched around the source.
            n_steps: Number of base grid points each side of the source, per orbital
                axis.

        Returns:
            The bank's subset, its lattice indices and shifts those of the full bank.
        """
        bank = cls.over_f0_interval(
            tiling, *tiling.prior.f0, velocity=velocity, window=window
        )
        # the sky lattice points nearest the source in mismatch, not in angle: near the
        # ecliptic a cell is many times taller in latitude than it is wide
        g_sky_diag, sky_eigenvectors = sky_metric(
            velocity, f0=bank.f0_centre, g_f0f0=tiling.g_f0f0
        )
        lattice_coords = (
            np.asarray(
                sky_direction(jnp.asarray(bank.alpha), jnp.asarray(bank.delta))
            ).T
            @ sky_eigenvectors
        )
        source_coords = (
            np.asarray(
                sky_direction(jnp.asarray([point.alpha]), jnp.asarray([point.delta]))
            ).T
            @ sky_eigenvectors
        )
        sky_mismatch = ((lattice_coords - source_coords) ** 2 * g_sky_diag).sum(axis=1)
        sky_kept = np.argsort(sky_mismatch)[:n_sky_neighbours]

        # the ap block the source's omega^2 points are laid on
        ap_block_highs = [ap_block.ap_high for ap_block in tiling.ap_blocks[:-1]]
        source_ap_block_idx = int(np.searchsorted(ap_block_highs, float(point.ap)))
        block_omega2 = np.flatnonzero(bank.ap_block_idxs == source_ap_block_idx)
        omega2_kept = block_omega2[
            idxs_around(
                bank.omega2[block_omega2], float(point.omega) ** 2, n_steps=n_steps
            )
        ]
        f0_kept = idxs_around(bank.f0, float(point.f0), n_steps=n_steps)
        depth_kept = idxs_around(
            bank.modulation_depth, float(point.modulation_depth), n_steps=n_steps
        )
        return bank._replace(
            f0=bank.f0[f0_kept],
            f0_idxs=bank.f0_idxs[f0_kept],
            modulation_depth=bank.modulation_depth[depth_kept],
            modulation_depth_idxs=bank.modulation_depth_idxs[depth_kept],
            omega2=bank.omega2[omega2_kept],
            omega2_idxs=bank.omega2_idxs[omega2_kept],
            omega2_step=bank.omega2_step[omega2_kept],
            ap_block_idxs=bank.ap_block_idxs[omega2_kept],
            alpha=bank.alpha[sky_kept],
            delta=bank.delta[sky_kept],
            sky_idxs=bank.sky_idxs[sky_kept],
        )
