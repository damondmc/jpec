"""Template bank: subgrids on a sky lattice, over an f0 interval or around a point."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int

from jpec.bank.lattice import OrbitalSubgrid
from jpec.bank.metric import sky_metric
from jpec.bank.tiling import LatticeTiling
from jpec.constant import ARC_PAD_BINS, PHASE_OVERSAMPLE
from jpec.detector.doppler import sky_direction
from jpec.params import Parameters, ap_from_modulation_depth_bins, omega_from_period
from jpec.waveform.harmonics import (
    harmonic_count,
    phase_sample_count,
    template_harmonics,
)
from jpec.waveform.plane import arc_bin_offsets
from jpec.waveform.window import Window


@partial(jax.jit, static_argnames=("n_theta", "j_max", "window", "batch_size"))
def template_harmonics_at_depths(
    bin_offsets: Float[Array, " n_freq"],
    modulation_depth_bins: Float[Array, " n_modulation_depth"],
    *,
    n_theta: int,
    j_max: int,
    window: Window,
    batch_size: int,
) -> Complex[Array, "n_modulation_depth n_harmonic n_freq"]:
    """`tau[j, k]` at every modulation depth on one grid, `batch_size` at a time.

    Args:
        bin_offsets: Bin centres, as offsets from f0 [bins].
        modulation_depth_bins: The modulation depths [bins].
        n_theta: Orbital phases sampled.
        j_max: Highest harmonic kept.
        window: The window the SFTs were made with.
        batch_size: Modulation depths built at once, which caps the planes held.

    Returns:
        `tau[j, k]` for every modulation depth, in their order.
    """

    def at_depth(
        depth_bins: Float[Array, ""],
    ) -> Complex[Array, "n_harmonic n_freq"]:
        return template_harmonics(
            bin_offsets,
            modulation_depth_bins=depth_bins,
            n_theta=n_theta,
            j_max=j_max,
            window=window,
        )

    return jax.lax.map(at_depth, modulation_depth_bins, batch_size=batch_size)


class OrbitalTemplateGrid(NamedTuple):
    """One orbital subgrid and what its template harmonics are built from.

    The same at every sky point.

    Attributes:
        orbital_subgrid: The orbital subgrid.
        tsft: SFT length [s].
        window: The window the SFTs were made with.
    """

    orbital_subgrid: OrbitalSubgrid
    tsft: float
    window: Window

    @property
    def j_max(self) -> int:
        """Highest harmonic kept, set by the deepest modulation depth."""
        return harmonic_count(float(self.orbital_subgrid.modulation_depth_bins[-1]))

    @property
    def n_phi(self) -> int:
        """Orbital phases the statistic is evaluated at."""
        return phase_sample_count(self.j_max, oversample=PHASE_OVERSAMPLE)

    @property
    def bin_offsets(self) -> Float[Array, " n_freq"]:
        """Bin centres spanning the deepest modulation depth, offsets from f0 [bins]."""
        return arc_bin_offsets(
            float(self.orbital_subgrid.modulation_depth_bins[-1]), pad_bins=ARC_PAD_BINS
        )

    def template_harmonics(
        self, *, sub_bin_offset: float, batch_size: int
    ) -> Complex[Array, "n_modulation_depth n_harmonic n_freq"]:
        """`tau[j, k]` at every modulation depth on the phase grid the statistic uses.

        Args:
            sub_bin_offset: Bins from the window's origin to the f0 points the
                templates serve.
            batch_size: Modulation depths built at once.

        Returns:
            `tau[j, k]` for every modulation depth of the subgrid.
        """
        bin_offsets = self.bin_offsets - sub_bin_offset
        return template_harmonics_at_depths(
            bin_offsets,
            jnp.asarray(self.orbital_subgrid.modulation_depth_bins),
            n_theta=self.n_phi,
            j_max=self.j_max,
            window=self.window,
            batch_size=batch_size,
        )

    def template_harmonics_at_offsets(
        self,
        sub_bin_offsets: Float[np.ndarray, " n_sub_bin_offset"],
        *,
        batch_size: int,
    ) -> Complex[Array, "n_sub_bin_offset n_modulation_depth n_harmonic n_freq"]:
        """`tau[j, k]` at every modulation depth, one stack per sub-bin offset.

        Args:
            sub_bin_offsets: Bins from the window's origin to the f0 points each
                stack serves.
            batch_size: Modulation depths built at once.

        Returns:
            template_harmonics() at each offset, in their order.
        """
        return jnp.stack(
            [
                self.template_harmonics(
                    sub_bin_offset=float(sub_bin_offset), batch_size=batch_size
                )
                for sub_bin_offset in sub_bin_offsets
            ]
        )


def idxs_around(
    values: Float[np.ndarray, " n_axis_point"], value: float, *, n_steps: int
) -> Int[np.ndarray, " n_kept"]:
    """Indices of the `n_steps` points each side of the one nearest `value`.

    They are clipped to the axis.

    Args:
        values: One axis of a subgrid, increasing.
        value: The point searched around.
        n_steps: Points kept each side.

    Returns:
        The indices, increasing.
    """
    nearest_idx = int(np.argmin(np.abs(values - value)))
    return np.arange(
        max(nearest_idx - n_steps, 0), min(nearest_idx + n_steps + 1, values.size)
    )


class TemplateBank(NamedTuple):
    """One job's templates: every orbital subgrid of the setup over its sky points.

    Attributes:
        template_grids: One per subgrid, in the order of `BCC_SUBGRID_OFFSETS`.
        alpha: Right ascension of each sky point [rad].
        delta: Declination of each sky point [rad].
    """

    template_grids: list[OrbitalTemplateGrid]
    alpha: Float[np.ndarray, " n_sky"]
    delta: Float[np.ndarray, " n_sky"]

    @property
    def n_templates(self) -> int:
        """Templates in the box, over every subgrid and the job's sky points."""
        n_templates_per_sky_point = sum(
            template_grid.orbital_subgrid.n_templates_per_sky_point
            for template_grid in self.template_grids
        )
        return n_templates_per_sky_point * int(self.alpha.size)

    def parameters(
        self,
        *,
        subgrid_idx: int,
        sky_idx: int,
        f0_idx: int,
        period_idx: int,
        modulation_depth_idx: int,
        phase_idx: int,
    ) -> Parameters:
        """The template the bank's indices point at.

        Args:
            subgrid_idx: Index of the subgrid.
            sky_idx: Index of the sky point.
            f0_idx: Index of the f0 point.
            period_idx: Index of the trial period.
            modulation_depth_idx: Index of the modulation depth.
            phase_idx: Index of the orbital phase, of the subgrid's `n_phi`.

        Returns:
            The template's parameters.
        """
        template_grid = self.template_grids[subgrid_idx]
        f0 = float(template_grid.orbital_subgrid.f0[f0_idx])
        period = float(template_grid.orbital_subgrid.period[period_idx])
        modulation_depth_bins = float(
            template_grid.orbital_subgrid.modulation_depth_bins[modulation_depth_idx]
        )
        ap = ap_from_modulation_depth_bins(
            modulation_depth_bins,
            f0=f0,
            omega=2.0 * np.pi / period,
            tsft=template_grid.tsft,
        )
        return Parameters(
            f0=f0,
            ap=float(ap),
            period=period,
            phi_b=2.0 * np.pi * phase_idx / template_grid.n_phi,
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
        """An f0 interval's subgrids over the sky lattice.

        Args:
            tiling: The setup's tiling.
            f0_low: Low edge of the interval [Hz].
            f0_high: High edge [Hz].
            velocity: Detector velocity `v/c` at each SFT, every detector, which
                sets the sky lattice.
            window: The window the SFTs were made with.

        Returns:
            The bank, its subgrids over the whole sky lattice.
        """
        alpha, delta = tiling.sky_points(velocity, f0=0.5 * (f0_low + f0_high))
        template_grids = [
            OrbitalTemplateGrid(
                orbital_subgrid=orbital_subgrid, tsft=tiling.tsft, window=window
            )
            for orbital_subgrid in tiling.orbital_subgrids(f0_low, f0_high)
        ]
        return cls(template_grids=template_grids, alpha=alpha, delta=delta)

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
            n_sky_neighbours: Sky lattice points searched around the source.
            n_steps: Lattice steps each side of the source, per orbital axis.

        Returns:
            The bank, each subgrid's `in_box` as the search has it.
        """
        # the sky lattice points nearest the source in mismatch, not in angle: near the
        # ecliptic a cell is many times taller in latitude than it is wide
        alpha, delta = tiling.sky_points(velocity, f0=float(point.f0))
        curvature, sky_eigenvectors = sky_metric(
            velocity, f0=float(point.f0), g_f0f0=tiling.g_f0f0
        )
        lattice_x = (
            np.asarray(sky_direction(jnp.asarray(alpha), jnp.asarray(delta))).T
            @ sky_eigenvectors
        )
        source_x = (
            np.asarray(sky_direction(point.alpha, point.delta)) @ sky_eigenvectors
        )
        sky_mismatch = ((lattice_x - source_x) ** 2 * curvature).sum(axis=1)
        nearest_sky_idxs = np.argsort(sky_mismatch)[:n_sky_neighbours]

        # the ap block the source's periods are laid on
        ap_block_highs = [ap_block.ap_high for ap_block in tiling.ap_blocks[:-1]]
        source_ap_block_idx = int(np.searchsorted(ap_block_highs, float(point.ap)))
        modulation_depth_bins = float(point.modulation_depth_bins(tiling.tsft))
        omega2 = float(point.omega) ** 2

        template_grids = []
        for orbital_subgrid in tiling.orbital_subgrids(*tiling.prior.f0):
            f0_idxs = idxs_around(orbital_subgrid.f0, float(point.f0), n_steps=n_steps)
            modulation_depth_idxs = idxs_around(
                orbital_subgrid.modulation_depth_bins,
                modulation_depth_bins,
                n_steps=n_steps,
            )
            ap_block_period_idxs = np.flatnonzero(
                orbital_subgrid.ap_block_idxs == source_ap_block_idx
            )
            grid_omega2 = (
                np.asarray(
                    omega_from_period(
                        jnp.asarray(orbital_subgrid.period[ap_block_period_idxs])
                    )
                )
                ** 2
            )
            period_idxs = ap_block_period_idxs[
                idxs_around(grid_omega2, omega2, n_steps=n_steps)
            ]
            source_orbital_subgrid = OrbitalSubgrid(
                f0=orbital_subgrid.f0[f0_idxs],
                modulation_depth_bins=orbital_subgrid.modulation_depth_bins[
                    modulation_depth_idxs
                ],
                period=orbital_subgrid.period[period_idxs],
                ap_block_idxs=orbital_subgrid.ap_block_idxs[period_idxs],
                in_box=orbital_subgrid.in_box[
                    np.ix_(period_idxs, modulation_depth_idxs)
                ],
            )
            template_grids.append(
                OrbitalTemplateGrid(
                    orbital_subgrid=source_orbital_subgrid,
                    tsft=tiling.tsft,
                    window=window,
                )
            )
        return cls(
            template_grids=template_grids,
            alpha=alpha[nearest_sky_idxs],
            delta=delta[nearest_sky_idxs],
        )
