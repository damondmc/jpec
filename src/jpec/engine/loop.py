"""The search loop: a template bank searched on one band, sky batch by sky batch."""

from collections.abc import Iterator
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float, Int

from jpec.bank.template_bank import TemplateBank
from jpec.config import SearchLoopConfig
from jpec.detector.doppler import solar_roemer_delay
from jpec.engine.candidates import TopTemplates
from jpec.engine.data import SearchData
from jpec.engine.layout import TemplateBatch, bank_layout, f0_idxs_per_toplist
from jpec.engine.match import (
    matched_filter_grid,
    placed_template_harmonics,
    template_harmonics_at_depths,
    template_norms_squared,
    top_k_templates,
)
from jpec.engine.transform import (
    data_harmonics_per_sky_point,
    weighted_excess_power_per_reference,
)
from jpec.stat.matched_filter import orbital_phase_mod_2pi


class SkyShifts(NamedTuple):
    """One sky point's grid shifts, depth classes and ap mask, per template batch.

    Attributes:
        whole_bin_shifts: Whole bins each (period, modulation depth) moves its f0
            grid by.
        subbin_shifts: Sub-bin offsets each (period, modulation depth) moves its f0
            grid by, past the whole bins.
        depth_classes: Depth class of each period.
        ap_in_range: Whether each (period, modulation depth) has ap in its ap block.
    """

    whole_bin_shifts: Int[
        Array, "n_batch n_batch_period n_batch_depth"
    ]
    subbin_shifts: Int[Array, "n_batch n_batch_period n_batch_depth"]
    depth_classes: Int[Array, "n_batch n_batch_period"]
    ap_in_range: Bool[Array, "n_batch n_batch_period n_batch_depth"]


class BandSearch:
    """A template bank searched on one loaded band.

    Attributes:
        search_data: The loaded band searched.
        config: How the search loop batches its work.
        n_top: Number of templates each match keeps per toplist.
        layout: Where the bank's templates read from the band, and their batches.
        toplist_f0_idxs: The base f0 points of each toplist, padded with `n_f0`.
        placed_template_harmonics: Each template's harmonics `tau[j, k]` at every
            sub-bin offset and whole-bin placement in a window `n_pad` bins wider,
            zero outside it, on the device.
        template_norms_squared: Each template's squared norm over its first
            harmonics, harmonic by harmonic, on the device.
        orbital_phase: Orbital phase in detector time at each trial period and SFT,
            per omega^2 class [rad], on the device.
        omega: Orbital angular frequency of each trial period, per omega^2 class
            [rad/s], on the device.
        f0_doppler_references: Frequency each base f0 point is Doppler-corrected at
            [Hz].
        ap_in_range_per_classes: Each template batch's ap mask, per combination of a
            sky point's classes, filled as sky points need them.
    """

    def __init__(
        self,
        bank: TemplateBank,
        search_data: SearchData,
        config: SearchLoopConfig,
        *,
        n_top: int,
        toplist_f0_edges: Float[np.ndarray, " n_toplist_edge"],
    ) -> None:
        """Lay a bank out on a band and build its template harmonics and phases.

        Args:
            bank: The template bank searched.
            search_data: The loaded band searched.
            config: How the search loop batches its work.
            n_top: Number of templates each match keeps per toplist.
            toplist_f0_edges: Edges of the toplists' f0 intervals [Hz], low to high.
        """
        self.search_data = search_data
        self.config = config
        self.n_top = n_top
        self.layout = bank_layout(
            bank,
            freq_low=search_data.freq_low,
            n_band_bins=search_data.n_band_bins,
            n_subbin_offsets=config.n_subbin_offsets,
            period_batch_size=config.period_batch_size,
            harmonic_fraction=config.harmonic_fraction,
            phase_mismatch=config.phase_mismatch,
            doppler_mismatch=config.doppler_mismatch,
        )
        layout = self.layout
        self.toplist_f0_idxs = jnp.asarray(
            f0_idxs_per_toplist(bank.f0, toplist_f0_edges)
        )
        # every depth of every depth class, in bins, one row per class
        depth_classes = np.arange(bank.depth_shift_fractions.size)
        depth_bins = bank.modulation_depth_at(depth_classes) * bank.tsft
        # tau[j, k] per (sub-bin offset, depth class, modulation depth)
        templates = jnp.stack(
            [
                template_harmonics_at_depths(
                    jnp.asarray(layout.bin_offsets - subbin_offset),
                    jnp.asarray(depth_bins.ravel()),
                    n_theta=layout.n_theta,
                    j_max=layout.j_max,
                    window=bank.window,
                    batch_size=config.modulation_depth_batch_size,
                ).reshape(*depth_bins.shape, layout.j_max, -1)
                for subbin_offset in layout.subbin_offsets
            ]
        )
        self.placed_template_harmonics = placed_template_harmonics(
            templates, n_pad=layout.n_pad
        )
        self.template_norms_squared = template_norms_squared(templates)
        self.orbital_phase = jnp.asarray(
            orbital_phase_mod_2pi(search_data.times, layout.omega)
        )
        self.omega = jnp.asarray(layout.omega)
        self.f0_doppler_references = np.asarray(layout.doppler_references)[
            np.asarray(layout.doppler_reference_idxs)
        ]
        self.ap_in_range_per_classes: dict[
            tuple[int, ...],
            Bool[Array, "n_batch n_batch_period n_batch_depth"],
        ] = {}

    def sky_shifts(self, sky_idx: int) -> SkyShifts:
        """How one sky point moves the base grids.

        Args:
            sky_idx: Index of the sky point in the bank.

        Returns:
            The shifts, over every template batch.
        """
        layout = self.layout
        bank = layout.bank
        n_offsets = layout.subbin_offsets.size
        # the f0 shift in sub-bin units u: u // n whole bins and u % n sub-bin offsets
        units = np.rint(bank.f0_shift(sky_idx) * bank.tsft * n_offsets).astype(
            np.int32
        )
        units = np.stack(layout.padded_runs(units, 0))
        omega2_class = int(bank.omega2_class[sky_idx])
        depth_class = bank.depth_class(sky_idx)
        # the mask depends on the sky point only through its classes
        classes = (omega2_class, *depth_class.tolist())
        if classes not in self.ap_in_range_per_classes:
            self.ap_in_range_per_classes[classes] = jnp.asarray(
                layout.ap_in_range(omega2_class, depth_class)
            )
        return SkyShifts(
            whole_bin_shifts=jnp.asarray(units // n_offsets),
            subbin_shifts=jnp.asarray(units % n_offsets),
            depth_classes=jnp.asarray(
                np.stack(layout.padded_runs(depth_class.astype(np.int32), 0))
            ),
            ap_in_range=self.ap_in_range_per_classes[classes],
        )

    def doppler_correct(
        self, sky_idx: int
    ) -> Float[Array, "n_ref n_sft n_transformed_bin"]:
        """Correct the band toward one sky point at every Doppler reference.

        Args:
            sky_idx: Index of the sky point in the bank.

        Returns:
            The weighted excess power at each Doppler reference, over the bins the
                transform reads.
        """
        search_data = self.search_data
        layout = self.layout
        alpha = float(layout.bank.alpha[sky_idx])
        delta = float(layout.bank.delta[sky_idx])
        return weighted_excess_power_per_reference(
            sft_time_series=search_data.sft_time_series,
            floor=search_data.floor,
            velocity_per_detector=search_data.velocity_per_detector,
            doppler_references=layout.doppler_references,
            transformed_start_idxs=layout.transformed_start_idxs,
            n_transformed_bins=layout.n_transformed_bins,
            tsft=search_data.tsft,
            alpha=alpha,
            delta=delta,
            sft_weights=search_data.sft_weights(alpha, delta),
            mean_power=search_data.mean_power,
        )

    def transform(
        self,
        weighted_excess_power: Float[
            Array, "n_sky n_ref n_sft n_transformed_bin"
        ],
        solar_roemer_delays: Float[Array, "n_sky n_sft"],
        omega2_class: int,
        template_batch: TemplateBatch,
    ) -> Complex[
        Array, "n_sky n_ref n_batch_period n_harmonic n_transformed_bin"
    ]:
        """Transform a sky batch at one template batch's periods.

        Args:
            weighted_excess_power: The Doppler-corrected band of each sky point.
            solar_roemer_delays: Solar System Roemer delay at each sky point and SFT
                [s].
            omega2_class: Omega^2 class of the sky batch's points.
            template_batch: The template batch whose periods the transform runs at.

        Returns:
            `F[j, k]` at each sky point, Doppler reference and trial period.
        """
        return data_harmonics_per_sky_point(
            weighted_excess_power,
            self.orbital_phase,
            self.omega,
            solar_roemer_delays,
            omega2_class,
            template_batch.period_start_idx,
            period_batch_size=self.layout.period_batch_size,
            j_max=template_batch.n_harmonics,
        )

    def match(
        self,
        data_harmonics: Complex[
            Array, "n_ref n_batch_period n_harmonic n_transformed_bin"
        ],
        sky_shifts: SkyShifts,
        batch_idx: int,
    ) -> tuple[
        Float[Array, "n_toplist n_kept"],
        Int[Array, "n_toplist n_kept"],
        Int[Array, "n_toplist n_kept"],
    ]:
        """Match one template batch at one sky point and keep each toplist's top.

        Args:
            data_harmonics: `F[j, k]` of the batch's periods at the sky point, per
                Doppler reference.
            sky_shifts: How the sky point moves the base grids.
            batch_idx: Index of the template batch.

        Returns:
            Per toplist, the top statistics, their flat index into the batch's
                templates, and their orbital phase index.
        """
        layout = self.layout
        max_statistic, phase_idxs = matched_filter_grid(
            data_harmonics=data_harmonics,
            placed_harmonics=self.placed_template_harmonics,
            norms_squared=self.template_norms_squared,
            window_start_idxs=layout.window_start_idxs,
            group_f0_idxs=layout.group_f0_idxs,
            group_doppler_reference_idxs=layout.group_doppler_reference_idxs,
            group_subbin_offset_idxs=layout.group_subbin_offset_idxs,
            modulation_depth_start_idx=layout.template_batches[
                batch_idx
            ].modulation_depth_start_idx,
            whole_bin_shifts=sky_shifts.whole_bin_shifts[batch_idx],
            subbin_shifts=sky_shifts.subbin_shifts[batch_idx],
            depth_classes=sky_shifts.depth_classes[batch_idx],
            n_batch_depths=layout.n_batch_depths,
            n_phi=layout.n_phi,
            f0_batch_size=self.config.f0_batch_size,
        )
        return top_k_templates(
            max_statistic,
            phase_idxs,
            sky_shifts.ap_in_range[batch_idx],
            self.toplist_f0_idxs,
            n_top=self.n_top,
        )

    def _unpack_top_k(
        self,
        top_k: tuple[
            Float[Array, "n_toplist n_kept"],
            Int[Array, "n_toplist n_kept"],
            Int[Array, "n_toplist n_kept"],
        ],
        batch_idx: int,
        sky_idx: int,
    ) -> Iterator[TopTemplates]:
        """Yield one match's top templates, one TopTemplates per toplist.

        Args:
            top_k: Per toplist, the top statistics, their flat index into the
                batch's templates, and their orbital phase index.
            batch_idx: Index of the template batch matched.
            sky_idx: Index of the sky point in the bank.

        Yields:
            The toplist's top templates, as indices on the bank's base grids.
        """
        layout = self.layout
        template_batch = layout.template_batches[batch_idx]
        n_f0 = layout.bank.f0.size
        max_statistic, flat_idxs, phase_idxs = (np.asarray(array) for array in top_k)
        f0_idxs, period_offsets, modulation_depth_offsets = np.unravel_index(
            flat_idxs,
            (n_f0, layout.period_batch_size, layout.n_batch_depths),
        )
        for toplist_idx in range(max_statistic.shape[0]):
            yield TopTemplates(
                sky_idx=sky_idx,
                toplist_idx=toplist_idx,
                max_statistic=max_statistic[toplist_idx],
                f0_idxs=f0_idxs[toplist_idx],
                period_idxs=template_batch.period_start_idx
                + period_offsets[toplist_idx],
                modulation_depth_idxs=template_batch.modulation_depth_start_idx
                + modulation_depth_offsets[toplist_idx],
                phase_idxs=phase_idxs[toplist_idx],
                doppler_references=self.f0_doppler_references[f0_idxs[toplist_idx]],
                n_phi=layout.n_phi,
            )

    def search_sky_batch(
        self, sky_idxs: Int[np.ndarray, " n_sky"]
    ) -> Iterator[TopTemplates]:
        """Search a batch of sky points and yield each match's top templates.

        Args:
            sky_idxs: Indices of the sky points in the bank, all of one omega^2 class.

        Yields:
            One TopTemplates per (template batch, sky point, toplist).
        """
        bank = self.layout.bank
        omega2_class = int(bank.omega2_class[sky_idxs[0]])
        solar_roemer_delays = solar_roemer_delay(
            self.search_data.position,
            alpha=jnp.asarray(bank.alpha[sky_idxs]),
            delta=jnp.asarray(bank.delta[sky_idxs]),
        )
        weighted_excess_power = jnp.stack(
            [self.doppler_correct(int(sky_idx)) for sky_idx in sky_idxs]
        )
        sky_shifts = [self.sky_shifts(int(sky_idx)) for sky_idx in sky_idxs]
        previous_match = None
        for batch_idx, template_batch in enumerate(self.layout.template_batches):
            data_harmonics = self.transform(
                weighted_excess_power, solar_roemer_delays, omega2_class, template_batch
            )
            for batch_sky_idx, sky_idx in enumerate(sky_idxs):
                top_k = self.match(
                    data_harmonics[batch_sky_idx], sky_shifts[batch_sky_idx], batch_idx
                )
                for array in top_k:
                    array.copy_to_host_async()
                if previous_match is not None:
                    # work on the host while the device runs this match
                    yield from self._unpack_top_k(*previous_match)
                previous_match = (top_k, batch_idx, int(sky_idx))
        # the last match
        if previous_match is not None:
            yield from self._unpack_top_k(*previous_match)
