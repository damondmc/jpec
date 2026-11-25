"""The search loop: every template at a batch of sky points scored on one band."""

from collections.abc import Iterator, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float, Int

from jpec.bank.template_bank import TemplateBank
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
from jpec.stat.matched_filter import barycenter_orbital_phase, orbital_phase_mod_2pi


class SkyPoint(NamedTuple):
    """One sky point as the loop matches it, on the device.

    Attributes:
        sky_idx: Index of the sky point in the bank.
        start_shifts: Whole bins each (period, modulation depth) of each template
            batch moves the f0 grid by.
        sub_bin_shifts: Sub-bin offsets each moves it by.
        depth_classes: The depth class of each period of each template batch.
        ap_in_range: Whether each (period, modulation depth) of each template batch
            has ap in range.
    """

    sky_idx: int
    start_shifts: Int[Array, "n_template_batch n_period n_matched_modulation_depth"]
    sub_bin_shifts: Int[Array, "n_template_batch n_period n_matched_modulation_depth"]
    depth_classes: Int[Array, "n_template_batch n_period"]
    ap_in_range: Bool[Array, "n_template_batch n_period n_matched_modulation_depth"]


class HarmonicBank:
    """A bank as the loop matches it: its band layout, template harmonics and phases.

    Built once for a band; a sky point adds its shifts, depth classes and ap mask.

    Attributes:
        bank: The bank.
        layout: The bank's layout on the band.
        placed_template_harmonics: placed_template_harmonics() of the layout's
            templates, on the device.
        template_norms_squared: template_norms_squared() of the same templates, on
            the device.
        orbital_phase: Orbital phase at each trial period and SFT per omega^2 class
            [rad], on the device.
        omega: `layout.omega`, on the device.
        f0_doppler_references: The Doppler reference of each f0 point [Hz].
        ap_in_range_per_classes: layout.ap_in_range() per (omega^2 class, depth
            classes), on the device.
    """

    def __init__(
        self,
        bank: TemplateBank,
        *,
        times: Float[np.ndarray, " n_sft"],
        freq_low: float,
        n_band_bins: int,
        n_sub_bin_offsets: int,
        period_batch_size: int,
        harmonic_fraction: float,
        phase_mismatch: float,
        doppler_mismatch: float,
        modulation_depth_batch_size: int,
    ) -> None:
        """Lay a bank out on a band and build its template harmonics and phases.

        Args:
            bank: The bank.
            times: SFT midpoints from the epoch `phi_b` is quoted at [s].
            freq_low: First frequency of the band [Hz].
            n_band_bins: Number of bins the band holds.
            n_sub_bin_offsets: Number of sub-bin template offsets across one bin.
            period_batch_size: Number of trial periods per transform call.
            harmonic_fraction: Share of each modulation depth's harmonics matched.
            phase_mismatch: Largest mismatch the orbital phase grid may leave.
            doppler_mismatch: Largest mismatch an f0 point's offset from its Doppler
                reference may leave.
            modulation_depth_batch_size: Number of modulation depths whose templates
                are built at once.
        """
        self.bank = bank
        self.layout = bank_layout(
            bank,
            freq_low=freq_low,
            n_band_bins=n_band_bins,
            n_sub_bin_offsets=n_sub_bin_offsets,
            period_batch_size=period_batch_size,
            harmonic_fraction=harmonic_fraction,
            phase_mismatch=phase_mismatch,
            doppler_mismatch=doppler_mismatch,
        )
        layout = self.layout
        # tau[j, k] per (sub-bin offset, depth class, modulation depth)
        templates = jnp.stack(
            [
                jnp.stack(
                    [
                        template_harmonics_at_depths(
                            jnp.asarray(layout.bin_offsets - sub_bin_offset),
                            jnp.asarray(
                                bank.modulation_depth_at(depth_class) * bank.tsft
                            ),
                            n_theta=layout.n_theta,
                            j_max=layout.j_max,
                            window=bank.window,
                            batch_size=modulation_depth_batch_size,
                        )
                        for depth_class in range(bank.depth_shift_fractions.size)
                    ]
                )
                for sub_bin_offset in layout.sub_bin_offsets
            ]
        )
        self.placed_template_harmonics = placed_template_harmonics(
            templates, n_pad=layout.n_pad
        )
        self.template_norms_squared = template_norms_squared(templates)
        self.orbital_phase = jnp.asarray(orbital_phase_mod_2pi(times, layout.omega))
        self.omega = jnp.asarray(layout.omega)
        self.f0_doppler_references = np.asarray(layout.doppler_references)[
            np.asarray(layout.doppler_reference_idxs)
        ]
        self.ap_in_range_per_classes: dict[
            tuple[int, ...],
            Bool[Array, "n_template_batch n_period n_matched_modulation_depth"],
        ] = {}

    def sky_point(self, sky_idx: int) -> SkyPoint:
        """One sky point's shifts, depth classes and ap mask, on the device.

        Args:
            sky_idx: Index of the sky point in the bank.

        Returns:
            The sky point, over every template batch.
        """
        layout = self.layout
        n_offsets = layout.sub_bin_offsets.size
        # the f0 shift in sub-bin units u: u // n whole bins and u % n sub-bin offsets
        units = np.rint(
            self.bank.f0_shift(sky_idx) * self.bank.tsft * n_offsets
        ).astype(np.int32)
        units = np.stack(layout.padded_runs(units, 0))
        omega2_class = int(self.bank.omega2_class[sky_idx])
        depth_class = self.bank.depth_class(sky_idx)
        # the mask depends on the sky point only through its classes
        classes = (omega2_class, *depth_class.tolist())
        if classes not in self.ap_in_range_per_classes:
            self.ap_in_range_per_classes[classes] = jnp.asarray(
                layout.ap_in_range(omega2_class, depth_class)
            )
        return SkyPoint(
            sky_idx=sky_idx,
            start_shifts=jnp.asarray(units // n_offsets),
            sub_bin_shifts=jnp.asarray(units % n_offsets),
            depth_classes=jnp.asarray(
                np.stack(layout.padded_runs(depth_class.astype(np.int32), 0))
            ),
            ap_in_range=self.ap_in_range_per_classes[classes],
        )


def match_at_sky_point(
    harmonic_bank: HarmonicBank,
    data_harmonics: Complex[
        Array, "n_doppler_reference n_period n_harmonic n_transformed_bin"
    ],
    sky_point: SkyPoint,
    template_batch: TemplateBatch,
    batch_idx: int,
    *,
    f0_batch_size: int,
) -> tuple[
    Float[Array, "n_f0 n_period n_matched_modulation_depth"],
    Int[Array, "n_f0 n_period n_matched_modulation_depth"],
]:
    """The match of one template batch at one sky point, over every f0 point.

    Args:
        harmonic_bank: The bank's template harmonics, laid out on the band.
        data_harmonics: `F[j, k]` of the batch's periods at the sky point, per
            Doppler reference.
        sky_point: The sky point's shifts, depth classes and ap mask.
        template_batch: The template batch matched.
        batch_idx: Index of the template batch in `layout.template_batches`.
        f0_batch_size: Number of f0 points matched at once.

    Returns:
        The maximum of `S` over the orbital phase and the phase index it sits at, per
            (f0, period, modulation depth) of the batch.
    """
    layout = harmonic_bank.layout
    return matched_filter_grid(
        data_harmonics=data_harmonics,
        placed_harmonics=harmonic_bank.placed_template_harmonics,
        norms_squared=harmonic_bank.template_norms_squared,
        window_start_idxs=layout.window_start_idxs,
        group_f0_idxs=layout.group_f0_idxs,
        group_doppler_reference_idxs=layout.group_doppler_reference_idxs,
        group_sub_bin_offset_idxs=layout.group_sub_bin_offset_idxs,
        modulation_depth_start_idx=template_batch.modulation_depth_start_idx,
        start_shifts=sky_point.start_shifts[batch_idx],
        sub_bin_shifts=sky_point.sub_bin_shifts[batch_idx],
        depth_classes=sky_point.depth_classes[batch_idx],
        n_matched_modulation_depths=layout.n_window_modulation_depths,
        n_phi=layout.n_phi,
        f0_batch_size=f0_batch_size,
    )


def top_templates(
    top_k: tuple[
        Float[Array, "n_toplist n_kept"],
        Int[Array, "n_toplist n_kept"],
        Int[Array, "n_toplist n_kept"],
    ],
    harmonic_bank: HarmonicBank,
    template_batch: TemplateBatch,
    sky_idx: int,
) -> Iterator[TopTemplates]:
    """The top templates inside the parameter space, toplist by toplist.

    Args:
        top_k: The statistic, flat grid index and phase index top_k_templates()
            returns.
        harmonic_bank: The bank the templates belong to.
        template_batch: The template batch they were matched over.
        sky_idx: Index of the sky point in the bank.

    Yields:
        One TopTemplates per toplist.
    """
    layout = harmonic_bank.layout
    n_f0 = harmonic_bank.bank.f0.size
    max_statistic, flat_idxs, phase_idxs = (np.asarray(array) for array in top_k)
    f0_idxs, period_offsets, modulation_depth_offsets = np.unravel_index(
        flat_idxs,
        (n_f0, layout.period_batch_size, layout.n_window_modulation_depths),
    )
    for toplist_idx in range(max_statistic.shape[0]):
        yield TopTemplates(
            sky_idx=sky_idx,
            toplist_idx=toplist_idx,
            max_statistic=max_statistic[toplist_idx],
            f0_idxs=f0_idxs[toplist_idx],
            period_idxs=template_batch.period_start_idx + period_offsets[toplist_idx],
            modulation_depth_idxs=template_batch.modulation_depth_start_idx
            + modulation_depth_offsets[toplist_idx],
            phase_idxs=phase_idxs[toplist_idx],
            doppler_references=harmonic_bank.f0_doppler_references[
                f0_idxs[toplist_idx]
            ],
            n_phi=layout.n_phi,
        )


def top_templates_at_sky_points(
    search_data: SearchData,
    harmonic_bank: HarmonicBank,
    *,
    sky_idxs: Sequence[int] | Int[np.ndarray, " n_batch_sky"],
    f0_batch_size: int,
    n_top: int,
    toplist_f0_edges: Float[np.ndarray, " n_toplist + 1"],
) -> Iterator[TopTemplates]:
    """Every template at a batch of sky points scored, each toplist's top kept.

    Args:
        search_data: The band searched, the one the bank is laid out on.
        harmonic_bank: The bank's template harmonics, laid out on the band.
        sky_idxs: Indices of the sky points in the bank, all of one omega^2 class,
            as layout.sky_batches() gives them.
        f0_batch_size: Number of f0 points matched at once.
        n_top: Number of templates kept per toplist from each match.
        toplist_f0_edges: Edges of the toplists' f0 intervals [Hz], low to high.

    Yields:
        One TopTemplates per (template batch, sky point, toplist).
    """
    bank = harmonic_bank.bank
    layout = harmonic_bank.layout
    sky_idx_array = np.asarray(sky_idxs)
    omega2_class = int(bank.omega2_class[sky_idx_array[0]])
    class_orbital_phase = harmonic_bank.orbital_phase[omega2_class]
    class_omega = harmonic_bank.omega[omega2_class]
    solar_roemer_delays = solar_roemer_delay(
        search_data.position,
        alpha=jnp.asarray(bank.alpha[sky_idx_array]),
        delta=jnp.asarray(bank.delta[sky_idx_array]),
    )
    # the f0 idxs each toplist holds
    toplist_f0_idxs = jnp.asarray(f0_idxs_per_toplist(bank.f0, toplist_f0_edges))
    # Doppler correction at every sky point, with its SFT weights
    weighted_excess_power = jnp.stack(
        [
            weighted_excess_power_per_reference(
                sft_time_series=search_data.sft_time_series,
                floor=search_data.floor,
                velocity_per_detector=search_data.velocity_per_detector,
                doppler_references=layout.doppler_references,
                transformed_start_idxs=layout.transformed_start_idxs,
                n_transformed_bins=layout.n_transformed_bins,
                tsft=search_data.tsft,
                alpha=float(bank.alpha[sky_idx]),
                delta=float(bank.delta[sky_idx]),
                sft_weights=search_data.sft_weights(
                    float(bank.alpha[sky_idx]), float(bank.delta[sky_idx])
                ),
                mean_power=search_data.mean_power,
            )
            for sky_idx in sky_idxs
        ]
    )
    sky_points = [harmonic_bank.sky_point(int(sky_idx)) for sky_idx in sky_idxs]
    previous_match = None
    for batch_idx, template_batch in enumerate(layout.template_batches):
        batch_orbital_phase = jax.lax.dynamic_slice_in_dim(
            class_orbital_phase,
            template_batch.period_start_idx,
            layout.period_batch_size,
        )
        batch_omega = jax.lax.dynamic_slice_in_dim(
            class_omega,
            template_batch.period_start_idx,
            layout.period_batch_size,
        )
        # one transform for every sky point and Doppler reference of the batch
        batch_barycenter_orbital_phase = barycenter_orbital_phase(
            batch_orbital_phase, batch_omega, solar_roemer_delays
        )
        # F[j, k] at every sky point and reference
        batch_data_harmonics = data_harmonics_per_sky_point(
            weighted_excess_power,
            batch_barycenter_orbital_phase,
            j_max=template_batch.n_harmonics,
        )
        for batch_sky_idx, sky_point in enumerate(sky_points):
            # one call for every f0 point of the batch, at the sky point's shifts
            max_statistic, phase_idxs = match_at_sky_point(
                harmonic_bank,
                batch_data_harmonics[batch_sky_idx],
                sky_point,
                template_batch,
                batch_idx,
                f0_batch_size=f0_batch_size,
            )
            top_k = top_k_templates(
                max_statistic,
                phase_idxs,
                sky_point.ap_in_range[batch_idx],
                toplist_f0_idxs,
                n_top=n_top,
            )
            for array in top_k:
                array.copy_to_host_async()
            if previous_match is not None:
                # work on the host while the GPU keeps working on this match
                yield from top_templates(*previous_match)
            previous_match = (
                top_k,
                harmonic_bank,
                template_batch,
                sky_point.sky_idx,
            )
    # the last match
    if previous_match is not None:
        yield from top_templates(*previous_match)
