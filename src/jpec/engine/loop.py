"""The search loop at one sky point: every template there scored against one band."""

from collections.abc import Iterator
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float, Int

from jpec.bank.template_bank import TemplateBank
from jpec.detector.doppler import doppler_corrected_power
from jpec.engine.candidates import LoudestTemplates
from jpec.engine.data import SearchData
from jpec.engine.layout import PeriodBatch, SubgridLayout, subgrid_layouts
from jpec.stat.matched_filter import (
    data_harmonics,
    max_over_phase,
    orbital_basis,
    orbital_phase_mod_2pi,
)


@jax.jit
def doppler_corrected_power_per_reference(
    sft_time_series: list[Complex[Array, "n_sft n_band"]],
    floor: list[Float[Array, "n_sft n_band"]],
    velocity_per_detector: list[Float[Array, "n_sft 3"]],
    doppler_references: Float[Array, " n_doppler_reference"],
    *,
    tsft: float,
    alpha: float,
    delta: float,
) -> Float[Array, "n_doppler_reference n_sft n_band"]:
    """The band corrected toward one sky position at every Doppler reference in turn.

    Args:
        sft_time_series: Inverse transform of the band, per detector.
        floor: Mean noise power in each SFT and bin, per detector.
        velocity_per_detector: Detector velocity `v/c` at each SFT, per detector.
        doppler_references: Frequencies the Doppler shift is taken at [Hz].
        tsft: SFT length [s].
        alpha: Right ascension [rad].
        delta: Declination [rad].

    Returns:
        doppler_corrected_power() at each Doppler reference.
    """

    def at_reference(
        doppler_reference: Float[Array, ""],
    ) -> Float[Array, "n_sft n_band"]:
        return doppler_corrected_power(
            sft_time_series,
            floor=floor,
            velocity_per_detector=velocity_per_detector,
            tsft=tsft,
            alpha=alpha,
            delta=delta,
            doppler_reference=doppler_reference,
        )

    # lax.map, not vmap: one reference's transforms in memory at a time
    return jax.lax.map(at_reference, doppler_references)


jitted_orbital_basis = jax.jit(orbital_basis, static_argnames="j_max")
jitted_data_harmonics_per_reference = jax.jit(
    jax.vmap(data_harmonics, in_axes=(0, None))
)


@partial(
    jax.jit, static_argnames=("n_matched_modulation_depths", "n_phi", "f0_batch_size")
)
def matched_filter_grid(
    data_harmonics: Complex[
        Array, "n_doppler_reference n_period n_harmonic n_transformed_bin"
    ],
    template_harmonics: Complex[
        Array, "n_sub_bin_offset n_modulation_depth n_harmonic n_freq"
    ],
    doppler_reference_idxs: Int[Array, " n_f0"],
    window_start_idxs: Int[Array, " n_f0"],
    sub_bin_offset_idxs: Int[Array, " n_f0"],
    modulation_depth_start_idx: Int[Array, ""],
    *,
    n_matched_modulation_depths: int,
    n_phi: int,
    f0_batch_size: int,
) -> tuple[
    Float[Array, "n_f0 n_period n_matched_modulation_depth"],
    Int[Array, "n_f0 n_period n_matched_modulation_depth"],
]:
    """Matched filter on every (f0, period, modulation depth), maximized over phase.

    Args:
        data_harmonics: `F[j, k]` at each period over the transformed bins, per
            Doppler reference.
        template_harmonics: `tau[j, k]` at every modulation depth, per sub-bin offset.
        doppler_reference_idxs: Which reference's data harmonics each f0 point reads.
        window_start_idxs: First transformed bin of each f0 point's template window.
        sub_bin_offset_idxs: Which of `template_harmonics` each f0 point takes.
        modulation_depth_start_idx: First modulation depth matched.
        n_matched_modulation_depths: Modulation depths matched, from
            `modulation_depth_start_idx` on.
        n_phi: Orbital phases the statistic is evaluated at.
        f0_batch_size: f0 points matched at once, which caps the memory held.

    Returns:
        The maximum of `S` over the orbital phase and the phase index it sits at.
    """
    n_periods, n_harmonics = data_harmonics.shape[1], data_harmonics.shape[2]
    n_window_bins = template_harmonics.shape[-1]
    matched_template_harmonics = jax.lax.dynamic_slice_in_dim(
        template_harmonics,
        modulation_depth_start_idx,
        n_matched_modulation_depths,
        axis=1,
    )
    # data first, then template: the inner vmap runs over the templates, the outer
    # over the periods
    over_templates = jax.vmap(partial(max_over_phase, n_phi=n_phi), in_axes=(None, 0))
    over_periods = jax.vmap(over_templates, in_axes=(0, None))

    def match_f0(
        f0_point: tuple[Int[Array, ""], Int[Array, ""], Int[Array, ""]],
    ) -> tuple[
        Float[Array, "n_period n_matched_modulation_depth"],
        Int[Array, "n_period n_matched_modulation_depth"],
    ]:
        doppler_reference_idx, window_start_idx, sub_bin_offset_idx = f0_point
        window_data_harmonics = jax.lax.dynamic_slice(
            data_harmonics[doppler_reference_idx],
            (0, 0, window_start_idx),
            (n_periods, n_harmonics, n_window_bins),
        )
        return over_periods(
            window_data_harmonics, matched_template_harmonics[sub_bin_offset_idx]
        )

    return jax.lax.map(
        match_f0,
        (doppler_reference_idxs, window_start_idxs, sub_bin_offset_idxs),
        batch_size=f0_batch_size,
    )


@partial(jax.jit, static_argnames="n_loudest")
def loudest_in_box(
    max_statistic: Float[Array, "n_f0 n_period n_matched_modulation_depth"],
    phase_idxs: Int[Array, "n_f0 n_period n_matched_modulation_depth"],
    in_box: Bool[Array, "n_period n_matched_modulation_depth"],
    *,
    n_loudest: int,
) -> tuple[
    Float[Array, " n_loudest"], Int[Array, " n_loudest"], Int[Array, " n_loudest"]
]:
    """The loudest in-box templates of a grid, so only they leave the device.

    Args:
        max_statistic: Maximum of `S` over the orbital phase at every template of
            the grid.
        phase_idxs: Index of the orbital phase each maximum sits at.
        in_box: Whether each (period, modulation depth) lies in the box, the same at
            every f0 point.
        n_loudest: Templates kept, at most the grid's size.

    Returns:
        Their statistic, loudest first and `-inf` past the in-box ones, their flat
            index into the grid, and their phase index.
    """
    # top_k ranks NaN above every number, so a NaN statistic is dropped with the
    # out-of-box ones
    is_kept = in_box[None] & ~jnp.isnan(max_statistic)
    in_box_max_statistic = jnp.where(is_kept, max_statistic, -jnp.inf).ravel()
    loudest_max_statistic, flat_idxs = jax.lax.top_k(in_box_max_statistic, n_loudest)
    return loudest_max_statistic, flat_idxs, phase_idxs.ravel()[flat_idxs]


class HarmonicBank(NamedTuple):
    """A bank as the loop matches it: each subgrid's harmonics, phases and band layout.

    Built once for every sky point and every data set on that band.

    Attributes:
        bank: The bank.
        layouts: Its subgrids in order, from subgrid_layout() on the band.
        template_harmonics: Each subgrid's `tau[j, k]` at its layout's sub-bin
            offsets.
        orbital_phases: Each subgrid's orbital phase.
    """

    bank: TemplateBank
    layouts: list[SubgridLayout]
    template_harmonics: list[
        Complex[Array, "n_sub_bin_offset n_modulation_depth n_harmonic n_freq"]
    ]
    orbital_phases: list[Float[Array, "n_period n_sft"]]

    @classmethod
    def from_bank(
        cls,
        bank: TemplateBank,
        *,
        times: Float[np.ndarray, " n_sft"],
        freq_low: float,
        n_band_bins: int,
        n_sub_bin_offsets: int,
        period_batch_size: int,
        modulation_depth_batch_size: int,
    ) -> "HarmonicBank":
        """Lay a bank out on a band and build its template harmonics and phases.

        Args:
            bank: The bank.
            times: SFT midpoints from the epoch `phi_b` is quoted at [s].
            freq_low: First frequency of the band [Hz].
            n_band_bins: Bins the band holds.
            n_sub_bin_offsets: Sub-bin template offsets across one bin.
            period_batch_size: Trial periods per transform call.
            modulation_depth_batch_size: Modulation depths whose templates are built at
                once.

        Returns:
            The bank and its subgrids laid out on the band.
        """
        layouts = subgrid_layouts(
            bank,
            freq_low=freq_low,
            n_band_bins=n_band_bins,
            n_sub_bin_offsets=n_sub_bin_offsets,
            period_batch_size=period_batch_size,
        )
        template_harmonics = [
            layout.template_grid.template_harmonics_at_offsets(
                layout.sub_bin_offsets, batch_size=modulation_depth_batch_size
            )
            for layout in layouts
        ]
        orbital_phases = [
            jnp.asarray(orbital_phase_mod_2pi(times, layout.omega))
            for layout in layouts
        ]
        return cls(
            bank=bank,
            layouts=layouts,
            template_harmonics=template_harmonics,
            orbital_phases=orbital_phases,
        )


def orbital_statistics(
    search_data: SearchData,
    harmonic_bank: HarmonicBank,
    *,
    sky_idx: int,
    f0_batch_size: int,
    n_loudest: int,
) -> Iterator[LoudestTemplates]:
    """Every template at one sky point scored on one band, each grid's loudest kept.

    Args:
        search_data: The band searched, the one the bank is laid out on.
        harmonic_bank: The bank's template harmonics, laid out on the band.
        sky_idx: Index of the sky point in the bank.
        f0_batch_size: f0 points matched at once, which caps the S(phi_b) the match
            holds.
        n_loudest: Templates kept of each grid, which caps what is copied to the host.

    Yields:
        One LoudestTemplates per (subgrid, period batch), subgrid outermost; each one
            batch after it was dispatched, so the device runs the next while the host
            reads it.
    """
    bank = harmonic_bank.bank
    alpha = float(bank.alpha[sky_idx])
    delta = float(bank.delta[sky_idx])

    def fetched(
        subgrid_idx: int,
        layout: SubgridLayout,
        batch: PeriodBatch,
        loudest: tuple[Array, Array, Array],
    ) -> LoudestTemplates:
        max_statistic, flat_idxs, phase_idxs = (np.asarray(array) for array in loudest)
        f0_idxs, period_offsets, modulation_depth_offsets = np.unravel_index(
            flat_idxs,
            (
                layout.window_start_idxs.shape[0],
                layout.period_batch_size,
                layout.n_window_modulation_depths,
            ),
        )
        return LoudestTemplates(
            subgrid_idx=subgrid_idx,
            sky_idx=sky_idx,
            max_statistic=max_statistic,
            f0_idxs=f0_idxs,
            period_idxs=batch.period_start_idx + period_offsets,
            modulation_depth_idxs=batch.modulation_depth_start_idx
            + modulation_depth_offsets,
            phase_idxs=phase_idxs,
        )

    pending = None
    for subgrid_idx, (
        layout,
        subgrid_template_harmonics,
        subgrid_orbital_phases,
    ) in enumerate(
        zip(
            harmonic_bank.layouts,
            harmonic_bank.template_harmonics,
            harmonic_bank.orbital_phases,
        )
    ):
        template_grid = layout.template_grid
        in_box = template_grid.orbital_subgrid.in_box
        n_window_modulation_depths = layout.n_window_modulation_depths
        n_grid_loudest = min(
            n_loudest,
            layout.window_start_idxs.shape[0]
            * layout.period_batch_size
            * n_window_modulation_depths,
        )
        corrected_sft_power = doppler_corrected_power_per_reference(
            sft_time_series=search_data.sft_time_series,
            floor=search_data.floor,
            velocity_per_detector=search_data.velocity_per_detector,
            doppler_references=layout.doppler_references,
            tsft=search_data.tsft,
            alpha=alpha,
            delta=delta,
        )
        corrected_sft_power = corrected_sft_power[..., layout.transformed_bin_idxs]
        for batch in layout.period_batches:
            # a full run of periods, whatever the batch holds, so every call has one
            # shape
            batch_orbital_phase = jax.lax.dynamic_slice_in_dim(
                subgrid_orbital_phases,
                batch.period_start_idx,
                layout.period_batch_size,
            )
            batch_orbital_basis = jitted_orbital_basis(
                batch_orbital_phase, j_max=template_grid.j_max
            )
            # one transform per Doppler reference, not per f0 point
            batch_data_harmonics = jitted_data_harmonics_per_reference(
                corrected_sft_power, batch_orbital_basis
            )
            # the match reads only F[j, k]
            del batch_orbital_basis
            # one call for every f0 point of the batch
            max_statistic, phase_idxs = matched_filter_grid(
                data_harmonics=batch_data_harmonics,
                template_harmonics=subgrid_template_harmonics,
                doppler_reference_idxs=layout.doppler_reference_idxs,
                window_start_idxs=layout.window_start_idxs,
                sub_bin_offset_idxs=layout.sub_bin_offset_idxs,
                modulation_depth_start_idx=batch.modulation_depth_start_idx,
                n_matched_modulation_depths=n_window_modulation_depths,
                n_phi=template_grid.n_phi,
                f0_batch_size=f0_batch_size,
            )
            # the batch's own periods in its window; the run past period_stop_idx is
            # out of its box
            n_batch_periods = batch.period_stop_idx - batch.period_start_idx
            batch_in_box = np.zeros(
                (layout.period_batch_size, n_window_modulation_depths), dtype=bool
            )
            batch_in_box[:n_batch_periods] = in_box[
                batch.period_start_idx : batch.period_stop_idx,
                batch.modulation_depth_start_idx : batch.modulation_depth_start_idx
                + n_window_modulation_depths,
            ]
            loudest = loudest_in_box(
                max_statistic,
                phase_idxs,
                jnp.asarray(batch_in_box),
                n_loudest=n_grid_loudest,
            )
            for array in loudest:
                array.copy_to_host_async()
            # the host reads the batch before while the device runs this one
            if pending is not None:
                yield fetched(*pending)
            pending = (subgrid_idx, layout, batch, loudest)
    if pending is not None:
        yield fetched(*pending)
