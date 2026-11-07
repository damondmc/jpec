"""The search loop: every template at a batch of sky points scored on one band."""

from collections.abc import Iterator, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float, Int

from jpec.bank.template_bank import TemplateBank
from jpec.detector.doppler import doppler_corrected_power, solar_roemer_delay
from jpec.engine.candidates import LoudestTemplates
from jpec.engine.data import SearchData
from jpec.engine.layout import PeriodBatch, SubgridLayout, subgrid_layouts
from jpec.stat.matched_filter import (
    barycenter_orbital_phase,
    data_harmonics,
    max_over_phase,
    orbital_phase_mod_2pi,
)


@partial(jax.jit, static_argnames="n_transformed_bins")
def doppler_corrected_power_per_reference(
    sft_time_series: list[Complex[Array, "n_sft n_band"]],
    floor: list[Float[Array, "n_sft n_band"]],
    velocity_per_detector: list[Float[Array, "n_sft 3"]],
    doppler_references: Float[Array, " n_doppler_reference"],
    transformed_start_idxs: Int[Array, " n_doppler_reference"],
    *,
    n_transformed_bins: int,
    tsft: float,
    alpha: float,
    delta: float,
) -> Float[Array, "n_doppler_reference n_sft n_transformed_bin"]:
    """The band corrected toward one sky position at every Doppler reference in turn.

    Args:
        sft_time_series: Inverse transform of the band, per detector.
        floor: Mean noise power in each SFT and bin, per detector.
        velocity_per_detector: Detector velocity `v/c` at each SFT, per detector.
        doppler_references: Frequencies the Doppler shift is taken at [Hz].
        transformed_start_idxs: First band bin each reference keeps.
        n_transformed_bins: Bins each reference keeps from its first.
        tsft: SFT length [s].
        alpha: Right ascension [rad].
        delta: Declination [rad].

    Returns:
        doppler_corrected_power() at each Doppler reference, over the bins it keeps.
    """

    def at_reference(
        reference_and_start_idx: tuple[Float[Array, ""], Int[Array, ""]],
    ) -> Float[Array, "n_sft n_transformed_bin"]:
        doppler_reference, transformed_start_idx = reference_and_start_idx
        corrected_power = doppler_corrected_power(
            sft_time_series,
            floor=floor,
            velocity_per_detector=velocity_per_detector,
            tsft=tsft,
            alpha=alpha,
            delta=delta,
            doppler_reference=doppler_reference,
        )
        return jax.lax.dynamic_slice_in_dim(
            corrected_power, transformed_start_idx, n_transformed_bins, axis=-1
        )

    # lax.map, not vmap: one reference's transforms in memory at a time
    return jax.lax.map(at_reference, (doppler_references, transformed_start_idxs))


@partial(jax.jit, static_argnames="j_max")
def data_harmonics_per_sky_point(
    corrected_sft_power: Float[Array, "n_sky n_doppler_reference n_sft n_freq"],
    orbital_phase: Float[Array, "n_sky n_period n_sft"],
    *,
    j_max: int,
) -> Complex[Array, "n_sky n_doppler_reference n_period n_harmonic n_freq"]:
    """data_harmonics() of the corrected power at each sky point and reference.

    Args:
        corrected_sft_power: The Doppler-corrected power at each sky point and
            reference.
        orbital_phase: barycenter_orbital_phase() at each sky point, SFT and trial
            period of a batch [rad].
        j_max: Highest harmonic to keep.

    Returns:
        `F[j, k]` at each sky point, reference and trial period.
    """
    per_reference = jax.vmap(partial(data_harmonics, j_max=j_max), in_axes=(0, None))
    return jax.vmap(per_reference)(corrected_sft_power, orbital_phase)


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
    window_start_idxs: Int[Array, " n_f0"],
    group_f0_idxs: Int[Array, "n_group n_member"],
    group_doppler_reference_idxs: Int[Array, " n_group"],
    group_sub_bin_offset_idxs: Int[Array, " n_group"],
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
        window_start_idxs: First bin of each f0 point's template window, among its
            reference's transformed bins.
        group_f0_idxs: The f0 points of each group, from f0_groups().
        group_doppler_reference_idxs: Which reference's data harmonics each group
            reads.
        group_sub_bin_offset_idxs: Which of `template_harmonics` each group takes.
        modulation_depth_start_idx: First modulation depth matched.
        n_matched_modulation_depths: Modulation depths matched, from
            `modulation_depth_start_idx` on.
        n_phi: Orbital phases the statistic is evaluated at.
        f0_batch_size: f0 points matched at once, in whole groups and at least one,
            which caps the memory held.

    Returns:
        The maximum of `S` over the orbital phase and the phase index it sits at,
            per f0 point.
    """
    n_periods, n_harmonics = data_harmonics.shape[1], data_harmonics.shape[2]
    n_window_bins = template_harmonics.shape[-1]
    matched_template_harmonics = jax.lax.dynamic_slice_in_dim(
        template_harmonics,
        modulation_depth_start_idx,
        n_matched_modulation_depths,
        axis=1,
    )[..., :n_harmonics, :]
    # data first, then template: the inner vmap runs over the templates, the outer
    # over the periods
    over_templates = jax.vmap(partial(max_over_phase, n_phi=n_phi), in_axes=(None, 0))
    over_periods = jax.vmap(over_templates, in_axes=(0, None))
    # a group's f0 points share one template
    over_members = jax.vmap(over_periods, in_axes=(0, None))

    def window_data_harmonics(
        doppler_reference_idx: Int[Array, ""], window_start_idx: Int[Array, ""]
    ) -> Complex[Array, "n_period n_harmonic n_freq"]:
        return jax.lax.dynamic_slice(
            data_harmonics[doppler_reference_idx],
            (0, 0, window_start_idx),
            (n_periods, n_harmonics, n_window_bins),
        )

    def match_group(
        group: tuple[Int[Array, " n_member"], Int[Array, ""], Int[Array, ""]],
    ) -> tuple[
        Float[Array, "n_member n_period n_matched_modulation_depth"],
        Int[Array, "n_member n_period n_matched_modulation_depth"],
    ]:
        f0_idxs, doppler_reference_idx, sub_bin_offset_idx = group
        member_data_harmonics = jax.vmap(window_data_harmonics, in_axes=(None, 0))(
            doppler_reference_idx, window_start_idxs[f0_idxs]
        )
        return over_members(
            member_data_harmonics, matched_template_harmonics[sub_bin_offset_idx]
        )

    n_members = group_f0_idxs.shape[1]
    group_max_statistic, group_phase_idxs = jax.lax.map(
        match_group,
        (group_f0_idxs, group_doppler_reference_idxs, group_sub_bin_offset_idxs),
        batch_size=max(f0_batch_size // n_members, 1),
    )
    # each f0 point back in its own row; a repeated member writes the same values
    flat_f0_idxs = group_f0_idxs.ravel()
    grid_shape = (window_start_idxs.shape[0], *group_max_statistic.shape[2:])
    max_statistic = (
        jnp.zeros(grid_shape, group_max_statistic.dtype)
        .at[flat_f0_idxs]
        .set(group_max_statistic.reshape(-1, *grid_shape[1:]))
    )
    phase_idxs = (
        jnp.zeros(grid_shape, group_phase_idxs.dtype)
        .at[flat_f0_idxs]
        .set(group_phase_idxs.reshape(-1, *grid_shape[1:]))
    )
    return max_statistic, phase_idxs


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


class HarmonicBank:
    """A bank as the loop matches it: each subgrid's harmonics, phases and band layout.

    Built once for every sky point and every data set on that band.

    Attributes:
        bank: The bank.
        layouts: Its subgrids in order, from subgrid_layout() on the band.
        template_harmonics: Each subgrid's `tau[j, k]` at its layout's sub-bin
            offsets.
        orbital_phases: Each subgrid's orbital phase in detector time.
        omegas: Each subgrid's orbital angular frequencies [rad/s], on the device.
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
            n_band_bins: Bins the band holds.
            n_sub_bin_offsets: Sub-bin template offsets across one bin.
            period_batch_size: Trial periods per transform call.
            harmonic_fraction: Share of each modulation depth's harmonics matched.
            phase_mismatch: Largest mismatch the orbital phase grid may leave.
            doppler_mismatch: Largest mismatch an f0 point's offset from its Doppler
                reference may leave.
            modulation_depth_batch_size: Modulation depths whose templates are built at
                once.
        """
        self.bank = bank
        self.layouts = subgrid_layouts(
            bank,
            freq_low=freq_low,
            n_band_bins=n_band_bins,
            n_sub_bin_offsets=n_sub_bin_offsets,
            period_batch_size=period_batch_size,
            harmonic_fraction=harmonic_fraction,
            phase_mismatch=phase_mismatch,
            doppler_mismatch=doppler_mismatch,
        )
        self.template_harmonics = [
            layout.template_grid.template_harmonics_at_offsets(
                layout.sub_bin_offsets, batch_size=modulation_depth_batch_size
            )
            for layout in self.layouts
        ]
        self.orbital_phases = [
            jnp.asarray(orbital_phase_mod_2pi(times, layout.omega))
            for layout in self.layouts
        ]
        self.omegas = [jnp.asarray(layout.omega) for layout in self.layouts]


def orbital_statistics(
    search_data: SearchData,
    harmonic_bank: HarmonicBank,
    *,
    sky_idxs: Sequence[int],
    f0_batch_size: int,
    n_loudest: int,
) -> Iterator[LoudestTemplates]:
    """Every template at a batch of sky points scored, each grid's loudest kept.

    Args:
        search_data: The band searched, the one the bank is laid out on.
        harmonic_bank: The bank's template harmonics, laid out on the band.
        sky_idxs: Indices of the sky points in the bank, transformed together.
        f0_batch_size: no. f0 points matched at once.
        n_loudest: Templates kept of each grid, which caps what is copied to the host.

    Yields:
        One LoudestTemplates per (subgrid, period batch, sky point).
    """
    bank = harmonic_bank.bank
    # each subgrid's f0 points' Doppler reference frequencies, on the host
    f0_doppler_references = [
        np.asarray(layout.doppler_references)[np.asarray(layout.doppler_reference_idxs)]
        for layout in harmonic_bank.layouts
    ]

    def fetched(
        subgrid_idx: int,
        layout: SubgridLayout,
        batch: PeriodBatch,
        sky_idx: int,
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
            doppler_references=f0_doppler_references[subgrid_idx][f0_idxs],
            n_phi=layout.n_phi,
        )

    sky_idx_array = np.asarray(sky_idxs)
    solar_roemer_delays = solar_roemer_delay(
        search_data.position,
        alpha=jnp.asarray(bank.alpha[sky_idx_array]),
        delta=jnp.asarray(bank.delta[sky_idx_array]),
    )
    pending = None
    for subgrid_idx, (
        layout,
        subgrid_template_harmonics,
        subgrid_orbital_phases,
        subgrid_omega,
    ) in enumerate(
        zip(
            harmonic_bank.layouts,
            harmonic_bank.template_harmonics,
            harmonic_bank.orbital_phases,
            harmonic_bank.omegas,
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
        corrected_sft_power = jnp.stack(
            [
                doppler_corrected_power_per_reference(
                    sft_time_series=search_data.sft_time_series,
                    floor=search_data.floor,
                    velocity_per_detector=search_data.velocity_per_detector,
                    doppler_references=layout.doppler_references,
                    transformed_start_idxs=layout.transformed_start_idxs,
                    n_transformed_bins=layout.n_transformed_bins,
                    tsft=search_data.tsft,
                    alpha=float(bank.alpha[sky_idx]),
                    delta=float(bank.delta[sky_idx]),
                )
                for sky_idx in sky_idxs
            ]
        )
        for batch in layout.period_batches:
            # a full run of periods, whatever the batch holds, so every call has one
            # shape
            batch_orbital_phase = jax.lax.dynamic_slice_in_dim(
                subgrid_orbital_phases,
                batch.period_start_idx,
                layout.period_batch_size,
            )
            batch_omega = jax.lax.dynamic_slice_in_dim(
                subgrid_omega, batch.period_start_idx, layout.period_batch_size
            )
            # one transform for every sky point and Doppler reference of the batch
            batch_barycenter_orbital_phase = barycenter_orbital_phase(
                batch_orbital_phase, batch_omega, solar_roemer_delays
            )
            batch_data_harmonics = data_harmonics_per_sky_point(
                corrected_sft_power,
                batch_barycenter_orbital_phase,
                j_max=batch.n_harmonics,
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
            device_batch_in_box = jnp.asarray(batch_in_box)
            for sky_offset, sky_idx in enumerate(sky_idxs):
                # one call for every f0 point of the batch
                max_statistic, phase_idxs = matched_filter_grid(
                    data_harmonics=batch_data_harmonics[sky_offset],
                    template_harmonics=subgrid_template_harmonics,
                    window_start_idxs=layout.window_start_idxs,
                    group_f0_idxs=layout.group_f0_idxs,
                    group_doppler_reference_idxs=layout.group_doppler_reference_idxs,
                    group_sub_bin_offset_idxs=layout.group_sub_bin_offset_idxs,
                    modulation_depth_start_idx=batch.modulation_depth_start_idx,
                    n_matched_modulation_depths=n_window_modulation_depths,
                    n_phi=layout.n_phi,
                    f0_batch_size=f0_batch_size,
                )
                loudest = loudest_in_box(
                    max_statistic,
                    phase_idxs,
                    device_batch_in_box,
                    n_loudest=n_grid_loudest,
                )
                for array in loudest:
                    array.copy_to_host_async()
                # the host reads the grid before while the device runs this one
                if pending is not None:
                    yield fetched(*pending)
                pending = (subgrid_idx, layout, batch, sky_idx, loudest)
    if pending is not None:
        yield fetched(*pending)
