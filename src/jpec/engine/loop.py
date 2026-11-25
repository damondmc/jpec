"""The search loop: every template at a batch of sky points scored on one band."""

from collections.abc import Iterator, Sequence
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float, Int

from jpec.bank.template_bank import TemplateBank
from jpec.detector.doppler import (
    doppler_corrected_mean_power,
    doppler_corrected_power,
    solar_roemer_delay,
)
from jpec.engine.candidates import TopTemplates
from jpec.engine.data import SearchData
from jpec.engine.layout import TemplateBatch, bank_layout, f0_idxs_per_toplist
from jpec.engine.match import (
    matched_filter_grid,
    placed_template_harmonics,
    template_norms_squared,
)
from jpec.stat.matched_filter import (
    barycenter_orbital_phase,
    data_harmonics,
    orbital_phase_mod_2pi,
)
from jpec.waveform.harmonics import template_harmonics
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
        n_theta: Number of orbital phases sampled.
        j_max: Highest harmonic kept.
        window: The window the SFTs were made with.
        batch_size: Number of modulation depths built at once, which caps the planes
            held.

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


def sft_weights_at_sky_point(
    search_data: SearchData, *, alpha: float, delta: float
) -> Float[Array, " n_sft"]:
    """The weight each SFT takes toward a sky position, of unit mean square.

        w[n] = (F_plus[n]^2 + F_cross[n]^2) / noise_level[n]

    the share of a signal's power the SFT is expected to hold, over
    `sqrt(<w^2>_n)`.

    Args:
        search_data: The band searched.
        alpha: Right ascension [rad].
        delta: Declination [rad].

    Returns:
        The weight per SFT, the detectors concatenated, in the floor's dtype.
    """
    antenna_power = np.concatenate(
        [
            sft.detector.antenna_power(
                sft.timestamps_midpoints, alpha=alpha, delta=delta
            )
            for sft in search_data.sfts
        ]
    )
    weights = antenna_power / search_data.sft_noise_level
    return jnp.asarray(
        weights / np.sqrt(np.mean(weights**2)), dtype=search_data.floor[0].dtype
    )


@partial(jax.jit, static_argnames="n_transformed_bins")
def weighted_excess_power_per_reference(
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
    sft_weights: Float[Array, " n_sft"],
    mean_power: list[Float[Array, " n_band"]],
) -> Float[Array, "n_doppler_reference n_sft n_transformed_bin"]:
    """The band corrected toward one sky position at every Doppler reference in turn.

    Each SFT's power less the power expected from its detector's time-averaged
    spectrum is multiplied by the SFT's weight:

        sft_weights[n] (power[n, k] - expected[n, k])

    Args:
        sft_time_series: Inverse transform of the band, per detector.
        floor: Mean noise power in each SFT and bin, per detector.
        velocity_per_detector: Detector velocity `v/c` at each SFT, per detector.
        doppler_references: Frequencies the Doppler shift is taken at [Hz].
        transformed_start_idxs: First band bin each reference keeps.
        n_transformed_bins: Number of bins each reference keeps from its first.
        tsft: SFT length [s].
        alpha: Right ascension [rad].
        delta: Declination [rad].
        sft_weights: Weight of each SFT toward the sky position, the detectors
            concatenated.
        mean_power: Normalized power averaged over the SFTs, per detector, in the
            detector frame.

    Returns:
        The weighted power at each Doppler reference, over the bins it keeps.
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
        corrected_power = jax.lax.dynamic_slice_in_dim(
            corrected_power, transformed_start_idx, n_transformed_bins, axis=-1
        )
        expected_power = doppler_corrected_mean_power(
            mean_power,
            velocity_per_detector=velocity_per_detector,
            tsft=tsft,
            alpha=alpha,
            delta=delta,
            doppler_reference=doppler_reference,
            start_idx=transformed_start_idx,
            n_bins=n_transformed_bins,
        )
        # the expected power is subtracted so the weights do not spread the floor and
        # steady lines from j = 0 into the harmonics matched
        return sft_weights[:, None] * (corrected_power - expected_power)

    # one reference's transforms in memory at a time
    return jax.lax.map(at_reference, (doppler_references, transformed_start_idxs))


@partial(jax.jit, static_argnames="j_max")
def data_harmonics_per_sky_point(
    weighted_excess_power: Float[Array, "n_sky n_doppler_reference n_sft n_freq"],
    orbital_phase: Float[Array, "n_sky n_period n_sft"],
    *,
    j_max: int,
) -> Complex[Array, "n_sky n_doppler_reference n_period n_harmonic n_freq"]:
    """data_harmonics() of the weighted excess power at each sky point and reference.

    Args:
        weighted_excess_power: weighted_excess_power_per_reference() at each sky
            point.
        orbital_phase: barycenter_orbital_phase() at each sky point, SFT and trial
            period of a batch [rad].
        j_max: Highest harmonic to keep.

    Returns:
        `F[j, k]` at each sky point, reference and trial period.
    """
    per_reference = jax.vmap(partial(data_harmonics, j_max=j_max), in_axes=(0, None))
    return jax.vmap(per_reference)(weighted_excess_power, orbital_phase)


@partial(jax.jit, static_argnames="n_top")
def top_k_templates(
    max_statistic: Float[Array, "n_f0 n_period n_matched_modulation_depth"],
    phase_idxs: Int[Array, "n_f0 n_period n_matched_modulation_depth"],
    ap_in_range: Bool[Array, "n_period n_matched_modulation_depth"],
    toplist_f0_idxs: Int[Array, "n_toplist n_member"],
    *,
    n_top: int,
) -> tuple[
    Float[Array, "n_toplist n_kept"],
    Int[Array, "n_toplist n_kept"],
    Int[Array, "n_toplist n_kept"],
]:
    """Each toplist's top in-range templates of a grid.

    Args:
        max_statistic: Maximum of `S` over the orbital phase at every template of
            the grid.
        phase_idxs: Index of the orbital phase each maximum sits at.
        ap_in_range: Whether each (period, modulation depth) has ap in range.
        toplist_f0_idxs: The f0 points each toplist covers.
        n_top: Number of templates kept per toplist.

    Returns:
        Per toplist, their statistic, top first and `-inf` past the in-range ones,
            their flat index into the grid, and their phase index.
    """
    n_f0, n_periods, n_modulation_depths = max_statistic.shape
    n_templates_per_f0 = n_periods * n_modulation_depths
    # top_k ranks NaN above every number, so a NaN statistic is dropped with the
    # out-of-box ones
    is_kept = ap_in_range[None] & ~jnp.isnan(max_statistic)
    ap_in_range_max_statistic = jnp.where(is_kept, max_statistic, -jnp.inf)
    # each toplist's f0 points, a padding point scoring -inf
    toplist_max_statistic = jnp.take(
        ap_in_range_max_statistic,
        toplist_f0_idxs,
        axis=0,
        mode="fill",
        fill_value=-jnp.inf,
    ).reshape(toplist_f0_idxs.shape[0], -1)
    # every template of the toplist when it holds fewer than n_top
    top_max_statistic, member_flat_idxs = jax.lax.top_k(
        toplist_max_statistic, min(n_top, toplist_max_statistic.shape[1])
    )
    f0_idxs = jnp.take_along_axis(
        toplist_f0_idxs, member_flat_idxs // n_templates_per_f0, axis=1
    )
    # a padding point's index lies past the grid and is clipped onto its last
    # template; its statistic is -inf
    flat_idxs = jnp.minimum(
        f0_idxs * n_templates_per_f0 + member_flat_idxs % n_templates_per_f0,
        n_f0 * n_templates_per_f0 - 1,
    )
    return top_max_statistic, flat_idxs, phase_idxs.ravel()[flat_idxs]


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
                sft_weights=sft_weights_at_sky_point(
                    search_data,
                    alpha=float(bank.alpha[sky_idx]),
                    delta=float(bank.delta[sky_idx]),
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
