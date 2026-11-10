"""Where a bank's subgrids read from one loaded band, and how their work is batched."""

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jpec.bank.metric import freq_bin_curvature
from jpec.bank.template_bank import OrbitalTemplateGrid, TemplateBank
from jpec.constant import N_CURVATURE_POSITIONS, V_MAX
from jpec.waveform.harmonics import harmonic_count, phase_count, template_harmonics
from jpec.waveform.window import Window

# a batch's harmonic count is rounded up to a multiple of this
HARMONIC_COUNT_STEP = 32


class PeriodBatch(NamedTuple):
    """Trial periods and the modulation-depth window holding their in-box templates.

    Attributes:
        period_start_idx: First trial period of the run.
        period_stop_idx: One past its last.
        modulation_depth_start_idx: First modulation depth of the window.
        n_harmonics: Harmonics the batch is transformed and matched with, the ones
            its deepest in-box template needs.
    """

    period_start_idx: int
    period_stop_idx: int
    modulation_depth_start_idx: int
    n_harmonics: int


class SubgridLayout(NamedTuple):
    """A subgrid's layout on one band: f0 windows, references, offsets and batches.

    Attributes:
        template_grid: The subgrid's orbital template grid.
        transformed_start_idxs: First band bin each Doppler reference transforms.
        n_transformed_bins: Bins every reference transforms from its first, enough
            for the template windows of all its f0 points.
        window_start_idxs: First bin of each f0 point's template window, counted from
            its reference's transformed start.
        doppler_references: Frequency each corrected band is Doppler-corrected at
            [Hz].
        doppler_reference_idxs: Which corrected band each f0 point reads.
        sub_bin_offsets: The sub-bin offsets some f0 point takes, the ones template
            harmonics are built at [bins].
        sub_bin_offset_idxs: Which of `sub_bin_offsets` each f0 point takes.
        group_f0_idxs: The f0 points that read one reference and take one offset,
            each group padded to the largest by repeating its last f0 point.
        group_doppler_reference_idxs: The reference each group reads.
        group_sub_bin_offset_idxs: The offset each group takes.
        omega: Orbital angular frequency of each trial period [rad/s], on the host
            in float64, the last repeated `period_batch_size - 1` times so every
            batch reads a full run.
        period_batches: The period batches that hold an in-box template.
        period_batch_size: Trial periods every batch is transformed and matched over,
            so each call has one shape; a batch shorter than that runs on into the
            periods after it, which are out of its box.
        n_window_modulation_depths: Modulation depths every batch is matched against.
        n_phi: Orbital phases the statistic is read at, from phase_count() of the
            deepest modulation depth's matched harmonics.
    """

    template_grid: OrbitalTemplateGrid
    transformed_start_idxs: Int[Array, " n_doppler_reference"]
    n_transformed_bins: int
    window_start_idxs: Int[Array, " n_f0"]
    doppler_references: Float[Array, " n_doppler_reference"]
    doppler_reference_idxs: Int[Array, " n_f0"]
    sub_bin_offsets: Float[np.ndarray, " n_sub_bin_offset"]
    sub_bin_offset_idxs: Int[Array, " n_f0"]
    group_f0_idxs: Int[Array, "n_group n_member"]
    group_doppler_reference_idxs: Int[Array, " n_group"]
    group_sub_bin_offset_idxs: Int[Array, " n_group"]
    omega: Float[np.ndarray, " n_period"]
    period_batches: list[PeriodBatch]
    period_batch_size: int
    n_window_modulation_depths: int
    n_phi: int


def split_periods(
    in_box: Bool[np.ndarray, "n_period n_modulation_depth"],
    ap_block_idxs: Int[np.ndarray, " n_period"],
    n_harmonics_per_depth: Int[np.ndarray, " n_modulation_depth"],
    batch_size: int,
) -> tuple[list[PeriodBatch], int]:
    """Periods of one ap block, `batch_size` at a time, with a modulation-depth window.

    The windows share one width.

    Args:
        in_box: Whether each (period, modulation depth) lies in its ap block.
        ap_block_idxs: The ap block each period belongs to, ap block after ap block.
        n_harmonics_per_depth: Harmonics each modulation depth's templates need.
        batch_size: Trial periods per transform call.

    Returns:
        The batches that hold an in-box template, and the modulation depths every one
            of them is matched against.
    """
    n_periods, n_modulation_depths = in_box.shape
    # a batch across two ap blocks would need both their windows, and widen
    # every batch's
    # the period index each ap block starts at, where ap_block_idxs changes
    blk_start_idxs = np.flatnonzero(
        np.diff(ap_block_idxs, prepend=ap_block_idxs[0] - 1)
    )
    # one past each ap block's last period: the next block's first, then n_periods
    blk_stop_idxs = np.append(blk_start_idxs[1:], n_periods)
    period_batches = []
    n_window_modulation_depths = 0
    for blk_start_idx, blk_stop_idx in zip(
        blk_start_idxs.tolist(), blk_stop_idxs.tolist()
    ):
        # the block's periods, batch_size at a time; the last batch may be shorter
        for period_start_idx in range(blk_start_idx, blk_stop_idx, batch_size):
            period_stop_idx = min(period_start_idx + batch_size, blk_stop_idx)
            in_box_idxs = np.flatnonzero(
                in_box[period_start_idx:period_stop_idx].any(axis=0)
            )
            # periods past the box, as an injection bank's edge has
            if in_box_idxs.size == 0:
                continue
            start_idx, stop_idx = int(in_box_idxs[0]), int(in_box_idxs[-1]) + 1
            n_harmonics = int(n_harmonics_per_depth[in_box_idxs].max())
            period_batches.append(
                PeriodBatch(period_start_idx, period_stop_idx, start_idx, n_harmonics)
            )
            n_window_modulation_depths = max(
                n_window_modulation_depths, stop_idx - start_idx
            )
    # every window takes the widest width, so one that would run past the last
    # modulation depth starts lower
    max_start_idx = n_modulation_depths - n_window_modulation_depths
    return [
        batch._replace(
            modulation_depth_start_idx=min(
                batch.modulation_depth_start_idx, max_start_idx
            )
        )
        for batch in period_batches
    ], n_window_modulation_depths


def f0_groups(
    doppler_reference_idxs: Int[np.ndarray, " n_f0"],
    sub_bin_offset_idxs: Int[np.ndarray, " n_f0"],
) -> tuple[
    Int[np.ndarray, "n_group n_member"],
    Int[np.ndarray, " n_group"],
    Int[np.ndarray, " n_group"],
]:
    """The f0 points that read one reference and take one offset, padded to one size.

    Args:
        doppler_reference_idxs: The reference each f0 point reads.
        sub_bin_offset_idxs: The sub-bin offset each f0 point takes.

    Returns:
        Each group's f0 points, its last repeated to the largest group's size, and
            each group's reference and offset.
    """
    group_keys = sorted(
        set(zip(doppler_reference_idxs.tolist(), sub_bin_offset_idxs.tolist()))
    )
    members = [
        np.flatnonzero(
            (doppler_reference_idxs == reference_idx)
            & (sub_bin_offset_idxs == offset_idx)
        )
        for reference_idx, offset_idx in group_keys
    ]
    n_members = max(member.size for member in members)
    group_f0_idxs = np.stack(
        [
            np.pad(member, (0, n_members - member.size), mode="edge")
            for member in members
        ]
    )
    group_doppler_reference_idxs = np.array(
        [reference_idx for reference_idx, _ in group_keys]
    )
    group_sub_bin_offset_idxs = np.array([offset_idx for _, offset_idx in group_keys])
    return group_f0_idxs, group_doppler_reference_idxs, group_sub_bin_offset_idxs


def doppler_reference_f0_width(
    *, tsft: float, window: Window, doppler_mismatch: float
) -> float:
    """The f0 span one Doppler reference serves.

    An f0 point `delta_f` from its reference is left a track error of
    `delta_f (v . n_hat) tsft` bins, which for a template held fixed costs at most

        mu_doppler = g_freq_bin (delta_f V_MAX tsft)^2

    so the f0 points may lie up to

        delta_f_max = sqrt(mu_doppler / g_freq_bin) / (V_MAX tsft)

    from it, and a reference at the centre of its span serves

        f0_width = 2 delta_f_max

    Args:
        tsft: SFT length [s].
        window: The window the SFTs were made with.
        doppler_mismatch: Largest mismatch the offset from the reference may leave,
            `mu_doppler`.

    Returns:
        The span [Hz], centred on the reference.
    """
    g_freq_bin = freq_bin_curvature(window, n_positions=N_CURVATURE_POSITIONS)
    return 2.0 * math.sqrt(doppler_mismatch / g_freq_bin) / (V_MAX * tsft)


def doppler_reference_count(
    f0_width: float, *, tsft: float, window: Window, doppler_mismatch: float
) -> int:
    """Doppler references enough to hold a subgrid's offsets to the mismatch.

    Args:
        f0_width: Span of the subgrid's f0 points [Hz].
        tsft: SFT length [s].
        window: The window the SFTs were made with.
        doppler_mismatch: Largest mismatch an offset from a reference may leave.

    Returns:
        The count, at least 1.
    """
    f0_width_per_reference = doppler_reference_f0_width(
        tsft=tsft, window=window, doppler_mismatch=doppler_mismatch
    )
    return max(1, math.ceil(f0_width / f0_width_per_reference))


def subgrid_layout(
    template_grid: OrbitalTemplateGrid,
    *,
    freq_low: float,
    n_band_bins: int,
    n_sub_bin_offsets: int,
    period_batch_size: int,
    harmonic_fraction: float,
    phase_mismatch: float,
    doppler_mismatch: float,
) -> SubgridLayout:
    """A subgrid's layout on one band: all the loop needs but data and harmonics.

    Args:
        template_grid: The subgrid's orbital template grid.
        freq_low: First frequency of the loaded band [Hz].
        n_band_bins: Bins the loaded band holds.
        n_sub_bin_offsets: Sub-bin template offsets across one bin.
        period_batch_size: Trial periods per transform call.
        harmonic_fraction: Share of each modulation depth's harmonics matched.
        phase_mismatch: Largest mismatch the orbital phase grid may leave.
        doppler_mismatch: Largest mismatch an f0 point's offset from its Doppler
            reference may leave.

    Returns:
        The subgrid with its f0 points placed on the band, their references and
            offsets chosen, and its periods batched.
    """
    tsft = template_grid.tsft
    f0 = template_grid.orbital_subgrid.f0
    # an f0 point is a whole-bin slice of the band plus a remainder the template carries
    bins_from_freq_low = (f0 - freq_low) * tsft
    nearest_bin_idx = np.round(bins_from_freq_low).astype(int)
    sub_bin_remainder = bins_from_freq_low - nearest_bin_idx  # in [-0.5, 0.5]
    window_start_idxs = nearest_bin_idx + int(template_grid.bin_offsets[0])
    n_window_bins = template_grid.bin_offsets.shape[0]
    # dynamic_slice clamps a window that runs off the band instead of failing
    if (
        window_start_idxs.min() < 0
        or window_start_idxs.max() + n_window_bins > n_band_bins
    ):
        raise ValueError(
            f"the subgrid's f0 points from {f0[0]:.5f} to {f0[-1]:.5f} Hz do not all "
            f"fit the {n_band_bins}-bin band loaded from {freq_low:.5f} Hz"
        )
    # Doppler references at the centres of equal pieces of the f0 range; each f0 point
    # takes the nearest
    n_doppler_references = doppler_reference_count(
        float(f0[-1] - f0[0]),
        tsft=tsft,
        window=template_grid.window,
        doppler_mismatch=doppler_mismatch,
    )
    reference_centres = (np.arange(n_doppler_references) + 0.5) / n_doppler_references
    doppler_references = f0[0] + reference_centres * (f0[-1] - f0[0])
    doppler_reference_idxs = np.argmin(
        np.abs(doppler_references[None, :] - f0[:, None]), axis=1
    )

    # each reference transforms only the bins its own f0 points' windows read, every
    # reference the same count so they stack into one array
    #
    #   band:            [------------------------------------------------]
    #   reference 0:       [== its f0 points' windows ==]
    #   reference 1:                          [== its f0 points' windows ==]
    #                      ^ transformed_start_idxs[0]
    #                      |---- n_transformed_bins ----|
    #
    # a reference whose bins would run past the band starts lower; window_start_idxs
    # is then counted from the f0 point's own reference's start
    reference_window_start_idxs = [
        window_start_idxs[doppler_reference_idxs == reference_idx]
        for reference_idx in range(n_doppler_references)
    ]
    n_transformed_bins = n_window_bins + max(
        int(starts.max() - starts.min()) for starts in reference_window_start_idxs
    )
    transformed_start_idxs = np.array(
        [
            min(int(starts.min()), n_band_bins - n_transformed_bins)
            for starts in reference_window_start_idxs
        ]
    )
    window_start_idxs = (
        window_start_idxs - transformed_start_idxs[doppler_reference_idxs]
    )

    # sub-bin offsets on one fixed grid across a bin; each f0 point takes the nearest
    # to its remainder
    sub_bin_offsets = (np.arange(n_sub_bin_offsets) + 0.5) / n_sub_bin_offsets - 0.5
    sub_bin_offset_idxs = np.argmin(
        np.abs(sub_bin_offsets[None, :] - sub_bin_remainder[:, None]), axis=1
    )
    # only the offsets some f0 point takes are kept, and the index points into those
    used_offset_idxs, sub_bin_offset_idxs = np.unique(
        sub_bin_offset_idxs, return_inverse=True
    )
    group_f0_idxs, group_doppler_reference_idxs, group_sub_bin_offset_idxs = f0_groups(
        doppler_reference_idxs, sub_bin_offset_idxs
    )
    # no wider than the subgrid, as a bank around one source is
    period = template_grid.orbital_subgrid.period
    period_batch_size = min(period_batch_size, period.size)
    # each depth's share of its harmonics, rounded up to the step and at most j_max
    n_kept_harmonics_per_depth = [
        math.ceil(round(harmonic_fraction * harmonic_count(float(depth_bins)), 9))
        for depth_bins in template_grid.orbital_subgrid.modulation_depth_bins
    ]
    n_harmonics_per_depth = np.array(
        [
            min(
                HARMONIC_COUNT_STEP * math.ceil(n_kept / HARMONIC_COUNT_STEP),
                template_grid.j_max,
            )
            for n_kept in n_kept_harmonics_per_depth
        ]
    )
    period_batches, n_window_modulation_depths = split_periods(
        template_grid.orbital_subgrid.in_box,
        template_grid.orbital_subgrid.ap_block_idxs,
        n_harmonics_per_depth,
        batch_size=period_batch_size,
    )
    # the phase grid the deepest modulation depth needs, over the most harmonics any
    # batch matches
    n_matched_harmonics = max(batch.n_harmonics for batch in period_batches)
    deepest_template_harmonics = np.asarray(
        template_harmonics(
            template_grid.bin_offsets,
            modulation_depth_bins=float(
                template_grid.orbital_subgrid.modulation_depth_bins[-1]
            ),
            n_theta=template_grid.n_theta,
            j_max=template_grid.j_max,
            window=template_grid.window,
        )
    )[:n_matched_harmonics]
    n_phi = phase_count(
        (np.abs(deepest_template_harmonics) ** 2).sum(axis=1),
        phase_mismatch=phase_mismatch,
    )
    # the last batch starts at most at the last period, so it reads
    # period_batch_size - 1 past it
    padded_period = np.pad(period, (0, period_batch_size - 1), mode="edge")
    return SubgridLayout(
        template_grid=template_grid,
        transformed_start_idxs=jnp.asarray(transformed_start_idxs),
        n_transformed_bins=n_transformed_bins,
        window_start_idxs=jnp.asarray(window_start_idxs),
        doppler_references=jnp.asarray(doppler_references),
        doppler_reference_idxs=jnp.asarray(doppler_reference_idxs),
        sub_bin_offsets=sub_bin_offsets[used_offset_idxs],
        sub_bin_offset_idxs=jnp.asarray(sub_bin_offset_idxs),
        group_f0_idxs=jnp.asarray(group_f0_idxs),
        group_doppler_reference_idxs=jnp.asarray(group_doppler_reference_idxs),
        group_sub_bin_offset_idxs=jnp.asarray(group_sub_bin_offset_idxs),
        omega=2.0 * np.pi / padded_period,
        period_batches=period_batches,
        period_batch_size=period_batch_size,
        n_window_modulation_depths=n_window_modulation_depths,
        n_phi=n_phi,
    )


def subgrid_layouts(
    bank: TemplateBank,
    *,
    freq_low: float,
    n_band_bins: int,
    n_sub_bin_offsets: int,
    period_batch_size: int,
    harmonic_fraction: float,
    phase_mismatch: float,
    doppler_mismatch: float,
) -> list[SubgridLayout]:
    """Every subgrid of a bank laid out on one loaded band.

    Args:
        bank: The bank.
        freq_low: First frequency of the loaded band [Hz].
        n_band_bins: Bins the loaded band holds.
        n_sub_bin_offsets: Sub-bin template offsets across one bin.
        period_batch_size: Trial periods per transform call.
        harmonic_fraction: Share of each modulation depth's harmonics matched.
        phase_mismatch: Largest mismatch the orbital phase grid may leave.
        doppler_mismatch: Largest mismatch an f0 point's offset from its Doppler
            reference may leave.

    Returns:
        One per subgrid, in the bank's order.
    """
    return [
        subgrid_layout(
            template_grid,
            freq_low=freq_low,
            n_band_bins=n_band_bins,
            n_sub_bin_offsets=n_sub_bin_offsets,
            period_batch_size=period_batch_size,
            harmonic_fraction=harmonic_fraction,
            phase_mismatch=phase_mismatch,
            doppler_mismatch=doppler_mismatch,
        )
        for template_grid in bank.template_grids
    ]
