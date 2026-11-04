"""Where a bank's subgrids read from one loaded band, and how their work is batched."""

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jpec.bank.template_bank import OrbitalTemplateGrid, TemplateBank
from jpec.constant import V_MAX
from jpec.waveform.harmonics import harmonic_count

# the Doppler residual allowed, as a fraction of the residual the sub-bin template
# harmonics leave
DOPPLER_TOLERANCE_FRACTION = 1.0 / 3.0

# a batch's harmonic count is rounded up to a multiple of this
HARMONIC_COUNT_STEP = 16


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
        transformed_bin_idxs: The band bins some template window reads, the only ones
            transformed.
        window_start_idxs: First bin of each f0 point's template window, counted from
            `transformed_bin_idxs.start`.
        doppler_references: Frequency each corrected band is Doppler-corrected at
            [Hz].
        doppler_reference_idxs: Which corrected band each f0 point reads.
        sub_bin_offsets: The sub-bin offsets some f0 point takes, the ones template
            harmonics are built at [bins].
        sub_bin_offset_idxs: Which of `sub_bin_offsets` each f0 point takes.
        omega: Orbital angular frequency of each trial period [rad/s], on the host
            in float64, the last repeated `period_batch_size - 1` times so every
            batch reads a full run.
        period_batches: The period batches that hold an in-box template.
        period_batch_size: Trial periods every batch is transformed and matched over,
            so each call has one shape; a batch shorter than that runs on into the
            periods after it, which are out of its box.
        n_window_modulation_depths: Modulation depths every batch is matched against.
    """

    template_grid: OrbitalTemplateGrid
    transformed_bin_idxs: slice
    window_start_idxs: Int[Array, " n_f0"]
    doppler_references: Float[Array, " n_doppler_reference"]
    doppler_reference_idxs: Int[Array, " n_f0"]
    sub_bin_offsets: Float[np.ndarray, " n_sub_bin_offset"]
    sub_bin_offset_idxs: Int[Array, " n_f0"]
    omega: Float[np.ndarray, " n_period"]
    period_batches: list[PeriodBatch]
    period_batch_size: int
    n_window_modulation_depths: int


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


def doppler_reference_f0_width(*, tsft: float, n_sub_bin_offsets: int) -> float:
    """The f0 span one Doppler reference serves.

        sub_bin_residual = 1 / (2 n_sub_bin_offsets)  [bins]
        tolerance = DOPPLER_TOLERANCE_FRACTION * sub_bin_residual
        f0_width = 2 tolerance / (V_MAX tsft)

    Args:
        tsft: SFT length [s].
        n_sub_bin_offsets: Sub-bin template offsets built.

    Returns:
        The span [Hz], centred on the reference.
    """
    # the nearest of n_sub_bin_offsets templates, 1 / n_sub_bin_offsets apart, is off
    # by up to half that
    sub_bin_residual = 1.0 / (2.0 * n_sub_bin_offsets)
    tolerance = DOPPLER_TOLERANCE_FRACTION * sub_bin_residual
    # the reference sits at the centre of its span, so an f0 point is at most half the
    # span from it
    return 2.0 * tolerance / (V_MAX * tsft)


def doppler_reference_count(
    f0_width: float, *, tsft: float, n_sub_bin_offsets: int
) -> int:
    """Doppler references enough to hold a subgrid's residual shift to the tolerance.

    Args:
        f0_width: Span of the subgrid's f0 points [Hz].
        tsft: SFT length [s].
        n_sub_bin_offsets: Sub-bin template offsets built.

    Returns:
        The count, at least 1.
    """
    f0_width_per_reference = doppler_reference_f0_width(
        tsft=tsft, n_sub_bin_offsets=n_sub_bin_offsets
    )
    return max(1, int(np.ceil(f0_width / f0_width_per_reference)))


def subgrid_layout(
    template_grid: OrbitalTemplateGrid,
    *,
    freq_low: float,
    n_band_bins: int,
    n_sub_bin_offsets: int,
    period_batch_size: int,
) -> SubgridLayout:
    """A subgrid's layout on one band: all the loop needs but data and harmonics.

    Args:
        template_grid: The subgrid's orbital template grid.
        freq_low: First frequency of the loaded band [Hz].
        n_band_bins: Bins the loaded band holds.
        n_sub_bin_offsets: Sub-bin template offsets across one bin.
        period_batch_size: Trial periods per transform call.

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
    # the margin beyond the windows is only for the Doppler shift, so the transform
    # skips it
    #
    #   band:          [------------------------------------------------]  n_band_bins
    #   lowest f0:          [--- n_window_bins ---]
    #   highest f0:                       [--- n_window_bins ---]
    #   transformed:        [====================================]
    #                       ^ transformed_start_idx              ^ transformed_stop_idx
    #
    # window_start_idxs is then counted from transformed_start_idx, the first bin of
    # the transformed slice
    transformed_start_idx = int(window_start_idxs.min())
    transformed_stop_idx = int(window_start_idxs.max()) + n_window_bins
    transformed_bin_idxs = slice(transformed_start_idx, transformed_stop_idx)
    window_start_idxs = window_start_idxs - transformed_start_idx

    # Doppler references at the centres of equal pieces of the f0 range; each f0 point
    # takes the nearest
    n_doppler_references = doppler_reference_count(
        float(f0[-1] - f0[0]), tsft=tsft, n_sub_bin_offsets=n_sub_bin_offsets
    )
    reference_centres = (np.arange(n_doppler_references) + 0.5) / n_doppler_references
    doppler_references = f0[0] + reference_centres * (f0[-1] - f0[0])
    doppler_reference_idxs = np.argmin(
        np.abs(doppler_references[None, :] - f0[:, None]), axis=1
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
    # no wider than the subgrid, as a bank around one source is
    period = template_grid.orbital_subgrid.period
    period_batch_size = min(period_batch_size, period.size)
    # each depth's harmonics, rounded up to the step and at most j_max
    n_harmonics_per_depth = np.array(
        [
            min(
                HARMONIC_COUNT_STEP
                * math.ceil(harmonic_count(float(depth_bins)) / HARMONIC_COUNT_STEP),
                template_grid.j_max,
            )
            for depth_bins in template_grid.orbital_subgrid.modulation_depth_bins
        ]
    )
    period_batches, n_window_modulation_depths = split_periods(
        template_grid.orbital_subgrid.in_box,
        template_grid.orbital_subgrid.ap_block_idxs,
        n_harmonics_per_depth,
        batch_size=period_batch_size,
    )
    # the last batch starts at most at the last period, so it reads
    # period_batch_size - 1 past it
    padded_period = np.pad(period, (0, period_batch_size - 1), mode="edge")
    return SubgridLayout(
        template_grid=template_grid,
        transformed_bin_idxs=transformed_bin_idxs,
        window_start_idxs=jnp.asarray(window_start_idxs),
        doppler_references=jnp.asarray(doppler_references),
        doppler_reference_idxs=jnp.asarray(doppler_reference_idxs),
        sub_bin_offsets=sub_bin_offsets[used_offset_idxs],
        sub_bin_offset_idxs=jnp.asarray(sub_bin_offset_idxs),
        omega=2.0 * np.pi / padded_period,
        period_batches=period_batches,
        period_batch_size=period_batch_size,
        n_window_modulation_depths=n_window_modulation_depths,
    )


def subgrid_layouts(
    bank: TemplateBank,
    *,
    freq_low: float,
    n_band_bins: int,
    n_sub_bin_offsets: int,
    period_batch_size: int,
) -> list[SubgridLayout]:
    """Every subgrid of a bank laid out on one loaded band.

    Args:
        bank: The bank.
        freq_low: First frequency of the loaded band [Hz].
        n_band_bins: Bins the loaded band holds.
        n_sub_bin_offsets: Sub-bin template offsets across one bin.
        period_batch_size: Trial periods per transform call.

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
        )
        for template_grid in bank.template_grids
    ]
