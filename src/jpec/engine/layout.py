"""Where a bank's templates read from one loaded band, and how their work is batched."""

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jpec.bank.template_bank import TemplateBank
from jpec.constant import ARC_PAD_BINS, V_MAX
from jpec.waveform.harmonics import (
    harmonic_count,
    phase_count,
    template_harmonics,
    theta_sample_count,
)
from jpec.waveform.plane import arc_bin_offsets
from jpec.waveform.window import Window, freq_bin_curvature

# a batch's harmonic count is rounded up to a multiple of this
HARMONIC_COUNT_STEP = 32


class TemplateBatch(NamedTuple):
    """Trial periods and the modulation depths holding their in-range templates.

    Attributes:
        period_start_idx: First trial period of the batch.
        period_stop_idx: One past its last.
        modulation_depth_start_idx: First modulation depth of the window.
        n_harmonics: Number of harmonics the batch is transformed and matched with.
    """

    period_start_idx: int
    period_stop_idx: int
    modulation_depth_start_idx: int
    n_harmonics: int


class BankLayout(NamedTuple):
    """A bank's layout on one band: f0 windows, references, offsets and batches.

    Attributes:
        bank: The template bank laid out.
        transformed_start_idxs: First band bin each Doppler reference transforms.
        n_transformed_bins: Number of bins every reference transforms.
        window_start_idxs: First bin of each f0 point's template window, counted from
            its reference's transformed start.
        doppler_references: Frequency of each Doppler reference [Hz].
        doppler_reference_idxs: Which Doppler reference each f0 point reads.
        subbin_offsets: The sub-bin offsets template harmonics are built at [bins].
        subbin_offset_idxs: Which sub-bin offset each f0 point takes.
        group_f0_idxs: The f0 points of each (reference, offset) group.
        group_doppler_reference_idxs: The reference each group reads.
        group_subbin_offset_idxs: The offset each group takes before any shift.
        bin_offsets: Bin centres of every template window, offsets from f0 [bins].
        j_max: Highest harmonic a template is built with.
        n_theta: Number of orbital phases a template is sampled at for its harmonics.
        n_pad: Number of bins each window is widened by for the f0 shifts.
        omega: Orbital angular frequency of each trial period per omega^2 class
            [rad/s].
        template_batches: The batches of trial periods.
        period_batch_size: Number of trial periods per batch.
        n_batch_depths: Number of modulation depths every batch is matched
            against.
        n_phi: Number of orbital phases the statistic is read at.
    """

    bank: TemplateBank
    transformed_start_idxs: Int[Array, " n_ref"]
    n_transformed_bins: int
    window_start_idxs: Int[Array, " n_f0"]
    doppler_references: Float[Array, " n_ref"]
    doppler_reference_idxs: Int[Array, " n_f0"]
    subbin_offsets: Float[np.ndarray, " n_subbin_offset"]
    subbin_offset_idxs: Int[Array, " n_f0"]
    group_f0_idxs: Int[Array, "n_group n_member"]
    group_doppler_reference_idxs: Int[Array, " n_group"]
    group_subbin_offset_idxs: Int[Array, " n_group"]
    bin_offsets: Float[np.ndarray, " n_freq"]
    j_max: int
    n_theta: int
    n_pad: int
    omega: Float[np.ndarray, "n_class n_period"]
    template_batches: list[TemplateBatch]
    period_batch_size: int
    n_batch_depths: int
    n_phi: int

    @property
    def template_harmonics_bytes(self) -> int:
        """Number of device bytes the loop's template harmonics take on this layout."""
        n_templates = (
            self.subbin_offsets.size
            * self.bank.depth_shift_fractions.size
            * self.bank.modulation_depth.size
        )
        placed_bytes = (
            (self.n_pad + 1)
            * n_templates
            * self.j_max
            * (self.bin_offsets.shape[0] + self.n_pad)
            * np.dtype(np.complex64).itemsize
        )
        norms_bytes = n_templates * self.j_max * np.dtype(np.float32).itemsize
        return placed_bytes + norms_bytes

    def sky_batches(self, sky_batch_size: int) -> list[Int[np.ndarray, " n_sky"]]:
        """The bank's sky points in batches, each of one omega^2 class.

        Args:
            sky_batch_size: Number of sky points per batch, at most.

        Returns:
            The sky indices of each batch, class after class, increasing within one.
        """
        omega2_class = self.bank.omega2_class
        batches = []
        for class_idx in range(self.omega.shape[0]):
            sky_idxs = np.flatnonzero(omega2_class == class_idx)
            batches.extend(
                sky_idxs[start_idx : start_idx + sky_batch_size]
                for start_idx in range(0, sky_idxs.size, sky_batch_size)
            )
        return batches

    def padded_runs(
        self, per_period: np.ndarray, fill_value: int
    ) -> list[np.ndarray]:
        """A per-(period, modulation depth) or per-period array over each batch's run.

        Args:
            per_period: The array, periods first and modulation depths second if any.
            fill_value: What the rows past the last period read.

        Returns:
            Each batch's run of `period_batch_size` periods, cut to its modulation-depth
                window when the array has that axis.
        """
        n_periods = per_period.shape[0]
        padded = np.full(
            (n_periods + self.period_batch_size - 1, *per_period.shape[1:]),
            fill_value,
            dtype=per_period.dtype,
        )
        padded[:n_periods] = per_period
        runs = []
        for batch in self.template_batches:
            run = padded[
                batch.period_start_idx : batch.period_start_idx + self.period_batch_size
            ]
            if per_period.ndim == 2:
                run = run[
                    :,
                    batch.modulation_depth_start_idx : batch.modulation_depth_start_idx
                    + self.n_batch_depths,
                ]
            runs.append(run)
        return runs

    def ap_in_range(
        self, omega2_class: int, depth_class: Int[np.ndarray, " n_period"]
    ) -> Bool[np.ndarray, "n_batch n_batch_period n_batch_depth"]:
        """Whether each (period, modulation depth) of each batch has ap in range.

        Args:
            omega2_class: Index of the sky point's omega^2 shift fraction.
            depth_class: Index of each period's depth shift fraction at the sky
                point.

        Returns:
            The mask over each batch's full run of periods and its window's depths.
        """
        ap_in_range = np.stack(
            self.padded_runs(self.bank.ap_in_range(omega2_class, depth_class), False)
        )
        # a batch shorter than period_batch_size runs on into periods outside it
        for batch_idx, batch in enumerate(self.template_batches):
            n_batch_periods = batch.period_stop_idx - batch.period_start_idx
            ap_in_range[batch_idx, n_batch_periods:] = False
        return ap_in_range


def split_periods(
    ap_in_range: Bool[np.ndarray, "n_period n_depth"],
    ap_block_idxs: Int[np.ndarray, " n_period"],
    n_harmonics_per_depth: Int[np.ndarray, " n_depth"],
    batch_size: int,
) -> tuple[list[TemplateBatch], int]:
    """Periods of one ap block, `batch_size` at a time, with the depths each matches.

    Args:
        ap_in_range: Whether each (period, modulation depth) lies in its ap block.
        ap_block_idxs: The ap block each period belongs to, ap block after ap block.
        n_harmonics_per_depth: Number of harmonics each modulation depth's templates
            need.
        batch_size: Number of trial periods per transform call.

    Returns:
        The batches that hold an in-range template, and the modulation depths every one
            of them is matched against.
    """
    n_periods, n_modulation_depths = ap_in_range.shape
    # batches never span two ap blocks: each block's periods run [start, stop)
    block_starts = np.flatnonzero(np.diff(ap_block_idxs, prepend=ap_block_idxs[0] - 1))
    block_stops = np.append(block_starts[1:], n_periods)
    template_batches = []
    n_batch_depths = 0
    for block_start, block_stop in zip(block_starts.tolist(), block_stops.tolist()):
        # the block's periods, batch_size at a time; the last batch may be shorter
        for period_start_idx in range(block_start, block_stop, batch_size):
            period_stop_idx = min(period_start_idx + batch_size, block_stop)
            depth_idxs = np.flatnonzero(
                ap_in_range[period_start_idx:period_stop_idx].any(axis=0)
            )
            # periods past the box, as an injection bank's edge has
            if depth_idxs.size == 0:
                continue
            start_idx, stop_idx = int(depth_idxs[0]), int(depth_idxs[-1]) + 1
            n_harmonics = int(n_harmonics_per_depth[depth_idxs].max())
            template_batches.append(
                TemplateBatch(period_start_idx, period_stop_idx, start_idx, n_harmonics)
            )
            n_batch_depths = max(
                n_batch_depths, stop_idx - start_idx
            )
    # every batch matches the widest batch's number of depths, so one that would run
    # past the last modulation depth starts lower
    max_start_idx = n_modulation_depths - n_batch_depths
    return [
        batch._replace(
            modulation_depth_start_idx=min(
                batch.modulation_depth_start_idx, max_start_idx
            )
        )
        for batch in template_batches
    ], n_batch_depths


def f0_groups(
    doppler_reference_idxs: Int[np.ndarray, " n_f0"],
    subbin_offset_idxs: Int[np.ndarray, " n_f0"],
) -> tuple[
    Int[np.ndarray, "n_group n_member"],
    Int[np.ndarray, " n_group"],
    Int[np.ndarray, " n_group"],
]:
    """The f0 points that read one reference and take one offset, padded to one size.

    Args:
        doppler_reference_idxs: The reference each f0 point reads.
        subbin_offset_idxs: The sub-bin offset each f0 point takes.

    Returns:
        Each group's f0 points, its last repeated to the largest group's size, and
            each group's reference and offset.
    """
    group_keys = sorted(
        set(
            zip(
                doppler_reference_idxs.tolist(),
                subbin_offset_idxs.tolist(),
                strict=True,
            )
        )
    )
    members = [
        np.flatnonzero(
            (doppler_reference_idxs == reference_idx)
            & (subbin_offset_idxs == offset_idx)
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
    group_subbin_offset_idxs = np.array([offset_idx for _, offset_idx in group_keys])
    return group_f0_idxs, group_doppler_reference_idxs, group_subbin_offset_idxs


def f0_idxs_per_toplist(
    f0: Float[np.ndarray, " n_f0"],
    toplist_f0_edges: Float[np.ndarray, " n_toplist_edge"],
) -> Int[np.ndarray, "n_toplist n_member"]:
    """The f0 points of each toplist.

    f0 outside the edges goes to the first or last toplist. Membership is by the base
    f0 grid: a template's reported f0, its sky point's shift added, may lie up to one
    f0 step above its base f0, and so past its toplist's high edge.

    Args:
        f0: Frequency of each base f0 point [Hz].
        toplist_f0_edges: Edges of the toplists' f0 intervals [Hz], low to high.

    Returns:
        Index of each toplist's f0 points, in f0 order, padded with `n_f0`.
    """
    n_toplists = toplist_f0_edges.size - 1
    # the toplist each f0 point goes to
    toplist_idxs = np.clip(
        np.searchsorted(toplist_f0_edges, f0, side="right") - 1, 0, n_toplists - 1
    )
    grouped_f0_idxs = [
        np.flatnonzero(toplist_idxs == toplist_idx) for toplist_idx in range(n_toplists)
    ]
    n_members = max(f0_idxs.size for f0_idxs in grouped_f0_idxs)
    padded_f0_idxs = np.full((n_toplists, n_members), f0.size)
    for toplist_idx, f0_idxs in enumerate(grouped_f0_idxs):
        padded_f0_idxs[toplist_idx, : f0_idxs.size] = f0_idxs
    return padded_f0_idxs


def doppler_reference_f0_width(
    *, tsft: float, window: Window, doppler_mismatch: float
) -> float:
    """The f0 span one Doppler reference serves.

        f0_width = 2 sqrt(mu_doppler / g_freq_bin) / (V_MAX tsft)

    Args:
        tsft: SFT length [s].
        window: The window the SFTs were made with.
        doppler_mismatch: Largest mismatch the offset from the reference may leave,
            `mu_doppler`.

    Returns:
        The span [Hz], centred on the reference.
    """
    g_freq_bin = freq_bin_curvature(window)
    return 2.0 * math.sqrt(doppler_mismatch / g_freq_bin) / (V_MAX * tsft)


def doppler_reference_count(
    f0_width: float, *, tsft: float, window: Window, doppler_mismatch: float
) -> int:
    """Number of Doppler references needed to hold a bank's offsets to the mismatch.

    Args:
        f0_width: Span of the bank's f0 points [Hz].
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


def bank_layout(
    bank: TemplateBank,
    *,
    freq_low: float,
    n_band_bins: int,
    n_subbin_offsets: int,
    period_batch_size: int,
    harmonic_fraction: float,
    phase_mismatch: float,
    doppler_mismatch: float,
) -> BankLayout:
    """A bank's layout on one band: all the loop needs but data and harmonics.

    Args:
        bank: The template bank to lay out.
        freq_low: First frequency of the loaded band [Hz].
        n_band_bins: Number of bins the loaded band holds.
        n_subbin_offsets: Number of sub-bin template offsets across one bin.
        period_batch_size: Number of trial periods per transform call.
        harmonic_fraction: Share of each modulation depth's harmonics matched.
        phase_mismatch: Largest mismatch the orbital phase grid may leave.
        doppler_mismatch: Largest mismatch an f0 point's offset from its Doppler
            reference may leave.

    Returns:
        The bank with its base f0 points placed on the band, their references and
            offsets chosen, and its periods batched.
    """
    tsft = bank.tsft
    f0 = bank.f0
    # templates are built one depth step past the deepest base depth, which holds
    # every depth class
    deepest_depth_bins = bank.max_modulation_depth * tsft
    j_max = harmonic_count(deepest_depth_bins)
    n_theta = theta_sample_count(j_max)
    bin_offsets = np.asarray(arc_bin_offsets(deepest_depth_bins, pad_bins=ARC_PAD_BINS))
    # a template at offset index i < n moved by u = rint(f0_shift tsft n) sub-bin
    # units, f0_shift in [0, f0_step], sits (i + u) // n bins past its window start;
    # u <= f0_step tsft n + 1/2, so (i + u) / n < f0_step tsft + 1 and the placement
    # is at most ceil(f0_step tsft)
    n_pad = math.ceil(bank.f0_step * tsft)
    # an f0 point is a whole-bin slice of the band plus a remainder the template carries
    bins_from_freq_low = (f0 - freq_low) * tsft
    nearest_bin_idx = np.round(bins_from_freq_low).astype(int)
    subbin_remainder = bins_from_freq_low - nearest_bin_idx  # in [-0.5, 0.5]
    band_window_start_idxs = nearest_bin_idx + int(bin_offsets[0])
    n_window_bins = bin_offsets.shape[0]
    # each window is read n_pad bins wider; dynamic_slice clamps a window that runs
    # off the band instead of failing
    if (
        band_window_start_idxs.min() < 0
        or band_window_start_idxs.max() + n_window_bins + n_pad > n_band_bins
    ):
        windows_low = freq_low + band_window_start_idxs.min() / tsft
        windows_high = (
            freq_low + (band_window_start_idxs.max() + n_window_bins + n_pad) / tsft
        )
        raise ValueError(
            f"the bank's windows [{windows_low:.5f}, {windows_high:.5f}] Hz are not "
            f"in the data [{freq_low:.5f}, {freq_low + n_band_bins / tsft:.5f}] Hz"
        )
    # Doppler references at the centres of equal pieces of the f0 range; each f0 point
    # takes the nearest
    n_doppler_refs = doppler_reference_count(
        float(f0[-1] - f0[0]),
        tsft=tsft,
        window=bank.window,
        doppler_mismatch=doppler_mismatch,
    )
    reference_centres = (np.arange(n_doppler_refs) + 0.5) / n_doppler_refs
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
    starts_per_reference = [
        band_window_start_idxs[doppler_reference_idxs == reference_idx]
        for reference_idx in range(n_doppler_refs)
    ]
    n_transformed_bins = (
        n_window_bins
        + n_pad
        + max(int(starts.max() - starts.min()) for starts in starts_per_reference)
    )
    transformed_start_idxs = np.array(
        [
            min(int(starts.min()), n_band_bins - n_transformed_bins)
            for starts in starts_per_reference
        ]
    )
    window_start_idxs = (
        band_window_start_idxs - transformed_start_idxs[doppler_reference_idxs]
    )

    # sub-bin offsets on one fixed grid across a bin; each f0 point takes the nearest
    # to its remainder, and a shift moves it along the whole grid
    subbin_offsets = (np.arange(n_subbin_offsets) + 0.5) / n_subbin_offsets - 0.5
    subbin_offset_idxs = np.argmin(
        np.abs(subbin_offsets[None, :] - subbin_remainder[:, None]), axis=1
    )
    group_f0_idxs, group_doppler_reference_idxs, group_subbin_offset_idxs = f0_groups(
        doppler_reference_idxs, subbin_offset_idxs
    )
    # no wider than the bank, as a bank around one source is
    n_periods = bank.omega2.size
    period_batch_size = min(period_batch_size, n_periods)
    # each depth's share of its harmonics at its deepest depth class, rounded up to
    # the step and at most j_max
    deepest_class_depth_bins = (
        bank.modulation_depth_at(np.asarray(bank.depth_shift_fractions.size - 1))
        * tsft
    )
    n_kept_harmonics_per_depth = [
        math.ceil(round(harmonic_fraction * harmonic_count(float(depth_bins)), 9))
        for depth_bins in deepest_class_depth_bins
    ]
    n_harmonics_per_depth = np.array(
        [
            min(HARMONIC_COUNT_STEP * math.ceil(n_kept / HARMONIC_COUNT_STEP), j_max)
            for n_kept in n_kept_harmonics_per_depth
        ]
    )
    # in range at any omega^2 class and any depth class, which each period takes
    # independently of the others
    n_omega2_classes = bank.omega2_shift_fractions.size
    ap_in_range = np.zeros((n_periods, bank.modulation_depth.size), dtype=bool)
    for omega2_class in range(n_omega2_classes):
        for depth_class in range(bank.depth_shift_fractions.size):
            ap_in_range |= bank.ap_in_range(
                omega2_class, np.full(n_periods, depth_class)
            )
    template_batches, n_batch_depths = split_periods(
        ap_in_range,
        bank.ap_block_idxs,
        n_harmonics_per_depth,
        batch_size=period_batch_size,
    )
    # the most harmonics any batch matches
    n_harmonics = max(batch.n_harmonics for batch in template_batches)
    templates = np.asarray(
        template_harmonics(
            jnp.asarray(bin_offsets),
            modulation_depth_bins=deepest_depth_bins,
            n_theta=n_theta,
            j_max=j_max,
            window=bank.window,
        )
    )[:n_harmonics]
    n_phi = phase_count(
        (np.abs(templates) ** 2).sum(axis=1),
        phase_mismatch=phase_mismatch,
    )
    # each omega^2 class's periods, padded so the last few periods can be passed with
    # the same batch size
    omega = np.stack(
        [
            np.pad(
                bank.omega_at(omega2_class), (0, period_batch_size - 1), mode="edge"
            )
            for omega2_class in range(n_omega2_classes)
        ]
    )
    return BankLayout(
        bank=bank,
        transformed_start_idxs=jnp.asarray(transformed_start_idxs),
        n_transformed_bins=n_transformed_bins,
        window_start_idxs=jnp.asarray(window_start_idxs),
        doppler_references=jnp.asarray(doppler_references),
        doppler_reference_idxs=jnp.asarray(doppler_reference_idxs),
        subbin_offsets=subbin_offsets,
        subbin_offset_idxs=jnp.asarray(subbin_offset_idxs),
        group_f0_idxs=jnp.asarray(group_f0_idxs),
        group_doppler_reference_idxs=jnp.asarray(group_doppler_reference_idxs),
        group_subbin_offset_idxs=jnp.asarray(group_subbin_offset_idxs),
        bin_offsets=bin_offsets,
        j_max=j_max,
        n_theta=n_theta,
        n_pad=n_pad,
        omega=omega,
        template_batches=template_batches,
        period_batch_size=period_batch_size,
        n_batch_depths=n_batch_depths,
        n_phi=n_phi,
    )
