r"""Cut the search a config asks for into jobs, and price each one.

Usage:

  jpec search-setup examples/search.yaml --sft-metadata setup/sft_metadata.json \
      --timing results/time_search_gpu_<host>.json --output setup/search_setup.json
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from jaxtyping import Float

from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import (
    DataSetup,
    JobSetup,
    SearchJobConfig,
    SearchLoopConfig,
    SearchSetup,
    SFTMetadata,
    read_config,
    read_json,
)
from jpec.constant import DAY, GB
from jpec.detector.site import Observation, reference_epoch
from jpec.engine.data import freq_band_edges
from jpec.engine.layout import BankLayout, bank_layout
from jpec.pipelines.monitor import RunLogger
from jpec.waveform.window import Window

COMPLEX64_BYTES = 8
FLOAT32_BYTES = 4


def f0_interval_layout(
    tiling: LatticeTiling,
    f0_low: float,
    f0_high: float,
    *,
    velocity: Float[np.ndarray, "n_sft 3"],
    window: Window,
    search_loop_config: SearchLoopConfig,
) -> tuple[BankLayout, float, float, int]:
    """The bank of an f0 interval laid out on the band a search over it loads.

    Args:
        tiling: The tiling the templates are built on.
        f0_low: Low edge of the interval [Hz].
        f0_high: High edge [Hz].
        velocity: Detector velocity `v/c` at each SFT, every detector.
        window: The window the SFTs were made with.
        search_loop_config: How the search loop batches its work.

    Returns:
        The layout of the bank over the whole sky, the band's low and high edges
            [Hz], and the number of bins the band holds.
    """
    bank = TemplateBank.over_f0_interval(
        tiling, f0_low, f0_high, velocity=velocity, window=window
    )
    freq_low, freq_high = freq_band_edges(
        bank.f0[0],
        bank.f0_high,
        max_modulation_depth=bank.max_modulation_depth,
        tsft=bank.tsft,
    )
    n_band_bins = round((freq_high - freq_low) * tiling.tsft)
    layout = bank_layout(
        bank,
        freq_low=freq_low,
        n_band_bins=n_band_bins,
        n_subbin_offsets=search_loop_config.n_subbin_offsets,
        period_batch_size=search_loop_config.period_batch_size,
        harmonic_fraction=search_loop_config.harmonic_fraction,
        phase_mismatch=search_loop_config.phase_mismatch,
        doppler_mismatch=search_loop_config.doppler_mismatch,
    )
    return layout, freq_low, freq_high, n_band_bins


def search_memory_gb(
    layout: BankLayout,
    *,
    n_band_bins: int,
    n_sfts: int,
    sky_batch_size: int,
    f0_batch_size: int,
) -> float:
    """Giga-bytes a search over a band holds at its peak.

    Args:
        layout: Its bank laid out on the band.
        n_band_bins: Number of bins of the band.
        n_sfts: Number of SFTs over every detector.
        sky_batch_size: Number of sky points per transform.
        f0_batch_size: Number of f0 points matched at once.

    Returns:
        Peak giga-bytes the search holds.
    """
    # the SFT time series and the floor
    loaded_bytes = n_sfts * n_band_bins * (COMPLEX64_BYTES + FLOAT32_BYTES)
    # every omega^2 class's trial periods
    orbital_phase_bytes = layout.omega.size * n_sfts * FLOAT32_BYTES
    n_padded_window_bins = layout.bin_offsets.shape[0] + layout.n_pad
    n_doppler_refs = int(layout.doppler_references.size)
    n_transformed_bins = layout.n_transformed_bins
    n_batch_depths = layout.n_batch_depths
    # the most harmonics any period batch transforms and matches
    n_batch_harmonics = max(batch.n_harmonics for batch in layout.template_batches)
    n_groups, n_members = layout.group_f0_idxs.shape
    n_matched_groups = min(n_groups, max(f0_batch_size // n_members, 1))
    n_matched_f0_points = n_matched_groups * n_members
    # the batch of sky points' transformed bins, at every reference
    stack_bytes = (
        sky_batch_size
        * n_doppler_refs
        * n_sfts
        * n_transformed_bins
        * FLOAT32_BYTES
    )
    # building the stack: one reference's whole corrected band in flight per sky
    # point, and the transformed bins as returned, expanded and stacked
    doppler_correction_bytes = (
        sky_batch_size * n_sfts * n_band_bins * FLOAT32_BYTES + 3 * stack_bytes
    )
    # the float32 cosine and sine bases at every sky point, the batch's harmonics
    # twice, the shifted templates and the data windows of the groups matched at once,
    # and the match's overlaps and S(phi_b)
    per_period_bytes = (
        2 * sky_batch_size * n_batch_harmonics * n_sfts * FLOAT32_BYTES
        + 2
        * sky_batch_size
        * n_doppler_refs
        * n_batch_harmonics
        * n_transformed_bins
        * COMPLEX64_BYTES
        + n_matched_groups
        * n_batch_depths
        * n_batch_harmonics
        * n_padded_window_bins
        * COMPLEX64_BYTES
        + n_matched_f0_points * n_batch_harmonics * n_padded_window_bins * COMPLEX64_BYTES
        + n_matched_f0_points
        * n_batch_depths
        * n_batch_harmonics
        * COMPLEX64_BYTES
        + n_matched_f0_points
        * n_batch_depths
        * layout.n_phi
        * FLOAT32_BYTES
    )
    # transforming: the stack twice, and a full run of periods
    transform_bytes = 2 * stack_bytes + layout.period_batch_size * per_period_bytes
    # one template per (subbin offset, depth class, depth)
    n_templates = (
        layout.subbin_offsets.size
        * layout.bank.depth_shift_fractions.size
        * layout.bank.modulation_depth.size
    )
    # the loop's placed template harmonics and their norms
    template_bytes = (
        (layout.n_pad + 1)
        * n_templates
        * layout.j_max
        * n_padded_window_bins
        * COMPLEX64_BYTES
        + n_templates * layout.j_max * FLOAT32_BYTES
    )
    # building the template table: about three unplaced tables in flight
    unplaced_template_bytes = (
        n_templates * layout.j_max * layout.bin_offsets.shape[0] * COMPLEX64_BYTES
    )
    return (
        loaded_bytes
        + template_bytes
        + orbital_phase_bytes
        + max(3 * unplaced_template_bytes, doppler_correction_bytes, transform_bytes)
    ) / GB


def search_hours(
    layout: BankLayout,
    *,
    n_band_bins: int,
    n_sfts: int,
    rates: dict[str, float],
) -> float:
    """Hours a search over a band takes.

    Args:
        layout: Its bank laid out on the band.
        n_band_bins: Number of bins of the band.
        n_sfts: Number of SFTs over every detector.
        rates: The `rates` of a jpec time-search result.

    Returns:
        Hours.
    """
    n_doppler_refs = layout.doppler_references.size
    # once per sky point, at every reference
    doppler_seconds = (
        n_doppler_refs * n_sfts * n_band_bins * rates["doppler_correction"]
    )
    # every batch is a full run of its omega^2 class's periods, at its own harmonics
    n_batch_harmonics = sum(batch.n_harmonics for batch in layout.template_batches)
    n_batch_templates = (
        layout.period_batch_size
        * layout.window_start_idxs.size
        * layout.n_batch_depths
    )
    # the transform per period and harmonic, at one sky point and every reference
    transform_seconds = (
        layout.period_batch_size
        * n_doppler_refs
        * n_sfts
        * layout.n_transformed_bins
        * rates["data_harmonics"]
    )
    # the overlaps per harmonic, over each template's window widened by n_pad
    overlap_seconds = (
        n_batch_templates
        * (layout.bin_offsets.shape[0] + layout.n_pad)
        * rates["match"]
    )
    # S(phi_b) and its maximum per template, at every orbital phase
    phase_seconds = n_batch_templates * layout.n_phi * rates["phase_maximum"]
    seconds_per_sky_point = float(
        doppler_seconds
        + n_batch_harmonics * (transform_seconds + overlap_seconds)
        + len(layout.template_batches) * phase_seconds
    )
    return layout.bank.alpha.size * seconds_per_sky_point / 3600.0


def run(args: argparse.Namespace) -> None:
    """Cut the config's box into jobs, price each one and write the setup.

    Args:
        args: The parsed flags.
    """
    # the metric and the lattice are taken in float64, on the host
    output = Path(args.output)
    run_logger = RunLogger("search_setup", output.parent, platform="cpu", x64=True)
    config = read_config(args.config, SearchJobConfig)
    sft_metadata = read_json(args.sft_metadata, SFTMetadata)
    template_bank_config, compute = config.template_bank, config.compute
    search_loop_config = config.search_loop

    # the run's SFTs, as the search will find them, their epoch and the detector motion
    window = sft_metadata.window
    tsft = sft_metadata.tsft
    n_sfts = sft_metadata.n_sfts
    t_ref = reference_epoch(sft_metadata.timestamps, tsft=tsft)
    observation = Observation(sft_metadata.timestamps, tsft=tsft, t_ref=t_ref)
    velocity = observation.velocity

    # one lattice per period range, each over the whole f0 range
    prior = config.parameter_space.prior()
    period_edges = config.job_period_edges()
    period_ranges = list(zip(period_edges, period_edges[1:]))
    tilings = [
        LatticeTiling.from_metric(
            prior._replace(period=(period_low * DAY, period_high * DAY)),
            observation=observation,
            mismatch=template_bank_config.mismatch,
            n_ap_blocks=template_bank_config.n_ap_blocks,
            window=window,
        )
        for period_low, period_high in period_ranges
    ]

    # the f0 range in jobs of job_f0_width, each over every period range
    f0_edges = config.job_f0_edges()
    n_f0_intervals = len(f0_edges) - 1
    timing = None if args.timing is None else json.loads(Path(args.timing).read_text())
    rates = None if timing is None else timing["rates"]
    run_logger.info(
        f"{n_f0_intervals * len(period_ranges)} jobs: {n_f0_intervals} f0 intervals "
        f"of {compute.job_f0_width:g} Hz x {len(period_ranges)} period ranges "
        f"{period_edges} d"
    )
    if timing is None:
        run_logger.info(
            "no timing given, so the hours are not priced: time the first job with "
            "jpec time-search and run this again with --timing"
        )
    else:
        run_logger.info(f"Hours at the rates of {timing['host']}")

    # what each job searches, loads, holds and takes
    job_setups = []
    for job_f0_low, job_f0_high in zip(f0_edges, f0_edges[1:]):
        for tiling_idx, (tiling, (period_low, period_high)) in enumerate(
            zip(tilings, period_ranges, strict=True)
        ):
            layout, freq_low, freq_high, n_band_bins = f0_interval_layout(
                tiling,
                job_f0_low,
                job_f0_high,
                velocity=velocity,
                window=window,
                search_loop_config=search_loop_config,
            )
            memory_gb = search_memory_gb(
                layout,
                n_band_bins=n_band_bins,
                n_sfts=n_sfts,
                sky_batch_size=search_loop_config.sky_batch_size,
                f0_batch_size=search_loop_config.f0_batch_size,
            )
            hours = None
            if rates is not None:
                hours = search_hours(
                    layout, n_band_bins=n_band_bins, n_sfts=n_sfts, rates=rates
                )
            job_setups.append(
                JobSetup(
                    job_idx=len(job_setups),
                    f0_low=job_f0_low,
                    f0_high=job_f0_high,
                    period_low=period_low,
                    period_high=period_high,
                    tiling_idx=tiling_idx,
                    freq_low=float(freq_low),
                    freq_high=float(freq_high),
                    files=sft_metadata.files_covering(
                        freq_low, freq_high, sft_glob=config.sft_glob
                    ),
                    n_sky_points=int(layout.bank.alpha.size),
                    n_templates=layout.bank.n_templates,
                    memory_gb=memory_gb,
                    hours=hours,
                )
            )

    n_templates_per_job = np.array(
        [job_setup.n_templates for job_setup in job_setups], dtype=float
    )
    n_templates_total = int(n_templates_per_job.sum())
    run_logger.info(f"Total templates: {n_templates_total:.3e}")
    run_logger.info(
        f"Per-job templates: {n_templates_per_job.min():.3e} to "
        f"{n_templates_per_job.max():.3e}"
    )
    memory_gb_per_job = np.array([job_setup.memory_gb for job_setup in job_setups])
    run_logger.info(
        f"Per-job memory: {memory_gb_per_job.min():.1f} to "
        f"{memory_gb_per_job.max():.1f} GB"
    )
    if rates is not None:
        hours_per_job = np.array([job_setup.hours for job_setup in job_setups])
        run_logger.info(
            f"Per-job hours: [{hours_per_job.min():.1f}, {hours_per_job.max():.1f}]"
        )
        run_logger.info(f"Total hours: {hours_per_job.sum():.1f}")
        for (period_low, period_high), period_hours in zip(
            period_ranges, hours_per_job.reshape(n_f0_intervals, -1).T, strict=True
        ):
            run_logger.info(
                f"  period {period_low:g}-{period_high:g} d: {period_hours.sum():.1f} h"
            )

    # the jobs over a limit
    over_limit = [
        job_setup
        for job_setup in job_setups
        if job_setup.memory_gb > compute.memory_gb
        or (job_setup.hours is not None and job_setup.hours > compute.hours)
    ]
    if over_limit:
        run_logger.info(
            f"WARNING: {len(over_limit)} of {len(job_setups)} jobs exceed "
            f"{compute.memory_gb:g} GB or {compute.hours:g} h:"
        )
        for job_setup in over_limit:
            hours_text = "" if job_setup.hours is None else f", {job_setup.hours:.1f} h"
            run_logger.info(
                f"  job {job_setup.job_idx}: f0 {job_setup.f0_low:g}-"
                f"{job_setup.f0_high:g} Hz, period {job_setup.period_low:g}-"
                f"{job_setup.period_high:g} d, {job_setup.memory_gb:.1f} GB{hours_text}"
            )

    setup = SearchSetup(
        jpec=importlib.metadata.version("jpec"),
        data=DataSetup(
            sft_glob=config.sft_glob,
            detectors=sft_metadata.detectors,
            n_sfts=n_sfts,
            tsft=tsft,
            window=window,
            t_ref=t_ref,
        ),
        tilings=[tiling.to_dict() for tiling in tilings],
        search_loop=search_loop_config,
        n_toplist=config.n_toplist,
        toplist_f0_width=config.toplist_f0_width,
        n_templates_total=n_templates_total,
        jobs=job_setups,
    )
    run_logger.save_setup(output.stem, setup)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the command's flags.

    Args:
        parser: The command's parser.
    """
    parser.add_argument(
        "config", help="YAML describing the search; see examples/search.yaml"
    )
    parser.add_argument(
        "--sft-metadata",
        required=True,
        help="jpec check-sft result, the data to search",
    )
    parser.add_argument(
        "--timing",
        default=None,
        help="jpec time-search result, used to estimate each job's hours; without "
        "it, only each job's memory is estimated",
    )
    parser.add_argument(
        "--output",
        default="setup/search_setup.json",
        help="JSON file the setup is written to",
    )
