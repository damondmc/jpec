r"""Cut the search a config asks for into jobs that each fit the card.

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

from jpec.bank.lattice import BCC_SUBGRID_STEP
from jpec.bank.template_bank import OrbitalTemplateGrid
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
from jpec.constant import GB
from jpec.detector.site import Observation, reference_epoch
from jpec.engine.data import band_edges_for_f0
from jpec.engine.layout import (
    SubgridLayout,
    doppler_reference_f0_width,
    subgrid_layout,
)
from jpec.pipelines.monitor import RunLogger
from jpec.waveform.window import Window

COMPLEX64_BYTES = 8
FLOAT32_BYTES = 4
# halvings to find the widest f0 interval that fits
N_BISECTIONS = 24


def template_and_sky_point_counts(
    tiling: LatticeTiling,
    velocity: Float[np.ndarray, "n_sft 3"],
    f0_low: float,
    f0_high: float,
) -> tuple[int, int]:
    """Templates an f0 interval holds over the whole sky, and the sky points it has.

    The sky points are those TemplateBank.over_f0_interval builds, at the interval's
    midpoint.

    Args:
        tiling: The tiling the templates are built on.
        velocity: Detector velocity `v/c` at each SFT, every detector.
        f0_low: Low edge of the interval [Hz].
        f0_high: High edge [Hz].

    Returns:
        (Number of templates over the box, number of sky points)
    """
    alpha, _ = tiling.sky_points(velocity, f0=0.5 * (f0_low + f0_high))
    n_templates_per_sky_point = sum(
        orbital_subgrid.n_templates_per_sky_point
        for orbital_subgrid in tiling.orbital_subgrids(f0_low, f0_high)
    )
    return n_templates_per_sky_point * int(alpha.size), int(alpha.size)


def f0_interval_layouts(
    tiling: LatticeTiling,
    f0_low: float,
    f0_high: float,
    *,
    window: Window,
    search_loop_config: SearchLoopConfig,
) -> tuple[list[SubgridLayout], int]:
    """Each subgrid of an f0 interval laid out on the band a search over it loads.

    Args:
        tiling: The tiling the templates are built on.
        f0_low: Low edge of the interval [Hz].
        f0_high: High edge [Hz].
        window: The window the SFTs were made with.
        search_loop_config: How the search loop batches its work.

    Returns:
        The layouts, in the bank's order, and the bins of the band.
    """
    freq_low, freq_high = band_edges_for_f0(
        tiling.prior, f0_low, f0_high, tsft=tiling.tsft
    )
    n_band_bins = round((freq_high - freq_low) * tiling.tsft)
    layouts = [
        subgrid_layout(
            OrbitalTemplateGrid(
                orbital_subgrid=orbital_subgrid, tsft=tiling.tsft, window=window
            ),
            freq_low=freq_low,
            n_band_bins=n_band_bins,
            n_sub_bin_offsets=search_loop_config.n_sub_bin_offsets,
            period_batch_size=search_loop_config.period_batch_size,
            harmonic_fraction=search_loop_config.harmonic_fraction,
        )
        for orbital_subgrid in tiling.orbital_subgrids(f0_low, f0_high)
    ]
    return layouts, n_band_bins


def search_memory_gb(
    layouts: list[SubgridLayout],
    *,
    n_band_bins: int,
    n_sfts: int,
    sky_batch_size: int,
    f0_batch_size: int,
) -> float:
    """Giga-bytes a search over a band holds at its peak.

    Args:
        layouts: Its subgrids laid out on the band.
        n_band_bins: Bins of the band.
        n_sfts: SFTs over every detector.
        sky_batch_size: Sky points per transform.
        f0_batch_size: f0 points matched at once.

    Returns:
        Peak giga-bytes the search holds.
    """
    # the SFT time series and the floor
    loaded_bytes = n_sfts * n_band_bins * (COMPLEX64_BYTES + FLOAT32_BYTES)
    template_harmonics_bytes = 0
    orbital_phase_bytes = 0
    subgrid_bytes = 0
    for layout in layouts:
        template_grid = layout.template_grid
        j_max = template_grid.j_max
        n_window_bins = template_grid.bin_offsets.shape[0]
        n_doppler_references = int(layout.doppler_references.size)
        n_transformed_bins = (
            layout.transformed_bin_idxs.stop - layout.transformed_bin_idxs.start
        )
        n_window_modulation_depths = layout.n_window_modulation_depths
        # the most harmonics any period batch transforms and matches
        n_batch_harmonics = max(batch.n_harmonics for batch in layout.period_batches)
        n_groups, n_members = layout.group_f0_idxs.shape
        n_matched_f0_points = min(n_groups, max(f0_batch_size // n_members, 1)) * (
            n_members
        )
        template_harmonics_bytes += (
            layout.sub_bin_offsets.size
            * template_grid.orbital_subgrid.modulation_depth_bins.size
            * j_max
            * n_window_bins
            * COMPLEX64_BYTES
        )
        orbital_phase_bytes += layout.omega.size * n_sfts * FLOAT32_BYTES
        # the batch of sky points' transformed bins, at every reference
        stack_bytes = (
            sky_batch_size
            * n_doppler_references
            * n_sfts
            * n_transformed_bins
            * FLOAT32_BYTES
        )
        # building the stack: every sky point's whole corrected band in flight, and
        # the transformed bins as slices, expanded and stacked
        doppler_correction_bytes = (
            sky_batch_size * n_doppler_references * n_sfts * n_band_bins * FLOAT32_BYTES
            + 3 * stack_bytes
        )
        # the float32 cosine and sine bases at every sky point, the batch's harmonics
        # twice, and the match's overlaps and S(phi_b)
        per_period_bytes = (
            2 * sky_batch_size * n_batch_harmonics * n_sfts * FLOAT32_BYTES
            + 2
            * sky_batch_size
            * n_doppler_references
            * n_batch_harmonics
            * n_transformed_bins
            * COMPLEX64_BYTES
            + n_matched_f0_points
            * n_window_modulation_depths
            * n_batch_harmonics
            * COMPLEX64_BYTES
            + n_matched_f0_points
            * n_window_modulation_depths
            * template_grid.n_phi
            * FLOAT32_BYTES
        )
        # transforming: the stack twice, and a full run of periods
        transform_bytes = 2 * stack_bytes + layout.period_batch_size * per_period_bytes
        subgrid_bytes = max(subgrid_bytes, doppler_correction_bytes, transform_bytes)
    return (
        loaded_bytes + template_harmonics_bytes + orbital_phase_bytes + subgrid_bytes
    ) / GB


def search_hours(
    layouts: list[SubgridLayout],
    *,
    n_band_bins: int,
    n_sky_points: int,
    n_sfts: int,
    rates: dict[str, float],
) -> float:
    """Hours a search over a band takes.

    Args:
        layouts: Its subgrids laid out on the band.
        n_band_bins: Bins of the band.
        n_sky_points: Sky points searched.
        n_sfts: SFTs over every detector.
        rates: The `rates` of a jpec time-search result.

    Returns:
        Hours.
    """
    seconds_per_sky_point = 0.0
    for layout in layouts:
        template_grid = layout.template_grid
        # once per subgrid and sky point, at every reference
        seconds_per_sky_point += float(
            layout.doppler_references.size
            * n_sfts
            * n_band_bins
            * rates["doppler_correction"]
        )
        n_transformed_bins = (
            layout.transformed_bin_idxs.stop - layout.transformed_bin_idxs.start
        )
        # every batch is a full run of periods, at its own harmonics
        n_batch_harmonics = sum(batch.n_harmonics for batch in layout.period_batches)
        n_batch_templates = (
            layout.period_batch_size
            * layout.window_start_idxs.size
            * layout.n_window_modulation_depths
        )
        # the transform and the overlaps per period and harmonic
        transform_seconds = (
            layout.period_batch_size
            * layout.doppler_references.size
            * n_sfts
            * n_transformed_bins
            * rates["data_harmonics"]
        )
        overlap_seconds = (
            n_batch_templates * template_grid.bin_offsets.shape[0] * rates["match"]
        )
        # S(phi_b) and its maximum per template, at every orbital phase
        phase_seconds = n_batch_templates * template_grid.n_phi * rates["phase_maximum"]
        seconds_per_sky_point += float(
            n_batch_harmonics * (transform_seconds + overlap_seconds)
            + len(layout.period_batches) * phase_seconds
        )
    return n_sky_points * seconds_per_sky_point / 3600.0


def job_cost(
    tiling: LatticeTiling,
    velocity: Float[np.ndarray, "n_sft 3"],
    f0_low: float,
    f0_high: float,
    *,
    window: Window,
    n_sfts: int,
    search_loop_config: SearchLoopConfig,
    rates: dict[str, float] | None,
) -> tuple[float, float]:
    """Giga-bytes a job over an f0 interval holds at its peak, and the hours it takes.

    Args:
        tiling: The tiling the templates are built on.
        velocity: Detector velocity `v/c` at each SFT, every detector.
        f0_low: Low edge of the interval [Hz].
        f0_high: High edge [Hz].
        window: The window the SFTs were made with.
        n_sfts: SFTs over every detector.
        search_loop_config: How the search loop batches its work.
        rates: The `rates` of a jpec time-search result; None prices no hours.

    Returns:
        (Giga-bytes, hours), the hours 0 without rates.
    """
    layouts, n_band_bins = f0_interval_layouts(
        tiling, f0_low, f0_high, window=window, search_loop_config=search_loop_config
    )
    gigabytes = search_memory_gb(
        layouts,
        n_band_bins=n_band_bins,
        n_sfts=n_sfts,
        sky_batch_size=search_loop_config.sky_batch_size,
        f0_batch_size=search_loop_config.f0_batch_size,
    )
    if rates is None:
        return gigabytes, 0.0
    # the sky points at the interval's midpoint, as TemplateBank.over_f0_interval builds
    alpha, _ = tiling.sky_points(velocity, f0=0.5 * (f0_low + f0_high))
    job_hours = search_hours(
        layouts,
        n_band_bins=n_band_bins,
        n_sky_points=int(alpha.size),
        n_sfts=n_sfts,
        rates=rates,
    )
    return gigabytes, job_hours


def split_job_by_f0(
    tiling: LatticeTiling,
    velocity: Float[np.ndarray, "n_sft 3"],
    *,
    window: Window,
    n_sfts: int,
    memory_gb: float,
    hours: float,
    search_loop_config: SearchLoopConfig,
    rates: dict[str, float] | None,
) -> list[tuple[float, float]]:
    """The f0 intervals, tiling the prior's f0 range, that each fit in `memory_gb`.

    Each also fits in `hours` when the rates are known.

    Args:
        tiling: The tiling the templates are built on.
        velocity: Detector velocity `v/c` at each SFT, every detector.
        window: The window the SFTs were made with.
        n_sfts: SFTs over every detector.
        memory_gb: Giga-bytes one job may hold.
        hours: Hours one job may take.
        search_loop_config: How the search loop batches its work.
        rates: The `rates` of a jpec time-search result; None splits on memory alone.

    Returns:
        `(f0_low, f0_high)` of each job, in order.
    """
    f0_low, f0_high = tiling.prior.f0
    subgrid_f0_step = tiling.f0_step * BCC_SUBGRID_STEP
    # a subgrid's f0 points run up to two of its steps past the job's edges
    f0_overhang = 2.0 * subgrid_f0_step
    # the f0 span one Doppler reference holds to the tolerance
    f0_width_per_reference = doppler_reference_f0_width(
        tsft=tiling.tsft, n_sub_bin_offsets=search_loop_config.n_sub_bin_offsets
    )
    f0_intervals: list[tuple[float, float]] = []
    low = f0_low
    while low < f0_high - 1e-9:

        def fits(high: float) -> bool:
            # the job being cut, from low up to high, within both limits
            gigabytes, job_hours = job_cost(
                tiling,
                velocity,
                low,
                high,
                window=window,
                n_sfts=n_sfts,
                search_loop_config=search_loop_config,
                rates=rates,
            )
            return gigabytes <= memory_gb and job_hours <= hours

        # the widest interval that fits
        if fits(f0_high):
            high = f0_high
        else:
            if not fits(low + subgrid_f0_step):
                limits = f"{memory_gb:.1f} GB"
                if rates is not None:
                    limits += f" or {hours:.1f} h"
                raise SystemExit(
                    f"one f0 step at {low:.3f} Hz already exceeds {limits}; "
                    "raise compute.memory_gb or compute.hours"
                )
            high, failing_high = low + subgrid_f0_step, f0_high
            for _ in range(N_BISECTIONS):
                middle = 0.5 * (high + failing_high)
                if fits(middle):
                    high = middle
                else:
                    failing_high = middle
        # the hours per Hz fall with the width until the next Doppler reference doubles
        # the transform, so the cheapest job is that one or one whose f0 points span
        # just under a whole number of references
        if rates is not None:
            reference_counts = np.arange(
                1, int((high - low + f0_overhang) / f0_width_per_reference) + 1
            )
            reference_edges = (
                low
                + f0_width_per_reference * (1.0 - 1e-9) * reference_counts
                - f0_overhang
            )
            candidate_f0_highs = [
                high,
                *reference_edges[reference_edges > low].tolist(),
            ]
            hours_per_hz = []
            for candidate_f0_high in candidate_f0_highs:
                _, job_hours = job_cost(
                    tiling,
                    velocity,
                    low,
                    candidate_f0_high,
                    window=window,
                    n_sfts=n_sfts,
                    search_loop_config=search_loop_config,
                    rates=rates,
                )
                hours_per_hz.append(job_hours / (candidate_f0_high - low))
            high = candidate_f0_highs[int(np.argmin(hours_per_hz))]
        f0_intervals.append((low, high))
        low = high

    return f0_intervals


def run(args: argparse.Namespace) -> None:
    """Split the config's f0 range into jobs and write the setup.

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

    tiling = LatticeTiling.from_metric(
        config.parameter_space.prior(),
        observation=observation,
        mismatch=template_bank_config.mismatch,
        n_ap_blocks=template_bank_config.n_ap_blocks,
        window=window,
    )

    # cut the f0 range into jobs that each fit the memory, and the time when the rates
    # are known
    timing = None if args.timing is None else json.loads(Path(args.timing).read_text())
    rates = None if timing is None else timing["rates"]
    f0_intervals = split_job_by_f0(
        tiling,
        velocity,
        window=window,
        n_sfts=n_sfts,
        memory_gb=compute.memory_gb,
        hours=compute.hours,
        search_loop_config=search_loop_config,
        rates=rates,
    )
    if timing is None:
        run_logger.info(
            f"{len(f0_intervals)} jobs under {compute.memory_gb:.1f} GB each; "
            "no timing given, so the hours are not priced: time the first job with "
            "jpec time-search and run this again with --timing"
        )
    else:
        run_logger.info(
            f"{len(f0_intervals)} jobs under {compute.memory_gb:.1f} GB "
            f"and {compute.hours:.1f} h each, at the rates of {timing['host']}."
        )

    # what each job searches, loads, holds and takes
    job_setups = []
    for job_idx, (job_f0_low, job_f0_high) in enumerate(f0_intervals):
        n_templates, n_sky_points = template_and_sky_point_counts(
            tiling, velocity, job_f0_low, job_f0_high
        )
        freq_low, freq_high = band_edges_for_f0(
            tiling.prior, job_f0_low, job_f0_high, tsft=tsft
        )
        layouts, n_band_bins = f0_interval_layouts(
            tiling,
            job_f0_low,
            job_f0_high,
            window=window,
            search_loop_config=search_loop_config,
        )
        job_setups.append(
            JobSetup(
                job_idx=job_idx,
                f0_low=job_f0_low,
                f0_high=job_f0_high,
                freq_low=float(freq_low),
                freq_high=float(freq_high),
                files=sft_metadata.files_covering(
                    freq_low, freq_high, sft_glob=config.sft_glob
                ),
                n_sky_points=n_sky_points,
                n_templates=n_templates,
                memory_gb=search_memory_gb(
                    layouts,
                    n_band_bins=n_band_bins,
                    n_sfts=n_sfts,
                    sky_batch_size=search_loop_config.sky_batch_size,
                    f0_batch_size=search_loop_config.f0_batch_size,
                ),
                hours=None
                if rates is None
                else search_hours(
                    layouts,
                    n_band_bins=n_band_bins,
                    n_sky_points=n_sky_points,
                    n_sfts=n_sfts,
                    rates=rates,
                ),
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
    if rates is not None:
        hours_per_job = np.array([job_setup.hours for job_setup in job_setups])
        run_logger.info(
            f"Per-job hours: {hours_per_job.min():.1f} to {hours_per_job.max():.1f}"
        )
        run_logger.info(f"Total hours: {hours_per_job.sum():.1f}")

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
        tiling=tiling.to_dict(),
        search_loop=search_loop_config,
        n_toplist=config.n_toplist,
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
        help="jpec time-search result, the rates the jobs are priced at; without one "
        "the jobs are cut on memory alone",
    )
    parser.add_argument(
        "--output",
        default="setup/search_setup.json",
        help="JSON file the setup is written to",
    )
