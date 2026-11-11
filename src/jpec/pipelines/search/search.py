r"""Run the complex TwoSpect search over the jobs of a setup.

Usage:

  jpec search --setup setup/search_setup.json --jobs 0 1 --platform gpu \
      --output results --progress-minutes 10
"""

import argparse
import time

import numpy as np

from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import JobSetup, SearchSetup, read_json
from jpec.engine.candidates import Toplist
from jpec.engine.data import SearchData
from jpec.engine.loop import HarmonicBank, top_templates_at_sky_points
from jpec.pipelines.monitor import RunLogger


def search_job(
    run_logger: RunLogger,
    setup: SearchSetup,
    job_setup: JobSetup,
    args: argparse.Namespace,
) -> None:
    """Search one job's f0 interval and period range over the sky, save its toplists.

    Args:
        run_logger: The run's log and output.
        setup: The search setup.
        job_setup: The job's row of it.
        args: The parsed flags.
    """
    started = time.perf_counter()
    search_loop_config = setup.search_loop

    search_data = SearchData.load(
        setup.data,
        job_setup.files,
        job_setup.freq_low,
        job_setup.freq_high,
        floor_width=search_loop_config.floor_width,
        floor_batch_size=search_loop_config.floor_batch_size,
    )

    # the bcc lattice's 2 subgrids over the sky lattice, on the job's period range
    bank = TemplateBank.over_f0_interval(
        LatticeTiling.from_dict(setup.tilings[job_setup.tiling_idx]),
        job_setup.f0_low,
        job_setup.f0_high,
        velocity=search_data.velocity,
        window=setup.data.window,
    )
    # tau[j, k] and orbital phases once per subgrid, shared by every sky point
    harmonic_bank = HarmonicBank(
        bank,
        times=search_data.times,
        freq_low=search_data.freq_low,
        n_band_bins=search_data.n_band_bins,
        n_sub_bin_offsets=search_loop_config.n_sub_bin_offsets,
        period_batch_size=search_loop_config.period_batch_size,
        harmonic_fraction=search_loop_config.harmonic_fraction,
        phase_mismatch=search_loop_config.phase_mismatch,
        doppler_mismatch=search_loop_config.doppler_mismatch,
        modulation_depth_batch_size=search_loop_config.modulation_depth_batch_size,
    )
    n_doppler_references_per_subgrid = [
        int(subgrid.layout.doppler_references.size)
        for subgrid in harmonic_bank.subgrids
    ]
    run_logger.search_summary(
        setup,
        job_setup,
        search_data,
        bank,
        setup_path=args.setup,
        n_doppler_references_per_subgrid=n_doppler_references_per_subgrid,
    )

    # one toplist per toplist_f0_width of the job's f0 interval
    toplist_f0_edges = np.asarray(setup.toplist_f0_edges(job_setup))
    n_toplists = toplist_f0_edges.size - 1
    toplists = [Toplist(setup.n_toplist) for _ in range(n_toplists)]
    run_logger.start_progress(
        bank, seconds=60.0 * args.progress_minutes, started=started
    )
    n_sky_points = int(bank.alpha.size)
    sky_batch_size = search_loop_config.sky_batch_size
    for sky_start_idx in range(0, n_sky_points, sky_batch_size):
        for top in top_templates_at_sky_points(
            search_data,
            harmonic_bank,
            sky_idxs=range(
                sky_start_idx, min(sky_start_idx + sky_batch_size, n_sky_points)
            ),
            f0_batch_size=search_loop_config.f0_batch_size,
            n_top=setup.n_toplist,
            toplist_f0_edges=toplist_f0_edges,
        ):
            run_logger.progress(top.sky_idx, top.subgrid_idx)
            toplists[top.toplist_idx].add_top(top, bank)

    toplist_dicts = [
        {
            "f0_low": float(toplist_f0_low),
            "f0_high": float(toplist_f0_high),
            # loudest first
            "candidates": [candidate.to_dict() for candidate in toplist.descending()],
        }
        for toplist, toplist_f0_low, toplist_f0_high in zip(
            toplists, toplist_f0_edges, toplist_f0_edges[1:]
        )
    ]
    elapsed = time.perf_counter() - started
    n_kept = sum(len(toplist.kept) for toplist in toplists)
    run_logger.info(
        f"Candidates kept: {n_kept} / {n_toplists * setup.n_toplist} in {n_toplists} "
        "toplists."
    )
    run_logger.info(f"Done: time used {elapsed:.1f} s")
    device_memory_gb = run_logger.device_memory_gb()
    if device_memory_gb is not None:
        run_logger.info(
            f"Peak device memory: {device_memory_gb[1]:.2f} GB; the setup models the "
            f"job at {job_setup.memory_gb:.2f} GB"
        )

    run_logger.save_result(
        f"search_job{job_setup.job_idx:05d}",
        {
            "setup": args.setup,
            "job_idx": job_setup.job_idx,
            "f0_low": job_setup.f0_low,
            "f0_high": job_setup.f0_high,
            "period_low": job_setup.period_low,
            "period_high": job_setup.period_high,
            "n_sky_points": n_sky_points,
            "t_ref": search_data.t_ref,
            "seconds": elapsed,
            "modelled_gb": job_setup.memory_gb,
            "n_doppler_references_per_subgrid": n_doppler_references_per_subgrid,
            "search_loop": search_loop_config.model_dump(),
            # in f0 order
            "toplists": toplist_dicts,
        },
    )


def run(args: argparse.Namespace) -> None:
    """Search the jobs the flags select.

    Args:
        args: The parsed flags.
    """
    run_logger = RunLogger("search", args.output, platform=args.platform, x64=False)
    setup = read_json(args.setup, SearchSetup)

    # the jobs overlapping the f0 range, or those in the job index range
    job_setups = setup.jobs
    if args.f0_range is not None:
        f0_low, f0_high = args.f0_range
        job_setups = [
            job_setup
            for job_setup in setup.jobs
            if job_setup.f0_low < f0_high and job_setup.f0_high > f0_low
        ]
        if not job_setups:
            raise SystemExit(
                f"no job overlaps {f0_low} to {f0_high} Hz; the setup covers "
                f"{setup.jobs[0].f0_low} to {setup.jobs[-1].f0_high} Hz"
            )
    elif args.jobs is not None:
        job_setups = setup.jobs[args.jobs[0] : args.jobs[1]]

    for job_setup in job_setups:
        search_job(run_logger, setup, job_setup, args)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the command's flags.

    Args:
        parser: The command's parser.
    """
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default=None,
        help="JAX backend; default is whatever JAX picks",
    )
    parser.add_argument(
        "--output", default="results", help="directory the JSON result is written into"
    )
    parser.add_argument("--setup", required=True, help="jpec search-setup result")
    parser.add_argument(
        "--jobs",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "STOP"),
        help="jobs [START, STOP); all by default",
    )
    parser.add_argument(
        "--f0-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="every job whose f0 interval overlaps [LOW, HIGH) [Hz]; takes precedence "
        "over --jobs",
    )
    parser.add_argument(
        "--progress-minutes",
        type=float,
        default=10.0,
        metavar="MINUTES",
        help="minutes between progress lines within a job; 0 prints none",
    )
