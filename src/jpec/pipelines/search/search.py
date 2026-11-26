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
from jpec.engine.loop import BandSearch
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

    # the A5* lattice over the sky, on the job's period range
    bank = TemplateBank.over_f0_interval(
        LatticeTiling.from_dict(setup.tilings[job_setup.tiling_idx]),
        job_setup.f0_low,
        job_setup.f0_high,
        velocity=search_data.velocity,
        window=setup.data.window,
    )
    # one toplist per toplist_f0_width of the job's f0 interval
    toplist_f0_edges = np.asarray(setup.toplist_f0_edges(job_setup))
    band_search = BandSearch(
        bank,
        search_data,
        search_loop_config,
        n_top=setup.n_toplist,
        toplist_f0_edges=toplist_f0_edges,
    )
    layout = band_search.layout
    run_logger.search_summary(
        setup=setup,
        job_setup=job_setup,
        search_data=search_data,
        layout=layout,
        setup_path=args.setup,
    )

    n_toplists = toplist_f0_edges.size - 1
    toplists = [Toplist(setup.n_toplist) for _ in range(n_toplists)]
    run_logger.start_progress(
        bank, seconds=60.0 * args.progress_minutes, started=started
    )
    n_sky_points = int(bank.alpha.size)
    n_sky_points_done = 0
    for sky_idxs in layout.sky_batches(search_loop_config.sky_batch_size):
        for top in band_search.search_sky_batch(sky_idxs):
            run_logger.progress(n_sky_points_done)
            toplists[top.toplist_idx].add_top(top, bank)
        n_sky_points_done += sky_idxs.size

    toplist_dicts = []
    for toplist, toplist_f0_low, toplist_f0_high in zip(
        toplists, toplist_f0_edges, toplist_f0_edges[1:]
    ):
        candidates = [candidate.to_dict() for candidate in toplist.descending()]
        toplist_dicts.append(
            {
                "f0_low": float(toplist_f0_low),
                "f0_high": float(toplist_f0_high),
                "candidates": candidates,
            }
        )
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
            "n_doppler_references": int(layout.doppler_references.size),
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
                f"[{f0_low}, {f0_high}] Hz is not in the setup's jobs "
                f"[{setup.jobs[0].f0_low}, {setup.jobs[-1].f0_high}] Hz"
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
