r"""Time each search loop stage on one sky point, for the rates jobs are priced at.

Usage:

  jpec time-search --setup setup/search_setup.json --platform gpu --batches 3 \
      --repeats 3 --output results
"""

import argparse
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import SearchSetup, read_json
from jpec.detector.doppler import solar_roemer_delay
from jpec.engine.data import SearchData
from jpec.engine.loop import BandSearch
from jpec.pipelines.monitor import RunLogger


def mean_seconds(function: Callable[..., Any], *args: Any, repeats: int) -> float:
    """The mean wall time of a call, its device work finished.

    Args:
        function: What to time; whatever it returns is waited on.
        *args: What it is called with.
        repeats: Timed calls, after one untimed call that compiles.

    Returns:
        Mean seconds per call.
    """
    jax.block_until_ready(function(*args))
    began = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(function(*args))
    return (time.perf_counter() - began) / repeats


def run(args: argparse.Namespace) -> None:
    """Time the stages on the first sky batch; save the rates and one sky point's cost.

    Args:
        args: The parsed flags.
    """
    run_logger = RunLogger(
        "time_search", args.output, platform=args.platform, x64=False
    )
    setup = read_json(args.setup, SearchSetup)
    job_setup = setup.jobs[0]
    search_loop_config = setup.search_loop

    search_data = SearchData.load(
        setup.data,
        job_setup.files,
        job_setup.freq_low,
        job_setup.freq_high,
        floor_width=search_loop_config.floor_width,
        floor_batch_size=search_loop_config.floor_batch_size,
    )
    bank = TemplateBank.over_f0_interval(
        LatticeTiling.from_dict(setup.tilings[job_setup.tiling_idx]),
        job_setup.f0_low,
        job_setup.f0_high,
        velocity=search_data.velocity,
        window=setup.data.window,
    )
    # the job's band search, toplists included, as jpec search builds it
    band_search = BandSearch(
        bank,
        search_data,
        search_loop_config,
        n_top=setup.n_toplist,
        toplist_f0_edges=np.asarray(setup.toplist_f0_edges(job_setup)),
    )
    layout = band_search.layout

    j_max = layout.j_max
    n_window_bins = layout.bin_offsets.shape[0]
    n_f0_points = int(layout.window_start_idxs.size)
    n_doppler_references = int(layout.doppler_references.size)
    n_window_modulation_depths = layout.n_window_modulation_depths
    period_batch_size = layout.period_batch_size
    # the bank's first sky batch, of one omega^2 class, and its first sky point
    sky_idxs = layout.sky_batches(search_loop_config.sky_batch_size)[0]
    sky_batch_size = int(sky_idxs.size)
    omega2_class = int(bank.omega2_class[sky_idxs[0]])
    sky_shifts = band_search.sky_shifts(int(sky_idxs[0]))
    n_period_batches = len(layout.template_batches)

    doppler_seconds = mean_seconds(
        band_search.doppler_correct, int(sky_idxs[0]), repeats=args.repeats
    )
    weighted_excess_power = jnp.stack(
        [band_search.doppler_correct(int(sky_idx)) for sky_idx in sky_idxs]
    )
    n_transformed_bins = int(weighted_excess_power.shape[-1])

    solar_roemer_delays = solar_roemer_delay(
        search_data.position,
        alpha=jnp.asarray(bank.alpha[sky_idxs]),
        delta=jnp.asarray(bank.delta[sky_idxs]),
    )

    n_templates = n_f0_points * period_batch_size * n_window_modulation_depths
    transform_seconds, match_seconds = [], []
    transform_work = []
    # every match timed, at the batch's harmonics and at half of them: its seconds,
    # its overlap multiply-adds over the windows widened by n_pad, and its orbital
    # phases evaluated
    match_fit_seconds, match_fit_work = [], []
    for batch_idx, batch in enumerate(
        layout.template_batches[: args.n_period_batches_timed]
    ):
        transform_args = (
            weighted_excess_power,
            solar_roemer_delays,
            omega2_class,
            batch,
        )
        transform_seconds.append(
            mean_seconds(band_search.transform, *transform_args, repeats=args.repeats)
        )
        batch_data_harmonics = band_search.transform(*transform_args)[0]

        # the match and the device-side cut to the top k, as the search runs them
        match_seconds.append(
            mean_seconds(
                band_search.match,
                batch_data_harmonics,
                sky_shifts,
                batch_idx,
                repeats=args.repeats,
            )
        )
        half_harmonics = max(batch.n_harmonics // 2, 1)
        half_data_harmonics = batch_data_harmonics[:, :, :half_harmonics]
        half_match_seconds = mean_seconds(
            band_search.match,
            half_data_harmonics,
            sky_shifts,
            batch_idx,
            repeats=args.repeats,
        )
        for n_harmonics, seconds in [
            (batch.n_harmonics, match_seconds[-1]),
            (half_harmonics, half_match_seconds),
        ]:
            match_fit_seconds.append(seconds)
            match_fit_work.append(
                [
                    n_templates * n_harmonics * (n_window_bins + layout.n_pad),
                    n_templates * layout.n_phi,
                ]
            )
        # multiply-adds of the transform at one sky point and reference
        transform_work.append(
            period_batch_size
            * batch.n_harmonics
            * search_data.n_sfts
            * n_transformed_bins
        )

    # seconds per call, and calls per sky point
    stages = {
        "doppler correction": (
            doppler_seconds / n_doppler_references,
            n_doppler_references,
        ),
        "transform": (
            float(np.mean(transform_seconds)) / (sky_batch_size * n_doppler_references),
            n_period_batches * n_doppler_references,
        ),
        "match": (float(np.mean(match_seconds)), n_period_batches),
    }
    seconds_per_sky_point = {
        name: call * calls for name, (call, calls) in stages.items()
    }
    sky_point_seconds = sum(seconds_per_sky_point.values())

    # the match's seconds per overlap multiply-add over the widened window and per
    # orbital phase of a template, fitted to every match timed
    (match_rate, phase_maximum_rate), *_ = np.linalg.lstsq(
        np.array(match_fit_work, dtype=float),
        np.array(match_fit_seconds),
        rcond=None,
    )

    # what jpec search-setup prices a job with: seconds per SFT bin of the Doppler
    # correction at one reference, per multiply-add of the transform at one sky point
    # and reference, its share of the batch's basis included, over the batches timed,
    # and the match's two rates
    rates = {
        "doppler_correction": stages["doppler correction"][0]
        / (search_data.n_sfts * search_data.n_band_bins),
        "data_harmonics": sum(transform_seconds)
        / (sky_batch_size * n_doppler_references)
        / sum(transform_work),
        "match": float(match_rate),
        "phase_maximum": float(phase_maximum_rate),
    }

    run_logger.info(
        f"{n_f0_points} f0 points, {n_period_batches} period batches of "
        f"{period_batch_size}, {sky_batch_size} sky points per transform, "
        f"{n_window_modulation_depths} window modulation depths, j_max {j_max}, "
        f"{n_window_bins} window bins + {layout.n_pad} for the f0 shifts, "
        f"{search_data.n_band_bins} band bins, {n_transformed_bins} transformed, "
        f"{search_data.n_sfts} SFTs, {n_doppler_references} Doppler references, "
        f"{bank.alpha.size} sky points"
    )
    run_logger.info(
        f"{'stage':<24}{'ms per call':>13}{'calls per sky point':>21}"
        f"{'s per sky point':>17}{'share':>8}"
    )
    for name, (call, calls) in stages.items():
        run_logger.info(
            f"{name:<24}{call * 1e3:>13.2f}{calls:>21}"
            f"{seconds_per_sky_point[name]:>17.2f}"
            f"{seconds_per_sky_point[name] / sky_point_seconds:>8.1%}"
        )
    run_logger.info(f"{'one sky point':<24}{'':>13}{'':>21}{sky_point_seconds:>17.2f}")
    # the device's peak against the memory model's for the job timed
    device_memory_gb = run_logger.device_memory_gb()
    if device_memory_gb is not None:
        run_logger.info(
            f"peak device memory {device_memory_gb[1]:.2f} GB; the setup models job 0 "
            f"at {job_setup.memory_gb:.2f} GB"
        )

    run_logger.save_result(
        "time_search",
        {
            "setup": args.setup,
            "n_f0_points": n_f0_points,
            "n_period_batches": n_period_batches,
            "sky_batch_size": sky_batch_size,
            "period_batch_size": period_batch_size,
            "n_window_modulation_depths": n_window_modulation_depths,
            "j_max": j_max,
            "n_window_bins": n_window_bins,
            "n_pad": layout.n_pad,
            "n_band_bins": search_data.n_band_bins,
            "n_transformed_bins": n_transformed_bins,
            "n_sfts": search_data.n_sfts,
            "n_doppler_references": n_doppler_references,
            "n_sky_points": int(bank.alpha.size),
            "seconds_per_call": {name: call for name, (call, _) in stages.items()},
            "calls_per_sky_point": {name: calls for name, (_, calls) in stages.items()},
            "seconds_per_sky_point": seconds_per_sky_point,
            "modelled_gb": job_setup.memory_gb,
            # read by jpec search-setup
            "rates": rates,
        },
        per_host=True,
    )


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
        "--batches",
        dest="n_period_batches_timed",
        type=int,
        default=3,
        help="period batches timed",
    )
    parser.add_argument("--repeats", type=int, default=3, help="timed calls per stage")
