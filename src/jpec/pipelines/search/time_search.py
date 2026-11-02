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
from jaxtyping import Array, Complex, Float, Int

from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import SearchSetup, read_json
from jpec.engine.data import SearchData
from jpec.engine.layout import subgrid_layout
from jpec.engine.loop import (
    doppler_corrected_power_per_reference,
    jitted_data_harmonics_per_reference,
    jitted_orbital_basis,
    loudest_in_box,
    matched_filter_grid,
)
from jpec.pipelines.monitor import RunLogger
from jpec.pipelines.search.search_setup import GB
from jpec.stat.matched_filter import orbital_phase_mod_2pi


def mean_seconds(function: Callable[[], Any], *, repeats: int) -> float:
    """The mean wall time of a call, its device work finished.

    Args:
        function: What to time; whatever it returns is waited on.
        repeats: Timed calls, after one untimed call that compiles.

    Returns:
        Mean seconds per call.
    """
    jax.block_until_ready(function())
    began = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(function())
    return (time.perf_counter() - began) / repeats


def run(args: argparse.Namespace) -> None:
    """Time the stages on the first subgrid; save the rates and one sky point's cost.

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
        LatticeTiling.from_dict(setup.tiling),
        job_setup.f0_low,
        job_setup.f0_high,
        velocity=search_data.velocity,
        window=setup.data.window,
    )
    layout = subgrid_layout(
        bank.template_grids[0],
        freq_low=search_data.freq_low,
        n_band_bins=search_data.n_band_bins,
        n_sub_bin_offsets=search_loop_config.n_sub_bin_offsets,
        period_batch_size=search_loop_config.period_batch_size,
    )
    template_grid = layout.template_grid
    template_harmonics = template_grid.template_harmonics_at_offsets(
        layout.sub_bin_offsets,
        batch_size=search_loop_config.modulation_depth_batch_size,
    )

    j_max = template_grid.j_max
    n_window_bins = template_grid.bin_offsets.shape[0]
    n_f0_points = int(layout.window_start_idxs.size)
    n_doppler_references = int(layout.doppler_references.size)
    n_window_modulation_depths = layout.n_window_modulation_depths
    period_batch_size = layout.period_batch_size
    n_subgrids = len(bank.template_grids)
    n_period_batches = len(layout.period_batches)
    in_box = jnp.ones((period_batch_size, n_window_modulation_depths), dtype=bool)

    def doppler_correction() -> Float[Array, "n_doppler_reference n_sft n_band"]:
        return doppler_corrected_power_per_reference(
            search_data.sft_time_series,
            search_data.floor,
            search_data.velocity_per_detector,
            layout.doppler_references,
            tsft=search_data.tsft,
            alpha=float(bank.alpha[0]),
            delta=float(bank.delta[0]),
        )

    doppler_seconds = mean_seconds(doppler_correction, repeats=args.repeats)
    corrected_sft_power = doppler_correction()[..., layout.transformed_bin_idxs]
    n_transformed_bins = int(corrected_sft_power.shape[-1])

    orbital_basis_seconds, transform_seconds, match_seconds = [], [], []
    for batch in layout.period_batches[: args.n_period_batches_timed]:
        period_idxs = slice(
            batch.period_start_idx, batch.period_start_idx + period_batch_size
        )

        def orbital_basis() -> Complex[Array, "n_period n_harmonic n_sft"]:
            # the host's float64 phase wrap counted in, as the search pays it per batch
            batch_orbital_phase = orbital_phase_mod_2pi(
                search_data.times, layout.omega[period_idxs]
            )
            return jitted_orbital_basis(jnp.asarray(batch_orbital_phase), j_max=j_max)

        batch_orbital_basis = orbital_basis()

        def transform() -> Complex[
            Array, "n_doppler_reference n_period n_harmonic n_transformed_bin"
        ]:
            return jitted_data_harmonics_per_reference(
                corrected_sft_power, batch_orbital_basis
            )

        batch_data_harmonics = transform()

        # the match and the device-side cut to the loudest, as the search runs them
        def match() -> tuple[
            Float[Array, " n_loudest"],
            Int[Array, " n_loudest"],
            Int[Array, " n_loudest"],
        ]:
            max_statistic, phase_idxs = matched_filter_grid(
                data_harmonics=batch_data_harmonics,
                template_harmonics=template_harmonics,
                doppler_reference_idxs=layout.doppler_reference_idxs,
                window_start_idxs=layout.window_start_idxs,
                sub_bin_offset_idxs=layout.sub_bin_offset_idxs,
                modulation_depth_start_idx=batch.modulation_depth_start_idx,
                n_matched_modulation_depths=n_window_modulation_depths,
                n_phi=template_grid.n_phi,
                f0_batch_size=search_loop_config.f0_batch_size,
            )
            return loudest_in_box(
                max_statistic,
                phase_idxs,
                in_box,
                n_loudest=min(setup.n_toplist, max_statistic.size),
            )

        orbital_basis_seconds.append(mean_seconds(orbital_basis, repeats=args.repeats))
        transform_seconds.append(mean_seconds(transform, repeats=args.repeats))
        match_seconds.append(mean_seconds(match, repeats=args.repeats))

    # seconds per call, and calls per sky point over every subgrid as this one
    stages = {
        "doppler correction": (
            doppler_seconds / n_doppler_references,
            n_subgrids * n_doppler_references,
        ),
        "orbital basis": (
            float(np.mean(orbital_basis_seconds)),
            n_subgrids * n_period_batches,
        ),
        "transform": (
            float(np.mean(transform_seconds)) / n_doppler_references,
            n_subgrids * n_period_batches * n_doppler_references,
        ),
        "match": (float(np.mean(match_seconds)), n_subgrids * n_period_batches),
    }
    seconds_per_sky_point = {
        name: call * calls for name, (call, calls) in stages.items()
    }
    sky_point_seconds = sum(seconds_per_sky_point.values())

    # what jpec search-setup prices a job with: seconds per SFT bin of the Doppler
    # correction at one reference, per element of the basis, per multiply-add of the
    # transform at one reference and of the match
    orbital_basis_work = period_batch_size * j_max * search_data.n_sfts
    match_work = (
        n_f0_points
        * period_batch_size
        * n_window_modulation_depths
        * j_max
        * n_window_bins
    )
    rates = {
        "doppler_correction": stages["doppler correction"][0]
        / (search_data.n_sfts * search_data.n_band_bins),
        "orbital_basis": stages["orbital basis"][0] / orbital_basis_work,
        "transform": stages["transform"][0] / (orbital_basis_work * n_transformed_bins),
        "match": stages["match"][0] / match_work,
    }

    run_logger.info(
        f"{n_f0_points} f0 points, {n_period_batches} period batches of "
        f"{period_batch_size}, "
        f"{n_window_modulation_depths} window modulation depths, j_max {j_max}, "
        f"{n_window_bins} window bins, {search_data.n_band_bins} band bins, "
        f"{n_transformed_bins} transformed, {search_data.n_sfts} SFTs, "
        f"{n_doppler_references} Doppler references, "
        f"{n_subgrids} subgrids"
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
    # the device's peak against the memory model's for the job timed; no stats on CPU
    memory_stats = jax.devices()[0].memory_stats()
    peak_gb = None if memory_stats is None else memory_stats["peak_bytes_in_use"] / GB
    if peak_gb is not None:
        run_logger.info(
            f"peak device memory {peak_gb:.2f} GB; the setup models job 0 at "
            f"{job_setup.memory_gb:.2f} GB"
        )

    run_logger.save_result(
        "time_search",
        {
            "setup": args.setup,
            "n_f0_points": n_f0_points,
            "n_period_batches": n_period_batches,
            "period_batch_size": period_batch_size,
            "n_window_modulation_depths": n_window_modulation_depths,
            "j_max": j_max,
            "n_window_bins": n_window_bins,
            "n_band_bins": search_data.n_band_bins,
            "n_transformed_bins": n_transformed_bins,
            "n_sfts": search_data.n_sfts,
            "n_doppler_references": n_doppler_references,
            "n_subgrids": n_subgrids,
            "seconds_per_call": {name: call for name, (call, _) in stages.items()},
            "calls_per_sky_point": {name: calls for name, (_, calls) in stages.items()},
            "seconds_per_sky_point": seconds_per_sky_point,
            "peak_gb": peak_gb,
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
