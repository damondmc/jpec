r"""Sensitivity depth of the search on real SFTs, measured by injection.

Usage:

  jpec sensitivity --setup setup/injection_204_mu0.8_setup.json --platform gpu \
      --seed 0 \
      --output results
"""

import argparse
import time

import jax
import numpy as np

from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import InjectionSetup, read_json
from jpec.constant import DAY
from jpec.engine.data import SearchData
from jpec.engine.loop import HarmonicBank, orbital_statistics
from jpec.injection.population import draw_injections
from jpec.injection.signal import simulated_source
from jpec.pipelines.monitor import RunLogger
from jpec.stat.efficiency import RecoveryCurve


def measure_sensitivity_depth(
    run_logger: RunLogger, setup: InjectionSetup, args: argparse.Namespace
) -> None:
    """Inject sources, search around each, and fit the sensitivity depth per threshold.

    Args:
        run_logger: The run's log and output.
        setup: The injection setup.
        args: The parsed flags, carrying `setup` and `seed`.
    """
    started = time.perf_counter()
    trials_config = setup.trials
    curve_config = setup.recovery_curve
    search_loop_config = setup.search_loop

    search_data = SearchData.load(
        setup.data,
        setup.files,
        setup.freq_low,
        setup.freq_high,
        floor_width=search_loop_config.floor_width,
        floor_batch_size=search_loop_config.floor_batch_size,
    )

    # sample the injection population
    prior = setup.parameter_space.prior()
    injections = draw_injections(
        prior,
        cos_iota=trials_config.cos_iota,
        psi=trials_config.psi,
        key=jax.random.key(args.seed),
        n_injections=trials_config.n_injections,
    )

    # the part of the band each source's search reads, and the source simulated on it
    source_data = [
        search_data.around_f0(prior, float(injection.parameters.f0))
        for injection in injections
    ]
    sources = [
        simulated_source(injection, band_data.sfts, t_ref=band_data.t_ref)
        for injection, band_data in zip(injections, source_data)
    ]

    # a template bank near each source laid out on the source's band, built once per
    # source, not once per h0
    tiling = LatticeTiling.from_dict(setup.tiling)
    source_harmonic_banks = [
        HarmonicBank(
            TemplateBank.around_point(
                tiling,
                injection.parameters,
                velocity=search_data.velocity,
                window=setup.data.window,
                n_sky_neighbours=trials_config.n_sky_neighbours,
                n_steps=trials_config.n_steps,
            ),
            times=band_data.times,
            freq_low=band_data.freq_low,
            n_band_bins=band_data.n_band_bins,
            n_sub_bin_offsets=search_loop_config.n_sub_bin_offsets,
            period_batch_size=search_loop_config.period_batch_size,
            harmonic_fraction=search_loop_config.harmonic_fraction,
            phase_mismatch=search_loop_config.phase_mismatch,
            doppler_mismatch=search_loop_config.doppler_mismatch,
            modulation_depth_batch_size=search_loop_config.modulation_depth_batch_size,
        )
        for injection, band_data in zip(injections, source_data)
    ]

    # result tracking: one recovery curve per threshold, all fed by the same loudest
    # statistics
    thresholds = setup.thresholds
    curves = [RecoveryCurve(efficiency=curve_config.efficiency) for _ in thresholds]
    run_logger.injection_summary(
        setup, search_data, setup_path=args.setup, seed=args.seed
    )

    # every h0 tried, in order, with the statistics at each
    h0_values: list[float] = []
    loudest_statistics: list[list[float]] = []

    def search_injections_at(h0: float) -> None:
        # noise + h0 x signal around every source, its floor taken before the injection,
        # then the search loop over its bank, only the loudest template of each grid
        # kept
        loudest = np.zeros(len(sources))
        for source_idx, (source, band_data, harmonic_bank) in enumerate(
            zip(sources, source_data, source_harmonic_banks)
        ):
            injected_data = SearchData(
                source.injected_sfts(band_data.sfts, h0=h0),
                floor=band_data.floor,
                t_ref=band_data.t_ref,
            )
            n_sky_points = int(harmonic_bank.bank.alpha.size)
            sky_batch_size = search_loop_config.sky_batch_size
            loudest[source_idx] = max(
                float(grid_loudest.max_statistic[0])
                for sky_start_idx in range(0, n_sky_points, sky_batch_size)
                for grid_loudest in orbital_statistics(
                    injected_data,
                    harmonic_bank,
                    sky_idxs=range(
                        sky_start_idx, min(sky_start_idx + sky_batch_size, n_sky_points)
                    ),
                    f0_batch_size=search_loop_config.f0_batch_size,
                    n_loudest=1,
                )
            )
        h0_values.append(h0)
        loudest_statistics.append(loudest.tolist())
        for curve, threshold in zip(curves, thresholds):
            curve.add(h0, loudest >= threshold)
        recovered = ", ".join(
            f"{np.mean(loudest >= threshold):.1%}" for threshold in thresholds
        )
        run_logger.info(
            f"  h0 {h0:.4e}  sensitivity depth {search_data.sqrt_sn / h0:>6.1f}  "
            f"recovered {recovered}"
        )

    # h0 loop: the first scan, then per threshold one point at its fitted h0_ul until
    # it is known to tolerance; every point added feeds every curve
    loud_depth, weak_depth = curve_config.sensitivity_depth
    for h0 in search_data.sqrt_sn / np.geomspace(
        weak_depth, loud_depth, curve_config.n_amplitudes
    ):
        search_injections_at(float(h0))
    h0_ul = np.full(len(thresholds), np.nan)
    h0_ul_error = np.full(len(thresholds), np.nan)
    for threshold_idx, curve in enumerate(curves):
        h0_ul[threshold_idx], h0_ul_error[threshold_idx] = curve.fit()
        for _ in range(curve_config.n_refinements):
            if (
                np.isfinite(h0_ul[threshold_idx])
                and h0_ul_error[threshold_idx] / h0_ul[threshold_idx]
                <= curve_config.tolerance
            ):
                break
            search_injections_at(curve.next_h0(h0_ul[threshold_idx]))
            h0_ul[threshold_idx], h0_ul_error[threshold_idx] = curve.fit()
    elapsed = time.perf_counter() - started

    sensitivity_depth = search_data.sqrt_sn / h0_ul
    sensitivity_depth_error = search_data.sqrt_sn * h0_ul_error / h0_ul**2
    for threshold, threshold_depth, threshold_depth_error, threshold_h0_ul in zip(
        thresholds, sensitivity_depth, sensitivity_depth_error, h0_ul
    ):
        run_logger.info(
            f"Sensitivity depth at S {threshold:.4g}: {threshold_depth:.1f} "
            f"+- {threshold_depth_error:.1f} (h0_ul {threshold_h0_ul:.3e})"
        )
    run_logger.info(f"Done: time used {elapsed:.1f} s")

    run_logger.save_result(
        f"sensitivity_mu{setup.tiling['mismatch']:g}",
        {
            "setup": args.setup,
            "seed": args.seed,
            "mismatch": setup.tiling["mismatch"],
            "thresholds": thresholds,
            "t_ref": search_data.t_ref,
            "sqrt_sn": search_data.sqrt_sn,
            "search_loop": search_loop_config.model_dump(),
            "efficiency": curve_config.efficiency,
            "seconds": elapsed,
            # every source, and at each h0 how loud it came out and whether that counted
            "injections": [
                {
                    "f0": float(injection.parameters.f0),
                    "ap": float(injection.parameters.ap),
                    "period_days": float(injection.parameters.period) / DAY,
                    "phi_b": float(injection.parameters.phi_b),
                    "alpha": float(injection.parameters.alpha),
                    "delta": float(injection.parameters.delta),
                    "cos_iota": float(injection.orientation.cos_iota),
                    "psi": float(injection.orientation.psi),
                }
                for injection in injections
            ],
            # in the order tried
            "h0": h0_values,
            # [h0, source]
            "loudest": loudest_statistics,
            # [threshold, h0, source]
            "recovered": [
                (np.array(loudest_statistics) >= threshold).tolist()
                for threshold in thresholds
            ],
            # per threshold
            "h0_ul": h0_ul.tolist(),
            "h0_ul_error": h0_ul_error.tolist(),
            "sensitivity_depth": sensitivity_depth.tolist(),
            "sensitivity_depth_error": sensitivity_depth_error.tolist(),
        },
    )


def run(args: argparse.Namespace) -> None:
    """Measure the sensitivity depth at the setup's mismatch, one per threshold.

    Args:
        args: The parsed flags.
    """
    run_logger = RunLogger(
        "sensitivity", args.output, platform=args.platform, x64=False
    )
    measure_sensitivity_depth(run_logger, read_json(args.setup, InjectionSetup), args)


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
    parser.add_argument("--setup", required=True, help="jpec injection-setup result")
    parser.add_argument("--seed", type=int, default=0, help="seed of the drawn sources")
