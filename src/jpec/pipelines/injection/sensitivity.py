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
from jpec.engine.data import SearchData, freq_band_edges
from jpec.engine.loop import HarmonicBank, top_templates_at_sky_points
from jpec.injection.population import draw_injections
from jpec.injection.signal import (
    sft_with_simulated_signal,
    simulated_signal_sft_data,
)
from jpec.pipelines.monitor import RunLogger
from jpec.stat.efficiency import fit_efficiency

# injections searched between two progress lines of the log
N_INJECTIONS_PER_PROGRESS_LINE = 100


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
    key = jax.random.key(args.seed)
    injections = draw_injections(
        prior,
        cos_iota=trials_config.cos_iota,
        psi=trials_config.psi,
        key=key,
        n_injections=trials_config.n_injections,
    )
    # one h0 per injection, its sensitivity depth uniform in log over the range
    low_depth, high_depth = trials_config.sensitivity_depth
    h0 = search_data.sqrt_sn / np.exp(
        np.asarray(
            jax.random.uniform(
                jax.random.fold_in(key, 1),
                (len(injections),),
                minval=np.log(low_depth),
                maxval=np.log(high_depth),
            ),
            dtype=np.float64,
        )
    )

    thresholds = setup.thresholds
    run_logger.injection_summary(
        setup, search_data, setup_path=args.setup, seed=args.seed
    )

    # noise + h0 x signal around each injection, its floor taken before the injection,
    # then the search loop over the bank around it, the top in-range template kept
    tiling = LatticeTiling.from_dict(setup.tiling)
    top_statistic = np.zeros(len(injections))
    for injection_idx, (injection, injection_h0) in enumerate(
        zip(injections, h0, strict=True)
    ):
        bank = TemplateBank.around_point(
            tiling,
            injection.parameters,
            velocity=search_data.velocity,
            window=setup.data.window,
            n_sky_neighbours=trials_config.n_sky_neighbours,
            n_steps=trials_config.n_steps,
        )
        band_data = search_data.sub_band(
            *freq_band_edges(
                bank.f0[0],
                bank.f0_high,
                max_modulation_depth=bank.max_modulation_depth,
                tsft=bank.tsft,
            )
        )
        harmonic_bank = HarmonicBank(
            bank,
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
        injected_data = SearchData(
            [
                sft_with_simulated_signal(
                    sft,
                    np.asarray(
                        simulated_signal_sft_data(sft, injection, t_ref=band_data.t_ref)
                    ),
                    h0=float(injection_h0),
                )
                for sft in band_data.sfts
            ],
            floor=band_data.floor,
            t_ref=band_data.t_ref,
        )
        top_statistic[injection_idx] = max(
            float(top.max_statistic[0])
            for sky_idxs in harmonic_bank.layout.sky_batches(
                search_loop_config.sky_batch_size
            )
            for top in top_templates_at_sky_points(
                injected_data,
                harmonic_bank,
                sky_idxs=sky_idxs,
                f0_batch_size=search_loop_config.f0_batch_size,
                n_top=1,
                # one toplist over every f0 point
                toplist_f0_edges=np.array([-np.inf, np.inf]),
            )
        )
        n_searched = injection_idx + 1
        if n_searched % N_INJECTIONS_PER_PROGRESS_LINE == 0:
            run_logger.info(
                f"  {n_searched} / {len(injections)} injections searched, "
                f"{time.perf_counter() - started:.0f} s"
            )

    # per threshold, the sigmoid of the recoveries against h0, read at the efficiency;
    # NaN where every injection is recovered or none is
    h0_ul = np.full(len(thresholds), np.nan)
    h0_ul_error = np.full(len(thresholds), np.nan)
    for threshold_idx, threshold in enumerate(thresholds):
        recovered = top_statistic >= threshold
        if recovered.all() or not recovered.any():
            continue
        fit = fit_efficiency(h0, recovered)
        h0_ul[threshold_idx], h0_ul_error[threshold_idx] = (
            float(value) for value in fit.h0_at(setup.efficiency)
        )
    elapsed = time.perf_counter() - started

    sensitivity_depth = search_data.sqrt_sn / h0_ul
    sensitivity_depth_error = search_data.sqrt_sn * h0_ul_error / h0_ul**2
    for threshold, threshold_depth, threshold_depth_error, threshold_h0_ul in zip(
        thresholds, sensitivity_depth, sensitivity_depth_error, h0_ul, strict=True
    ):
        run_logger.info(
            f"Sensitivity depth at S {threshold:.4g}: {threshold_depth:.1f} "
            f"+- {threshold_depth_error:.1f} (h0_ul {threshold_h0_ul:.3e}), "
            f"{np.mean(top_statistic >= threshold):.1%} of the injections recovered"
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
            "efficiency": setup.efficiency,
            "seconds": elapsed,
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
            # per injection
            "h0": h0.tolist(),
            "top_statistic": top_statistic.tolist(),
            # [threshold, injection]
            "recovered": [
                (top_statistic >= threshold).tolist() for threshold in thresholds
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
