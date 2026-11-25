r"""Prepare an injection run: the band, its files, and the bank at one mismatch.

Usage:

  jpec injection-setup examples/injection.yaml \
      --sft-metadata setup/sfdb_204_metadata.json --mismatch 0.8 \
      --output setup/injection_204_mu0.8_setup.json
"""

import argparse
import importlib.metadata
from pathlib import Path

from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import (
    DataSetup,
    InjectionConfig,
    InjectionSetup,
    SFTMetadata,
    read_config,
    read_json,
)
from jpec.detector.site import Observation, reference_epoch
from jpec.engine.data import freq_band_edges
from jpec.pipelines.monitor import RunLogger


def run(args: argparse.Namespace) -> None:
    """Build the injection band, files and lattice at one mismatch; write the setup.

    Args:
        args: The parsed flags.
    """
    # the metric is taken in float64, on the host
    output = Path(args.output)
    run_logger = RunLogger("injection_setup", output.parent, platform="cpu", x64=True)
    config = read_config(args.config, InjectionConfig)
    sft_metadata = read_json(args.sft_metadata, SFTMetadata)
    box_config = config.parameter_space

    # the run's SFTs, epoch and detector motion
    tsft = sft_metadata.tsft
    t_ref = reference_epoch(sft_metadata.timestamps, tsft=tsft)
    observation = Observation(sft_metadata.timestamps, tsft=tsft, t_ref=t_ref)

    # the lattice, at the flag's mismatch if one is given
    injection_prior = box_config.prior()
    mismatch = (
        args.mismatch if args.mismatch is not None else config.template_bank.mismatch
    )
    tiling = LatticeTiling.from_metric(
        injection_prior,
        observation=observation,
        mismatch=mismatch,
        n_ap_blocks=config.template_bank.n_ap_blocks,
        window=sft_metadata.window,
    )

    # the band every source's bank reads, each a subset of the bank over the box, and
    # the files holding it
    box_bank = TemplateBank.over_f0_interval(
        tiling,
        *box_config.f0,
        velocity=observation.velocity,
        window=sft_metadata.window,
    )
    run_logger.info(f"Mismatch {mismatch:g}: f0 step {box_bank.f0_step:.3e} Hz")
    freq_low, freq_high = freq_band_edges(
        box_bank.f0[0],
        box_bank.f0_high,
        max_modulation_depth=box_bank.max_modulation_depth,
        tsft=box_bank.tsft,
    )

    setup = InjectionSetup(
        jpec=importlib.metadata.version("jpec"),
        data=DataSetup(
            sft_glob=config.sft_glob,
            detectors=sft_metadata.detectors,
            n_sfts=sft_metadata.n_sfts,
            tsft=tsft,
            window=sft_metadata.window,
            t_ref=t_ref,
        ),
        freq_low=float(freq_low),
        freq_high=float(freq_high),
        files=sft_metadata.files_covering(
            freq_low, freq_high, sft_glob=config.sft_glob
        ),
        parameter_space=box_config,
        tiling=tiling.to_dict(),
        trials=config.trials,
        search_loop=config.search_loop,
        thresholds=config.thresholds,
        efficiency=config.efficiency,
    )
    run_logger.save_setup(output.stem, setup)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the command's flags.

    Args:
        parser: The command's parser.
    """
    parser.add_argument(
        "config", help="YAML describing the injection run; see examples/injection.yaml"
    )
    parser.add_argument(
        "--sft-metadata",
        required=True,
        help="jpec check-sft result, the data to inject into",
    )
    parser.add_argument(
        "--mismatch",
        type=float,
        default=None,
        help="mismatch of the bank, in place of the config's template_bank.mismatch",
    )
    parser.add_argument(
        "--output",
        default="setup/injection_setup.json",
        help="JSON file the setup is written to",
    )
