r"""Check the SFT files agree band to band, and write what a setup needs of them.

Usage:

  jpec check-sft "data_sfdb/*Fmin203.0*.sft" --window tukey 0.5 \
      --output setup/sfdb_204_metadata.json
"""

import argparse
import os
from pathlib import Path

import numpy as np

from jpec.config import SFTBand, SFTMetadata
from jpec.detector.sft import SFTData, narrow_bands, read_sft_band
from jpec.pipelines.monitor import RunLogger
from jpec.waveform.window import Window

# Hz into a band, so the read is clear of its low edge
READ_OFFSET = 0.1
# Hz read of each band
READ_WIDTH = 0.1


def checked_sft_metadata(
    run_logger: RunLogger, sft_glob: str, *, unrecorded_window: Window | None
) -> SFTMetadata:
    """Read every band, check they agree, and return the metadata.

    Args:
        run_logger: Where the lines go.
        sft_glob: Glob of SFT files.
        unrecorded_window: The window to record when the SFT headers carry none.

    Returns:
        The SFT metadata: the glob, the SFT length, the window, the bands found
        and each detector's timestamps.
    """
    # so the files a job is given do not move with the directory this was run from
    sft_glob = os.path.abspath(os.path.expanduser(sft_glob))
    files_per_band = narrow_bands(sft_glob)
    if not files_per_band:
        raise SystemExit(f"no SFT files match {sft_glob}")
    band_edges = sorted(files_per_band)
    run_logger.info(
        f"{sum(len(files) for files in files_per_band.values())} files, "
        f"{len(files_per_band)} narrow bands from {min(files_per_band)[0]:.1f} Hz"
    )

    first_band_sfts: dict[str, SFTData] = {}
    disagree: list[str] = []
    for freq_low, freq_high in band_edges:
        read_low = freq_low + READ_OFFSET
        band_sfts = read_sft_band(
            files_per_band[(freq_low, freq_high)],
            read_low,
            read_low + READ_WIDTH,
            unrecorded_window=unrecorded_window,
        )
        if not first_band_sfts:
            first_band_sfts = band_sfts
            run_logger.info(
                f"  {freq_low:>7.1f} Hz  "
                + ", ".join(
                    f"{detector_name} n={sft.timestamps.size} tsft={sft.tsft:.0f} "
                    f"window={sft.window.kind}/{sft.window.beta}"
                    for detector_name, sft in sorted(band_sfts.items())
                )
            )
            continue
        if set(band_sfts) != set(first_band_sfts):
            disagree.append(
                f"{freq_low:.1f} Hz has detectors {sorted(band_sfts)}, "
                f"{min(band_edges)[0]:.1f} Hz has {sorted(first_band_sfts)}"
            )
            continue
        for detector_name, sft in sorted(band_sfts.items()):
            first_band_sft = first_band_sfts[detector_name]
            if sft.tsft != first_band_sft.tsft:
                disagree.append(
                    f"{detector_name} at {freq_low:.1f} Hz has tsft {sft.tsft}, "
                    f"not {first_band_sft.tsft}"
                )
            if sft.window != first_band_sft.window:
                disagree.append(
                    f"{detector_name} at {freq_low:.1f} Hz has window {sft.window}, "
                    f"not {first_band_sft.window}"
                )
            if not np.array_equal(sft.timestamps, first_band_sft.timestamps):
                n_shared = np.intersect1d(
                    sft.timestamps, first_band_sft.timestamps
                ).size
                disagree.append(
                    f"{detector_name} at {freq_low:.1f} Hz has "
                    f"{sft.timestamps.size} timestamps "
                    f"against {first_band_sft.timestamps.size}, {n_shared} shared"
                )
    if disagree:
        raise SystemExit(
            "the SFT files do not agree band to band, so one metadata file cannot "
            "serve the run:\n  " + "\n  ".join(disagree)
        )

    window = next(iter(first_band_sfts.values())).window
    tsft = next(iter(first_band_sfts.values())).tsft
    for detector_name, sft in sorted(first_band_sfts.items()):
        span = float(sft.timestamps[-1] + sft.tsft - sft.timestamps[0])
        # overlapping SFTs cover their spacing, not their length
        covered = float(np.minimum(np.diff(sft.timestamps), sft.tsft).sum() + sft.tsft)
        run_logger.info(
            f"  {detector_name}: {sft.timestamps.size} SFTs over "
            f"{span / 86400.0:.1f} days, "
            f"duty cycle {covered / span:.1%}"
        )
    return SFTMetadata(
        sft_glob=sft_glob,
        tsft=tsft,
        window=window,
        bands=[
            SFTBand(
                freq_low=freq_low,
                freq_high=freq_high,
                files=files_per_band[(freq_low, freq_high)],
            )
            for freq_low, freq_high in sorted(files_per_band)
        ],
        detectors=sorted(first_band_sfts),
        timestamps={
            detector_name: first_band_sfts[detector_name].timestamps.tolist()
            for detector_name in sorted(first_band_sfts)
        },
    )


def run(args: argparse.Namespace) -> None:
    """Check the files and write their metadata.

    Args:
        args: The parsed flags.
    """
    output = Path(args.output)
    run_logger = RunLogger("check_sft", output.parent, platform="cpu", x64=False)
    unrecorded_window = (
        Window(kind=args.window[0], beta=float(args.window[1])) if args.window else None
    )
    sft_metadata = checked_sft_metadata(
        run_logger, args.sft_glob, unrecorded_window=unrecorded_window
    )
    run_logger.save_setup(output.stem, sft_metadata)


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Add the command's flags.

    Args:
        p: The command's parser.
    """
    p.add_argument("sft_glob", help="glob of SFT files, e.g. 'data/*/*.sft'")
    p.add_argument(
        "--window",
        nargs=2,
        default=None,
        metavar=("KIND", "BETA"),
        help="the window the SFTs were made with, e.g. tukey 0.5; taken only when "
        "their headers record none, as SFTs converted from another format do not",
    )
    p.add_argument(
        "--output",
        default="setup/sft_metadata.json",
        help="JSON file the metadata is written to",
    )
