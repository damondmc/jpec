"""SFTs: one detector's amplitudes, and reading, checking and writing SFT files."""

import glob
from dataclasses import dataclass, replace
from typing import Any

import lal as _lal
import lalpulsar as _lalpulsar
import numpy as np
from jaxtyping import Complex, Float

from jpec.detector.site import Detector
from jpec.waveform.window import Window

lal: Any = _lal
lalpulsar: Any = _lalpulsar

# strain x s per unit amplitude, so SFT power is ~1 in float32
STRAIN_UNIT = 6e-23


@dataclass(frozen=True)
class SFTData:
    """The SFTs from one detector.

    Attributes:
        detector: The ground-based interferometer that data was collected from.
        sft_data: Complex Fourier amplitudes of the SFTs, in units of `strain_unit`.
        timestamps: SFT start times, absolute GPS [s].
        freq_low: Start frequency of the first frequency bin [Hz].
        tsft: SFT length [s].
        window: The window the SFTs were made with.
        strain_unit: Strain x s per unit of `sft_data`; 1 for the SFTs as recorded.
    """

    detector: Detector
    sft_data: Complex[np.ndarray, "n_sft n_freq"]
    timestamps: Float[np.ndarray, " n_sft"]
    freq_low: float
    tsft: float
    window: Window
    strain_unit: float = 1.0

    @property
    def sft_power(self) -> Float[np.ndarray, "n_sft n_freq"]:
        """`2*|sft_data|**2`, the observed plane.

        Take this only after any Doppler correction: squaring is irreversible.
        """
        return 2.0 * np.abs(self.sft_data) ** 2

    @property
    def delta_f(self) -> float:
        """Frequency bin width [Hz], the reciprocal of the SFT length."""
        return 1.0 / self.tsft

    @property
    def n_sfts(self) -> int:
        """SFTs this detector contributed."""
        return self.timestamps.size

    @property
    def timestamps_midpoints(self) -> Float[np.ndarray, " n_sft"]:
        """SFT midpoints, absolute GPS [s], where the Doppler shift is evaluated."""
        return self.timestamps + self.tsft / 2.0

    @property
    def velocity(self) -> Float[np.ndarray, "n_sft 3"]:
        """Detector velocity `v/c` at each SFT midpoint, cached on the timestamps."""
        return self.detector.velocity(self.timestamps_midpoints)

    def rescaled(self, strain_unit: float) -> "SFTData":
        """These SFTs with their amplitudes divided by `strain_unit`.

        Args:
            strain_unit: Strain x s per unit of the amplitudes the result holds.

        Returns:
            The SFTs, `strain_unit` recorded so the strain can be recovered.
        """
        return replace(
            self,
            sft_data=self.sft_data / strain_unit,
            strain_unit=self.strain_unit * strain_unit,
        )

    def narrowed(self, bin_idxs: slice) -> "SFTData":
        """These SFTs over a narrower band.

        Args:
            bin_idxs: Columns kept.

        Returns:
            The SFTs on those columns, their `freq_low` moved to the first of them.
        """
        freq_low = self.freq_low + bin_idxs.start * self.delta_f
        return replace(self, sft_data=self.sft_data[:, bin_idxs], freq_low=freq_low)


def narrow_bands(pattern: str) -> dict[tuple[float, float], list[str]]:
    """The narrow bands the files of a glob hold, each with the files carrying it.

    Args:
        pattern: Glob of SFT files, narrow-band or broadband.

    Returns:
        `(first, last)` bin frequency [Hz] of each band to the files holding it.
    """
    bands: dict[tuple[float, float], list[str]] = {}
    for path in sorted(glob.glob(pattern)):
        bands.setdefault(sft_file_bin_range(path), []).append(path)
    return bands


def read_sft_band(
    paths: list[str],
    freq_low: float,
    freq_high: float,
    *,
    unrecorded_window: Window | None = None,
) -> dict[str, SFTData]:
    """A band read from exactly these files, one entry per detector.

    Args:
        paths: The files to read.
        freq_low: Low edge of the band [Hz].
        freq_high: High edge of the band [Hz].
        unrecorded_window: The window of the SFT used when not recorded in the file.

    Returns:
        One SFTData per detector, keyed by prefix, at `strain_unit` 1.
    """
    catalog = lalpulsar.SFTdataFind(";".join(paths), None)
    names = lalpulsar.ListIFOsInCatalog(catalog)
    loaded = lalpulsar.LoadMultiSFTs(catalog, fMin=freq_low, fMax=freq_high)

    per_detector = {}
    for detector_idx, detector_name in enumerate(names.data):
        sfts = loaded.data[detector_idx]
        sft_data = np.array([sft.data.data for sft in sfts.data], dtype=np.complex64)
        timestamps = np.array(
            [
                sft.epoch.gpsSeconds + 1e-9 * sft.epoch.gpsNanoSeconds
                for sft in sfts.data
            ],
            dtype=np.float64,
        )
        # a row per SFT by construction; rows in ascending time, and column k the bin at
        # freq_low + k / tsft of one ascending set of bins in every row
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError(f"{detector_name} SFTs are not in strictly ascending time")
        bins = {(float(sft.f0), float(sft.deltaF)) for sft in sfts.data}
        if len(bins) != 1 or float(sfts.data[0].deltaF) <= 0.0:
            raise ValueError(
                f"{detector_name} SFTs do not share one ascending set of bins: "
                f"{sorted(bins)}"
            )
        windows = {
            (entry.window_type, float(entry.window_param))
            for entry in catalog.data
            if entry.header.name == detector_name
        }
        if len(windows) != 1:
            raise ValueError(
                f"{detector_name} SFTs have multiple windows: {sorted(windows)}"
            )
        (window_kind, window_beta) = windows.pop()
        if window_kind != "unknown":
            recorded_window = Window(kind=window_kind, beta=window_beta)
        elif unrecorded_window is not None:
            recorded_window = unrecorded_window
        else:
            raise ValueError(
                f"{detector_name} SFTs record no window;"
                " pass the one they were made with"
            )

        per_detector[detector_name] = SFTData(
            detector=Detector(detector_name),
            sft_data=sft_data,
            timestamps=timestamps,
            freq_low=float(sfts.data[0].f0),
            tsft=1.0 / float(sfts.data[0].deltaF),
            window=recorded_window,
        )
    return per_detector


def sft_file_bin_range(path: str) -> tuple[float, float]:
    """The frequency bins one SFT file holds, from its name or else its header.

    Args:
        path: The SFT file.

    Returns:
        Its first and last bin frequency [Hz], not the band's edges.
    """
    spec = lalpulsar.SFTFilenameSpec()
    try:
        lalpulsar.ParseSFTFilenameIntoSpec(spec, path)
        # a broadband name carries no band
        named = bool(spec.nbBinWidthFreq)
    except RuntimeError:
        # not a standard SFT name, as files converted from another format are not
        named = False
    if not named:
        first = lalpulsar.SFTdataFind(path, None).data[0]
        low = float(first.header.f0)
        return low, low + (first.numBins - 1) * float(first.header.deltaF)
    if spec.nbFirstBinRem or spec.nbBinWidthRem:
        raise ValueError(
            f"{path}: a narrow band whose edges are not whole Hz is not handled"
        )
    low = float(spec.nbFirstBinFreq)
    return low, low + spec.nbBinWidthFreq - 1.0 / spec.SFTtimebase


def load_sft_data(
    paths: list[str],
    freq_low: float,
    freq_high: float,
    *,
    window: Window,
    tsft: float,
    detectors: list[str],
    n_sfts: int,
) -> list[SFTData]:
    """A band read from exactly these files, checked against its metadata and rescaled.

    Args:
        paths: The files holding the band, at least one per detector.
        freq_low: Low edge of the band [Hz].
        freq_high: High edge of the band [Hz].
        window: The window the SFTs were made with, used too when a file records none.
        tsft: SFT length [s].
        detectors: The detectors, in order.
        n_sfts: SFTs over every detector.

    Returns:
        One SFTData per detector on the same frequency bins, at `STRAIN_UNIT`.
    """
    recorded_sfts = list(
        read_sft_band(paths, freq_low, freq_high, unrecorded_window=window).values()
    )
    recorded_detectors = [sft.detector.name for sft in recorded_sfts]
    recorded_n_sfts = sum(sft.n_sfts for sft in recorded_sfts)
    recorded_tsft = recorded_sfts[0].tsft
    problems = []
    if len({(sft.freq_low, sft.sft_data.shape[1]) for sft in recorded_sfts}) != 1:
        problems.append("the detectors' bands differ")
    if recorded_sfts[0].window != window:
        problems.append(f"window {recorded_sfts[0].window}, expected {window}")
    if recorded_tsft != tsft:
        problems.append(f"tsft {recorded_tsft:.0f} s, expected {tsft:.0f} s")
    if recorded_detectors != detectors:
        problems.append(f"detectors {recorded_detectors}, expected {detectors}")
    if recorded_n_sfts != n_sfts:
        problems.append(f"{recorded_n_sfts} SFTs, expected {n_sfts}")
    if problems:
        raise ValueError(
            "the SFTs loaded are not the data the metadata describes:\n  "
            + "\n  ".join(problems)
        )
    return [sft.rescaled(STRAIN_UNIT) for sft in recorded_sfts]


def write_sfts(sft: SFTData, path: str, *, comment: str) -> None:
    """Write one detector's SFTs to a file, in strain x s as recorded SFTs hold them.

    Args:
        sft: The SFTs.
        path: The file written.
        comment: Description recorded in the file.
    """
    sft_vector = lalpulsar.CreateSFTVector(sft.n_sfts, sft.sft_data.shape[1])
    for lal_sft, start, sft_data in zip(sft_vector.data, sft.timestamps, sft.sft_data):
        lal_sft.name = sft.detector.name
        lal_sft.epoch = lal.LIGOTimeGPS(float(start))
        lal_sft.f0 = sft.freq_low
        lal_sft.deltaF = sft.delta_f
        lal_sft.data.data[:] = (sft.strain_unit * sft_data).astype(np.complex64)
    lalpulsar.WriteSFTVector2NamedFile(
        sft_vector, path, sft.window.kind, sft.window.beta, comment
    )
