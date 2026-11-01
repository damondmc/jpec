from typing import Any

import lal as _lal
import lalpulsar as _lalpulsar
import numpy as np
import pytest

from jpec.detector.sft import narrow_bands, read_sft_band
from jpec.waveform.window import Window

lal: Any = _lal
lalpulsar: Any = _lalpulsar

TSTART = 1238166018
TSFT = 1800.0
SEGMENTS = 20
GAP_AFTER, GAP_SEGMENTS = 10, 3  # segments dropped after the tenth, as a run would
F0 = 150.0
BAND = 1.0
WINDOW, WINDOW_PARAM = "tukey", 0.001  # LVK production SFTs


def timestamps() -> np.ndarray:
    starts = np.arange(SEGMENTS + GAP_SEGMENTS) * TSFT
    kept = np.r_[starts[:GAP_AFTER], starts[GAP_AFTER + GAP_SEGMENTS :]]
    return TSTART + kept


@pytest.fixture(scope="session")
def written(tmp_path_factory):
    """Two detectors' worth of Gaussian-noise SFTs, written to disk with LAL."""
    outdir = tmp_path_factory.mktemp("sfts")
    n_band_bins = int(BAND * TSFT)
    rng = np.random.default_rng(0)
    for detector_name in ("H1", "L1"):
        sfts = lalpulsar.CreateSFTVector(SEGMENTS, n_band_bins)
        for sft, start in zip(sfts.data, timestamps()):
            sft.name = detector_name
            sft.epoch = lal.LIGOTimeGPS(float(start))
            sft.f0 = F0 - BAND / 2.0
            sft.deltaF = 1.0 / TSFT
            sft.data.data[:] = (
                rng.standard_normal(n_band_bins) + 1j * rng.standard_normal(n_band_bins)
            ).astype(np.complex64)
        span = int(timestamps()[-1] + TSFT - TSTART)
        path = outdir / (
            f"{detector_name[0]}-{SEGMENTS}_{detector_name}_{int(TSFT)}SFT_probe-"
            f"{TSTART}-{span}.sft"
        )
        lalpulsar.WriteSFTVector2NamedFile(
            sfts, str(path), WINDOW, WINDOW_PARAM, "probe"
        )
    return str(outdir / "*.sft")


@pytest.fixture(scope="session")
def sfts(written):
    # broadband names, so one band holds both detectors
    (paths,) = narrow_bands(written).values()
    return read_sft_band(paths, F0 - 0.4, F0 + 0.4)


def test_the_band_read_is_the_band_asked_for(sfts):
    for sft in sfts.values():
        assert sft.freq_low >= F0 - 0.5
        assert sft.freq_low + sft.delta_f * sft.sft_data.shape[1] <= F0 + 0.5


def test_both_detectors_are_found(sfts):
    assert set(sfts) == {"H1", "L1"}


def test_segment_length(sfts):
    for sft in sfts.values():
        assert sft.tsft == TSFT
        assert sft.delta_f == pytest.approx(1.0 / TSFT)


def test_window(sfts):
    for sft in sfts.values():
        assert sft.window.kind == WINDOW
        assert sft.window.beta == pytest.approx(WINDOW_PARAM)
        assert sft.window == Window(kind=WINDOW, beta=WINDOW_PARAM)


def test_one_row_per_segment(sfts):
    for sft in sfts.values():
        assert sft.sft_data.shape[0] == SEGMENTS
        assert sft.timestamps.shape == (SEGMENTS,)


def test_timestamps(sfts):
    for sft in sfts.values():
        assert np.array_equal(sft.timestamps, timestamps())
        assert np.diff(sft.timestamps).max() == (GAP_SEGMENTS + 1) * TSFT


def test_timestamps_are_absolute_gps_in_float64(sfts):
    for sft in sfts.values():
        assert sft.timestamps.dtype == np.float64
        assert sft.timestamps[0] > 1e9


def test_amplitudes_are_complex64(sfts):
    for sft in sfts.values():
        assert sft.sft_data.dtype == np.complex64
        assert np.all(np.isfinite(sft.sft_data))


def test_power(sfts):
    for sft in sfts.values():
        assert np.all(sft.sft_power >= 0.0)
        assert np.allclose(sft.sft_power, 2.0 * np.abs(sft.sft_data) ** 2)


def test_a_fractional_shift_is_exact_on_the_amplitudes(sfts):
    sft = sfts["H1"]
    data = sft.sft_data[0]
    shift = 0.37
    sample_idxs = np.arange(data.size)
    moved = np.fft.fft(
        np.fft.ifft(data) * np.exp(2j * np.pi * shift * sample_idxs / data.size)
    )
    roundtrip = np.fft.fft(
        np.fft.ifft(moved) * np.exp(-2j * np.pi * shift * sample_idxs / data.size)
    )
    assert np.max(np.abs(roundtrip - data)) / np.max(np.abs(data)) < 1e-6
