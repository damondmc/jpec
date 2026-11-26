"""Inject a source into Gaussian noise, search for it, and plot the top templates.

Usage:

  uv run --extra nb python examples/inject_and_search.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import SearchLoopConfig
from jpec.constant import DAY
from jpec.detector.site import Detector, Observation, reference_epoch
from jpec.engine.candidates import Candidate, Toplist
from jpec.engine.data import SearchData, freq_band_edges
from jpec.engine.loop import BandSearch
from jpec.injection.signal import (
    Injection,
    Orientation,
    sft_with_simulated_signal,
    simulated_signal_sft_data,
)
from jpec.noise.floor import noise_floor
from jpec.noise.gaussian import gaussian_noise_series_sft
from jpec.params import Parameters, Prior
from jpec.stat.null import null_sigma
from jpec.waveform.window import Window

# a gap-free run of both LIGO detectors, starting at the O4a start [GPS s]
RUN_START = 1368975618.0
DAYS = 20.0
TSFT = 1800.0
DETECTORS = ("H1", "L1")
WINDOW = Window(kind="tukey", beta=0.5)
# noise amplitude spectral density [1/sqrt(Hz)], and the source's sensitivity depth
# sqrt(Sn) / h0
SQRT_SN = 1e-23
SENSITIVITY_DEPTH = 15.0
# sampling rate of the baseband noise series [Hz], above the band's width
RATE = 1.0
# the source's sky position [rad], the one sky point searched
ALPHA = 1.2
DELTA = 0.3
SOURCE = Parameters(
    f0=150.012, ap=6.3, period=9.1 * DAY, phi_b=1.0, alpha=ALPHA, delta=DELTA
)
# the box searched around it
PRIOR = Prior(
    f0=(150.0, 150.02),
    ap=(5.5, 7.0),
    period=(8.5 * DAY, 9.5 * DAY),
    phi_b=(0.0, 2.0 * np.pi),
    alpha=(ALPHA, ALPHA),
    delta=(DELTA, DELTA),
)
MISMATCH = 0.4
FIGURE = Path(__file__).with_name("inject_and_search.png")


def top_candidates(
    band_search: BandSearch,
    search_loop_config: SearchLoopConfig,
    n_toplist: int,
) -> list[Candidate]:
    """The toplist of a search over the bank.

    Args:
        band_search: The bank laid out on the band searched.
        search_loop_config: How the search loop batches its work.
        n_toplist: Templates kept.

    Returns:
        The top templates, highest statistic first.
    """
    toplist = Toplist(n_toplist)
    layout = band_search.layout
    for sky_idxs in layout.sky_batches(search_loop_config.sky_batch_size):
        for top in band_search.search_sky_batch(sky_idxs):
            toplist.add_top(top, layout.bank)
    return toplist.descending()


def main() -> None:
    """Make the noise, inject the source, search with and without it, and plot both."""
    search_loop_config = SearchLoopConfig()
    timestamps = {
        detector_name: RUN_START + np.arange(0.0, DAYS * DAY, TSFT)
        for detector_name in DETECTORS
    }
    t_ref = reference_epoch(timestamps, tsft=TSFT)
    observation = Observation(timestamps, tsft=TSFT, t_ref=t_ref)

    # the lattice over the box, its steps from the metric of this run
    tiling = LatticeTiling.from_metric(
        PRIOR,
        observation=observation,
        mismatch=MISMATCH,
        n_ap_blocks=1,
        window=WINDOW,
    )

    # the bank over the box
    bank = TemplateBank.over_f0_interval(
        tiling, *PRIOR.f0, velocity=observation.velocity, window=WINDOW
    )
    print(f"{bank.n_templates} templates")

    # Gaussian noise on the band the search loads, each SFT cut from one continuous
    # series
    freq_low, freq_high = freq_band_edges(
        bank.f0[0],
        bank.f0_high,
        max_modulation_depth=bank.max_modulation_depth,
        tsft=TSFT,
    )
    n_band_bins = round((freq_high - freq_low) * TSFT)
    detector_keys = jax.random.split(jax.random.key(0), len(DETECTORS))
    noise_sfts = [
        gaussian_noise_series_sft(
            timestamps[detector_name],
            Detector(detector_name),
            key=detector_key,
            freq_low=freq_low,
            n_band_bins=n_band_bins,
            tsft=TSFT,
            sqrt_sn=SQRT_SN,
            window=WINDOW,
            rate=RATE,
        )
        for detector_name, detector_key in zip(DETECTORS, detector_keys)
    ]
    # the floor taken before the injection, as jpec sensitivity takes it
    floor = [
        noise_floor(
            jnp.asarray(sft.sft_power),
            width=search_loop_config.floor_width,
            batch_size=search_loop_config.floor_batch_size,
        )
        for sft in noise_sfts
    ]
    noise_data = SearchData(noise_sfts, floor=floor, t_ref=t_ref)

    # the source, face on, injected at its sensitivity depth
    injection = Injection(
        parameters=SOURCE, orientation=Orientation(cos_iota=1.0, psi=0.0)
    )
    h0 = noise_data.sqrt_sn / SENSITIVITY_DEPTH
    signal_data = SearchData(
        [
            sft_with_simulated_signal(
                sft,
                np.asarray(simulated_signal_sft_data(sft, injection, t_ref=t_ref)),
                h0=h0,
            )
            for sft in noise_sfts
        ],
        floor=floor,
        t_ref=t_ref,
    )

    # every template kept, in noise alone and with the injection
    sigma_s = float(null_sigma(noise_data.n_sfts, power_variance=1.0))
    # one toplist over every f0 point
    toplist_f0_edges = np.array([-np.inf, np.inf])
    noise_candidates = top_candidates(
        BandSearch(
            bank,
            noise_data,
            search_loop_config,
            n_top=bank.n_templates,
            toplist_f0_edges=toplist_f0_edges,
        ),
        search_loop_config,
        bank.n_templates,
    )
    signal_candidates = top_candidates(
        BandSearch(
            bank,
            signal_data,
            search_loop_config,
            n_top=bank.n_templates,
            toplist_f0_edges=toplist_f0_edges,
        ),
        search_loop_config,
        bank.n_templates,
    )
    top = signal_candidates[0]
    print(
        f"top with the injection: S = {top.statistic / sigma_s:.1f} sigma_S at "
        f"f0 {float(top.parameters.f0):.5f} Hz, "
        f"ap {float(top.parameters.ap):.2f} ls, "
        f"P {float(top.parameters.period) / DAY:.3f} d; "
        f"injected {SOURCE.f0:.5f} Hz, "
        f"{SOURCE.ap:.2f} ls, {SOURCE.period / DAY:.3f} d"
    )
    print(
        "top in noise alone: "
        f"S = {noise_candidates[0].statistic / sigma_s:.1f} sigma_S"
    )

    plt.style.use("jpec.style")
    figure, (histogram_axes, track_axes) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    noise_statistic = np.array([candidate.statistic for candidate in noise_candidates])
    signal_statistic = np.array(
        [candidate.statistic for candidate in signal_candidates]
    )
    statistic_bins = np.linspace(
        noise_statistic.min() / sigma_s, signal_statistic.max() / sigma_s, 100
    )
    histogram_axes.hist(
        noise_statistic / sigma_s,
        bins=statistic_bins,
        density=True,
        histtype="step",
        label="noise alone",
    )
    histogram_axes.hist(
        signal_statistic / sigma_s,
        bins=statistic_bins,
        density=True,
        histtype="step",
        label=(
            "with the injection, "
            rf"$\mathcal{{D}} = {SENSITIVITY_DEPTH:g}\,\mathrm{{Hz}}^{{-1/2}}$"
        ),
    )
    histogram_axes.set_yscale("log")
    histogram_axes.set_xlabel(r"$S / \sigma_S$")
    histogram_axes.set_ylabel("probability density")
    histogram_axes.legend()

    # the peak in (f0, P): every template at the top template's modulation depth
    top_modulation_depth_bins = float(
        top.parameters.modulation_depth_bins(TSFT)
    )
    slice_candidates = [
        candidate
        for candidate in signal_candidates
        if np.isclose(
            float(candidate.parameters.modulation_depth_bins(TSFT)),
            top_modulation_depth_bins,
        )
    ]
    f0 = np.array([float(candidate.parameters.f0) for candidate in slice_candidates])
    period_days = np.array(
        [float(candidate.parameters.period) / DAY for candidate in slice_candidates]
    )
    statistic = np.array([candidate.statistic for candidate in slice_candidates])
    points = track_axes.scatter(
        f0, period_days, c=statistic / sigma_s, s=20, cmap="viridis"
    )
    track_axes.scatter(
        [SOURCE.f0],
        [SOURCE.period / DAY],
        marker="o",
        s=160,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="Injection",
    )
    # framed, since the templates fill the axes
    track_axes.legend(loc="upper left", frameon=True)
    track_axes.ticklabel_format(axis="x", useOffset=False)
    track_axes.set_xlabel(r"$f_0$ [Hz]")
    track_axes.set_ylabel(r"$P$ [d]")
    figure.colorbar(points, ax=track_axes, label=r"$S / \sigma_S$")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=150)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()
