"""Reference signals the test suite runs against."""

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jpec.constant import ARC_PAD_BINS, DAY, THETA_OVERSAMPLE
from jpec.detector.doppler import doppler_corrected_power, solar_roemer_delay
from jpec.params import Parameters
from jpec.stat.matched_filter import data_harmonics, orbital_phase_mod_2pi
from jpec.waveform.harmonics import (
    harmonic_count,
    template_harmonics,
    theta_sample_count,
)
from jpec.waveform.plane import arc_bin_offsets
from jpec.waveform.track import frequency_offset
from jpec.waveform.window import Window


@dataclass(frozen=True)
class Case:
    """A point together with the SFT length it is meant to be observed at."""

    name: str
    parameters: Parameters
    tsft: float
    window: Window = Window()  # the window the SFTs are taken to have been made with

    @property
    def modulation_depth_bins(self) -> float:
        """Half-width of the arc [bins]."""
        return float(self.parameters.modulation_depth_bins(self.tsft))

    @property
    def smearing(self) -> float:
        """Frequency drift within one SFT [bins]."""
        return float(
            self.parameters.modulation_depth * self.parameters.omega * self.tsft**2
        )

    @property
    def smearing_limit(self) -> float:
        """Largest modulation_depth_bins an SFT of this length can resolve [bins]."""
        return math.sqrt(float(self.parameters.f0) * float(self.parameters.ap))


CASES = [
    Case(
        "shallow",
        Parameters(
            f0=100.0, ap=5.0, period=15.0 * DAY, phi_b=0.0, alpha=0.0, delta=0.0
        ),
        tsft=1800.0,
    ),
    Case(
        "nominal",
        Parameters(
            f0=150.0, ap=6.6, period=9.0 * DAY, phi_b=2.3248, alpha=0.0, delta=0.0
        ),
        tsft=1800.0,
    ),
    Case(
        "deep",
        Parameters(
            f0=350.0, ap=15.0, period=7.0 * DAY, phi_b=6.1, alpha=0.0, delta=0.0
        ),
        tsft=1024.0,
    ),
]


def orbit_offset(point, times):
    """The point's frequency offset from f0 at each time, the orbit alone [Hz]."""
    return frequency_offset(
        times,
        modulation_depth=point.modulation_depth,
        omega=point.omega,
        phi_b=point.phi_b,
    )


def search_template_harmonics(modulation_depth_bins, window, *, bin_offsets=None):
    """`tau[j, k]` on the phase grid and harmonic count a modulation depth needs."""
    j_max = harmonic_count(modulation_depth_bins)
    if bin_offsets is None:
        bin_offsets = arc_bin_offsets(modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    return template_harmonics(
        bin_offsets,
        modulation_depth_bins=modulation_depth_bins,
        n_theta=theta_sample_count(j_max, oversample=THETA_OVERSAMPLE),
        j_max=j_max,
        window=window,
    )


def harmonic_transform(plane):
    """Reference harmonics of a plane on uniform phases: its mean-normalized rfft."""
    return jnp.fft.rfft(plane, axis=0) / plane.shape[0]


def search_data_harmonics(
    sft, point, *, floor, velocity, times, half_width_bins, j_max
):
    """Data `F[j, k]` at a point via the search loop, and its bin offsets from f0."""
    f0 = float(point.f0)
    sft_power = doppler_corrected_power(
        [jnp.fft.ifft(jnp.asarray(sft.sft_data), axis=-1)],
        floor=[floor],
        velocity_per_detector=[jnp.asarray(velocity)],
        tsft=sft.tsft,
        alpha=float(point.alpha),
        delta=float(point.delta),
        doppler_reference=f0,
    )
    frequencies = sft.freq_low + np.arange(sft.sft_data.shape[1]) / sft.tsft
    bin_offsets = (frequencies - f0) * sft.tsft  # differenced in Hz, so f0 lands on 0
    centre_idx = int(np.argmin(np.abs(bin_offsets)))
    assert half_width_bins <= centre_idx < bin_offsets.size - half_width_bins
    kept_bin_idxs = slice(
        centre_idx - half_width_bins, centre_idx + half_width_bins + 1
    )
    delay = solar_roemer_delay(
        jnp.asarray(sft.position), alpha=float(point.alpha), delta=float(point.delta)
    )
    sft_orbital_phase = jnp.asarray(
        orbital_phase_mod_2pi(np.asarray(times) + np.asarray(delay), point.omega)
    )
    return (
        data_harmonics(sft_power[:, kept_bin_idxs], sft_orbital_phase, j_max=j_max),
        bin_offsets[kept_bin_idxs],
    )
