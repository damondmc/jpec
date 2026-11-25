"""The band one job searches: its width, its SFTs on the device, and its epoch."""

import math
from functools import cached_property

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from jpec.config import DataSetup
from jpec.constant import ARC_PAD_BINS, V_MAX
from jpec.detector.sft import SFTData, load_sft_data
from jpec.noise.floor import noise_floor
from jpec.waveform.plane import arc_bin_offsets


def freq_band_edges(
    f0_low: float, f0_high: float, *, max_modulation_depth: float, tsft: float
) -> tuple[float, float]:
    """The band a search over [f0_low, f0_high] reads, with Doppler shift and window.

    Args:
        f0_low: Lowest template f0 [Hz].
        f0_high: Highest template f0, with its shift [Hz].
        max_modulation_depth: Deepest template depth, with its shift [Hz].
        tsft: SFT length [s].

    Returns:
        The band's low and high edge [Hz].
    """
    n_window_bins = arc_bin_offsets(
        max_modulation_depth * tsft, pad_bins=ARC_PAD_BINS
    ).shape[0]
    n_doppler_bins = math.ceil(f0_high * V_MAX * tsft)
    # a window of 2h + 1 bins runs h bins below its nearest bin and h + 1 from it up;
    # the f0 shift is read in whole bins, up to one past f0_high
    n_low_bins = n_doppler_bins + n_window_bins // 2
    n_high_bins = n_doppler_bins + (n_window_bins - n_window_bins // 2) + 1
    return f0_low - n_low_bins / tsft, f0_high + n_high_bins / tsft


class SearchData:
    """What one job searches on: the band on the device and its templates' epoch."""

    def __init__(
        self,
        sfts: list[SFTData],
        *,
        floor: list[Float[Array, "n_sft n_band"]],
        t_ref: float,
    ) -> None:
        """Hold the band's SFTs and floor; velocities and positions go on the device.

        Args:
            sfts: The SFTs per detector, scaled.
            floor: Mean noise power in each SFT and bin, per detector.
            t_ref: Epoch `phi_b` is quoted at, absolute GPS [s].
        """
        self.sfts = sfts
        self.floor = floor
        self.t_ref = t_ref
        self.velocity_per_detector = [jnp.asarray(sft.velocity) for sft in sfts]
        self.velocity = np.concatenate([sft.velocity for sft in sfts])
        self.times = np.concatenate([sft.timestamps_midpoints for sft in sfts]) - t_ref
        self.position = jnp.asarray(np.concatenate([sft.position for sft in sfts]))

    @classmethod
    def load(
        cls,
        data_setup: DataSetup,
        files: list[str],
        freq_low: float,
        freq_high: float,
        *,
        floor_width: int,
        floor_batch_size: int,
    ) -> "SearchData":
        """A band read from SFT files, its noise floor taken per detector.

        Args:
            data_setup: The SFTs' detectors, count, length, window and epoch.
            files: The files holding the band, at least one per detector.
            freq_low: Low edge of the band [Hz].
            freq_high: High edge of the band [Hz].
            floor_width: Number of bins in the running median the floor is taken over.
            floor_batch_size: Number of SFTs whose floor is taken at once.

        Returns:
            The band, ready to search.
        """
        sfts = load_sft_data(
            files,
            freq_low,
            freq_high,
            window=data_setup.window,
            tsft=data_setup.tsft,
            detectors=data_setup.detectors,
            n_sfts=data_setup.n_sfts,
        )
        floor = [
            noise_floor(
                jnp.asarray(sft.sft_power),
                width=floor_width,
                batch_size=floor_batch_size,
            )
            for sft in sfts
        ]
        return cls(sfts, floor=floor, t_ref=data_setup.t_ref)

    def sub_band(self, freq_low: float, freq_high: float) -> "SearchData":
        """A narrower band of this one, its SFTs and floor cut together.

        Args:
            freq_low: Low edge of the narrower band [Hz].
            freq_high: High edge [Hz].

        Returns:
            The bins nearest the edges and those between, at the same epoch.
        """
        start_idx = int(round((freq_low - self.freq_low) * self.tsft))
        stop_idx = int(round((freq_high - self.freq_low) * self.tsft))
        if start_idx < 0 or stop_idx > self.n_band_bins:
            raise ValueError(
                f"[{freq_low:.5f}, {freq_high:.5f}] Hz is not in the data "
                f"[{self.freq_low:.5f}, "
                f"{self.freq_low + self.n_band_bins / self.tsft:.5f}] Hz"
            )
        bin_idxs = slice(start_idx, stop_idx)
        return SearchData(
            [sft.sub_band(bin_idxs) for sft in self.sfts],
            floor=[
                floor_per_detector[:, bin_idxs] for floor_per_detector in self.floor
            ],
            t_ref=self.t_ref,
        )

    @cached_property
    def sft_time_series(self) -> list[Complex[Array, "n_sft n_band"]]:
        """Inverse FFT of each SFT's band, per detector."""
        return [jnp.fft.ifft(jnp.asarray(sft.sft_data), axis=-1) for sft in self.sfts]

    @cached_property
    def mean_power(self) -> list[Float[Array, " n_band"]]:
        """Floor-normalized power averaged over the SFTs, per detector."""
        return [
            jnp.mean(jnp.asarray(sft.sft_power) / floor_per_detector, axis=0)
            for sft, floor_per_detector in zip(self.sfts, self.floor, strict=True)
        ]

    @cached_property
    def sft_noise_level(self) -> Float[np.ndarray, " n_sft"]:
        """Each SFT's floor averaged over the frequency band, detectors concatenated."""
        return np.concatenate(
            [
                np.asarray(jnp.mean(floor_per_detector, axis=1))
                for floor_per_detector in self.floor
            ]
        )

    def sft_weights(self, alpha: float, delta: float) -> Float[Array, " n_sft"]:
        """The weight each SFT takes toward a sky position, of unit mean square.

            w[n] proportional to (F_plus[n]^2 + F_cross[n]^2) / noise_level[n],
            scaled to <w^2>_n = 1

        Args:
            alpha: Right ascension [rad].
            delta: Declination [rad].

        Returns:
            The weight per SFT, the detectors concatenated, in the floor's dtype.
        """
        antenna_power = np.concatenate(
            [
                sft.detector.antenna_power(
                    sft.timestamps_midpoints, alpha=alpha, delta=delta
                )
                for sft in self.sfts
            ]
        )
        weights = antenna_power / self.sft_noise_level
        return jnp.asarray(
            weights / np.sqrt(np.mean(weights**2)), dtype=self.floor[0].dtype
        )

    @property
    def tsft(self) -> float:
        """SFT length [s]."""
        return self.sfts[0].tsft

    @property
    def freq_low(self) -> float:
        """Frequency of the band's first bin [Hz]."""
        return self.sfts[0].freq_low

    @property
    def n_sfts(self) -> int:
        """Number of SFTs over every detector."""
        return int(sum(sft.n_sfts for sft in self.sfts))

    @property
    def n_band_bins(self) -> int:
        """Number of frequency bins loaded."""
        return self.sfts[0].sft_data.shape[1]

    @property
    def sqrt_sn(self) -> float:
        """Noise amplitude spectral density over the band and the run [1/sqrt(Hz)].

            P_n = <floor[n, k]>_k,  P = 1 / <1 / P_n>_n,
            sqrt_sn = sqrt(P / tsft) * strain_unit

        n runs over every SFT of every detector and k over the band; floor estimates
        2|X|^2 = Sn tsft, so P / tsft is Sn in units of strain_unit^2.
        """
        per_sft = jnp.concatenate(
            [jnp.mean(floor_per_detector, axis=1) for floor_per_detector in self.floor]
        )
        mean_power = float(1.0 / jnp.mean(1.0 / per_sft))
        return float(np.sqrt(mean_power / self.tsft)) * self.sfts[0].strain_unit
