"""The band one job searches: its width, its SFTs on the device, and its epoch."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jpec.config import DataSetup
from jpec.constant import ARC_PAD_BINS, SHIFT_MARGIN_BINS, V_MAX
from jpec.detector.sft import SFTData, load_sft_data
from jpec.noise.floor import noise_floor
from jpec.params import Prior, omega_from_period
from jpec.waveform.plane import arc_bin_offsets


def band_edges_for_f0(
    prior: Prior, f0_low: float, f0_high: float, *, tsft: float
) -> tuple[float, float]:
    """The frequency band to load for an f0 interval.

    Args:
        prior: The region searched, whose deepest orbit sets the widest template.
        f0_low: Low edge of the interval [Hz].
        f0_high: High edge [Hz].
        tsft: SFT length [s].

    Returns:
        The band's low and high edge [Hz].
    """
    ap_high, period_low = prior.ap[1], prior.period[0]
    n_window_bins = arc_bin_offsets(
        f0_high * ap_high * float(omega_from_period(jnp.asarray(period_low))) * tsft,
        pad_bins=ARC_PAD_BINS,
    ).shape[0]
    margin = (
        np.ceil(f0_high * V_MAX * tsft) + n_window_bins // 2 + SHIFT_MARGIN_BINS
    ) / tsft
    return f0_low - margin, f0_high + margin


class SearchData:
    """What one job searches on: the band on the device and its templates' epoch."""

    def __init__(
        self,
        sfts: list[SFTData],
        *,
        floor: list[Float[Array, "n_sft n_band"]],
        t_ref: float,
    ) -> None:
        """Put the band, its time series, velocities and times on the device.

        Args:
            sfts: The SFTs per detector, scaled.
            floor: Mean noise power in each SFT and bin, per detector.
            t_ref: Epoch `phi_b` is quoted at, absolute GPS [s].
        """
        self.sfts = sfts
        self.floor = floor
        self.t_ref = t_ref
        # the inverse transform the Doppler correction shifts, the same at every
        # sky and f0
        self.sft_time_series = [
            jnp.fft.ifft(jnp.asarray(sft.sft_data), axis=-1) for sft in sfts
        ]
        self.velocity_per_detector = [jnp.asarray(sft.velocity) for sft in sfts]
        # the velocity [v/c] the sky lattice is built from, detectors
        # concatenated, on the host
        self.velocity = np.concatenate([sft.velocity for sft in sfts])
        # every SFT midpoint from t_ref [s], detectors concatenated, for the
        # orbital basis
        self.times = jnp.asarray(
            np.concatenate([sft.timestamps_midpoints for sft in sfts]) - t_ref
        )

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
            floor_width: Bins of the running median the floor is taken over.
            floor_batch_size: SFTs whose floor is taken at once.

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

    def around_f0(self, prior: Prior, f0: float) -> "SearchData":
        """The band the search around one f0 reads, its SFTs and floor cut together.

        Args:
            prior: The region searched, whose deepest orbit sets the widest template.
            f0: The frequency searched around [Hz].

        Returns:
            The narrower band, wide enough for the template, the Doppler shift and
                the f0 shift, at the same epoch.
        """
        search_freq_low, search_freq_high = band_edges_for_f0(
            prior, f0, f0, tsft=self.tsft
        )
        start_idx = int(round((search_freq_low - self.freq_low) * self.tsft))
        stop_idx = int(round((search_freq_high - self.freq_low) * self.tsft))
        if start_idx < 0 or stop_idx > self.n_band_bins:
            raise ValueError(
                f"the search around {f0:.5f} Hz needs {search_freq_low:.5f} "
                f"to {search_freq_high:.5f} Hz, "
                f"and the band holds {self.freq_low:.5f} "
                f"to {self.freq_low + self.n_band_bins / self.tsft:.5f} Hz; "
                "widen the loaded band"
            )
        bin_idxs = slice(start_idx, stop_idx)
        return SearchData(
            [sft.narrowed(bin_idxs) for sft in self.sfts],
            floor=[
                floor_per_detector[:, bin_idxs] for floor_per_detector in self.floor
            ],
            t_ref=self.t_ref,
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
        """SFTs over every detector."""
        return int(sum(sft.n_sfts for sft in self.sfts))

    @property
    def n_band_bins(self) -> int:
        """Frequency bins loaded."""
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
