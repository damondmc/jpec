"""Simulated signals: a source's SFT amplitudes, and SFTs with the source injected."""

from dataclasses import replace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex

from jpec.detector.doppler import sky_direction
from jpec.detector.sft import SFTData
from jpec.params import Parameters
from jpec.typing import Scalar
from jpec.waveform.plane import amplitude_from_arc
from jpec.waveform.track import frequency_offset


class Orientation(NamedTuple):
    """A source's orientation: how loud it presents, nothing about where its track goes.

    Attributes:
        cos_iota: Cosine of the inclination, in `[-1, 1]`.
        psi: Polarization angle [rad].
    """

    cos_iota: Scalar
    psi: Scalar


class Injection(NamedTuple):
    """One source: where it sits in the search space and how it is oriented.

    Attributes:
        parameters: Its track: f0, the orbit and the sky, `phi_b` at the reference
            epoch.
        orientation: Its inclination and polarization, which set only how loud it is.
    """

    parameters: Parameters
    orientation: Orientation


def simulated_signal_sft_data(
    sft: SFTData, injection: Injection, *, t_ref: float
) -> Complex[Array, "n_sft n_freq"]:
    """The SFT data a simulated source leaves at unit strain, in closed form.

        s[n, k] = (A_+ F_+[n] - i A_x F_x[n]) (tsft / 2)
                  W(k - arc[n]) exp(-i pi (k - arc[n]))

    Args:
        sft: The SFTs whose bins, times, detector and window the signal is laid on.
        injection: The source, its `phi_b` quoted at `t_ref`.
        t_ref: Epoch `phi_b` is quoted at, absolute GPS [s].

    Returns:
        The source's SFT data in strain x s, for `h0 = 1`.
    """
    parameters, orientation = injection
    f0 = float(parameters.f0)
    alpha, delta = float(parameters.alpha), float(parameters.delta)
    times = jnp.asarray(sft.timestamps_midpoints - t_ref)

    doppler = jnp.asarray(sft.velocity) @ sky_direction(alpha, delta)
    offset = frequency_offset(
        times,
        modulation_depth=float(parameters.modulation_depth),
        omega=float(parameters.omega),
        phi_b=parameters.phi_b,
    )
    # [bins] from freq_low; f0 - freq_low on the host, so float32 never holds f0
    # itself
    f0_from_freq_low = f0 - sft.freq_low
    arc = (f0_from_freq_low + f0 * doppler + offset) * sft.tsft
    bins = jnp.arange(sft.sft_data.shape[1], dtype=float)
    response = amplitude_from_arc(arc, bins, window=sft.window)

    f_plus, f_cross = sft.detector.antenna_pattern(
        sft.timestamps_midpoints, alpha=alpha, delta=delta, psi=float(orientation.psi)
    )
    cos_iota = float(orientation.cos_iota)
    a_plus, a_cross = 0.5 * (1.0 + cos_iota**2), cos_iota
    polarization = (a_plus * f_plus - 1j * a_cross * f_cross) * (sft.tsft / 2.0)
    return response * polarization[:, None]


def sft_with_simulated_signal(
    sft: SFTData,
    unit_sft_data: Complex[np.ndarray, "n_sft n_freq"],
    *,
    h0: float,
) -> SFTData:
    """SFTs, real or simulated noise, with a simulated signal added at amplitude `h0`.

    Args:
        sft: The SFTs, at their own `strain_unit`.
        unit_sft_data: simulated_signal_sft_data() on the same bins and times, in
            strain x s.
        h0: Strain amplitude.

    Returns:
        The SFTs holding the signal, at the same `strain_unit`.
    """
    added = h0 * unit_sft_data / sft.strain_unit
    sft_data = (sft.sft_data + added).astype(sft.sft_data.dtype)
    return replace(sft, sft_data=sft_data)


class SimulatedSource(NamedTuple):
    """One source and its unit-strain SFT data, the same at any amplitude and mismatch.

    Attributes:
        injection: The source.
        unit_sft_data: Its SFT data at unit strain on the SFTs it was simulated on, per
            detector.
    """

    injection: Injection
    unit_sft_data: list[Complex[np.ndarray, "n_sft n_freq"]]

    def injected_sfts(self, sfts: list[SFTData], *, h0: float) -> list[SFTData]:
        """The SFTs it was simulated on, with it added at amplitude `h0`.

        Args:
            sfts: The SFTs, per detector, on the bins and times it was simulated on.
            h0: Strain amplitude.

        Returns:
            The SFTs holding the source.
        """
        return [
            sft_with_simulated_signal(sft, unit_sft_data, h0=h0)
            for sft, unit_sft_data in zip(sfts, self.unit_sft_data)
        ]


def simulated_source(
    injection: Injection, sfts: list[SFTData], *, t_ref: float
) -> SimulatedSource:
    """One source simulated on the bins and at the times of some SFTs, at unit strain.

    Args:
        injection: The source, its `phi_b` quoted at `t_ref`.
        sfts: The SFTs per detector, real or simulated noise, it is laid on.
        t_ref: Epoch `phi_b` is quoted at, absolute GPS [s].

    Returns:
        The source, to be scaled by h0 and added to those SFTs.
    """
    unit_sft_data = [
        np.asarray(simulated_signal_sft_data(sft, injection, t_ref=t_ref))
        for sft in sfts
    ]
    return SimulatedSource(injection=injection, unit_sft_data=unit_sft_data)
