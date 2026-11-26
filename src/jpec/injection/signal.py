"""Simulated signals: a source's SFT amplitudes, and SFTs with the source injected."""

from dataclasses import replace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex

from jpec.detector.doppler import line_of_sight_velocity, solar_roemer_delay
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

        s[n, k] = (A_+ F_+[n] - i A_x F_x[n]) (tsft / 2) centre_amplitude
                  W(k - arc[n]) exp(-i pi (k - arc[n]))

    with `W` the window's response, 1 on the centre, and `centre_amplitude` the
    window's amplitude on the centre in SFTs normalized to the same noise power
    whatever the window.

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
    sky_alpha, sky_delta = jnp.array([alpha]), jnp.array([delta])
    delay = solar_roemer_delay(
        jnp.asarray(sft.position), alpha=sky_alpha, delta=sky_delta
    )[0]
    # SFT midpoints from t_ref in barycenter time, which the orbit runs in
    barycenter_times = jnp.asarray(sft.timestamps_midpoints - t_ref) + delay

    doppler = line_of_sight_velocity(
        jnp.asarray(sft.velocity), alpha=sky_alpha, delta=sky_delta
    )[0]
    offset = frequency_offset(
        barycenter_times,
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
    polarization = (
        (a_plus * f_plus - 1j * a_cross * f_cross)
        * (sft.tsft / 2.0)
        * sft.window.centre_amplitude()
    )
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
