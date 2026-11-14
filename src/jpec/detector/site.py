"""Detectors: velocity and position from the ephemeris, the epoch, and the SFT times."""

import warnings
from collections.abc import Mapping
from functools import cache
from typing import Any

import lal as _lal
import lalpulsar as _lalpulsar
import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike

lal: Any = _lal
lalpulsar: Any = _lalpulsar


@cache
def _ephemeris() -> Any:
    """The DE405 Earth and Sun tables spanning 2000 to 2040, loaded once and shared."""
    return lalpulsar.InitBarycenter("earth00-40-DE405.dat.gz", "sun00-40-DE405.dat.gz")


@cache
def _detector_states(
    detector_name: str, timestamps: bytes
) -> tuple[Float[np.ndarray, "n_time 3"], Float[np.ndarray, "n_time 3"]]:
    """Detector velocity and position per timestamp, keyed on the timestamps' bytes.

    Args:
        detector_name: Detector name.
        timestamps: Absolute GPS times [s], as the bytes of a float64 array.

    Returns:
        Cartesian `v/c` and position [light-s] from the Solar System Barycenter per
        time, read-only.
    """
    gps_times = np.frombuffer(timestamps, dtype=np.float64)
    vector = lalpulsar.CreateTimestampVector(gps_times.size)
    for time_idx, time in enumerate(gps_times):
        vector.data[time_idx] = lal.LIGOTimeGPS(float(time))
    states = lalpulsar.GetDetectorStates(
        vector, lal.cached_detector_by_prefix[detector_name], _ephemeris(), 0.0
    )
    velocity = np.array([state.vDetector for state in states.data], dtype=np.float64)
    position = np.array([state.rDetector for state in states.data], dtype=np.float64)
    velocity.flags.writeable = False
    position.flags.writeable = False
    return velocity, position


@cache
def _sidereal_times(timestamps: bytes) -> Float[np.ndarray, " n_time"]:
    """Greenwich mean sidereal time per timestamp, keyed on the timestamps' bytes.

    Args:
        timestamps: Absolute GPS times [s], as the bytes of a float64 array.

    Returns:
        The sidereal time in `[0, 2 pi)` [rad], read-only.
    """
    gps_times = np.frombuffer(timestamps, dtype=np.float64)
    sidereal_times = np.array(
        [
            lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(time)))
            for time in gps_times
        ]
    )
    sidereal_times = np.mod(sidereal_times, 2.0 * np.pi)
    sidereal_times.flags.writeable = False
    return sidereal_times


class Detector:
    """One interferometer: where it sits and how it moves."""

    def __init__(self, name: str) -> None:
        """Take the detector's name, warning when it is not H1 or L1.

        Args:
            name: Detector name, `"H1"` or `"L1"`.
        """
        if name not in ("H1", "L1"):
            warnings.warn(
                f'{name!r} is not supported; only "H1", "L1" are', stacklevel=2
            )
        self.name = name
        self.site: Any = lal.cached_detector_by_prefix[name]
        self.ephemeris = _ephemeris()

    def velocity(
        self, gps_times: Float[np.ndarray, " n_time"]
    ) -> Float[np.ndarray, "n_time 3"]:
        """Velocity of the site in the Solar System Barycenter frame, `v/c`.

        Cached on the timestamps, so a run walked twice costs one pass through LAL.

        Args:
            gps_times: Absolute GPS times [s].

        Returns:
            Cartesian `v/c` per time, ~1e-4, read-only.
        """
        return _detector_states(
            self.name, np.ascontiguousarray(gps_times, dtype=np.float64).tobytes()
        )[0]

    def position(
        self, gps_times: Float[np.ndarray, " n_time"]
    ) -> Float[np.ndarray, "n_time 3"]:
        """Position of the site from the Solar System Barycenter, in light-seconds.

        Cached on the timestamps with the velocity, so both cost one pass through LAL.

        Args:
            gps_times: Absolute GPS times [s].

        Returns:
            Cartesian position [light-s] per time, ~500, read-only.
        """
        return _detector_states(
            self.name, np.ascontiguousarray(gps_times, dtype=np.float64).tobytes()
        )[1]

    def antenna_pattern(
        self,
        gps_times: Float[np.ndarray, " n_time"],
        *,
        alpha: float,
        delta: float,
        psi: float,
    ) -> tuple[Float[np.ndarray, " n_time"], Float[np.ndarray, " n_time"]]:
        """How much of each polarization the site sees, at each time.

        Args:
            gps_times: Absolute GPS times [s].
            alpha: Right ascension [rad].
            delta: Declination [rad].
            psi: Polarization angle [rad].

        Returns:
            `F_+` and `F_x`, each in `[-1, 1]`.
        """
        f_plus, f_cross = np.empty(gps_times.size), np.empty(gps_times.size)
        for time_idx, time in enumerate(gps_times):
            gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(time)))
            f_plus[time_idx], f_cross[time_idx] = lal.ComputeDetAMResponse(
                self.site.response, alpha, delta, psi, gmst
            )
        return f_plus, f_cross

    def antenna_power(
        self, gps_times: Float[np.ndarray, " n_time"], *, alpha: float, delta: float
    ) -> Float[np.ndarray, " n_time"]:
        """`F_+^2 + F_x^2` at each time, which does not depend on polarization angle.

        Follows Anderson, Brady, Creighton and Flanagan, Phys. Rev. D 63, 042003
        (2001), arXiv:gr-qc/0008066, Appendix B, which lal.ComputeDetAMResponse()
        computes. Here we set the polarization angle to zero. The wave-frame axes in
        Earth-fixed coordinates, with `h = gmst - alpha` the hour angle, are

            X = (-sin h, -cos h, 0),  Y = (-cos h sin delta, sin h sin delta, cos delta)

            F_+ = X.D.X - Y.D.Y,  F_x = 2 X.D.Y

        Args:
            gps_times: Absolute GPS times [s].
            alpha: Right ascension [rad].
            delta: Declination [rad].

        Returns:
            `F_+^2 + F_x^2` per time, in `[0, 1]`.
        """
        sidereal_times = _sidereal_times(
            np.ascontiguousarray(gps_times, dtype=np.float64).tobytes()
        )
        hour_angle = sidereal_times - alpha
        sin_h, cos_h = np.sin(hour_angle), np.cos(hour_angle)
        x = np.stack([-sin_h, -cos_h, np.zeros_like(sin_h)], axis=1)
        y = np.stack(
            [
                -cos_h * np.sin(delta),
                sin_h * np.sin(delta),
                np.full_like(sin_h, np.cos(delta)),
            ],
            axis=1,
        )
        response = np.asarray(self.site.response, dtype=np.float64)
        response_x, response_y = x @ response, y @ response
        f_plus = np.sum(response_x * x, axis=1) - np.sum(response_y * y, axis=1)
        f_cross = 2.0 * np.sum(response_x * y, axis=1)
        return f_plus**2 + f_cross**2


def reference_epoch(timestamps: Mapping[str, ArrayLike], *, tsft: float) -> float:
    """The middle of the SFTs' span, the epoch `phi_b` is quoted at.

    Args:
        timestamps: SFT start times per detector, absolute GPS [s].
        tsft: SFT length [s].

    Returns:
        `t_ref`, absolute GPS [s].
    """
    starts = np.concatenate(
        [np.asarray(each, dtype=float) for each in timestamps.values()]
    )
    return float((starts.max() + starts.min() + tsft) / 2.0)


class Observation:
    """The SFT midpoints of a run, and where the detectors are and move at each.

    Attributes:
        tsft: SFT length [s].
        times: SFT midpoints from `t_ref` in detector time [s], every detector.
        velocity: Detector velocity `v/c` at each midpoint.
        position: Detector position [light-s] at each midpoint.
    """

    def __init__(
        self, timestamps: Mapping[str, ArrayLike], *, tsft: float, t_ref: float
    ) -> None:
        """Take the midpoints and the detector motion there from the SFT start times.

        Args:
            timestamps: SFT start times per detector, absolute GPS [s].
            tsft: SFT length [s].
            t_ref: Epoch the midpoints are measured from, absolute GPS [s].
        """
        self.tsft = tsft
        gps_midpoints = {
            detector_name: np.asarray(starts, dtype=float) + tsft / 2.0
            for detector_name, starts in timestamps.items()
        }
        # the detectors concatenated in the dict's order
        self.times = np.concatenate(list(gps_midpoints.values())) - t_ref
        self.velocity = np.concatenate(
            [
                Detector(detector_name).velocity(midpoints)
                for detector_name, midpoints in gps_midpoints.items()
            ]
        )
        self.position = np.concatenate(
            [
                Detector(detector_name).position(midpoints)
                for detector_name, midpoints in gps_midpoints.items()
            ]
        )
