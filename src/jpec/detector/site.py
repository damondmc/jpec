"""The detectors: their velocity from the ephemeris, the reference epoch, and the SFT times."""

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
    """The DE405 Earth and Sun tables spanning 2000 to 2040, a ``lalpulsar.EphemerisData`` loaded once and shared."""
    return lalpulsar.InitBarycenter("earth00-40-DE405.dat.gz", "sun00-40-DE405.dat.gz")


@cache
def _velocity(detector_name: str, timestamps: bytes) -> Float[np.ndarray, "n_time 3"]:
    """Detector velocity at each timestamp, keyed on their bytes so a run is walked once.

    Args:
        detector_name: Detector name.
        timestamps: Absolute GPS times [s], as the bytes of a float64 array.

    Returns:
        Cartesian ``v/c`` per time, read-only.
    """
    gps_times = np.frombuffer(timestamps, dtype=np.float64)
    vector = lalpulsar.CreateTimestampVector(gps_times.size)
    for time_idx, time in enumerate(gps_times):
        vector.data[time_idx] = lal.LIGOTimeGPS(float(time))
    states = lalpulsar.GetDetectorStates(
        vector, lal.cached_detector_by_prefix[detector_name], _ephemeris(), 0.0
    )
    velocity = np.array([state.vDetector for state in states.data], dtype=np.float64)
    velocity.flags.writeable = False
    return velocity


class Detector:
    """One interferometer: where it sits and how it moves.

    Args:
        name: Detector name, ``"H1"`` or ``"L1"``.
    """

    def __init__(self, name: str) -> None:
        """Take the detector's name, warning when it is not H1 or L1."""
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
        """Velocity of the site in the solar-system barycenter frame, ``v/c``.

        Cached on the timestamps, so a run walked twice costs one pass through LAL.

        Args:
            gps_times: Absolute GPS times [s].

        Returns:
            Cartesian ``v/c`` per time, ~1e-4, read-only.
        """
        return _velocity(
            self.name, np.ascontiguousarray(gps_times, dtype=np.float64).tobytes()
        )

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
            ``F_plus`` and ``F_cross``, each in ``[-1, 1]``.
        """
        f_plus, f_cross = np.empty(gps_times.size), np.empty(gps_times.size)
        for time_idx, time in enumerate(gps_times):
            gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(time)))
            f_plus[time_idx], f_cross[time_idx] = lal.ComputeDetAMResponse(
                self.site.response, alpha, delta, psi, gmst
            )
        return f_plus, f_cross


def reference_epoch(timestamps: Mapping[str, ArrayLike], *, tsft: float) -> float:
    """The middle of the SFTs' span, the epoch ``phi_b`` is quoted at.

    Args:
        timestamps: SFT start times per detector, absolute GPS [s].
        tsft: SFT length [s].

    Returns:
        ``t_ref``, absolute GPS [s].
    """
    starts = np.concatenate(
        [np.asarray(each, dtype=float) for each in timestamps.values()]
    )
    return float((starts.max() + starts.min() + tsft) / 2.0)


def times_and_velocity(
    timestamps: Mapping[str, ArrayLike], *, tsft: float, t_ref: float
) -> tuple[Float[np.ndarray, " n_sft"], Float[np.ndarray, "n_sft 3"]]:
    """The SFT midpoints from the reference epoch, and the detector velocities there.

    Args:
        timestamps: SFT start times per detector, absolute GPS [s].
        tsft: SFT length [s].
        t_ref: Epoch the midpoints are measured from, absolute GPS [s].

    Returns:
        The midpoints from ``t_ref`` [s] and ``v/c`` at each, the detectors
        concatenated in the dict's order.
    """
    gps_midpoints = {
        detector_name: np.asarray(starts, dtype=float) + tsft / 2.0
        for detector_name, starts in timestamps.items()
    }
    velocity = np.concatenate(
        [
            Detector(detector_name).velocity(midpoints)
            for detector_name, midpoints in gps_midpoints.items()
        ]
    )
    return np.concatenate(list(gps_midpoints.values())) - t_ref, velocity
