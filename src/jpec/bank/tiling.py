"""Lattice tiling of a box, its curvatures from the metric at the top of the box."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from jaxtyping import Float

from jpec.bank.metric import orbital_metric, physical_metric, project_out
from jpec.detector.site import Observation
from jpec.params import Parameters, Prior
from jpec.waveform.window import Window, freq_bin_curvature

# reference sky position (alpha, delta) [rad] the orbital metric is taken at
METRIC_SKY_POSITION = (1.2, 0.3)
# f0, modulation depth, omega^2 and two sky coordinates share the mismatch
N_AXES = 5


class ApBlock(NamedTuple):
    """One ap range of the box and the omega^2 curvature it is tiled with.

    Attributes:
        ap_low: Low edge of the ap block [light-seconds].
        ap_high: High edge [light-seconds], the ap its curvature is taken at.
        g_ww: The omega^2 curvature at `ap_high` and the box's top f0 [s^4/rad^4].
    """

    ap_low: float
    ap_high: float
    g_ww: float


@dataclass(frozen=True)
class LatticeTiling:
    """The bank's five axes, one A5* lattice under the metric.

    Attributes:
        prior: The region searched; its `phi_b` interval is not used.
        mismatch: Covering mismatch over every axis together.
        tsft: SFT length [s].
        g_f0f0: The f0 curvature at the box's top [1/Hz^2].
        g_dd: The modulation-depth curvature at the box's top [1/Hz^2].
        ap_blocks: One per ap block, in order.
    """

    prior: Prior
    mismatch: float
    tsft: float
    g_f0f0: float
    g_dd: float
    ap_blocks: list[ApBlock]

    @property
    def f0_reference(self) -> float:
        """Frequency the curvatures were taken at, the top of the box [Hz]."""
        return self.prior.f0[1]

    @property
    def mismatch_per_axis(self) -> float:
        """The mismatch over the axes, at which a cubic step is taken."""
        return self.mismatch / N_AXES

    @classmethod
    def from_metric(
        cls,
        prior: Prior,
        *,
        observation: Observation,
        mismatch: float,
        n_ap_blocks: int,
        window: Window,
    ) -> "LatticeTiling":
        """The lattice the metric of one observation asks for over a box.

        Args:
            prior: The region searched.
            observation: The run's SFT length, midpoints and detector motion.
            mismatch: Covering mismatch over every axis together.
            n_ap_blocks: Number of ap blocks the box is cut into.
            window: The window the SFTs were made with.

        Returns:
            The lattice, its curvatures measured.
        """
        g_freq_bin = freq_bin_curvature(window)
        top = Parameters(
            f0=prior.f0[1],
            ap=prior.ap[1],
            period=prior.period[1],
            phi_b=1.0,
            alpha=METRIC_SKY_POSITION[0],
            delta=METRIC_SKY_POSITION[1],
        )

        metric = project_out(physical_metric(top, observation, g_freq_bin=g_freq_bin))
        g_f0f0, g_dd, g_ww = np.diag(orbital_metric(metric, top))

        # ap enters the track only through the modulation depth, so the omega^2
        # curvature goes exactly as ap^2 and the others do not change with it
        edges = np.linspace(*prior.ap, n_ap_blocks + 1)
        ap_blocks = [
            ApBlock(
                ap_low=float(low),
                ap_high=float(high),
                g_ww=float(g_ww) * (float(high) / prior.ap[1]) ** 2,
            )
            for low, high in zip(edges[:-1], edges[1:])
        ]
        return cls(
            prior=prior,
            mismatch=mismatch,
            tsft=observation.tsft,
            g_f0f0=float(g_f0f0),
            g_dd=float(g_dd),
            ap_blocks=ap_blocks,
        )

    def lattice_metric(
        self,
        velocity: Float[np.ndarray, "n_sft 3"],
        *,
        f0: float,
        sky_eigenvectors: Float[np.ndarray, "3 3"],
    ) -> Float[np.ndarray, "n_ap_block 5 5"]:
        """The 5x5 metric on `(f0, modulation_depth, omega^2, x_a, x_b)` per ap block.

                     f0      depth   omega^2   x_a     x_b
            f0     [ g_00    .       .         g_0a    g_0b ]
            depth  [ .       g_dd    .         .       .    ]
            omega^2[ .       .       g_ww      .       .    ]
            x_a    [ g_0a    .       .         g_aa    g_ab ]
            x_b    [ g_0b    .       .         g_ab    g_bb ]

        `x_a`, `x_b` are direction cosines along the sky metric's two largest
        eigenvectors `e_a`, `e_b`, and f0 and the sky follow the track
        `f0 (1 + v . n)`, with `<.>` the mean over every SFT:

            g_00 = g_f0f0,  g_0a = g_f0f0 f0 <v . e_a>,
            g_ab = g_f0f0 f0^2 <(v . e_a)(v . e_b)>

        and `g_ww` each ap block's, scaled by `(f0 / f0_reference)^2`.

        Args:
            velocity: Detector velocity `v/c` at each SFT, every detector.
            f0: Frequency the metric is for [Hz].
            sky_eigenvectors: sky_metric()'s eigenvectors as columns, largest first.

        Returns:
            The metric of each ap block.
        """
        # v . e_a and v . e_b at each SFT
        sky_velocity = np.asarray(velocity) @ sky_eigenvectors[:, :2]
        g_f0_sky = self.g_f0f0 * f0 * sky_velocity.mean(axis=0)
        # <(v . e_i)(v . e_j)>, the mean over SFTs
        sky_second_moment = sky_velocity.T @ sky_velocity / len(sky_velocity)
        g_sky = self.g_f0f0 * f0**2 * sky_second_moment
        g_ww = np.array([ap_block.g_ww for ap_block in self.ap_blocks])
        g_ww = g_ww * (f0 / self.f0_reference) ** 2

        metric = np.zeros((len(self.ap_blocks), N_AXES, N_AXES))
        metric[:, 0, 0] = self.g_f0f0
        metric[:, 1, 1] = self.g_dd
        metric[:, 2, 2] = g_ww
        metric[:, 0, 3:] = g_f0_sky
        metric[:, 3:, 0] = g_f0_sky
        metric[:, 3:, 3:] = g_sky
        return metric

    def to_dict(self) -> dict[str, Any]:
        """The lattice as a record.

        Returns:
            The record, the prior field by field in its own units, the period in
                seconds.
        """
        return {
            "lattice": "a5_star",
            "prior": {
                name: list(interval) for name, interval in self.prior._asdict().items()
            },
            "mismatch": self.mismatch,
            "tsft": self.tsft,
            "g_f0f0": self.g_f0f0,
            "g_dd": self.g_dd,
            "ap_blocks": [ap_block._asdict() for ap_block in self.ap_blocks],
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "LatticeTiling":
        """The lattice a record describes.

        Args:
            record: What to_dict() wrote.

        Returns:
            The lattice.
        """
        if record.get("lattice") != "a5_star":
            raise ValueError(
                "the record is not an a5_star lattice; "
                "rerun jpec search-setup"
            )
        prior = Prior(
            **{name: tuple(interval) for name, interval in record["prior"].items()}
        )
        return cls(
            prior=prior,
            mismatch=record["mismatch"],
            tsft=record["tsft"],
            g_f0f0=record["g_f0f0"],
            g_dd=record["g_dd"],
            ap_blocks=[ApBlock(**ap_block) for ap_block in record["ap_blocks"]],
        )
