"""The lattice tiling of a box, its steps taken from the metric at the top of the box."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jpec.bank.lattice import (
    BCC_SUBGRID_OFFSETS,
    ApBlock,
    OrbitalSubgrid,
    cubic_step,
    orbital_subgrid,
    sky_lattice,
)
from jpec.bank.metric import (
    freq_bin_curvature,
    orbital_lattice_metric,
    physical_metric,
    project_out,
)
from jpec.constant import CURVATURE_POSITIONS
from jpec.params import Parameters, Prior
from jpec.waveform.window import Window

# (alpha, delta) [rad]: the orbital metric depends on the sky only through the 1 + v.n of its
# f0 entry, which moves it by under 2e-4 over the sky, so it is taken at one position
METRIC_SKY_POSITION = (1.2, 0.3)
# f0, modulation depth, omega^2 and two sky coordinates share the mismatch
N_AXES = 5


@dataclass(frozen=True)
class LatticeTiling:
    """The bank's five axes: a body-centred cubic lattice on the orbit and a hexagonal one on the sky.

    Args:
        prior: The region searched; its ``phi_b`` interval is not used.
        mismatch: Mismatch budget over every axis.
        tsft: SFT length [s].
        ap_blocks: One per ap block, in order, all from the metric at the top of the box.
    """

    prior: Prior
    mismatch: float
    tsft: float
    ap_blocks: list[ApBlock]

    @property
    def f0_reference(self) -> float:
        """Frequency the omega^2 steps were taken at, the top of the box [Hz]."""
        return self.prior.f0[1]

    @property
    def mismatch_per_axis(self) -> float:
        """Mismatch one axis is gridded at."""
        return self.mismatch / N_AXES

    @property
    def g_f0f0(self) -> float:
        """The f0 curvature at the top of the box, which the f0 grid and the sky lattice are set by."""
        return self.ap_blocks[-1].g_f0f0

    @property
    def f0_step(self) -> float:
        """Cubic f0 step at the top of the box [Hz]."""
        return float(cubic_step(jnp.asarray(self.g_f0f0), self.mismatch_per_axis))

    @property
    def modulation_depth_step(self) -> float:
        """Cubic modulation-depth step at the top of the box [bins]."""
        return self.ap_blocks[-1].modulation_depth_step

    @classmethod
    def from_metric(
        cls,
        prior: Prior,
        *,
        times: Float[Array, " n_sft"],
        velocity: Float[Array, "n_sft 3"],
        tsft: float,
        mismatch: float,
        n_ap_blocks: int,
        window: Window,
    ) -> "LatticeTiling":
        """The lattice the metric of one observation asks for over a box.

        Args:
            prior: The region searched.
            times: SFT midpoints from the reference epoch [s], every detector.
            velocity: Detector velocity ``v/c`` at each of them.
            tsft: SFT length [s].
            mismatch: Mismatch budget over every axis.
            n_ap_blocks: ap blocks the box is cut into.
            window: The window the data was made with.

        Returns:
            The lattice, its steps measured.
        """
        mismatch_per_axis = mismatch / N_AXES
        g_freq_bin = freq_bin_curvature(window, n_positions=CURVATURE_POSITIONS)
        top = Parameters(
            f0=prior.f0[1],
            ap=prior.ap[1],
            period=prior.period[1],
            phi_b=1.0,
            alpha=METRIC_SKY_POSITION[0],
            delta=METRIC_SKY_POSITION[1],
        )

        metric = project_out(
            physical_metric(
                top,
                times=times,
                velocity=velocity,
                tsft=tsft,
                g_freq_bin=g_freq_bin,
            )
        )
        orbital_metric = orbital_lattice_metric(metric, top, tsft=tsft)
        orbital_steps = np.asarray(
            cubic_step(jnp.diag(orbital_metric), mismatch_per_axis)
        )

        # ap enters the track only through the modulation depth, so the omega^2 step goes exactly
        # as 1 / ap and the other steps do not change with it
        edges = np.linspace(*prior.ap, n_ap_blocks + 1)
        ap_blocks = [
            ApBlock(
                ap_low=float(low),
                ap_high=float(high),
                modulation_depth_step=float(orbital_steps[1]),
                omega2_step=float(orbital_steps[2]) * prior.ap[1] / float(high),
                g_f0f0=float(metric[0, 0]),
            )
            for low, high in zip(edges[:-1], edges[1:])
        ]
        return cls(prior=prior, mismatch=mismatch, tsft=tsft, ap_blocks=ap_blocks)

    def orbital_subgrids(self, f0_low: float, f0_high: float) -> list[OrbitalSubgrid]:
        """Every subgrid of the orbital lattice over an f0 band, in the order of ``BCC_SUBGRID_OFFSETS``.

        Args:
            f0_low: Low edge of the band [Hz].
            f0_high: High edge [Hz].

        Returns:
            One orbital_subgrid() per offset.
        """
        return [
            orbital_subgrid(
                f0_low,
                f0_high,
                ap_low=self.prior.ap[0],
                ap_high=self.prior.ap[1],
                period_low=self.prior.period[0],
                period_high=self.prior.period[1],
                f0_step=self.f0_step,
                modulation_depth_step=self.modulation_depth_step,
                ap_blocks=self.ap_blocks,
                f0_reference=self.f0_reference,
                tsft=self.tsft,
                offset=offset,
            )
            for offset in BCC_SUBGRID_OFFSETS
        ]

    def sky_points(
        self, velocity: Float[np.ndarray, "n_sft 3"], *, f0: float
    ) -> tuple[Float[np.ndarray, " n_sky"], Float[np.ndarray, " n_sky"]]:
        """The sky lattice at one frequency, on the same mismatch per axis.

        Args:
            velocity: Detector velocity ``v/c`` at each SFT, every detector.
            f0: Frequency the lattice is for [Hz].

        Returns:
            Right ascension and declination of each position [rad].
        """
        return sky_lattice(
            velocity,
            f0=f0,
            g_f0f0=self.g_f0f0,
            mismatch=self.mismatch_per_axis,
            box=(*self.prior.alpha, *self.prior.delta),
        )

    def to_dict(self) -> dict[str, Any]:
        """The lattice as a record, with its derived steps written out for a reader.

        Returns:
            The record, the prior field by field in its own units, the period in seconds.
        """
        return {
            "lattice": "bcc",
            "prior": {
                name: list(interval) for name, interval in self.prior._asdict().items()
            },
            "mismatch": self.mismatch,
            "tsft": self.tsft,
            "f0_reference": self.f0_reference,
            "f0_step": self.f0_step,
            "modulation_depth_step": self.modulation_depth_step,
            "g_f0f0": self.g_f0f0,
            "ap_blocks": [ap_block._asdict() for ap_block in self.ap_blocks],
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "LatticeTiling":
        """The lattice a record describes.

        Args:
            record: What to_dict() wrote.

        Returns:
            The lattice; its derived steps are recomputed, not read.
        """
        if record.get("lattice") != "bcc" or "prior" not in record:
            raise ValueError(
                "the record is not a bcc lattice this class wrote; rerun jpec search-setup"
            )
        prior = Prior(
            **{name: tuple(interval) for name, interval in record["prior"].items()}
        )
        return cls(
            prior=prior,
            mismatch=record["mismatch"],
            tsft=record["tsft"],
            ap_blocks=[ApBlock(**ap_block) for ap_block in record["ap_blocks"]],
        )
