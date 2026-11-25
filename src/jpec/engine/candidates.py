"""What the search loop scores, and what is kept of it."""

import heapq
from typing import NamedTuple

import numpy as np
from jaxtyping import Float, Int

from jpec.bank.template_bank import TemplateBank
from jpec.constant import DAY
from jpec.params import Parameters


class Candidate(NamedTuple):
    """One template and what it scored; the statistic leads, so candidates order by it.

    Attributes:
        statistic: Maximum of `S` over the orbital phase.
        parameters: The template, its `phi_b` where that maximum sits.
        doppler_reference: Frequency the template's band was Doppler-corrected at
            [Hz].
    """

    statistic: float
    parameters: Parameters
    doppler_reference: float

    def to_dict(self) -> dict[str, float]:
        """Convert the candidate to a dictionary."""
        return {
            "f0": float(self.parameters.f0),
            "ap": float(self.parameters.ap),
            "period_days": float(self.parameters.period) / DAY,
            "phi_b": float(self.parameters.phi_b),
            "alpha": float(self.parameters.alpha),
            "delta": float(self.parameters.delta),
            "S": self.statistic,
            "doppler_reference": self.doppler_reference,
        }


class TopTemplates(NamedTuple):
    """The top in-range templates of one toplist's f0 points in one match.

    Attributes:
        sky_idx: Index of the sky point in the bank.
        toplist_idx: Index of the toplist whose f0 points they are.
        max_statistic: Maximum of `S` over the orbital phase, top first;
            `-inf` once the in-range templates run out.
        f0_idxs: Index of each template's point on the bank's base f0 grid.
        period_idxs: Index of its point on the base omega^2 grid.
        modulation_depth_idxs: Index of its point on the base modulation-depth grid.
        phase_idxs: Index of the orbital phase its maximum sits at.
        doppler_references: Frequency its band was Doppler-corrected at [Hz].
        n_phi: Number of orbital phases the statistic was read at.
    """

    sky_idx: int
    toplist_idx: int
    max_statistic: Float[np.ndarray, " n_top"]
    f0_idxs: Int[np.ndarray, " n_top"]
    period_idxs: Int[np.ndarray, " n_top"]
    modulation_depth_idxs: Int[np.ndarray, " n_top"]
    phase_idxs: Int[np.ndarray, " n_top"]
    doppler_references: Float[np.ndarray, " n_top"]
    n_phi: int

    def candidate(self, bank: TemplateBank, rank: int) -> Candidate:
        """One of the templates as a candidate.

        Args:
            bank: The bank the templates were scored on.
            rank: Rank of the template, 0 for the top one.

        Returns:
            The template's statistic and parameters.
        """
        parameters = bank.parameters(
            sky_idx=self.sky_idx,
            f0_idx=int(self.f0_idxs[rank]),
            period_idx=int(self.period_idxs[rank]),
            modulation_depth_idx=int(self.modulation_depth_idxs[rank]),
            phase_idx=int(self.phase_idxs[rank]),
            n_phi=self.n_phi,
        )
        return Candidate(
            statistic=float(self.max_statistic[rank]),
            parameters=parameters,
            doppler_reference=float(self.doppler_references[rank]),
        )


class Toplist:
    """A heap queue of the top templates, at most `n_toplist` of them."""

    def __init__(self, n_toplist: int) -> None:
        """Start the toplist empty.

        Args:
            n_toplist: Number of templates kept.
        """
        self.n_toplist = n_toplist
        self.kept: list[Candidate] = []

    @property
    def lowest_kept_statistic(self) -> float:
        """The statistic to beat: the lowest kept once the list is full."""
        if len(self.kept) < self.n_toplist:
            return -np.inf
        return self.kept[0].statistic

    def add(self, candidate: Candidate) -> None:
        """Keep a candidate if the list has room, or if it beats the lowest kept."""
        if len(self.kept) < self.n_toplist:
            heapq.heappush(self.kept, candidate)
        elif candidate.statistic > self.kept[0].statistic:
            heapq.heappushpop(self.kept, candidate)

    def add_top(self, top: TopTemplates, bank: TemplateBank) -> None:
        """Keep every one of the top templates that beats the lowest kept.

        Args:
            top: The top templates, top first.
            bank: The bank they were scored on.
        """
        for rank, statistic in enumerate(top.max_statistic.tolist()):
            # top first, so once one fails to beat the lowest kept every later one
            # does
            if statistic <= self.lowest_kept_statistic:
                break
            self.add(top.candidate(bank, rank))

    def descending(self) -> list[Candidate]:
        """What it kept, top first."""
        return sorted(self.kept, reverse=True)
