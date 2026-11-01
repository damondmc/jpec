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
    """

    statistic: float
    parameters: Parameters

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
        }


class LoudestTemplates(NamedTuple):
    """The loudest in-box templates of one f0 x period x modulation depth block.

    Attributes:
        subgrid_idx: Index of the subgrid in the bank.
        sky_idx: Index of the sky point in the bank.
        max_statistic: Maximum of `S` over the orbital phase, loudest first;
            `-inf` once the grid's in-box templates run out.
        f0_idxs: Index of each template's f0 point in the subgrid.
        period_idxs: Index of its trial period.
        modulation_depth_idxs: Index of its modulation depth.
        phase_idxs: Index of the orbital phase its maximum sits at.
    """

    subgrid_idx: int
    sky_idx: int
    max_statistic: Float[np.ndarray, " n_loudest"]
    f0_idxs: Int[np.ndarray, " n_loudest"]
    period_idxs: Int[np.ndarray, " n_loudest"]
    modulation_depth_idxs: Int[np.ndarray, " n_loudest"]
    phase_idxs: Int[np.ndarray, " n_loudest"]

    def candidate(self, bank: TemplateBank, loudest_idx: int) -> Candidate:
        """One of the templates as a candidate.

        Args:
            bank: The bank the templates were scored on.
            loudest_idx: Rank of the template, 0 for the loudest.

        Returns:
            The template's statistic and parameters.
        """
        return Candidate(
            statistic=float(self.max_statistic[loudest_idx]),
            parameters=bank.parameters(
                subgrid_idx=self.subgrid_idx,
                sky_idx=self.sky_idx,
                f0_idx=int(self.f0_idxs[loudest_idx]),
                period_idx=int(self.period_idxs[loudest_idx]),
                modulation_depth_idx=int(self.modulation_depth_idxs[loudest_idx]),
                phase_idx=int(self.phase_idxs[loudest_idx]),
            ),
        )


class Toplist:
    """A heap queue of the loudest templates, at most `n_toplist` of them."""

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

    def add_loudest(self, loudest: LoudestTemplates, bank: TemplateBank) -> None:
        """Keep every one of a grid's loudest templates that beats the lowest kept.

        Args:
            loudest: The grid's loudest templates, loudest first.
            bank: The bank they were scored on.
        """
        for loudest_idx, statistic in enumerate(loudest.max_statistic.tolist()):
            # loudest first, so once one fails to beat the lowest kept every later
            # one does
            if statistic <= self.lowest_kept_statistic:
                break
            self.add(loudest.candidate(bank, loudest_idx))

    def descending(self) -> list[Candidate]:
        """What it kept, loudest first."""
        return sorted(self.kept, reverse=True)
