"""Pipeline inputs and outputs: search and injection YAML, SFT metadata and setups."""

import json
import math
from itertools import takewhile
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from jpec.constant import DAY
from jpec.params import Prior
from jpec.waveform.window import Window

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class ParameterSpaceConfig(BaseModel):
    """The box in parameter space, each axis a `[low, high]` pair."""

    model_config = ConfigDict(extra="forbid")

    f0: tuple[float, float]
    ap: tuple[float, float]
    period: tuple[float, float]  # days
    phi_b: tuple[float, float] = (0.0, 2.0 * math.pi)
    alpha: tuple[float, float] = (0.0, 2.0 * math.pi)
    delta: tuple[float, float] = (-0.5 * math.pi, 0.5 * math.pi)

    def prior(self) -> Prior:
        """The box as the library takes it, the period in seconds."""
        period_low, period_high = self.period
        return Prior(
            f0=self.f0,
            ap=self.ap,
            period=(period_low * DAY, period_high * DAY),
            phi_b=self.phi_b,
            alpha=self.alpha,
            delta=self.delta,
        )

    @model_validator(mode="after")
    def _low_below_high(self) -> "ParameterSpaceConfig":
        for axis in ("f0", "phi_b"):
            low, high = getattr(self, axis)
            if low >= high:
                raise ValueError(
                    f"parameter_space.{axis} is [{low}, {high}] and needs low < high"
                )
        for axis in ("ap", "period", "alpha", "delta"):
            low, high = getattr(self, axis)
            if low > high:
                raise ValueError(
                    f"parameter_space.{axis} is [{low}, {high}] and needs low <= high"
                )
        if (self.alpha[0] == self.alpha[1]) != (self.delta[0] == self.delta[1]):
            raise ValueError(
                "parameter_space.alpha and delta give a line on the sky; give both "
                "as intervals, or both as one value for a point"
            )
        return self


class TemplateBankConfig(BaseModel):
    """How the bank is laid out over the box."""

    model_config = ConfigDict(extra="forbid")

    mismatch: float = Field(default=0.2, gt=0.0)
    n_ap_blocks: int = Field(default=1, ge=1)


class SearchLoopConfig(BaseModel):
    """How the search loop batches its work, and how finely it places an f0 point."""

    model_config = ConfigDict(extra="forbid")

    # sky points per transform, which share its basis
    sky_batch_size: int = Field(default=8, ge=1)
    # trial periods per transform
    period_batch_size: int = Field(default=16, ge=1)
    # f0 points matched at once
    f0_batch_size: int = Field(default=32, ge=1)
    # template offsets across one bin
    n_sub_bin_offsets: int = Field(default=8, ge=1)
    # share of each modulation depth's harmonics matched
    harmonic_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    # modulation depths whose templates are built at once
    modulation_depth_batch_size: int = Field(default=16, ge=1)
    # running-median window of the noise floor [bins], odd
    floor_width: int = Field(default=101, ge=1)
    # SFTs whose floor windows are sorted at once
    floor_batch_size: int = Field(default=512, ge=1)

    @model_validator(mode="after")
    def _floor_width_odd(self) -> "SearchLoopConfig":
        if self.floor_width % 2 == 0:
            raise ValueError(
                f"search_loop.floor_width is {self.floor_width} and needs to be odd"
            )
        return self


class ComputeConfig(BaseModel):
    """What one job may hold and take, which cuts the box into jobs.

    The rates come from a jpec time-search run.
    """

    model_config = ConfigDict(extra="forbid")

    memory_gb: float = Field(default=8.0, gt=0.0)
    hours: float = Field(default=24.0, gt=0.0)


class SearchJobConfig(BaseModel):
    """A search: what to search, how to cut it up, what to keep, and the SFT glob."""

    model_config = ConfigDict(extra="forbid")

    parameter_space: ParameterSpaceConfig
    template_bank: TemplateBankConfig = Field(default_factory=TemplateBankConfig)
    compute: ComputeConfig
    search_loop: SearchLoopConfig = Field(default_factory=SearchLoopConfig)
    n_toplist: int = Field(default=1000, ge=1)
    # absolute glob of the SFTs, as the search will find them
    sft_glob: str


class InjectionTrialConfig(BaseModel):
    """Trials of an injection run: how many sources, searched how far around each."""

    model_config = ConfigDict(extra="forbid")

    # sources drawn once and reused at every amplitude
    n_injections: int = Field(default=200, ge=1)
    # lattice steps each side of the source, per orbital axis
    n_steps: int = Field(default=2, ge=1)
    # sky lattice points searched around it, nearest in mismatch
    n_sky_neighbours: int = Field(default=7, ge=1)
    # cosine of the inclination angle the sources are drawn over
    cos_iota: tuple[float, float] = (-1.0, 1.0)
    # polarization angle the sources are drawn over [rad]
    psi: tuple[float, float] = (0.0, math.pi)

    @model_validator(mode="after")
    def _orientation_in_range(self) -> "InjectionTrialConfig":
        low, high = self.cos_iota
        if not -1.0 <= low < high <= 1.0:
            raise ValueError(
                f"trials.cos_iota is [{low}, {high}] and needs -1 <= low < high <= 1"
            )
        low, high = self.psi
        if low >= high:
            raise ValueError(f"trials.psi is [{low}, {high}] and needs low < high")
        return self


class RecoveryCurveConfig(BaseModel):
    """The sigmoid of the recovered fraction against amplitude.

    Where it is sampled, and when the amplitude fitted at the efficiency is good enough.
    """

    model_config = ConfigDict(extra="forbid")

    # sqrt(Sn) / h0 range the first scan spans
    sensitivity_depth: tuple[float, float] = (15.0, 45.0)
    # h0 points of the first scan, evenly spaced in log over that range
    n_amplitudes: int = Field(default=4, ge=2)
    # the recovered fraction the quoted amplitude h0_ul is read at
    efficiency: float = Field(default=0.95, gt=0.0, lt=1.0)
    # standard error of the fitted amplitude, relative to it, that ends the refinement
    tolerance: float = Field(default=0.05, gt=0.0)
    # h0 points each threshold may add, one at a time, before its fit meets the
    # tolerance
    n_refinements: int = Field(default=4, ge=0)


class InjectionConfig(BaseModel):
    """An injection run: box, banks, trials, recovery curve, and what counts as found.

    The box is the one the sources are drawn from and searched over.
    """

    model_config = ConfigDict(extra="forbid")

    parameter_space: ParameterSpaceConfig
    template_bank: TemplateBankConfig = Field(default_factory=TemplateBankConfig)
    trials: InjectionTrialConfig = Field(default_factory=InjectionTrialConfig)
    recovery_curve: RecoveryCurveConfig = Field(default_factory=RecoveryCurveConfig)
    search_loop: SearchLoopConfig = Field(default_factory=SearchLoopConfig)
    thresholds: list[float] = Field(min_length=1)  # thresholds on the statistic S
    sft_glob: str  # glob of the SFT files, as the run will find them


class SFTBand(BaseModel):
    """One frequency band of the SFT set and the files holding it.

    Attributes:
        freq_low: Low edge of the band [Hz].
        freq_high: High edge [Hz].
        files: The files, one or more per detector.
    """

    model_config = ConfigDict(extra="forbid")

    freq_low: float
    freq_high: float
    files: list[str]


class SFTMetadata(BaseModel):
    """What jpec check-sft reads off the SFT files and the setups need of them.

    Attributes:
        sft_glob: Glob the files were found by, which the search loads amplitudes from.
        tsft: SFT length [s].
        window: The window they were made with.
        bands: Each narrow band found and the files holding it, in frequency order.
        detectors: The detectors, in the order the timestamps are concatenated.
        timestamps: SFT start times per detector, absolute GPS [s].
    """

    model_config = ConfigDict(extra="forbid")

    sft_glob: str
    tsft: float = Field(gt=0.0)
    window: Window
    bands: list[SFTBand]
    detectors: list[str]
    timestamps: dict[str, list[float]]

    @model_validator(mode="after")
    def _every_detector_has_timestamps(self) -> "SFTMetadata":
        missing = [
            detector_name
            for detector_name in self.detectors
            if not self.timestamps.get(detector_name)
        ]
        if missing:
            raise ValueError(f"no timestamps recorded for {missing}")
        return self

    @property
    def n_sfts(self) -> int:
        """SFTs over every detector."""
        return sum(
            len(self.timestamps[detector_name]) for detector_name in self.detectors
        )

    def files_covering(
        self, freq_low: float, freq_high: float, *, sft_glob: str
    ) -> list[str]:
        """The files of the lowest narrow band covering freq_low to freq_high.

        They are given as sft_glob finds them.

        Args:
            freq_low: Low edge of the band to read [Hz].
            freq_high: High edge [Hz].
            sft_glob: Glob the files are read through, which keeps each file's path
                below the checked glob's wildcards.

        Returns:
            The files, at least one per detector.
        """
        # the directory above each glob's first wildcard
        checked_root, read_root = (
            Path(
                *takewhile(lambda part: not set("*?[") & set(part), Path(pattern).parts)
            )
            for pattern in (self.sft_glob, sft_glob)
        )
        for band in sorted(self.bands, key=lambda band: band.freq_low):
            if band.freq_low <= freq_low and freq_high <= band.freq_high:
                return [
                    str(read_root / Path(file).relative_to(checked_root))
                    for file in band.files
                ]
        raise SystemExit(
            f"{freq_low:.4f} to {freq_high:.4f} Hz is not inside one checked band of "
            f"{self.sft_glob}"
        )


class DataSetup(BaseModel):
    """The SFTs a run reads: glob, detectors, count, length, window, and the epoch.

    Attributes:
        sft_glob: Glob of the SFT files, as the run will find them.
        detectors: The detectors, in the order the timestamps are concatenated.
        n_sfts: SFTs over every detector.
        tsft: SFT length [s].
        window: The window they were made with.
        t_ref: Epoch `phi_b` is quoted at, absolute GPS [s].
    """

    model_config = ConfigDict(extra="forbid")

    sft_glob: str
    detectors: list[str]
    n_sfts: int
    tsft: float
    window: Window
    t_ref: float


class JobSetup(BaseModel):
    """One search job: f0 interval, band and files loaded, and what it holds and takes.

    Attributes:
        job_idx: Index of the job in the setup.
        f0_low: Low edge of the f0 interval [Hz].
        f0_high: High edge [Hz].
        freq_low: Low edge of the band loaded [Hz].
        freq_high: High edge [Hz].
        files: The files holding the band, at least one per detector.
        n_sky_points: Sky points of the job's bank.
        n_templates: Templates of the job's bank.
        memory_gb: Peak giga-bytes the job holds.
        hours: Hours the job takes at the timing's rates; None when the setup was made
            without one.
    """

    model_config = ConfigDict(extra="forbid")

    job_idx: int
    f0_low: float
    f0_high: float
    freq_low: float
    freq_high: float
    files: list[str]
    n_sky_points: int
    n_templates: int
    memory_gb: float
    hours: float | None


class SearchSetup(BaseModel):
    """A search cut into jobs, as jpec search-setup writes it.

    Attributes:
        jpec: The jpec version the setup was made with.
        data: The SFTs searched.
        tiling: LatticeTiling.to_dict() of the lattice.
        search_loop: How the search loop batches its work.
        n_toplist: Templates each job keeps.
        n_templates_total: Templates over every job.
        jobs: The jobs, in f0 order.
    """

    model_config = ConfigDict(extra="forbid")

    jpec: str
    data: DataSetup
    tiling: dict[str, Any]
    search_loop: SearchLoopConfig
    n_toplist: int
    n_templates_total: int
    jobs: list[JobSetup]


class InjectionSetup(BaseModel):
    """An injection run, as jpec injection-setup writes it.

    Attributes:
        jpec: The jpec version the setup was made with.
        data: The SFTs injected into.
        freq_low: Low edge of the band loaded [Hz].
        freq_high: High edge [Hz].
        files: The files holding the band, at least one per detector.
        parameter_space: The box the sources are drawn from and searched over.
        tiling: LatticeTiling.to_dict() of the lattice.
        trials: How many sources, searched how far around each.
        recovery_curve: Where the recovered fraction is sampled, and when its fit is
            good enough.
        search_loop: How the search loop batches its work.
        thresholds: Thresholds on the statistic `S`, one sensitivity depth each.
    """

    model_config = ConfigDict(extra="forbid")

    jpec: str
    data: DataSetup
    freq_low: float
    freq_high: float
    files: list[str]
    parameter_space: ParameterSpaceConfig
    tiling: dict[str, Any]
    trials: InjectionTrialConfig
    recovery_curve: RecoveryCurveConfig
    search_loop: SearchLoopConfig
    thresholds: list[float]


def read_config(path: str, model: type[ConfigT]) -> ConfigT:
    """The run a YAML file asks for, validated.

    Args:
        path: The YAML file.
        model: What it describes, `SearchJobConfig` or `InjectionConfig`.

    Returns:
        The config, defaults filled in.
    """
    try:
        return model.model_validate(yaml.safe_load(Path(path).read_text()))
    except ValidationError as error:
        raise SystemExit(f"{path}:\n{error}") from error


def read_json(path: str, model: type[ConfigT]) -> ConfigT:
    """A JSON file a pipeline wrote, validated.

    Args:
        path: The JSON file.
        model: What it holds: `SFTMetadata`, `SearchSetup` or `InjectionSetup`.

    Returns:
        The model.
    """
    try:
        return model.model_validate(json.loads(Path(path).read_text()))
    except ValidationError as error:
        raise SystemExit(f"{path}:\n{error}") from error
