"""What a run logs and saves: its backend, a summary, its progress, and its results."""

import importlib.metadata
import json
import logging
import socket
import sys
import time
from pathlib import Path
from platform import machine
from typing import Any, Optional, Union

import jax
from pydantic import BaseModel

from jpec.bank.template_bank import TemplateBank
from jpec.config import InjectionSetup, JobSetup, SearchLoopConfig, SearchSetup
from jpec.constant import GB
from jpec.engine.data import SearchData
from jpec.engine.layout import BankLayout


class Progress:
    """How far one job has got, a line no more often than every `seconds`."""

    def __init__(self, bank: TemplateBank, *, seconds: float, started: float) -> None:
        """Count the bank's sky points.

        Args:
            bank: The job's bank, whose sky points are counted.
            seconds: Minimum gap between lines; 0 or less gives none.
            started: When the job began, from `time.perf_counter`.
        """
        self.n_sky_points = int(bank.alpha.size)
        self.seconds = seconds
        self.started = started
        self.due = started + seconds

    def report(self, n_sky_points_done: int) -> Optional[str]:
        """The progress line, if one is due.

        Args:
            n_sky_points_done: Number of sky points whose every template is scored.

        Returns:
            The fraction and sky points done, the time spent and the time left; None
            when no line is due.
        """
        now = time.perf_counter()
        if self.seconds <= 0.0 or now < self.due:
            return None
        self.due = now + self.seconds
        fraction_done = n_sky_points_done / self.n_sky_points
        elapsed = now - self.started
        time_left = (
            f"{elapsed * (1.0 - fraction_done) / fraction_done / 60.0:.1f} min to go"
            if fraction_done > 0.0
            else "time left unknown"
        )
        return (
            f"  {fraction_done:6.1%}  sky {n_sky_points_done}/{self.n_sky_points}  "
            f"{elapsed / 60.0:.1f} min in, {time_left}"
        )


class RunLogger:
    """One command's run: its JAX backend, logs, job progress and saved results."""

    def __init__(
        self,
        name: str,
        output: Union[str, Path],
        *,
        platform: Optional[str],
        x64: bool,
    ) -> None:
        """Select the backend and precision, record the run metadata, open the log.

        Args:
            name: Name of the command, which the log file is named for.
            output: Directory the log and the results are written into, created if
                missing.
            platform: JAX backend, cpu or gpu; None for whatever JAX picks.
            x64: Whether JAX runs in float64.
        """
        # the backend and precision before any array exists
        if platform is not None:
            jax.config.update("jax_platforms", {"cpu": "cpu", "gpu": "cuda"}[platform])
        jax.config.update("jax_enable_x64", x64)
        host = socket.gethostname()
        self.short_host = host.split(".")[0]
        self.run_metadata = {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_kind": jax.devices()[0].device_kind,
            "x64": x64,
            "jax": jax.__version__,
            "jpec": importlib.metadata.version("jpec"),
            "host": host,
            "machine": machine(),
        }
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

        # plain lines on the terminal, time-stamped ones in a log of this run alone
        log_path = (
            self.output
            / f"{name}_{time.strftime('%Y%m%dT%H%M%S')}_{self.short_host}.log"
        )
        self.logger = logging.getLogger(f"jpec.{name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        log_file = logging.FileHandler(log_path)
        log_file.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(log_file)

        # the job in progress, set by start_progress()
        self.job_progress: Optional[Progress] = None

        self.info(
            f"backend {self.run_metadata['backend']}  "
            f"devices {self.run_metadata['devices']} "
            f"({self.run_metadata['device_kind']})  "
            f"x64 {x64}  jax {jax.__version__}  host {self.run_metadata['host']}"
        )
        if platform == "gpu" and jax.default_backend() != "gpu":
            raise SystemExit(
                "asked for gpu but JAX has no CUDA backend: uv sync --extra gpu"
            )

    def info(self, message: str) -> None:
        """Log a line.

        Args:
            message: The line.
        """
        self.logger.info(message)

    def device_memory_gb(self) -> Optional[tuple[float, float]]:
        """Giga-bytes the first device holds now and has held at most.

        Returns:
            In use and peak, or None on a backend that keeps no memory stats.
        """
        memory_stats = jax.devices()[0].memory_stats()
        if memory_stats is None:
            return None
        return (
            memory_stats["bytes_in_use"] / GB,
            memory_stats["peak_bytes_in_use"] / GB,
        )

    def save_result(
        self, name: str, payload: dict[str, Any], *, per_host: bool = False
    ) -> Path:
        """Write what the run measured to JSON, after the run's metadata.

        Args:
            name: Name of the result, which the file is named for.
            payload: What to write after the run metadata.
            per_host: Whether to name the file for the backend and host too, so results
                measured on different machines sit side by side.

        Returns:
            The path written.
        """
        stem = name
        if per_host:
            stem = f"{name}_{self.run_metadata['backend']}_{self.short_host}"
        path = self.output / f"{stem}.json"
        device_memory_gb = self.device_memory_gb()
        peak_gb = None if device_memory_gb is None else device_memory_gb[1]
        path.write_text(
            json.dumps(
                {"name": name, **self.run_metadata, "peak_gb": peak_gb, **payload},
                indent=2,
            )
        )
        self.info(f"wrote {path}")
        return path

    def save_setup(self, name: str, setup: BaseModel) -> Path:
        """Write what a later command reads to JSON, as read_json() reads it back.

        Args:
            name: Name of the file, without the extension.
            setup: The SFT metadata, or a search or injection setup.

        Returns:
            The path written.
        """
        path = self.output / f"{name}.json"
        path.write_text(setup.model_dump_json(indent=2) + "\n")
        self.info(f"wrote {path}")
        return path

    def start_progress(
        self, bank: TemplateBank, *, seconds: float, started: float
    ) -> None:
        """Start tracking a job, so progress() can log how far it has got.

        Args:
            bank: The job's bank, whose sky points are counted.
            seconds: Minimum gap between progress lines; 0 or less logs none.
            started: When the job began, from `time.perf_counter`.
        """
        self.job_progress = Progress(bank, seconds=seconds, started=started)

    def progress(self, n_sky_points_done: int) -> None:
        """Log how far the job has got, if a line is due.

        Args:
            n_sky_points_done: Number of sky points whose every template is scored.
        """
        if self.job_progress is None:
            return
        report = self.job_progress.report(n_sky_points_done)
        if report is None:
            return
        device_memory_gb = self.device_memory_gb()
        if device_memory_gb is not None:
            in_use_gb, peak_gb = device_memory_gb
            report += f"  device {in_use_gb:.2f} GB, peak {peak_gb:.2f} GB"
        self.info(report)

    def data_summary(self, search_data: SearchData) -> None:
        """Log the band as loaded, the lines a search and an injection run share.

        Args:
            search_data: The band as loaded.
        """
        first_detector_sfts = search_data.sfts[0]
        sfts_per_detector = ", ".join(
            f"{sft.detector.name} {sft.n_sfts}" for sft in search_data.sfts
        )
        freq_low = first_detector_sfts.freq_low
        freq_high = freq_low + search_data.n_band_bins / search_data.tsft
        self.info(f"Loaded band: [{freq_low:.5f}, {freq_high:.5f}] Hz")
        self.info(f"Frequency bins loaded: {search_data.n_band_bins}")
        self.info(f"SFTs per detector: {sfts_per_detector}")
        self.info(
            f"SFT length: {search_data.tsft:.0f} s, "
            f"{first_detector_sfts.window.kind} window "
            f"beta {first_detector_sfts.window.beta}"
        )
        self.info(f"Reference time: {search_data.t_ref:.0f}")
        self.info(f"Noise sqrt(Sn): {search_data.sqrt_sn:.4e} /sqrt(Hz)")

    def search_loop_summary(self, search_loop_config: SearchLoopConfig) -> None:
        """Log how the search loop batches its work.

        Args:
            search_loop_config: How the search loop batches its work.
        """
        self.info(f"Sub-bin template offsets: {search_loop_config.n_sub_bin_offsets}")
        self.info(f"Sky points per transform: {search_loop_config.sky_batch_size}")
        self.info(f"Periods per transform: {search_loop_config.period_batch_size}")
        self.info(f"f0 points per match batch: {search_loop_config.f0_batch_size}")
        self.info(f"Share of harmonics matched: {search_loop_config.harmonic_fraction}")

    def search_summary(
        self,
        setup: SearchSetup,
        job_setup: JobSetup,
        search_data: SearchData,
        layout: BankLayout,
        *,
        setup_path: str,
    ) -> None:
        """Log what a job searches, before it starts.

        Args:
            setup: The search setup.
            job_setup: The job's row of it.
            search_data: The band as loaded.
            layout: The job's bank laid out on the band.
            setup_path: Where the setup was read from.
        """
        bank = layout.bank
        self.info(f"Job {job_setup.job_idx} of {len(setup.jobs)} from {setup_path}")
        self.info(
            f"Search band: [{job_setup.f0_low:.5f}, {job_setup.f0_high:.5f}] Hz"
        )
        self.info(
            f"Period range: [{job_setup.period_low:g}, {job_setup.period_high:g}] d"
        )
        self.data_summary(search_data)
        self.info(
            f"Frequency bins of the deepest template: {layout.bin_offsets.shape[0]}, "
            f"{layout.n_pad} more for the f0 shifts"
        )
        tiling = setup.tilings[job_setup.tiling_idx]
        self.info(f"Lattice: {tiling['lattice']}, mismatch {tiling['mismatch']}")
        self.info(f"Sky points: {bank.alpha.size}")
        self.info(f"Templates: {bank.n_templates:.3e}")
        self.info(f"Doppler references: {layout.doppler_references.size}")
        self.search_loop_summary(setup.search_loop)
        self.info(
            f"Toplists: {setup.n_toplist} templates per {setup.toplist_f0_width:g} Hz"
        )

    def injection_summary(
        self,
        setup: InjectionSetup,
        search_data: SearchData,
        *,
        setup_path: str,
        seed: int,
    ) -> None:
        """Log what an injection run measures, before it starts.

        Args:
            setup: The injection setup.
            search_data: The band as loaded.
            setup_path: Where the setup was read from.
            seed: Seed of the drawn sources.
        """
        trials_config = setup.trials
        h_label = f"h{100.0 * setup.efficiency:.0f}"
        self.info(f"Injection run from {setup_path}, seed {seed}")
        self.data_summary(search_data)
        self.info(f"Sources: {trials_config.n_injections}")
        self.info(f"Mismatch: {setup.tiling['mismatch']:g}")
        self.info(
            f"Bank around each source: {trials_config.n_steps} lattice steps "
            f"each side per orbital axis, {trials_config.n_sky_neighbours} sky points"
        )
        thresholds = ", ".join(f"{threshold:.4g}" for threshold in setup.thresholds)
        self.info(f"Thresholds: S {thresholds}, one depth each")
        self.search_loop_summary(setup.search_loop)
        self.info(
            f"Amplitudes: one h0 per source, sensitivity depth "
            f"{trials_config.sensitivity_depth[0]} to "
            f"{trials_config.sensitivity_depth[1]}, uniform in log"
        )
        self.info(f"Fit: {h_label} at {setup.efficiency:.0%} recovery")
