import jax
import jax.numpy as jnp
import numpy as np
import pytest
from signals import DAY, Case, search_data_harmonics, search_template_harmonics

from jpec.bank.metric import freq_bin_curvature, physical_metric, project_out
from jpec.constant import (
    N_CURVATURE_POSITIONS,
    SHIFT_MARGIN_BINS,
    V_MAX,
)
from jpec.detector.site import Detector, Observation
from jpec.injection.signal import (
    Injection,
    Orientation,
    sft_with_simulated_signal,
    simulated_signal_sft_data,
)
from jpec.noise.gaussian import gaussian_noise_sft
from jpec.params import Parameters
from jpec.stat.matched_filter import max_over_phase
from jpec.waveform.harmonics import harmonic_count

T0 = 1238166018.0
ALPHA, DELTA = 1.1, 0.3
MISMATCH = 0.05
N_ORBIT = 4
N_PHI = 4096  # fine enough that the phase grid adds nothing to the measured mismatch
PHI_B_IDX = Parameters._fields.index("phi_b")
WITHOUT_PHI_B_IDXS = [i for i in range(6) if i != PHI_B_IDX]


class ObservedSignal:
    """A noise-free signal over one span, and the statistic a template scores on it."""

    def __init__(self, case: Case, span: float):
        parameters = case.parameters
        self.case = case
        self.tsft = case.tsft
        starts = T0 + np.arange(0.0, span, case.tsft)
        self.t_ref = float(np.mean(starts + case.tsft / 2.0))
        self.parameters = parameters._replace(alpha=ALPHA, delta=DELTA)
        self.scale = np.array(
            [parameters.f0, parameters.ap, parameters.period, 1.0, 1.0, 1.0]
        )

        # a band wide enough for the Doppler shift, the arc and the correction's margin
        modulation_depth_bins = float(self.parameters.modulation_depth_bins(case.tsft))
        self.half_width_bins = int(np.ceil(1.1 * modulation_depth_bins)) + 8
        margin = (
            int(np.ceil(float(parameters.f0) * V_MAX * case.tsft))
            + self.half_width_bins
            + SHIFT_MARGIN_BINS
        )
        self.j_max = harmonic_count(1.1 * modulation_depth_bins)
        empty = gaussian_noise_sft(
            starts,
            Detector("H1"),
            key=jax.random.key(0),
            freq_low=float(parameters.f0) - margin / case.tsft,
            n_band_bins=2 * margin + 1,
            tsft=case.tsft,
            sqrt_sn=0.0,
            window=case.window,
        )
        face_on = Injection(self.parameters, Orientation(cos_iota=1.0, psi=0.0))
        unit_sft_data = simulated_signal_sft_data(empty, face_on, t_ref=self.t_ref)
        self.sft = sft_with_simulated_signal(empty, np.asarray(unit_sft_data), h0=1.0)
        self.times = jnp.asarray(self.sft.timestamps_midpoints - self.t_ref)
        self.floor = jnp.ones(self.sft.sft_data.shape)  # ones for a noise-free signal
        self.peak = self.statistic(self.parameters)

    def statistic(self, template: Parameters) -> float:
        """The template's statistic on this signal, maximized over the orbital phase."""
        data_harmonics, offsets = search_data_harmonics(
            self.sft,
            template,
            floor=self.floor,
            velocity=self.sft.velocity,
            times=self.times,
            half_width_bins=self.half_width_bins,
            j_max=self.j_max,
        )
        template_harmonics = search_template_harmonics(
            float(template.modulation_depth_bins(self.tsft)),
            self.sft.window,
            bin_offsets=jnp.asarray(offsets),
        )
        value, _ = max_over_phase(
            data_harmonics=data_harmonics[: template_harmonics.shape[0]],
            template_harmonics=template_harmonics,
            n_phi=N_PHI,
        )
        return float(value)

    def metric(self, true: Parameters, t_ref: float) -> np.ndarray:
        """The closed-form 6x6 metric at `true`, epoch at `t_ref`, absolute units."""
        observation = Observation(
            {"H1": self.sft.timestamps}, tsft=self.tsft, t_ref=t_ref
        )
        return np.asarray(
            physical_metric(
                true,
                observation,
                g_freq_bin=freq_bin_curvature(
                    self.case.window, n_positions=N_CURVATURE_POSITIONS
                ),
            )
        )

    def relative_without_phi_b(self, g: np.ndarray) -> np.ndarray:
        """The 5x5 metric per (d lambda / lambda)^2, angles in rad, phi_b maximized."""
        scale = self.scale[WITHOUT_PHI_B_IDXS]
        return np.asarray(project_out(jnp.asarray(g))) * np.outer(scale, scale)


_cache: dict[tuple[str, float], ObservedSignal] = {}


def observed_signal(case: Case, span: float) -> ObservedSignal:
    """One instance per signal and span; building one generates its SFTs."""
    key = (case.name, span)
    if key not in _cache:
        _cache[key] = ObservedSignal(case, span)
    return _cache[key]


@pytest.fixture
def closed_orbits(case: Case) -> ObservedSignal:
    """Exactly N_ORBIT orbits on the SFT grid, so the data grid closes."""
    return observed_signal(case, N_ORBIT * float(case.parameters.period))


def test_projection_is_epoch_invariant(case: Case):
    """Maximizing over phi_b cannot depend on where phi_b is quoted."""
    observed = observed_signal(case, 60.0 * DAY)
    omega = float(case.parameters.omega)
    shift = float(observed.times[0])
    at_start = observed.parameters._replace(
        phi_b=observed.parameters.phi_b - omega * shift
    )
    epoch_at_start = observed.relative_without_phi_b(
        observed.metric(at_start, observed.t_ref + shift)
    )
    epoch_at_middle = observed.relative_without_phi_b(
        observed.metric(observed.parameters, observed.t_ref)
    )
    assert (
        np.abs(epoch_at_start - epoch_at_middle).max()
        < 1e-10 * np.abs(epoch_at_middle).max()
    )


@pytest.mark.parametrize("axis", ["f0", "ap", "period", "alpha", "delta"])
def test_quadratic_form_predicts_the_mismatch(closed_orbits: ObservedSignal, axis: str):
    """Displaced along each axis by sqrt(mu / g'_aa), the mismatch the search path
    measures is mu.

    The closed form is the track's error in bins, not the statistic's Hessian;
    measured at 0.75 to 1.21 of mu over the cases and axes on four closed orbits,
    delta the lowest and the shallowest case's period the highest.
    """
    axis_idx = Parameters._fields.index(axis)
    relative_metric = closed_orbits.relative_without_phi_b(
        closed_orbits.metric(closed_orbits.parameters, closed_orbits.t_ref)
    )
    direction = np.zeros(5)
    direction[WITHOUT_PHI_B_IDXS.index(axis_idx)] = 1.0
    step = np.sqrt(MISMATCH / (direction @ relative_metric @ direction))

    displaced = (
        float(closed_orbits.parameters[axis_idx]) + step * closed_orbits.scale[axis_idx]
    )
    template = closed_orbits.parameters._replace(**{axis: displaced})
    measured_mismatch = 1.0 - closed_orbits.statistic(template) / closed_orbits.peak
    assert 0.7 * MISMATCH <= measured_mismatch <= 1.3 * MISMATCH
