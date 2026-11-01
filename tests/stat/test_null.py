import jax.numpy as jnp
import numpy as np
import pytest
from signals import CASES

from jpec.constant import ARC_PAD_BINS, PHASE_OVERSAMPLE
from jpec.stat.matched_filter import (
    data_harmonics,
    harmonic_overlap,
    orbital_basis,
    statistic_over_phase,
)
from jpec.stat.null import (
    detection_threshold,
    false_alarm_probability,
    null_sigma,
    rms_harmonic_number,
)
from jpec.waveform.harmonics import (
    harmonic_count,
    phase_sample_count,
    template_harmonics,
)
from jpec.waveform.plane import arc_bin_offsets

N_SFTS = 512
TRIALS = 400


# not yet reviewed
@pytest.fixture(scope="module")
def background():
    """One template, and the peak of a noise-only scan many times over."""
    case = CASES[0]
    parameters = case.parameters
    j_max = harmonic_count(case.modulation_depth_bins)
    n_phi = phase_sample_count(j_max, oversample=PHASE_OVERSAMPLE)
    offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    case_template_harmonics = template_harmonics(
        offsets,
        modulation_depth_bins=case.modulation_depth_bins,
        n_theta=n_phi,
        j_max=j_max,
        window=case.window,
    )
    times = jnp.arange(N_SFTS) * (parameters.period / n_phi)
    sft_orbital_basis = orbital_basis(times, omega=parameters.omega, j_max=j_max)

    rng = np.random.default_rng(3)
    peaks, values = np.empty(TRIALS), []
    for i in range(TRIALS):
        # floor-normalized power under noise: exponential with unit mean and variance
        sft_power = jnp.asarray(rng.exponential(1.0, size=(N_SFTS, offsets.size)))
        noise_data_harmonics = data_harmonics(sft_power, sft_orbital_basis)
        statistic = statistic_over_phase(
            harmonic_overlap(
                data_harmonics=noise_data_harmonics,
                template_harmonics=case_template_harmonics,
            ),
            n_phi=n_phi,
        )
        peaks[i] = float(jnp.max(statistic))
        values.append(np.asarray(statistic))
    return {
        "j_rms": rms_harmonic_number(case_template_harmonics),
        "peaks": peaks,
        "values": np.concatenate(values),
    }


def test_null_sigma_matches_a_noise_realization(background):
    """The claim the threshold rests on: Var(S) = Var(P) / (2*n_sfts)."""
    assert background["values"].std() == pytest.approx(
        null_sigma(N_SFTS, power_variance=1.0), rel=0.05
    )


def test_threshold_round_trips_through_fap(background):
    for fap in (0.5, 0.1, 0.01, 1e-4):
        threshold = detection_threshold(
            fap, n_sfts=N_SFTS, j_rms=background["j_rms"], power_variance=1.0
        )
        assert false_alarm_probability(
            threshold, n_sfts=N_SFTS, j_rms=background["j_rms"], power_variance=1.0
        ) == pytest.approx(fap)


def test_threshold_rejects_a_fap_it_cannot_deliver():
    with pytest.raises(ValueError, match="crossings can deliver"):
        detection_threshold(1.0 - 1e-12, n_sfts=N_SFTS, j_rms=1e-3, power_variance=1.0)
