import jax.numpy as jnp
import numpy as np
import pytest
from signals import Case, harmonic_transform, orbit_offset

from jpec.constant import ARC_PAD_BINS, DAY
from jpec.stat.matched_filter import (
    data_harmonics,
    harmonic_overlap,
    max_over_phase,
    orbital_phase_mod_2pi,
    statistic_over_phase,
)
from jpec.waveform.harmonics import (
    harmonic_count,
    theta_sample_count,
)
from jpec.waveform.plane import (
    arc_bin_offsets,
    plane_from_arc,
    template_plane,
)


def test_statistic_phi_b_peaks(case: Case, setup):
    """The claim the whole method rests on, and it pins the sign convention."""
    coefficients = harmonic_overlap(
        data_harmonics=setup["data_harmonics"],
        template_harmonics=setup["template_harmonics"],
    )
    phase_idx = int(
        jnp.argmax(statistic_over_phase(coefficients, n_phi=setup["n_theta"]))
    )
    peak = 2.0 * np.pi * phase_idx / setup["n_theta"]
    error = (float(peak) - case.parameters.phi_b + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(error) <= np.pi / setup["n_theta"]


def test_statistic_stands_above_the_rest(setup):
    values = statistic_over_phase(
        harmonic_overlap(
            data_harmonics=setup["data_harmonics"],
            template_harmonics=setup["template_harmonics"],
        ),
        n_phi=setup["n_theta"],
    )
    assert float(jnp.max(values) / jnp.sqrt(jnp.mean(values**2))) > 3.0


def test_shifting_the_peak(case: Case, setup):
    """The shift theorem, which is what makes `phi_b` free."""
    parameters = case.parameters
    shift = 1.0
    moved = parameters._replace(phi_b=parameters.phi_b + shift)
    times = jnp.arange(setup["n_theta"]) * (parameters.period / setup["n_theta"])
    plane = plane_from_arc(
        orbit_offset(moved, times) * case.tsft,
        setup["bin_offsets"],
        window=case.window,
    )
    moved_data_harmonics = harmonic_transform(plane)[1 : setup["j_max"] + 1]
    statistic = statistic_over_phase(
        harmonic_overlap(
            data_harmonics=moved_data_harmonics,
            template_harmonics=setup["template_harmonics"],
        ),
        n_phi=setup["n_theta"],
    )
    peak = 2.0 * np.pi * int(jnp.argmax(statistic)) / setup["n_theta"]
    error = (float(peak) - (parameters.phi_b + shift) + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(error) <= np.pi / setup["n_theta"]


def test_harmonic_shape_mismatch(setup):
    with pytest.raises(ValueError, match="shape mismatch"):
        harmonic_overlap(
            data_harmonics=setup["data_harmonics"][:, :-1],
            template_harmonics=setup["template_harmonics"],
        )


def test_statistic_over_phase_shape(setup):
    coefficients = harmonic_overlap(
        data_harmonics=setup["data_harmonics"],
        template_harmonics=setup["template_harmonics"],
    )
    assert statistic_over_phase(coefficients, n_phi=256).shape == (256,)


def test_max_over_phase_is_the_peak_of_the_statistic(setup):
    """max_over_phase returns the peak of statistic_over_phase and its index."""
    statistic = statistic_over_phase(
        harmonic_overlap(
            data_harmonics=setup["data_harmonics"],
            template_harmonics=setup["template_harmonics"],
        ),
        n_phi=setup["n_theta"],
    )
    value, phase_idx = max_over_phase(
        data_harmonics=setup["data_harmonics"],
        template_harmonics=setup["template_harmonics"],
        n_phi=setup["n_theta"],
    )
    assert int(phase_idx) == int(jnp.argmax(statistic))
    assert float(value) == pytest.approx(float(jnp.max(statistic)), rel=1e-12)


def test_one_period_and_a_batch_of_periods_agree(case: Case, setup):
    """An omega array gives `F[j, k]` each, so a period batch is many single periods."""
    times = jnp.arange(setup["n_theta"]) * (2.0 * jnp.pi / setup["n_theta"])
    omegas = jnp.array([1.0, 1.5, 2.0])
    plane = jnp.ones((setup["n_theta"], setup["bin_offsets"].size))
    batch = data_harmonics(
        plane, jnp.asarray(orbital_phase_mod_2pi(times, omegas)), j_max=4
    )
    for row, omega in enumerate(omegas):
        single_omega_harmonics = data_harmonics(
            plane, jnp.asarray(orbital_phase_mod_2pi(times, omega)), j_max=4
        )
        np.testing.assert_allclose(
            batch[row], single_omega_harmonics, rtol=1e-12, atol=1e-14
        )


def test_wrapped_phase_holds_the_highest_harmonic_in_float32():
    """Over all of O4, at a one-day period, float32 holds the wrapped phase only."""
    # every 50th SFT midpoint of a 909 day run from its middle: the error grows with
    # |t|, not the count
    times = np.arange(-454.5 * DAY, 454.5 * DAY, 50 * 1800.0)
    omega = 2.0 * np.pi / 83941.0
    j_max = 313
    j = np.arange(1, j_max + 1)
    exact = j[:, None] * omega * times
    # unit power at one SFT per bin, so F[j, k] is the basis at SFT k over n_sft
    unit_power = jnp.eye(times.size, dtype=jnp.float32)

    wrapped = data_harmonics(
        unit_power,
        jnp.asarray(orbital_phase_mod_2pi(times, omega), dtype=jnp.float32),
        j_max=j_max,
    )
    assert wrapped.dtype == jnp.complex64
    wrapped_error = np.abs(np.angle(np.asarray(wrapped) * np.exp(1j * exact)))
    assert wrapped_error.max() < 1e-3

    # the phase formed in float32, unwrapped
    unwrapped_phase = jnp.float32(omega) * jnp.asarray(times, dtype=jnp.float32)
    unwrapped = data_harmonics(unit_power, unwrapped_phase, j_max=j_max)
    unwrapped_error = np.abs(np.angle(np.asarray(unwrapped) * np.exp(1j * exact)))
    assert unwrapped_error.max() > 10.0 * wrapped_error.max()


def test_det_stat_in_float32(case: Case):
    parameters = case.parameters
    j_max = harmonic_count(case.modulation_depth_bins)
    n = theta_sample_count(j_max)

    def run(dtype):
        offsets = arc_bin_offsets(
            case.modulation_depth_bins, pad_bins=ARC_PAD_BINS
        ).astype(dtype)
        times = (jnp.arange(n) * (parameters.period / n)).astype(dtype)
        theta = (jnp.arange(n) * (2.0 * jnp.pi / n)).astype(dtype)
        arc = orbit_offset(parameters, times) * case.tsft
        signal_data_harmonics = harmonic_transform(
            plane_from_arc(arc, offsets, window=case.window)
        )[1 : j_max + 1]
        plane_transform = harmonic_transform(
            template_plane(
                offsets,
                theta=theta,
                modulation_depth_bins=case.modulation_depth_bins,
                window=case.window,
            )
        )
        return statistic_over_phase(
            harmonic_overlap(
                data_harmonics=signal_data_harmonics,
                template_harmonics=plane_transform[1 : j_max + 1],
            ),
            n_phi=n,
        )

    cheap, exact = run(jnp.float32), run(jnp.float64)
    assert cheap.dtype == jnp.float32
    assert float(jnp.max(jnp.abs(cheap - exact)) / jnp.max(exact)) < 1e-5
