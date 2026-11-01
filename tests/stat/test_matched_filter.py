import jax.numpy as jnp
import numpy as np
import pytest
from signals import Case, harmonic_transform, orbit_offset

from jpec.constant import ARC_PAD_BINS, PHASE_OVERSAMPLE
from jpec.stat.matched_filter import (
    data_harmonics,
    harmonic_overlap,
    max_over_phase,
    orbital_basis,
    statistic_over_phase,
)
from jpec.waveform.harmonics import (
    harmonic_count,
    phase_sample_count,
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
    """The shift theorem, which is what makes ``phi_b`` free."""
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
    """max_over_phase returns the maximum of statistic_over_phase and the index it sits at."""
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


def test_one_basis_and_a_batch_of_bases_agree(case: Case, setup):
    """An array of omegas gives one basis each, so a period batch is many single templates at once."""
    times = jnp.arange(setup["n_theta"]) * (2.0 * jnp.pi / setup["n_theta"])
    omegas = jnp.array([1.0, 1.5, 2.0])
    plane = jnp.ones((setup["n_theta"], setup["bin_offsets"].size))
    batch = data_harmonics(plane, orbital_basis(times, omega=omegas, j_max=4))
    for row, omega in enumerate(omegas):
        single_omega_harmonics = data_harmonics(
            plane, orbital_basis(times, omega=omega, j_max=4)
        )
        np.testing.assert_allclose(
            batch[row], single_omega_harmonics, rtol=1e-12, atol=1e-14
        )


def test_det_stat_in_float32(case: Case):
    parameters = case.parameters
    j_max = harmonic_count(case.modulation_depth_bins)
    n = phase_sample_count(j_max, oversample=PHASE_OVERSAMPLE)

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
