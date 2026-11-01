from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from signals import Case

from jpec.constant import ARC_PAD_BINS
from jpec.waveform.harmonics import harmonic_count
from jpec.waveform.plane import (
    arc_bin_offsets,
    plane_from_arc,
    template_plane,
)
from jpec.waveform.window import Window


@pytest.fixture
def offsets(case: Case):
    return arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)


def test_bin_grid_spans_the_arc_plus_its_pad(case: Case):
    grid = arc_bin_offsets(case.modulation_depth_bins, pad_bins=8.0)
    half = np.ceil(case.modulation_depth_bins) + 8.0
    assert float(grid[0]) == -half
    assert float(grid[-1]) == half
    assert jnp.all(jnp.diff(grid) == 1.0)


def test_bin_grid_is_symmetric_about_f0(case: Case):
    grid = arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    assert jnp.array_equal(grid, -grid[::-1])


def test_plane_is_real_non_negative_and_correctly_shaped(case: Case, theta, offsets):
    plane = template_plane(
        offsets,
        theta=theta,
        modulation_depth_bins=case.modulation_depth_bins,
        window=case.window,
    )
    assert plane.shape == (theta.size, offsets.size)
    assert jnp.all(plane >= 0.0)
    assert jnp.all(jnp.isfinite(plane))


def test_brightest_bin_follows_the_arc(case: Case, theta, offsets):
    plane = template_plane(
        offsets,
        theta=theta,
        modulation_depth_bins=case.modulation_depth_bins,
        window=case.window,
    )
    arc = -case.modulation_depth_bins * jnp.cos(theta)
    brightest = offsets[jnp.argmax(plane, axis=1)]
    assert float(jnp.max(jnp.abs(brightest - arc))) <= 0.5 + 1e-9


def test_arc_dtype_decides_the_plane_dtype(case: Case, theta):
    """One dtype boundary, so a float64 bin grid cannot double a float32 plane."""
    offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    arc = -case.modulation_depth_bins * jnp.cos(theta)

    cheap = plane_from_arc(
        arc.astype(jnp.float32), offsets.astype(jnp.float64), window=case.window
    )
    assert cheap.dtype == jnp.float32

    exact = plane_from_arc(
        arc.astype(jnp.float64), offsets.astype(jnp.float32), window=case.window
    )
    assert exact.dtype == jnp.float64

    assert float(jnp.max(jnp.abs(cheap.astype(jnp.float64) - exact))) < 1e-5


@pytest.mark.parametrize("beta", [0.25, 0.5, 1.0])
def test_taper_reaches_the_kernel_from_every_level(case: Case, theta, beta):
    """A dropped keyword would silently fall back to 0.5, not raise."""
    offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    window = Window(kind="tukey", beta=beta)
    direct = window.power(offsets)
    plane = template_plane(
        offsets,
        theta=jnp.zeros(1),
        modulation_depth_bins=case.modulation_depth_bins,
        window=window,
    )
    arc = -case.modulation_depth_bins * jnp.cos(jnp.zeros(1))
    np.testing.assert_array_equal(plane[0], window.power(offsets - arc[0]))
    assert not jnp.allclose(direct, Window(kind="tukey", beta=0.99).power(offsets)), (
        "a different taper must give a different kernel"
    )


def test_arc_turns_at_the_full_depth(case: Case, offsets):
    """``theta = 0`` and ``pi`` are the turning points of the frequency sweep."""
    plane = template_plane(
        offsets,
        theta=jnp.array([0.0, np.pi]),
        modulation_depth_bins=case.modulation_depth_bins,
        window=case.window,
    )
    brightest = offsets[jnp.argmax(plane, axis=1)]
    assert float(brightest[0]) == pytest.approx(-np.round(case.modulation_depth_bins))
    assert float(brightest[1]) == pytest.approx(np.round(case.modulation_depth_bins))


def test_template_plane_is_plane_from_arc_at_the_orbital_arc(
    case: Case, theta, offsets
):
    direct = plane_from_arc(
        -case.modulation_depth_bins * jnp.cos(theta), offsets, window=case.window
    )
    np.testing.assert_array_equal(
        template_plane(
            offsets,
            theta=theta,
            modulation_depth_bins=case.modulation_depth_bins,
            window=case.window,
        ),
        direct,
    )


def test_template_depends_on_depth_only(case: Case, theta, offsets):
    """Why the harmonics taken from this are free of period and phase.

    Three sources with different f0, radius and period, sharing one
    modulation_depth_bins, give one template.
    """
    parameters = case.parameters
    twins = [
        parameters,
        parameters._replace(f0=2.0 * parameters.f0, ap=parameters.ap / 2.0),
        parameters._replace(ap=2.0 * parameters.ap, period=2.0 * parameters.period),
    ]
    modulation_depths = [float(t.modulation_depth_bins(case.tsft)) for t in twins]
    assert modulation_depths == pytest.approx([case.modulation_depth_bins] * 3)
    reference = template_plane(
        offsets,
        theta=theta,
        modulation_depth_bins=modulation_depths[0],
        window=case.window,
    )
    for modulation_depth_bins in modulation_depths[1:]:
        np.testing.assert_array_equal(
            template_plane(
                offsets,
                theta=theta,
                modulation_depth_bins=modulation_depth_bins,
                window=case.window,
            ),
            reference,
        )


def test_plane_reproduces_the_frequency(case: Case, theta, offsets):
    """Cross-check against ``f(t) = f0 - df*cos(theta)`` written out in full.

    The badly-conditioned route, kept only to pin the sign convention. It agrees
    to its own float64 cancellation floor, not to machine epsilon.
    """
    parameters = case.parameters
    freqs = parameters.f0 + np.asarray(offsets) / case.tsft
    arc = parameters.f0 - parameters.modulation_depth * np.cos(np.asarray(theta))
    legacy = np.sinc((freqs[None, :] - arc[:, None]) * case.tsft) ** 2
    ours = np.asarray(
        template_plane(
            offsets,
            theta=theta,
            modulation_depth_bins=case.modulation_depth_bins,
            window=Window(kind="rectangular"),
        )
    )
    assert np.max(np.abs(ours - legacy)) < 1e-9


def test_plane_vmaps(theta, offsets):
    modulation_depths = jnp.linspace(2.0, 20.0, 6)

    window = Window()
    banked = jax.jit(
        jax.vmap(partial(template_plane, offsets, theta=theta, window=window))
    )
    planes = banked(modulation_depth_bins=modulation_depths)
    assert planes.shape == (6, theta.size, offsets.size)

    np.testing.assert_allclose(
        planes[4],
        template_plane(
            offsets,
            theta=theta,
            modulation_depth_bins=modulation_depths[4],
            window=window,
        ),
        rtol=1e-8,
    )


def test_harmonics_cutoff_gets_most_of_the_power(case: Case, theta, offsets):
    """Ties the plane back to :func:`harmonic_count`."""
    plane_transform = jnp.fft.fft(
        template_plane(
            offsets,
            theta=theta,
            modulation_depth_bins=case.modulation_depth_bins,
            window=case.window,
        ),
        axis=0,
    )
    power = jnp.sum(jnp.abs(plane_transform) ** 2, axis=1).at[0].set(0.0)
    cutoff = harmonic_count(case.modulation_depth_bins)
    half = power[: theta.size // 2]
    assert cutoff < theta.size // 2, "orbit undersampled for this modulation_depth_bins"
    assert float(jnp.sum(half[:cutoff]) / jnp.sum(half)) > 0.99


def test_naive_cutoff_would_lose_most_of_the_power(case: Case, theta, offsets):
    plane_transform = jnp.fft.fft(
        template_plane(
            offsets,
            theta=theta,
            modulation_depth_bins=case.modulation_depth_bins,
            window=case.window,
        ),
        axis=0,
    )
    power = jnp.sum(jnp.abs(plane_transform) ** 2, axis=1).at[0].set(0.0)
    half = power[: theta.size // 2]
    naive = max(int(np.floor(case.modulation_depth_bins)), 1)
    assert float(jnp.sum(half[:naive]) / jnp.sum(half)) < 0.7


def test_truncating_the_grid_costs_little(case: Case, theta):
    """The pad is insurance, not a requirement, and this quantifies both ends."""
    wide_offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=128.0)
    wide = jnp.fft.fft(
        template_plane(
            wide_offsets,
            theta=theta,
            modulation_depth_bins=case.modulation_depth_bins,
            window=case.window,
        ),
        axis=0,
    )
    mismatches = []
    for pad in (0.0, 8.0):
        narrow_offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=pad)
        narrow = jnp.fft.fft(
            template_plane(
                narrow_offsets,
                theta=theta,
                modulation_depth_bins=case.modulation_depth_bins,
                window=case.window,
            ),
            axis=0,
        )
        low_idx = int(jnp.searchsorted(wide_offsets, narrow_offsets[0]))
        padded = (
            jnp.zeros_like(wide)
            .at[:, low_idx : low_idx + narrow_offsets.size]
            .set(narrow)
        )
        overlap = jnp.abs(jnp.sum(padded.conj() * wide))
        norm = jnp.sqrt(jnp.sum(jnp.abs(padded) ** 2) * jnp.sum(jnp.abs(wide) ** 2))
        mismatches.append(1.0 - float(overlap / norm))
    assert mismatches[0] < 1e-3
    assert mismatches[1] < mismatches[0] / 10.0
