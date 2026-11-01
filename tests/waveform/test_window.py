import jax.numpy as jnp
import numpy as np
import pytest

from jpec.waveform.window import (
    Window,
    rectangular_spectrum,
    rectangular_window,
    tukey_spectrum,
    tukey_window,
)


def test_rectangular_window_is_the_zero_taper_limit():
    np.testing.assert_array_equal(rectangular_window(512), 1.0)
    np.testing.assert_array_equal(tukey_window(512, 1e-9), 1.0)


def test_rectangular_spectrum_is_sinc():
    """W(0) = 1, and exact nulls on every other bin."""
    x = jnp.array([0.0, 1.0, 2.0, 5.0])
    assert float(rectangular_spectrum(x)[0]) == pytest.approx(1.0)
    np.testing.assert_allclose(rectangular_spectrum(x)[1:], 0.0, atol=1e-5)


def test_hann_window_is_the_textbook_hann():
    """The full-taper Tukey window is sin^2 over the whole SFT."""
    for n in (1024, 4096):
        y = (2.0 * np.arange(n) - (n - 1)) / (n - 1)
        textbook = np.sin(0.5 * np.pi * (y + 1.0)) ** 2
        np.testing.assert_allclose(Window(kind="hann").samples(n), textbook, atol=1e-5)


@pytest.mark.parametrize("beta", [0.001, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("length", [1024, 4096])
def test_window_is_lals_window(length, beta):
    """Elementwise against the implementation that made the SFTs we read."""
    import lal  # pyright: ignore[reportMissingTypeStubs]

    ours = np.asarray(tukey_window(length, beta))
    window = lal.CreateTukeyREAL8Window(length, beta)  # pyright: ignore[reportAttributeAccessIssue]
    theirs = np.asarray(window.data.data)
    assert np.max(np.abs(ours - theirs)) < 1e-5


def test_window_taper_is_a_whole_number_of_samples():
    """``round(beta * length)`` split across both ends, so the flat part is 1 - beta."""
    for beta in (0.25, 0.5):
        w = np.asarray(tukey_window(100001, beta))
        assert w[w.size // 2] == pytest.approx(1.0)
        assert w[0] == pytest.approx(0.0, abs=1e-5)
        assert float(np.mean(w == 1.0)) == pytest.approx(1.0 - beta, abs=1e-5)


@pytest.mark.parametrize("beta", [0.001, 0.25, 0.5, 1.0])
def test_spectrum_is_the_transform_of_the_window(beta):
    """The closed form is the continuous limit of the sampled window."""
    n = 20000
    w = tukey_window(n, beta)
    t = (2.0 * jnp.arange(n) - (n - 1)) / (2.0 * (n - 1))
    offsets = jnp.array([0.0, 0.5, 1.0, 2.0, 3.7, -2.5])
    numeric = jnp.array(
        [jnp.real(jnp.sum(w * jnp.exp(-2j * jnp.pi * x * t))) / n for x in offsets]
    )
    np.testing.assert_allclose(numeric, tukey_spectrum(offsets, beta=beta), atol=1e-3)


def test_rectangular_spectrum_is_the_transform_of_its_window():
    """The pair must agree, or one of the two is wrong."""
    n = 20000
    w = rectangular_window(n)
    t = (2.0 * jnp.arange(n) - (n - 1)) / (2.0 * (n - 1))
    offsets = jnp.array([0.0, 0.5, 1.0, 2.0, 3.7, -2.5])
    numeric = jnp.array(
        [jnp.real(jnp.sum(w * jnp.exp(-2j * jnp.pi * x * t))) / n for x in offsets]
    )
    np.testing.assert_allclose(numeric, rectangular_spectrum(offsets), atol=1e-3)
    assert float(rectangular_spectrum(jnp.array([0.0]))[0]) == pytest.approx(
        1.0, abs=1e-5
    )


def test_tukey_reproduces_rectangular_at_zero_taper():
    """Rectangular is the Tukey window at beta -> 0."""
    x = jnp.array([0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 3.7, -2.5])
    as_rectangular = (tukey_spectrum(x, beta=1e-12) / (1.0 - 0.5e-12)) ** 2
    np.testing.assert_allclose(
        as_rectangular, Window(kind="rectangular").power(x), atol=1e-5
    )


def test_every_window_peaks_at_one():
    for window in (
        Window(kind="rectangular"),
        Window(kind="tukey", beta=0.5),
        Window(kind="hann"),
    ):
        assert float(window.power(jnp.array([0.0]))[0]) == pytest.approx(1.0)


def test_unknown_window_is_rejected():
    with pytest.raises(ValueError, match="window must be"):
        Window(kind="blackman")


@pytest.mark.parametrize("beta", [0.0, 1.5, -0.2])
def test_taper_fraction_is_validated(beta):
    """0 raised ZeroDivisionError and out-of-range values returned quiet nonsense."""
    with pytest.raises(ValueError, match=r"beta must be in \(0, 1\]"):
        Window(kind="tukey", beta=beta)


def test_hann_power_special_values():
    """Goetz & Riles (2011) Eq. 20, with its 4:1 center-to-neighbor ratio."""
    got = Window(kind="hann").power(jnp.array([0.0, 1.0, -1.0]))
    assert float(got[0]) == pytest.approx(1.0)
    assert float(got[1]) == pytest.approx(0.25)
    assert float(got[2]) == pytest.approx(0.25)


def test_hann_power_is_continuous_through_its_singularity():
    """``x = +/-1`` is removable; the limit is substituted, not evaluated."""
    near = Window(kind="hann").power(jnp.array([1.0 - 1e-5, 1.0, 1.0 + 1e-5]))
    assert jnp.all(jnp.isfinite(near))
    assert jnp.allclose(near, 0.25, atol=1e-4)


def test_lvk_production_taper_is_rectangular_for_our_purposes():
    """beta = 0.001 is the LAL default: 4e-4 from no window at all.

    Small enough that a rectangular template matches such data to 1.0000, which
    is why the SFTs in ``data/`` need no Tukey kernel.
    """
    x = jnp.linspace(-30.0, 30.0, 6001)
    lvk = Window(kind="tukey", beta=0.001).power(x)
    assert float(jnp.max(jnp.abs(lvk - Window(kind="rectangular").power(x)))) < 1e-3


def test_tukey_leaks_between_rect_and_hann():
    """Its whole reason for existing: SFDB-derived SFTs are neither."""
    x = jnp.array([1.0, 2.0, 3.0])
    rect = Window(kind="rectangular").power(x)
    tukey = Window(kind="tukey", beta=0.5).power(x)
    hann = Window(kind="hann").power(x)
    assert jnp.all(rect < tukey)
    assert float(tukey[0]) == pytest.approx(0.0801, abs=1e-3)
    assert float(hann[0]) == pytest.approx(0.25)
