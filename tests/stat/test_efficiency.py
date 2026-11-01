import jax.numpy as jnp
import numpy as np
import pytest

from jpec.stat.efficiency import detection_efficiency, fit_efficiency

H0_50 = 2.0e-23
SLOPE = 6.0
N_H0_VALUES = 12
N_DRAWS = 400


def draw(n_draws: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Injections recovered with the known efficiency, at h0 around its midpoint."""
    h0 = np.geomspace(0.3 * H0_50, 3.0 * H0_50, N_H0_VALUES)
    truth = np.asarray(
        detection_efficiency(jnp.log(h0), intercept=-SLOPE * np.log(H0_50), slope=SLOPE)
    )
    rng = np.random.default_rng(seed)
    return h0, rng.random((n_draws, N_H0_VALUES)) < truth


def test_the_fit_recovers_the_curve_it_was_drawn_from():
    """The midpoint and the steepness come back within the error the fit reports."""
    h0, recovered = draw(N_DRAWS)
    fit = fit_efficiency(h0, recovered)
    assert float(jnp.exp(fit.log_h0_at(0.5)[0])) == pytest.approx(H0_50, rel=0.05)
    assert float(fit.slope) == pytest.approx(SLOPE, rel=0.2)

    h95, error = fit.h0_at(0.95)
    truth = H0_50 * np.exp(np.log(0.95 / 0.05) / SLOPE)
    assert abs(float(h95) - truth) < 3.0 * float(error)


def test_more_injections_tighten_the_limit():
    """The error on h95 falls as 1/sqrt(draws), as the covariance must carry."""
    few = fit_efficiency(*draw(100, seed=1)).h0_at(0.95)[1]
    many = fit_efficiency(*draw(1600, seed=1)).h0_at(0.95)[1]
    assert float(few / many) == pytest.approx(4.0, rel=0.5)


def test_saturated_amplitudes_do_not_break_the_fit():
    """Least squares on fractions needs a floor at p = 0, 1; the likelihood does not."""
    h0 = np.geomspace(1e-3 * H0_50, 1e3 * H0_50, N_H0_VALUES)
    recovered = np.asarray(h0 > H0_50)[None, :].repeat(N_DRAWS, axis=0)
    fit = fit_efficiency(h0, recovered)
    assert np.isfinite(float(fit.log_h0_at(0.5)[0]))
    assert float(fit.slope) > 0.0
