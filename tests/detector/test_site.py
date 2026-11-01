import numpy as np
import pytest

from jpec.detector.site import Detector

T0 = 1238166018.0


@pytest.fixture(scope="module")
def times():
    """Midpoints spread over a year, so the Earth's orbit is sampled."""
    return T0 + np.arange(0.0, 365.0 * 86400.0, 7.0 * 86400.0)


@pytest.fixture(scope="module")
def h1(times):
    return Detector("H1").velocity(times)


def test_untested_detectors_warn():
    with pytest.warns(UserWarning, match="is not supported"):
        Detector("V1")


def test_velocity_is_one_vector_per_time(times, h1):
    assert h1.shape == (times.size, 3)
    assert h1.dtype == np.float64


def test_speed_order(h1):
    """Components of v/c peak at the Earth's orbital speed, 1e-4."""
    assert 5e-5 < np.abs(h1).max() < 2e-4


def test_detectors_differ(times, h1):
    l1 = Detector("L1").velocity(times)
    assert not np.allclose(h1, l1, rtol=1e-3)
