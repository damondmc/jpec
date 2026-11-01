import jax.numpy as jnp
import numpy as np
from signals import Case

from jpec.waveform.track import frequency_offset, observed_frequency


def test_observed_frequency_is_the_orbit_plus_the_earth(case: Case, times):
    """f = f0 (1 + v.n/c) + orbital offset, and f0 plus the orbit with no Earth motion."""
    parameters = case.parameters
    doppler = 1e-4 * jnp.sin(2.0 * jnp.pi * times / (365.25 * 86400.0))
    observed = observed_frequency(
        times,
        f0=parameters.f0,
        ap=parameters.ap,
        omega=parameters.omega,
        phi_b=parameters.phi_b,
        doppler=doppler,
    )
    orbit = frequency_offset(
        times,
        modulation_depth=parameters.modulation_depth,
        omega=parameters.omega,
        phi_b=parameters.phi_b,
    )
    np.testing.assert_allclose(
        observed, parameters.f0 * (1.0 + doppler) + orbit, rtol=1e-14
    )
    at_rest = observed_frequency(
        times,
        f0=parameters.f0,
        ap=parameters.ap,
        omega=parameters.omega,
        phi_b=parameters.phi_b,
        doppler=jnp.zeros_like(times),
    )
    np.testing.assert_allclose(at_rest, parameters.f0 + orbit, rtol=1e-14)
