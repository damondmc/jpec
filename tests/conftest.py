"""Test-wide configuration and the fixtures every module shares."""

import jax
import jax.numpy as jnp
import pytest
from signals import CASES, DAY, Case

# The library does not require this. Nothing on the search path accumulates,
# and test_search_path_runs_in_float32 checks that end to end.
#
# The suite does: 41 tests fail without it. Some are about dtype and cannot
# state their own premise in its absence, since astype(float64) silently
# returns float32 when the flag is off. The rest assert identities at machine
# precision, which needs a float64 floor to assert against.
jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=CASES, ids=[c.name for c in CASES])
def case(request) -> Case:
    """Each reference signal in turn."""
    return request.param


@pytest.fixture
def times(case: Case):
    """Twenty days of SFT start times for this case."""
    return jnp.arange(0.0, 20.0 * DAY, case.tsft)


@pytest.fixture
def theta():
    """Orbital phase over one orbit, uniform so its transform is a plain FFT."""
    return jnp.arange(2048) * 2.0 * jnp.pi / 2048
