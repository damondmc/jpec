import jax.numpy as jnp
import pytest
from signals import Case, harmonic_transform, orbit_offset

from jpec.constant import ARC_PAD_BINS
from jpec.waveform.harmonics import (
    harmonic_count,
    template_harmonics,
    theta_sample_count,
)
from jpec.waveform.plane import arc_bin_offsets, plane_from_arc


@pytest.fixture
def setup(case: Case):
    """One template and one orbit of data, on a shared harmonic grid."""
    parameters = case.parameters
    j_max = harmonic_count(case.modulation_depth_bins)
    n = theta_sample_count(j_max)
    offsets = arc_bin_offsets(case.modulation_depth_bins, pad_bins=ARC_PAD_BINS)
    times = jnp.arange(n) * (parameters.period / n)
    plane = plane_from_arc(
        orbit_offset(parameters, times) * case.tsft, offsets, window=case.window
    )
    return {
        "j_max": j_max,
        "n_theta": n,
        "bin_offsets": offsets,
        "data_harmonics": harmonic_transform(plane)[1 : j_max + 1],
        "template_harmonics": template_harmonics(
            offsets,
            modulation_depth_bins=case.modulation_depth_bins,
            n_theta=n,
            j_max=j_max,
            window=case.window,
        ),
    }
