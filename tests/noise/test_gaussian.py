import jax
import numpy as np
import pytest

from jpec.detector.site import Detector
from jpec.noise.gaussian import gaussian_noise_sft


def test_sqrt_sn_is_recovered_from_the_power():
    """<2|x|^2> = sqrt_sn^2 * tsft, so sqrt(<2|x|^2> / tsft) gives sqrt_sn back."""
    tsft = 1800.0
    sqrt_sn = 3.0
    timestamps = 1.3e9 + tsft * np.arange(200)
    sft = gaussian_noise_sft(
        timestamps,
        Detector("H1"),
        key=jax.random.key(0),
        freq_low=100.0,
        n_band_bins=512,
        tsft=tsft,
        sqrt_sn=sqrt_sn,
    )
    assert sft.sft_data.shape == (200, 512)
    assert float(sft.sft_power.mean()) == pytest.approx(sqrt_sn**2 * tsft, rel=0.01)
