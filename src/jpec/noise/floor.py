"""The noise floor: a running median of the SFT power, corrected for its bias."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def median_bias(width: int) -> float:
    """Ratio of the running median to the mean, for scaled chi2_2 power.

        bias = sum(1/i for i in range((width+1)//2, width+1))

    which tends to ln 2 = 0.693 as the width grows.

    Args:
        width: Window length in bins, odd so the median is a single sample.

    Returns:
        Expected median of scaled chi2_2 power with unit mean.
    """
    if width < 1 or width % 2 == 0:
        raise ValueError(f"width must be a positive odd number of bins, got {width}")
    return float(sum(1.0 / i for i in range((width + 1) // 2, width + 1)))


def running_median(
    values: Float[Array, " n_band"], *, width: int
) -> Float[Array, " n_band"]:
    """Median of the `width` values around each, the sequence mirrored past its ends.

    Args:
        values: The frequency bins of one SFT.
        width: Window length, odd.

    Returns:
        The running median, same length as `values`.
    """
    pad = width // 2
    padded = jnp.pad(values, (pad, pad), mode="symmetric")
    windows = jnp.stack(
        [padded[i : i + values.shape[0]] for i in range(width)], axis=-1
    )
    return jnp.median(windows, axis=-1)


def noise_floor(
    sft_power: Float[Array, "n_sft n_band"],
    *,
    width: int,
    batch_size: int,
) -> Float[Array, "n_sft n_band"]:
    """Mean noise power at each point of the plane, estimated by a running median.

    The median of the `width` frequency bins around each, divided by
    `median_bias` so it is unbiased for the mean. Never along the time axis:
    a glitch that lifts one SFT is absorbed, a line that sits in one frequency
    bin is not. Past the band's ends the frequency bins are mirrored, so the
    window near an edge still holds `width` distinct samples.

    Args:
        sft_power: Observed power plane.
        width: Running-median window in bins, odd.
        batch_size: SFTs whose windows are sorted at once; memory is
            `batch_size * n_band_bins * width`.

    Returns:
        Estimated mean noise power, same shape as `sft_power`.
    """
    if sft_power.shape[1] < width:
        raise ValueError(
            f"need at least width={width} bins to estimate a floor, "
            f"got {sft_power.shape[1]}"
        )
    median = jax.lax.map(
        partial(running_median, width=width),
        jnp.asarray(sft_power),
        batch_size=batch_size,
    )
    return median / median_bias(width)
