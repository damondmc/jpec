import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jpec.engine.loop import loudest_in_box, matched_filter_grid

N_DOPPLER_REFERENCES = 2
N_PERIODS = 4
N_HARMONICS = 6
N_TRANSFORMED_BINS = 40
N_SUB_BIN_OFFSETS = 3
N_MODULATION_DEPTHS = 7
N_WINDOW_BINS = 12
N_F0_POINTS = 9
N_MATCHED_MODULATION_DEPTHS = 5
N_PHI = 16


@pytest.fixture(scope="module")
def grid():
    keys = jax.random.split(jax.random.key(0), 4)
    data_harmonics = jax.random.normal(
        keys[0],
        (N_DOPPLER_REFERENCES, N_PERIODS, N_HARMONICS, N_TRANSFORMED_BINS),
        dtype=jnp.complex64,
    )
    template_harmonics = jax.random.normal(
        keys[1],
        (N_SUB_BIN_OFFSETS, N_MODULATION_DEPTHS, N_HARMONICS, N_WINDOW_BINS),
        dtype=jnp.complex64,
    )
    return {
        "data_harmonics": data_harmonics,
        "template_harmonics": template_harmonics,
        "doppler_reference_idxs": jax.random.randint(
            keys[2], (N_F0_POINTS,), 0, N_DOPPLER_REFERENCES
        ),
        "window_start_idxs": jnp.arange(N_F0_POINTS) * 3,
        "sub_bin_offset_idxs": jax.random.randint(
            keys[3], (N_F0_POINTS,), 0, N_SUB_BIN_OFFSETS
        ),
    }


def match(grid, f0_idxs: slice, *, f0_batch_size: int):
    return matched_filter_grid(
        data_harmonics=grid["data_harmonics"],
        template_harmonics=grid["template_harmonics"],
        doppler_reference_idxs=grid["doppler_reference_idxs"][f0_idxs],
        window_start_idxs=grid["window_start_idxs"][f0_idxs],
        sub_bin_offset_idxs=grid["sub_bin_offset_idxs"][f0_idxs],
        modulation_depth_start_idx=jnp.asarray(1),
        n_matched_modulation_depths=N_MATCHED_MODULATION_DEPTHS,
        n_phi=N_PHI,
        f0_batch_size=f0_batch_size,
    )


def test_one_call_matches_one_f0_point_per_call(grid):
    """Every f0 point in one batched call scores as each f0 point on its own."""
    max_statistic, phase_idxs = match(grid, slice(None), f0_batch_size=4)
    for f0_idx in range(N_F0_POINTS):
        f0_max_statistic, f0_phase_idxs = match(
            grid, slice(f0_idx, f0_idx + 1), f0_batch_size=1
        )
        np.testing.assert_allclose(
            max_statistic[f0_idx], f0_max_statistic[0], rtol=1e-5
        )
        np.testing.assert_array_equal(phase_idxs[f0_idx], f0_phase_idxs[0])


def test_loudest_in_box_is_the_sorted_in_box_grid(grid):
    """The device-side cut keeps in-box templates a host sort would, loudest first."""
    max_statistic, phase_idxs = match(grid, slice(None), f0_batch_size=4)
    in_box = jax.random.bernoulli(
        jax.random.key(1), 0.5, (N_PERIODS, N_MATCHED_MODULATION_DEPTHS)
    )
    n_loudest = 10
    loudest_max_statistic, flat_idxs, loudest_phase_idxs = loudest_in_box(
        max_statistic, phase_idxs, in_box, n_loudest=n_loudest
    )
    in_box_max_statistic = np.where(
        np.asarray(in_box)[None], np.asarray(max_statistic), -np.inf
    ).ravel()
    expected_flat_idxs = np.argsort(-in_box_max_statistic, kind="stable")[:n_loudest]
    np.testing.assert_array_equal(flat_idxs, expected_flat_idxs)
    np.testing.assert_array_equal(
        loudest_max_statistic, in_box_max_statistic[expected_flat_idxs]
    )
    np.testing.assert_array_equal(
        loudest_phase_idxs, np.asarray(phase_idxs).ravel()[expected_flat_idxs]
    )
