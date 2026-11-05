import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jpec.bank.template_bank import OrbitalTemplateGrid, TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import SearchLoopConfig
from jpec.constant import DAY
from jpec.detector.site import Detector, Observation, reference_epoch
from jpec.engine.data import SearchData, band_edges_for_f0
from jpec.engine.layout import f0_groups
from jpec.engine.loop import (
    HarmonicBank,
    loudest_in_box,
    matched_filter_grid,
    orbital_statistics,
)
from jpec.noise.floor import noise_floor
from jpec.noise.gaussian import gaussian_noise_sft
from jpec.params import Prior
from jpec.stat.matched_filter import max_over_phase
from jpec.waveform.window import Window

# the O4a start [GPS s], and the one sky point the small search is at [rad]
RUN_START = 1368975618.0
SKY_ALPHA, SKY_DELTA = 1.2, 0.3

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
    group_f0_idxs, group_doppler_reference_idxs, group_sub_bin_offset_idxs = f0_groups(
        np.asarray(grid["doppler_reference_idxs"][f0_idxs]),
        np.asarray(grid["sub_bin_offset_idxs"][f0_idxs]),
    )
    return matched_filter_grid(
        data_harmonics=grid["data_harmonics"],
        template_harmonics=grid["template_harmonics"],
        window_start_idxs=grid["window_start_idxs"][f0_idxs],
        group_f0_idxs=jnp.asarray(group_f0_idxs),
        group_doppler_reference_idxs=jnp.asarray(group_doppler_reference_idxs),
        group_sub_bin_offset_idxs=jnp.asarray(group_sub_bin_offset_idxs),
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


def test_each_template_scores_as_max_over_phase_alone(grid):
    """The grouped match gives every template what max_over_phase() gives it alone."""
    max_statistic, phase_idxs = match(grid, slice(None), f0_batch_size=4)
    depth_start_idx = 1
    for f0_idx in range(N_F0_POINTS):
        window_start_idx = int(grid["window_start_idxs"][f0_idx])
        reference_data_harmonics = grid["data_harmonics"][
            int(grid["doppler_reference_idxs"][f0_idx])
        ]
        offset_template_harmonics = grid["template_harmonics"][
            int(grid["sub_bin_offset_idxs"][f0_idx])
        ]
        for period_idx in range(N_PERIODS):
            window_data_harmonics = reference_data_harmonics[
                period_idx, :, window_start_idx : window_start_idx + N_WINDOW_BINS
            ]
            for depth_offset in range(N_MATCHED_MODULATION_DEPTHS):
                template_max_statistic, template_phase_idx = max_over_phase(
                    window_data_harmonics,
                    offset_template_harmonics[depth_start_idx + depth_offset],
                    n_phi=N_PHI,
                )
                np.testing.assert_allclose(
                    max_statistic[f0_idx, period_idx, depth_offset],
                    template_max_statistic,
                    rtol=1e-5,
                )
                assert (
                    phase_idxs[f0_idx, period_idx, depth_offset] == template_phase_idx
                )


def test_f0_groups_hold_every_f0_point_once_with_one_reference_and_offset():
    """Each group shares a reference and offset, and every f0 point is in one group."""
    doppler_reference_idxs = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])
    sub_bin_offset_idxs = np.array([2, 0, 2, 1, 0, 2, 1, 0, 2])
    group_f0_idxs, group_doppler_reference_idxs, group_sub_bin_offset_idxs = f0_groups(
        doppler_reference_idxs, sub_bin_offset_idxs
    )
    for f0_idxs, reference_idx, offset_idx in zip(
        group_f0_idxs, group_doppler_reference_idxs, group_sub_bin_offset_idxs
    ):
        assert np.all(doppler_reference_idxs[f0_idxs] == reference_idx)
        assert np.all(sub_bin_offset_idxs[f0_idxs] == offset_idx)
    assert sorted(set(group_f0_idxs.ravel().tolist())) == list(
        range(doppler_reference_idxs.size)
    )


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


@pytest.fixture(scope="module")
def small_search():
    """Ten days of one detector's Gaussian noise and a small bank at one sky point."""
    tsft = 1800.0
    window = Window(kind="tukey", beta=0.5)
    search_loop_config = SearchLoopConfig()
    timestamps = {"H1": RUN_START + np.arange(0.0, 10.0 * DAY, tsft)}
    t_ref = reference_epoch(timestamps, tsft=tsft)
    observation = Observation(timestamps, tsft=tsft, t_ref=t_ref)
    prior = Prior(
        f0=(150.0, 150.004),
        ap=(5.5, 6.0),
        period=(8.5 * DAY, 9.5 * DAY),
        phi_b=(0.0, 2.0 * np.pi),
        alpha=(SKY_ALPHA, SKY_ALPHA),
        delta=(SKY_DELTA, SKY_DELTA),
    )
    tiling = LatticeTiling.from_metric(
        prior,
        observation=observation,
        mismatch=0.4,
        n_ap_blocks=1,
        window=window,
    )
    freq_low, freq_high = band_edges_for_f0(prior, *prior.f0, tsft=tsft)
    sft = gaussian_noise_sft(
        timestamps["H1"],
        Detector("H1"),
        key=jax.random.key(0),
        freq_low=freq_low,
        n_band_bins=round((freq_high - freq_low) * tsft),
        tsft=tsft,
        sqrt_sn=1.0,
        window=window,
    )
    floor = noise_floor(
        jnp.asarray(sft.sft_power),
        width=search_loop_config.floor_width,
        batch_size=search_loop_config.floor_batch_size,
    )
    search_data = SearchData([sft], floor=[floor], t_ref=t_ref)
    template_grids = [
        OrbitalTemplateGrid(orbital_subgrid=orbital_subgrid, tsft=tsft, window=window)
        for orbital_subgrid in tiling.orbital_subgrids(*prior.f0)
    ]
    bank = TemplateBank(
        template_grids=template_grids,
        alpha=np.array([SKY_ALPHA]),
        delta=np.array([SKY_DELTA]),
    )
    harmonic_bank = HarmonicBank(
        bank,
        times=search_data.times,
        freq_low=search_data.freq_low,
        n_band_bins=search_data.n_band_bins,
        n_sub_bin_offsets=search_loop_config.n_sub_bin_offsets,
        period_batch_size=4,
        harmonic_fraction=search_loop_config.harmonic_fraction,
        modulation_depth_batch_size=search_loop_config.modulation_depth_batch_size,
    )
    return search_data, harmonic_bank


def loudest_per_grid(search_data, harmonic_bank, sky_idxs=(0,)):
    return list(
        orbital_statistics(
            search_data,
            harmonic_bank,
            sky_idxs=sky_idxs,
            f0_batch_size=8,
            n_loudest=20,
        )
    )


def test_a_sky_batch_scores_as_each_sky_point_alone(small_search):
    """Sky points transformed together score as each one transformed on its own."""
    search_data, harmonic_bank = small_search
    bank = harmonic_bank.bank
    three_sky_bank = copy.copy(harmonic_bank)
    three_sky_bank.bank = bank._replace(
        alpha=SKY_ALPHA + np.array([0.0, 0.05, 0.1]),
        delta=SKY_DELTA + np.array([0.0, -0.05, 0.05]),
    )
    batched = loudest_per_grid(search_data, three_sky_bank, sky_idxs=range(3))
    for sky_idx in range(3):
        alone = loudest_per_grid(search_data, three_sky_bank, sky_idxs=[sky_idx])
        batched_at_sky = [grid for grid in batched if grid.sky_idx == sky_idx]
        assert len(batched_at_sky) == len(alone)
        for batched_grid, alone_grid in zip(batched_at_sky, alone):
            assert batched_grid.subgrid_idx == alone_grid.subgrid_idx
            np.testing.assert_allclose(
                batched_grid.max_statistic, alone_grid.max_statistic, rtol=1e-5
            )
            np.testing.assert_array_equal(batched_grid.f0_idxs, alone_grid.f0_idxs)
            np.testing.assert_array_equal(
                batched_grid.period_idxs, alone_grid.period_idxs
            )


def test_harmonics_per_batch_leave_the_statistic_unchanged(small_search):
    """Each batch at its own harmonics scores as every batch at the subgrid's j_max."""
    search_data, harmonic_bank = small_search
    assert any(
        batch.n_harmonics < layout.template_grid.j_max
        for layout in harmonic_bank.layouts
        for batch in layout.period_batches
    ), "every batch needs j_max, so there is nothing to cut"

    full_bank = copy.copy(harmonic_bank)
    full_bank.layouts = [
        layout._replace(
            period_batches=[
                batch._replace(n_harmonics=layout.template_grid.j_max)
                for batch in layout.period_batches
            ]
        )
        for layout in harmonic_bank.layouts
    ]
    per_batch = loudest_per_grid(search_data, harmonic_bank)
    full = loudest_per_grid(search_data, full_bank)
    assert len(per_batch) == len(full)
    for per_batch_grid, full_grid in zip(per_batch, full):
        np.testing.assert_allclose(
            per_batch_grid.max_statistic, full_grid.max_statistic, rtol=1e-5
        )
        for field in ("f0_idxs", "period_idxs", "modulation_depth_idxs", "phase_idxs"):
            np.testing.assert_array_equal(
                getattr(per_batch_grid, field), getattr(full_grid, field)
            )


def test_band_trim_leaves_the_statistic_unchanged(small_search):
    """Transforming only the bins the windows read scores as the whole band does."""
    search_data, harmonic_bank = small_search
    n_band_bins = search_data.n_band_bins
    for layout in harmonic_bank.layouts:
        assert 0 <= layout.transformed_bin_idxs.start
        assert layout.transformed_bin_idxs.stop <= n_band_bins
    assert any(
        layout.transformed_bin_idxs.stop - layout.transformed_bin_idxs.start
        < n_band_bins
        for layout in harmonic_bank.layouts
    ), "the band holds only the windows, so there is nothing to trim"

    # the whole band transformed, each window counted from the band's first bin
    untrimmed_bank = copy.copy(harmonic_bank)
    untrimmed_bank.layouts = [
        layout._replace(
            transformed_bin_idxs=slice(0, n_band_bins),
            window_start_idxs=layout.window_start_idxs
            + layout.transformed_bin_idxs.start,
        )
        for layout in harmonic_bank.layouts
    ]
    trimmed = loudest_per_grid(search_data, harmonic_bank)
    untrimmed = loudest_per_grid(search_data, untrimmed_bank)
    assert len(trimmed) == len(untrimmed)
    for trimmed_grid, untrimmed_grid in zip(trimmed, untrimmed):
        np.testing.assert_allclose(
            trimmed_grid.max_statistic, untrimmed_grid.max_statistic, rtol=1e-10
        )
        for field in ("f0_idxs", "period_idxs", "modulation_depth_idxs", "phase_idxs"):
            np.testing.assert_array_equal(
                getattr(trimmed_grid, field), getattr(untrimmed_grid, field)
            )
