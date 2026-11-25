import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.config import SearchLoopConfig
from jpec.constant import DAY
from jpec.detector.doppler import solar_roemer_delay
from jpec.detector.site import Detector, Observation, reference_epoch
from jpec.engine.data import SearchData, freq_band_edges
from jpec.engine.loop import BandSearch
from jpec.engine.match import (
    matched_filter_grid,
    template_harmonics_at_depths,
    top_k_templates,
)
from jpec.noise.floor import noise_floor
from jpec.noise.gaussian import gaussian_noise_sft
from jpec.params import Prior
from jpec.stat.matched_filter import max_over_phase
from jpec.waveform.window import Window

# the O4a start [GPS s], and the one sky point the small search is at [rad]
RUN_START = 1368975618.0
SKY_ALPHA, SKY_DELTA = 1.2, 0.3
# a Doppler mismatch small enough that the small search's f0 range needs several
# Doppler references
MULTI_REFERENCE_DOPPLER_MISMATCH = 1e-8
# sky lattice indices of the three-sky-point bank, all of omega^2 class 0, sharing
# one depth-class pattern and each with its own f0 shifts
THREE_SKY_LATTICE_IDXS = np.array([[0, 0], [1, 1], [2, -1]])
PERIOD_BATCH_SIZE = 4
F0_BATCH_SIZE = 8
N_TOP = 20
# one toplist over the whole band [Hz]
ONE_TOPLIST_F0_EDGES = np.array([-np.inf, np.inf])
# every this many f0 points checked template by template
F0_STRIDE = 16

N_PERIODS = 4
N_MATCHED_MODULATION_DEPTHS = 5
N_F0_POINTS = 9
N_PHI = 16


def test_top_k_templates_is_the_sorted_ap_in_range_grid():
    """The device-side cut keeps in-range templates a host sort would, top first."""
    keys = jax.random.split(jax.random.key(0), 3)
    grid_shape = (N_F0_POINTS, N_PERIODS, N_MATCHED_MODULATION_DEPTHS)
    max_statistic = jax.random.normal(keys[0], grid_shape)
    phase_idxs = jax.random.randint(keys[1], grid_shape, 0, N_PHI)
    ap_in_range = jax.random.bernoulli(
        keys[2], 0.5, (N_PERIODS, N_MATCHED_MODULATION_DEPTHS)
    )
    n_top = 10
    # one toplist over every f0 point
    all_f0_idxs = jnp.arange(N_F0_POINTS)[None]
    top_max_statistic, flat_idxs, top_phase_idxs = top_k_templates(
        max_statistic, phase_idxs, ap_in_range, all_f0_idxs, n_top=n_top
    )
    ap_in_range_max_statistic = np.where(
        np.asarray(ap_in_range)[None], np.asarray(max_statistic), -np.inf
    ).ravel()
    expected_flat_idxs = np.argsort(-ap_in_range_max_statistic, kind="stable")[:n_top]
    np.testing.assert_array_equal(flat_idxs[0], expected_flat_idxs)
    np.testing.assert_array_equal(
        top_max_statistic[0], ap_in_range_max_statistic[expected_flat_idxs]
    )
    np.testing.assert_array_equal(
        top_phase_idxs[0], np.asarray(phase_idxs).ravel()[expected_flat_idxs]
    )


def band_search_on(
    search_data: SearchData,
    bank: TemplateBank,
    *,
    doppler_mismatch: float = SearchLoopConfig().doppler_mismatch,
    toplist_f0_edges: np.ndarray = ONE_TOPLIST_F0_EDGES,
) -> BandSearch:
    """A bank searched on the small search's band."""
    return BandSearch(
        bank,
        search_data,
        SearchLoopConfig(
            period_batch_size=PERIOD_BATCH_SIZE,
            f0_batch_size=F0_BATCH_SIZE,
            doppler_mismatch=doppler_mismatch,
        ),
        n_top=N_TOP,
        toplist_f0_edges=toplist_f0_edges,
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
    bank = TemplateBank.over_f0_interval(
        tiling, *prior.f0, velocity=observation.velocity, window=window
    )
    freq_low, freq_high = freq_band_edges(
        bank.f0[0],
        bank.f0_high,
        max_modulation_depth=bank.max_modulation_depth,
        tsft=bank.tsft,
    )
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
    return search_data, band_search_on(search_data, bank)


@pytest.fixture(scope="module")
def three_sky_search(small_search):
    """The small search at three sky points of one omega^2 class, each shifted."""
    search_data, band_search = small_search
    three_sky_bank = band_search.layout.bank._replace(
        alpha=SKY_ALPHA + np.array([0.0, 0.05, 0.1]),
        delta=SKY_DELTA + np.array([0.0, -0.05, 0.05]),
        sky_lattice_idxs=THREE_SKY_LATTICE_IDXS,
    )
    return search_data, band_search_on(search_data, three_sky_bank)


def top_template_list(band_search: BandSearch, sky_idxs=(0,)):
    return list(band_search.search_sky_batch(np.asarray(sky_idxs)))


def batch_data_harmonics(search_data, band_search, sky_idx, template_batch):
    """`F[j, k]` of a template batch at one sky point, as the loop transforms it."""
    bank = band_search.layout.bank
    return band_search.transform(
        band_search.doppler_correct(sky_idx)[None],
        solar_roemer_delay(
            search_data.position,
            alpha=jnp.asarray(bank.alpha[[sky_idx]]),
            delta=jnp.asarray(bank.delta[[sky_idx]]),
        ),
        int(bank.omega2_class[sky_idx]),
        template_batch,
    )[0]


def test_each_shifted_template_scores_as_max_over_phase_alone(three_sky_search):
    """The match gives every template what max_over_phase() gives it alone.

    Each template is built on its own at its shifted sub-bin offset and depth class,
    placed at its shifted bin in its f0 point's widened window.
    """
    search_data, band_search = three_sky_search
    layout = band_search.layout
    bank = layout.bank
    n_subbin_offsets = layout.subbin_offsets.size
    n_padded_bins = layout.bin_offsets.shape[0] + layout.n_pad
    window_start_idxs = np.asarray(layout.window_start_idxs)
    doppler_reference_idxs = np.asarray(layout.doppler_reference_idxs)
    subbin_offset_idxs = np.asarray(layout.subbin_offset_idxs)
    # tau[j, k] at every base depth, per (shifted sub-bin offset, depth class)
    templates = {}
    batch_idx = 0
    template_batch = layout.template_batches[batch_idx]
    for sky_idx in range(bank.alpha.size):
        data_harmonics = batch_data_harmonics(
            search_data, band_search, sky_idx, template_batch
        )
        sky_shifts = band_search.sky_shifts(sky_idx)
        max_statistic, phase_idxs = matched_filter_grid(
            data_harmonics=data_harmonics,
            placed_harmonics=band_search.placed_template_harmonics,
            norms_squared=band_search.template_norms_squared,
            window_start_idxs=layout.window_start_idxs,
            group_f0_idxs=layout.group_f0_idxs,
            group_doppler_reference_idxs=layout.group_doppler_reference_idxs,
            group_subbin_offset_idxs=layout.group_subbin_offset_idxs,
            modulation_depth_start_idx=template_batch.modulation_depth_start_idx,
            whole_bin_shifts=sky_shifts.whole_bin_shifts[batch_idx],
            subbin_shifts=sky_shifts.subbin_shifts[batch_idx],
            depth_classes=sky_shifts.depth_classes[batch_idx],
            n_batch_depths=layout.n_batch_depths,
            n_phi=layout.n_phi,
            f0_batch_size=F0_BATCH_SIZE,
        )
        whole_bin_shifts, subbin_shifts, depth_classes = (
            np.asarray(shifts[batch_idx])
            for shifts in (
                sky_shifts.whole_bin_shifts,
                sky_shifts.subbin_shifts,
                sky_shifts.depth_classes,
            )
        )
        for f0_idx in range(0, bank.f0.size, F0_STRIDE):
            window_start_idx = int(window_start_idxs[f0_idx])
            reference_data_harmonics = np.asarray(data_harmonics)[
                doppler_reference_idxs[f0_idx]
            ]
            for period_offset in range(layout.period_batch_size):
                depth_class = int(depth_classes[period_offset])
                window_data_harmonics = reference_data_harmonics[
                    period_offset,
                    :,
                    window_start_idx : window_start_idx + n_padded_bins,
                ]
                for depth_offset in range(layout.n_batch_depths):
                    shifted_offset_idx = int(
                        subbin_offset_idxs[f0_idx]
                        + subbin_shifts[period_offset, depth_offset]
                    )
                    offset_idx = shifted_offset_idx % n_subbin_offsets
                    placement = (
                        int(whole_bin_shifts[period_offset, depth_offset])
                        + shifted_offset_idx // n_subbin_offsets
                    )
                    if (offset_idx, depth_class) not in templates:
                        templates[offset_idx, depth_class] = np.asarray(
                            template_harmonics_at_depths(
                                jnp.asarray(
                                    layout.bin_offsets
                                    - layout.subbin_offsets[offset_idx]
                                ),
                                jnp.asarray(
                                    bank.modulation_depth_at(np.asarray(depth_class))
                                    * bank.tsft
                                ),
                                n_theta=layout.n_theta,
                                j_max=layout.j_max,
                                window=bank.window,
                                batch_size=16,
                            )
                        )
                    template = templates[offset_idx, depth_class][
                        template_batch.modulation_depth_start_idx + depth_offset,
                        : template_batch.n_harmonics,
                    ]
                    placed_template = np.zeros(
                        (template.shape[0], n_padded_bins), template.dtype
                    )
                    placed_template[
                        :, placement : placement + layout.bin_offsets.shape[0]
                    ] = template
                    template_max_statistic, template_phase_idx = max_over_phase(
                        jnp.asarray(window_data_harmonics),
                        jnp.asarray(placed_template),
                        n_phi=layout.n_phi,
                    )
                    np.testing.assert_allclose(
                        max_statistic[f0_idx, period_offset, depth_offset],
                        template_max_statistic,
                        rtol=1e-5,
                    )
                    assert (
                        phase_idxs[f0_idx, period_offset, depth_offset]
                        == template_phase_idx
                    )


def test_toplists_hold_the_single_toplists_templates_per_f0_interval(small_search):
    """Split into toplists, each f0 interval keeps what one toplist keeps of it."""
    search_data, band_search = small_search
    f0 = band_search.layout.bank.f0
    middle_f0 = 0.5 * (f0.min() + f0.max())
    split_search = band_search_on(
        search_data,
        band_search.layout.bank,
        toplist_f0_edges=np.array([-np.inf, middle_f0, np.inf]),
    )
    split = top_template_list(split_search)
    single = top_template_list(band_search)
    assert len(split) == 2 * len(single)
    for single_grid, low_grid, high_grid in zip(single, split[::2], split[1::2]):
        assert (low_grid.toplist_idx, high_grid.toplist_idx) == (0, 1)
        is_found = np.isfinite(single_grid.max_statistic)
        is_low = f0[single_grid.f0_idxs] < middle_f0
        # the single toplist's templates, cut at middle_f0, lead each split one
        for grid, is_member in (
            (low_grid, is_found & is_low),
            (high_grid, is_found & ~is_low),
        ):
            n_members = int(is_member.sum())
            for field in (
                "max_statistic",
                "f0_idxs",
                "period_idxs",
                "modulation_depth_idxs",
                "phase_idxs",
            ):
                np.testing.assert_array_equal(
                    getattr(grid, field)[:n_members],
                    getattr(single_grid, field)[is_member],
                )


def test_a_sky_batch_scores_as_each_sky_point_alone(three_sky_search):
    """Sky points transformed together score as each one transformed on its own."""
    _, band_search = three_sky_search
    assert np.unique(band_search.layout.bank.omega2_class).size == 1
    batched = top_template_list(band_search, sky_idxs=range(3))
    for sky_idx in range(3):
        alone = top_template_list(band_search, sky_idxs=[sky_idx])
        batched_at_sky = [grid for grid in batched if grid.sky_idx == sky_idx]
        assert len(batched_at_sky) == len(alone)
        for batched_grid, alone_grid in zip(batched_at_sky, alone):
            np.testing.assert_allclose(
                batched_grid.max_statistic, alone_grid.max_statistic, rtol=1e-5
            )
            np.testing.assert_array_equal(batched_grid.f0_idxs, alone_grid.f0_idxs)
            np.testing.assert_array_equal(
                batched_grid.period_idxs, alone_grid.period_idxs
            )


def test_harmonics_per_batch_leave_the_statistic_unchanged(small_search):
    """Each batch at its own harmonics scores as every batch at the layout's j_max."""
    _, band_search = small_search
    layout = band_search.layout
    assert any(
        batch.n_harmonics < layout.j_max for batch in layout.template_batches
    ), "every batch needs j_max, so there is nothing to cut"

    full_search = copy.copy(band_search)
    full_search.layout = layout._replace(
        template_batches=[
            batch._replace(n_harmonics=layout.j_max)
            for batch in layout.template_batches
        ]
    )
    per_batch = top_template_list(band_search)
    full = top_template_list(full_search)
    assert len(per_batch) == len(full)
    for per_batch_grid, full_grid in zip(per_batch, full):
        np.testing.assert_allclose(
            per_batch_grid.max_statistic, full_grid.max_statistic, rtol=1e-5
        )
        for field in ("f0_idxs", "period_idxs", "modulation_depth_idxs", "phase_idxs"):
            np.testing.assert_array_equal(
                getattr(per_batch_grid, field), getattr(full_grid, field)
            )


@pytest.fixture(scope="module")
def multi_reference_search(small_search):
    """The small search with a Doppler mismatch small enough to need several refs."""
    search_data, band_search = small_search
    return search_data, band_search_on(
        search_data,
        band_search.layout.bank,
        doppler_mismatch=MULTI_REFERENCE_DOPPLER_MISMATCH,
    )


def test_band_trim_leaves_the_statistic_unchanged(multi_reference_search):
    """Each reference transforming only its windows' bins scores as the whole band."""
    search_data, band_search = multi_reference_search
    n_band_bins = search_data.n_band_bins
    layout = band_search.layout
    starts = np.asarray(layout.transformed_start_idxs)
    windows = np.asarray(layout.window_start_idxs)
    n_padded_bins = layout.bin_offsets.shape[0] + layout.n_pad
    assert (starts >= 0).all()
    assert (starts + layout.n_transformed_bins <= n_band_bins).all()
    assert (windows >= 0).all()
    assert (windows + n_padded_bins <= layout.n_transformed_bins).all()
    assert layout.n_transformed_bins < n_band_bins, (
        "the band holds only the windows, so there is nothing to trim"
    )
    assert layout.doppler_references.size > 1, (
        "one reference, so the references' bins are not split"
    )

    # every reference transforms the whole band, each window counted from its first
    # bin
    untrimmed_search = copy.copy(band_search)
    untrimmed_search.layout = layout._replace(
        transformed_start_idxs=jnp.zeros_like(layout.transformed_start_idxs),
        n_transformed_bins=n_band_bins,
        window_start_idxs=layout.window_start_idxs
        + layout.transformed_start_idxs[layout.doppler_reference_idxs],
    )
    trimmed = top_template_list(band_search)
    untrimmed = top_template_list(untrimmed_search)
    assert len(trimmed) == len(untrimmed)
    for trimmed_grid, untrimmed_grid in zip(trimmed, untrimmed):
        np.testing.assert_allclose(
            trimmed_grid.max_statistic, untrimmed_grid.max_statistic, rtol=1e-10
        )
        for field in ("f0_idxs", "period_idxs", "modulation_depth_idxs", "phase_idxs"):
            np.testing.assert_array_equal(
                getattr(trimmed_grid, field), getattr(untrimmed_grid, field)
            )
