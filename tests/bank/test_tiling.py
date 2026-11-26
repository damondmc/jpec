from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jpec.bank.lattice import a5star_generator
from jpec.bank.metric import sky_metric
from jpec.bank.template_bank import TemplateBank
from jpec.bank.tiling import LatticeTiling
from jpec.constant import DAY
from jpec.detector.doppler import sky_direction
from jpec.detector.site import Observation
from jpec.params import Parameters, Prior, ap_from_modulation_depth
from jpec.waveform.window import Window

T0 = 1368975618.0
TSFT = 1800.0
DAYS = 20.0
BOX = Prior(
    f0=(150.0, 150.1),
    ap=(5.0, 15.0),
    period=(7.0 * DAY, 15.0 * DAY),
    phi_b=(0.0, 2.0 * np.pi),
    alpha=(0.0, 2.0 * np.pi),
    delta=(-0.5 * np.pi, 0.5 * np.pi),
)
MISMATCH = 0.2
N_AP_BLOCKS = 2
# the f0 band, narrow enough to check template by template
F0_LOW, F0_HIGH = 150.0, 150.02
N_SAMPLED_SKY_POINTS = 20
N_SAMPLED_TEMPLATES = 20
SOURCE = Parameters(
    f0=150.05, ap=9.0, period=10.0 * DAY, phi_b=1.0, alpha=1.2, delta=0.3
)
N_SKY_NEIGHBOURS, N_STEPS = 6, 2


@pytest.fixture(scope="module")
def observation() -> Observation:
    """A short run of one detector."""
    starts = T0 + np.arange(0.0, DAYS * DAY, TSFT)
    return Observation(
        {"H1": starts}, tsft=TSFT, t_ref=float(np.mean(starts + TSFT / 2.0))
    )


@pytest.fixture(scope="module")
def tiling(observation: Observation) -> LatticeTiling:
    """One tiling from the metric over the run."""
    return LatticeTiling.from_metric(
        BOX,
        observation=observation,
        mismatch=MISMATCH,
        n_ap_blocks=N_AP_BLOCKS,
        window=Window(),
    )


@pytest.fixture(scope="module")
def bank(tiling: LatticeTiling, observation: Observation) -> TemplateBank:
    """The bank over the narrow band and the whole sky."""
    return TemplateBank.over_f0_interval(
        tiling,
        f0_low=F0_LOW,
        f0_high=F0_HIGH,
        velocity=observation.velocity,
        window=Window(),
    )


def test_curvatures_are_the_top_of_the_box(
    tiling: LatticeTiling, bank: TemplateBank
):
    """The bank's steps are the A5* diagonal under the top of the box's curvatures.

    Each ap block's omega^2 curvature scales as ap^2.
    """
    top = tiling.ap_blocks[-1]
    generator = a5star_generator()
    assert len(tiling.ap_blocks) == N_AP_BLOCKS
    assert top.ap_high == BOX.ap[1]
    assert bank.f0_step == pytest.approx(
        np.sqrt(MISMATCH / tiling.g_f0f0) * generator[0, 0]
    )
    assert bank.modulation_depth_step == pytest.approx(
        np.sqrt(MISMATCH / tiling.g_dd) * generator[1, 1]
    )
    for ap_block in tiling.ap_blocks:
        assert ap_block.g_ww / ap_block.ap_high**2 == pytest.approx(
            top.g_ww / top.ap_high**2
        )
    assert tiling.f0_reference == BOX.f0[1]


def test_record_round_trips(tiling: LatticeTiling):
    """to_dict then from_dict is the same tiling, the prior written in its own units."""
    record = tiling.to_dict()
    assert record["lattice"] == "a5star"
    assert record["prior"]["period"] == list(BOX.period)
    assert LatticeTiling.from_dict(record) == tiling


def test_a_point_sky_gives_the_point(
    tiling: LatticeTiling, observation: Observation
):
    """A point sky gives that point alone, at the lattice's sky origin."""
    point_tiling = replace(
        tiling, prior=BOX._replace(alpha=(1.2, 1.2), delta=(0.3, 0.3))
    )
    bank = TemplateBank.over_f0_interval(
        point_tiling,
        f0_low=F0_LOW,
        f0_high=F0_HIGH,
        velocity=observation.velocity,
        window=Window(),
    )
    assert bank.alpha.tolist() == [1.2]
    assert bank.delta.tolist() == [0.3]
    assert bank.sky_lattice_idxs.tolist() == [[0, 0]]


def test_a_foreign_record_is_rejected(tiling: LatticeTiling):
    """A record another tiling wrote must not be read as this one."""
    with pytest.raises(ValueError, match="rerun jpec search-setup"):
        LatticeTiling.from_dict({**tiling.to_dict(), "lattice": "bcc"})


def test_templates_stay_inside_the_seam_margin(tiling: LatticeTiling, bank):
    """In-range templates overrun ap by at most one modulation-depth step.

    The ap is that of each sky point's shifted depth and omega^2, at every class.
    """
    ap_low, ap_high = tiling.prior.ap
    worst = 0.0
    for omega2_class, omega2_fraction in enumerate(bank.omega2_shift_fractions):
        omega = np.sqrt(bank.omega2 + omega2_fraction * bank.omega2_step)
        for depth_class, depth_fraction in enumerate(bank.depth_shift_fractions):
            ap_in_range = bank.ap_in_range(
                omega2_class, np.full(bank.omega2.size, depth_class)
            )
            assert ap_in_range.any()
            ap = ap_from_modulation_depth(
                bank.modulation_depth[None, :]
                + depth_fraction * bank.modulation_depth_step,
                f0=bank.f0_centre,
                omega=omega[:, None],
            )
            step_ap = np.broadcast_to(
                ap_from_modulation_depth(
                    np.asarray(bank.modulation_depth_step),
                    f0=bank.f0_centre,
                    omega=omega[:, None],
                ),
                ap.shape,
            )
            excess = np.maximum(np.maximum(ap_low - ap, ap - ap_high), 0.0)
            worst = max(
                worst, float(np.max(excess[ap_in_range] / step_ap[ap_in_range]))
            )
    assert worst <= 1.0 + 1e-12


def test_every_template_is_a_lattice_point(
    tiling: LatticeTiling, bank: TemplateBank, observation: Observation
):
    """parameters() of any indices is `origin + T n` for integer `n`.

    The sky points inside the disc sit at the sky rows of `T` times their indices;
    each ap block's `T` is the first's with `T[2, 2:]` scaled by its omega^2 step.
    """
    generator = bank.generator
    _, sky_eigenvectors = sky_metric(
        observation.velocity, f0=bank.f0_centre, g_f0f0=tiling.g_f0f0
    )
    sky_x = (
        np.asarray(sky_direction(jnp.asarray(bank.alpha), jnp.asarray(bank.delta))).T
        @ sky_eigenvectors[:, :2]
    )
    in_disc = (sky_x**2).sum(axis=1) < 1.0 - 1e-9
    assert in_disc.any()
    np.testing.assert_allclose(
        sky_x[in_disc],
        bank.sky_lattice_idxs[in_disc] @ generator[3:, 3:].T,
        atol=1e-12,
    )

    # the origin's depth, from any base depth and its lattice index
    modulation_depth_low = (
        bank.modulation_depth[0]
        - bank.modulation_depth_step * bank.modulation_depth_lattice_idxs[0]
    )
    rng = np.random.default_rng(0)
    for sky_idx in rng.integers(0, bank.alpha.size, N_SAMPLED_SKY_POINTS):
        sky_lattice_idxs = bank.sky_lattice_idxs[sky_idx]
        for f0_idx, period_idx, modulation_depth_idx in zip(
            rng.integers(0, bank.f0.size, N_SAMPLED_TEMPLATES),
            rng.integers(0, bank.omega2.size, N_SAMPLED_TEMPLATES),
            rng.integers(0, bank.modulation_depth.size, N_SAMPLED_TEMPLATES),
            strict=True,
        ):
            template = bank.parameters(
                sky_idx=int(sky_idx),
                f0_idx=int(f0_idx),
                period_idx=int(period_idx),
                modulation_depth_idx=int(modulation_depth_idx),
                phase_idx=0,
                n_phi=8,
            )
            block_generator = generator.copy()
            block_generator[2, 2:] *= bank.omega2_step[period_idx] / generator[2, 2]
            omega2_low = (
                bank.omega2[period_idx]
                - bank.omega2_step[period_idx] * bank.omega2_lattice_idxs[period_idx]
            )
            # theta - origin on the orbital axes, then n by back substitution
            offset = np.array(
                [
                    template.f0 - F0_LOW,
                    float(template.modulation_depth) - modulation_depth_low,
                    float(template.omega) ** 2 - omega2_low,
                ]
            )
            idxs = np.zeros(3)
            for axis in (2, 1, 0):
                idxs[axis] = (
                    offset[axis]
                    - block_generator[axis, axis + 1 : 3]
                    @ np.rint(idxs[axis + 1 :])
                    - block_generator[axis, 3:] @ sky_lattice_idxs
                ) / block_generator[axis, axis]
            np.testing.assert_allclose(idxs, np.rint(idxs), atol=1e-6)


def test_around_point_is_a_subset_of_the_full_bank(
    tiling: LatticeTiling, observation: Observation
):
    """The bank around a source holds the full bank's templates at the same indices."""
    f0_low, f0_high = tiling.prior.f0
    full_bank = TemplateBank.over_f0_interval(
        tiling,
        f0_low=f0_low,
        f0_high=f0_high,
        velocity=observation.velocity,
        window=Window(),
    )
    around_bank = TemplateBank.around_point(
        tiling,
        SOURCE,
        velocity=observation.velocity,
        window=Window(),
        n_sky_neighbours=N_SKY_NEIGHBOURS,
        n_steps=N_STEPS,
    )
    assert around_bank.alpha.size == N_SKY_NEIGHBOURS
    assert around_bank.f0.size == 2 * N_STEPS + 1
    # each subset point's index in the full bank, by its lattice index
    full_f0_idxs = np.searchsorted(
        full_bank.f0_lattice_idxs, around_bank.f0_lattice_idxs
    )
    np.testing.assert_array_equal(
        full_bank.f0_lattice_idxs[full_f0_idxs], around_bank.f0_lattice_idxs
    )
    full_modulation_depth_idxs = np.searchsorted(
        full_bank.modulation_depth_lattice_idxs,
        around_bank.modulation_depth_lattice_idxs,
    )
    full_period_idxs = [
        int(
            np.flatnonzero(
                (full_bank.omega2_lattice_idxs == omega2_idx)
                & (full_bank.ap_block_idxs == ap_block_idx)
            )[0]
        )
        for omega2_idx, ap_block_idx in zip(
            around_bank.omega2_lattice_idxs, around_bank.ap_block_idxs, strict=True
        )
    ]
    full_sky_idxs = [
        int(np.flatnonzero((full_bank.alpha == alpha) & (full_bank.delta == delta))[0])
        for alpha, delta in zip(around_bank.alpha, around_bank.delta, strict=True)
    ]
    for sky_idx, full_sky_idx in enumerate(full_sky_idxs):
        assert around_bank.omega2_class[sky_idx] == full_bank.omega2_class[full_sky_idx]
        for period_idx, full_period_idx in enumerate(full_period_idxs):
            for modulation_depth_idx, full_modulation_depth_idx in enumerate(
                full_modulation_depth_idxs
            ):
                for f0_idx, full_f0_idx in enumerate(full_f0_idxs):
                    assert around_bank.parameters(
                        sky_idx=sky_idx,
                        f0_idx=f0_idx,
                        period_idx=period_idx,
                        modulation_depth_idx=modulation_depth_idx,
                        phase_idx=0,
                        n_phi=8,
                    ) == full_bank.parameters(
                        sky_idx=full_sky_idx,
                        f0_idx=int(full_f0_idx),
                        period_idx=full_period_idx,
                        modulation_depth_idx=int(full_modulation_depth_idx),
                        phase_idx=0,
                        n_phi=8,
                    )
