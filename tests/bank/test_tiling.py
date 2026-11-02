from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jpec.bank.lattice import BCC_SUBGRID_STEP
from jpec.bank.tiling import LatticeTiling
from jpec.constant import DAY
from jpec.detector.site import Detector
from jpec.params import Prior, ap_from_modulation_depth_bins
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
BAND = (150.0, 150.02)  # narrow enough to check template by template


@pytest.fixture(scope="module")
def tiling() -> LatticeTiling:
    """One tiling from the metric over a short run."""
    midpoints = np.arange(0.0, DAYS * DAY, TSFT) + TSFT / 2.0
    times = jnp.asarray(midpoints - midpoints.mean())
    velocity = jnp.asarray(Detector("H1").velocity(T0 + midpoints))
    return LatticeTiling.from_metric(
        BOX,
        times=times,
        velocity=velocity,
        tsft=TSFT,
        mismatch=MISMATCH,
        n_ap_blocks=N_AP_BLOCKS,
        window=Window(),
    )


def test_steps_are_the_top_of_the_box(tiling: LatticeTiling):
    """Steps are the top of the box's, each ap block's omega^2 step scaled as 1 / ap."""
    top = tiling.ap_blocks[-1]
    assert len(tiling.ap_blocks) == N_AP_BLOCKS
    assert top.ap_high == BOX.ap[1]
    assert tiling.g_f0f0 == top.g_f0f0
    assert tiling.f0_step == pytest.approx(
        2.0 * np.sqrt(tiling.mismatch_per_axis / tiling.g_f0f0)
    )
    assert tiling.modulation_depth_step == top.modulation_depth_step
    for ap_block in tiling.ap_blocks:
        assert ap_block.omega2_step * ap_block.ap_high == pytest.approx(
            top.omega2_step * top.ap_high
        )
    assert tiling.f0_reference == BOX.f0[1]


def test_record_round_trips(tiling: LatticeTiling):
    """to_dict then from_dict is the same tiling, the prior written in its own units."""
    record = tiling.to_dict()
    assert record["lattice"] == "bcc"
    assert record["prior"]["period"] == list(BOX.period)
    assert LatticeTiling.from_dict(record) == tiling


def test_a_point_sky_gives_the_point(tiling: LatticeTiling):
    """A point sky gives that point alone."""
    point_tiling = replace(
        tiling, prior=BOX._replace(alpha=(1.2, 1.2), delta=(0.3, 0.3))
    )
    midpoints = np.arange(0.0, DAYS * DAY, TSFT) + TSFT / 2.0
    velocity = Detector("H1").velocity(T0 + midpoints)
    alpha, delta = point_tiling.sky_points(velocity, f0=BOX.f0[1])
    assert alpha.tolist() == [1.2]
    assert delta.tolist() == [0.3]


def test_a_foreign_record_is_rejected(tiling: LatticeTiling):
    """A record another tiling wrote must not be read as this one."""
    with pytest.raises(ValueError, match="rerun jpec search-setup"):
        LatticeTiling.from_dict({**tiling.to_dict(), "lattice": "cubic"})


def test_templates_stay_inside_the_seam_margin(tiling: LatticeTiling):
    """In-box templates overrun ap by at most one subgrid modulation-depth step."""
    ap_low, ap_high = tiling.prior.ap
    f0_centre = 0.5 * (BAND[0] + BAND[1])
    worst = 0.0
    for orbital_subgrid in tiling.orbital_subgrids(*BAND):
        omega = 2.0 * np.pi / orbital_subgrid.period
        ap = np.asarray(
            ap_from_modulation_depth_bins(
                jnp.asarray(orbital_subgrid.modulation_depth_bins[None, :]),
                f0=jnp.asarray(f0_centre),
                omega=jnp.asarray(omega[:, None]),
                tsft=tiling.tsft,
            )
        )
        step_ap = np.broadcast_to(
            np.asarray(
                ap_from_modulation_depth_bins(
                    jnp.asarray(BCC_SUBGRID_STEP * tiling.modulation_depth_step),
                    f0=jnp.asarray(f0_centre),
                    omega=jnp.asarray(omega[:, None]),
                    tsft=tiling.tsft,
                )
            ),
            ap.shape,
        )
        excess = np.maximum(np.maximum(ap_low - ap, ap - ap_high), 0.0)[
            orbital_subgrid.in_box
        ]
        assert orbital_subgrid.in_box.any()
        worst = max(worst, float(np.max(excess / step_ap[orbital_subgrid.in_box])))
    assert worst <= 1.0
