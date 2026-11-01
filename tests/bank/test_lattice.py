import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree

from jpec.bank.lattice import (
    BCC_SUBGRID_OFFSETS,
    ApBlock,
    orbital_subgrid,
    sky_lattice,
)
from jpec.bank.metric import (
    freq_bin_curvature,
    physical_metric,
    project_out,
    sky_metric,
)
from jpec.constant import CURVATURE_POSITIONS, DAY
from jpec.detector.doppler import sky_direction
from jpec.detector.site import Detector
from jpec.params import Parameters
from jpec.waveform.window import Window

T0 = 1368975618.0
TSFT = 1024.0
DAYS = 60.0
F0, AP, PERIOD = 150.0, 5.0, 15.0 * DAY
MISMATCH_PER_AXIS = 0.04
WHOLE_SKY = (0.0, 2.0 * np.pi, -0.5 * np.pi, 0.5 * np.pi)


def test_sky_lattice_covers_the_sphere_within_the_mismatch():
    """Any sky position's nearest lattice template is within two axes' budget, on the sky metric it is laid on."""
    midpoints = np.arange(0.0, DAYS * DAY, TSFT) + TSFT / 2.0
    times = jnp.asarray(midpoints - midpoints.mean())
    velocity = Detector("H1").velocity(T0 + midpoints)
    reference = Parameters(f0=F0, ap=AP, period=PERIOD, phi_b=1.0, alpha=1.2, delta=0.3)
    metric = project_out(
        physical_metric(
            reference,
            times=times,
            velocity=jnp.asarray(velocity),
            tsft=TSFT,
            g_freq_bin=freq_bin_curvature(Window(), n_positions=CURVATURE_POSITIONS),
        )
    )
    g_f0f0 = metric[0, 0]

    alpha, delta = sky_lattice(
        velocity,
        f0=F0,
        g_f0f0=float(g_f0f0),
        mismatch=MISMATCH_PER_AXIS,
        box=WHOLE_SKY,
    )
    curvature, sky_eigenvectors = sky_metric(velocity, f0=F0, g_f0f0=float(g_f0f0))

    def scaled(directions: np.ndarray) -> np.ndarray:
        """Directions in the eigenbasis, scaled so squared distance is mismatch."""
        return (directions @ sky_eigenvectors) * np.sqrt(curvature)

    templates = np.asarray(sky_direction(jnp.asarray(alpha), jnp.asarray(delta))).T
    rng = np.random.default_rng(0)
    n = 20000
    signals = np.asarray(
        sky_direction(
            jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, n)),
            jnp.asarray(np.arcsin(rng.uniform(-1.0, 1.0, n))),
        )
    ).T
    distance, _ = KDTree(scaled(templates)).query(scaled(signals))
    assert np.max(distance**2) <= 2.0 * MISMATCH_PER_AXIS


def test_orbital_lattice_covers_band_edges_blocks_and_faces_within_the_budget():
    """Any point of two neighbouring f0 bands has an in-box template within the three orbital axes' budget.

    In cubic steps a template ``u`` away costs ``4 mu_axis |u|^2``, so the budget ``3 mu_axis`` is
    ``|u|^2 <= 3/4``. The omega^2 step is the lower band's, the larger of the two, so a template
    within the budget in its own band's step is within it in this one.
    """
    period_low, period_high = 7.0 * DAY, 15.0 * DAY
    omega_low, omega_high = 2.0 * np.pi / period_high, 2.0 * np.pi / period_low
    ap_edges = np.linspace(5.0, 15.0, 4)
    f0_step, modulation_depth_step = 2e-4, 1.0
    omega2_steps = (omega_high**2 - omega_low**2) / np.array([40.0, 50.0, 60.0])
    # orbital_subgrid reads only the ap range and the omega^2 step
    ap_blocks = [
        ApBlock(
            ap_low=float(low),
            ap_high=float(high),
            modulation_depth_step=modulation_depth_step,
            omega2_step=float(step),
            g_f0f0=0.0,
        )
        for low, high, step in zip(ap_edges[:-1], ap_edges[1:], omega2_steps)
    ]
    bands = ((F0, F0 + 0.01), (F0 + 0.01, F0 + 0.02))
    f0_reference = bands[1][1]

    # every in-box template of both bands and both subgrids, in (f0, modulation depth, omega^2)
    lattice = []
    for band in bands:
        for offset in BCC_SUBGRID_OFFSETS:
            band_orbital_subgrid = orbital_subgrid(
                *band,
                ap_low=5.0,
                ap_high=15.0,
                period_low=period_low,
                period_high=period_high,
                f0_step=f0_step,
                modulation_depth_step=modulation_depth_step,
                ap_blocks=ap_blocks,
                f0_reference=f0_reference,
                tsft=TSFT,
                offset=offset,
            )
            period_idxs, modulation_depth_idxs = np.nonzero(band_orbital_subgrid.in_box)
            pairs = np.column_stack(
                [
                    band_orbital_subgrid.modulation_depth_bins[modulation_depth_idxs],
                    (2.0 * np.pi / band_orbital_subgrid.period[period_idxs]) ** 2,
                ]
            )
            lattice.append(
                np.column_stack(
                    [
                        np.repeat(band_orbital_subgrid.f0, len(pairs)),
                        np.tile(pairs, (band_orbital_subgrid.f0.size, 1)),
                    ]
                )
            )
    lattice = np.concatenate(lattice)

    # signals over both bands, the whole box in ap and P: the outer f0 edges, the shared one,
    # the ap block boundaries and every face of the box are all inside it
    rng = np.random.default_rng(0)
    n = 20000
    f0 = rng.uniform(bands[0][0], bands[1][1], n)
    ap = rng.uniform(5.0, 15.0, n)
    omega = 2.0 * np.pi / rng.uniform(period_low, period_high, n)
    signals = np.column_stack([f0, f0 * ap * omega * TSFT, omega**2])
    ap_block_idx = np.digitize(ap, ap_edges[1:-1])

    lower_centre = 0.5 * sum(bands[0])
    for block_idx, ap_block in enumerate(ap_blocks):
        scale = np.array(
            [
                1.0 / f0_step,
                1.0 / modulation_depth_step,
                1.0 / (ap_block.omega2_step * f0_reference / lower_centre),
            ]
        )
        distance, _ = KDTree(lattice * scale).query(
            signals[ap_block_idx == block_idx] * scale
        )
        assert np.max(distance**2) <= 0.75
