import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import KDTree

from jpec.bank.lattice import (
    a5star_generator,
    cubic_step,
    shift_fractions,
    sky_lattice,
    split_shift,
    triangular_generator,
)
from jpec.bank.metric import sky_metric
from jpec.bank.tiling import LatticeTiling
from jpec.constant import DAY
from jpec.detector.doppler import sky_direction
from jpec.detector.site import Observation
from jpec.params import Prior
from jpec.waveform.window import Window

T0 = 1368975618.0
TSFT = 1024.0
DAYS = 60.0
F0 = 150.0
MISMATCH = 0.2
WHOLE_SKY_ALPHA = (0.0, 2.0 * np.pi)
WHOLE_SKY_DELTA = (-0.5 * np.pi, 0.5 * np.pi)
# lattice indices each side of the origin enumerated for the covering check
N_COVERING_STEPS = 4


def test_a5star_offsets_are_exact_fractions_of_each_step():
    """R is upper triangular, each row's offsets fifths to halves of its step."""
    generator = a5star_generator()
    assert np.allclose(np.tril(generator, k=-1), 0.0)
    assert (np.diag(generator) > 0.0).all()
    np.testing.assert_allclose(
        generator[0, 1:] / generator[0, 0], [0.8, 0.6, 0.4, 0.2], atol=1e-12
    )
    np.testing.assert_allclose(
        generator[1, 2:] / generator[1, 1], [0.75, 0.5, 0.25], atol=1e-12
    )
    np.testing.assert_allclose(
        generator[2, 3:] / generator[2, 2], [2.0 / 3.0, 1.0 / 3.0], atol=1e-12
    )
    # the sky block is hexagonal
    assert generator[3, 4] / generator[3, 3] == pytest.approx(0.5)
    assert generator[4, 4] / generator[3, 3] == pytest.approx(np.sqrt(3.0) / 2.0)


def test_a5star_covers_at_radius_one():
    """Every point of a cell lies within distance 1 of a lattice point, some near 1."""
    generator = a5star_generator()
    idxs = np.array(
        list(
            itertools.product(
                range(-N_COVERING_STEPS, N_COVERING_STEPS + 1), repeat=5
            )
        )
    )
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, (20000, 5)) @ generator.T
    distance, _ = KDTree(idxs @ generator.T).query(points)
    assert distance.max() <= 1.0 + 1e-9
    assert distance.max() > 0.95
    assert np.mean(distance**2) == pytest.approx(0.55, abs=0.02)


def test_shift_fractions_are_the_rows_classes():
    """Rows 0 to 3 give 5, 4, 3 and 2 shift fractions, evenly spaced from 0."""
    generator = a5star_generator()
    for row, n_fractions in zip(range(4), (5, 4, 3, 2), strict=True):
        fractions = shift_fractions(generator[row, row + 1 :] / generator[row, row])
        np.testing.assert_allclose(
            fractions, np.arange(n_fractions) / n_fractions, atol=1e-12
        )


def test_a_shift_splits_into_its_fraction_and_whole_steps():
    """A row's shift at integer indices is one of its fractions plus whole steps."""
    generator = a5star_generator()
    rng = np.random.default_rng(0)
    for row in range(4):
        ratios = generator[row, row + 1 :] / generator[row, row]
        fractions = shift_fractions(ratios)
        shift = rng.integers(-50, 51, (1000, ratios.size)) @ ratios
        fraction_idxs, whole_steps = split_shift(shift, fractions)
        np.testing.assert_allclose(
            fractions[fraction_idxs] + whole_steps, shift, atol=1e-9
        )


def test_triangular_generator_whitens_to_the_scaled_generator():
    """T is upper triangular and `T^T g T = mu R^T R`, so it covers `g` to `mu`."""
    rng = np.random.default_rng(0)
    root = rng.normal(size=(5, 5))
    metric = root @ root.T + 5.0 * np.eye(5)
    generator = a5star_generator()
    triangular = triangular_generator(metric, MISMATCH, generator)
    assert np.allclose(np.tril(triangular, k=-1), 0.0)
    np.testing.assert_allclose(
        triangular.T @ metric @ triangular,
        MISMATCH * generator.T @ generator,
        atol=1e-12,
    )


def test_sky_lattice_covers_the_sphere_within_its_share_of_the_mismatch():
    """Every sky point is within `mu R_33^2 / 3`, the hexagonal sky block's covering.

    The mismatch is on the sky metric, f0 projected out, which is the Schur
    complement the sky block of `T` covers.
    """
    starts = T0 + np.arange(0.0, DAYS * DAY, TSFT)
    observation = Observation(
        {"H1": starts}, tsft=TSFT, t_ref=float(np.mean(starts + TSFT / 2.0))
    )
    velocity = observation.velocity
    prior = Prior(
        f0=(F0 - 0.01, F0),
        ap=(5.0, 5.0),
        period=(15.0 * DAY, 15.0 * DAY),
        phi_b=(0.0, 2.0 * np.pi),
        alpha=(0.0, 2.0 * np.pi),
        delta=(-0.5 * np.pi, 0.5 * np.pi),
    )
    tiling = LatticeTiling.from_metric(
        prior,
        observation=observation,
        mismatch=MISMATCH,
        n_ap_blocks=1,
        window=Window(),
    )
    curvature, sky_eigenvectors = sky_metric(velocity, f0=F0, g_f0f0=tiling.g_f0f0)
    metrics = tiling.lattice_metric(velocity, f0=F0, sky_eigenvectors=sky_eigenvectors)
    generator = a5star_generator()
    triangular = triangular_generator(metrics[0], MISMATCH, generator)
    alpha, delta, _ = sky_lattice(
        triangular[3:, 3:],
        sky_eigenvectors=sky_eigenvectors,
        cubic_steps=cubic_step(curvature[:2], tiling.mismatch_per_axis),
        alpha_interval=WHOLE_SKY_ALPHA,
        delta_interval=WHOLE_SKY_DELTA,
        margin=0.0,
    )

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
    assert np.max(distance**2) <= MISMATCH * generator[3, 3] ** 2 / 3.0
