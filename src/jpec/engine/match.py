"""The match: templates built, scored against the data, and the top ones kept."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float, Int

from jpec.stat.matched_filter import statistic_over_phase
from jpec.waveform.harmonics import template_harmonics
from jpec.waveform.window import Window


@partial(jax.jit, static_argnames=("n_theta", "j_max", "window", "batch_size"))
def template_harmonics_at_depths(
    bin_offsets: Float[Array, " n_freq"],
    modulation_depth_bins: Float[Array, " n_modulation_depth"],
    *,
    n_theta: int,
    j_max: int,
    window: Window,
    batch_size: int,
) -> Complex[Array, "n_modulation_depth n_harmonic n_freq"]:
    """`tau[j, k]` at every modulation depth on one grid, `batch_size` at a time.

    Args:
        bin_offsets: Bin centres, as offsets from f0 [bins].
        modulation_depth_bins: The modulation depths [bins].
        n_theta: Number of orbital phases sampled.
        j_max: Highest harmonic kept.
        window: The window the SFTs were made with.
        batch_size: Number of modulation depths built at once, which caps the planes
            held.

    Returns:
        `tau[j, k]` for every modulation depth, in their order.
    """

    def at_depth(
        depth_bins: Float[Array, ""],
    ) -> Complex[Array, "n_harmonic n_freq"]:
        return template_harmonics(
            bin_offsets,
            modulation_depth_bins=depth_bins,
            n_theta=n_theta,
            j_max=j_max,
            window=window,
        )

    return jax.lax.map(at_depth, modulation_depth_bins, batch_size=batch_size)


@partial(jax.jit, static_argnames="n_pad")
def placed_template_harmonics(
    template_harmonics: Complex[
        Array, "n_sub_bin_offset n_depth_class n_modulation_depth n_harmonic n_freq"
    ],
    *,
    n_pad: int,
) -> Complex[
    Array, "n_placed n_depth_class n_modulation_depth n_harmonic n_padded_freq"
]:
    """Template harmonics at every sub-bin offset and placement in a widened window.

    Index `s` holds the templates of offset index `s % n_sub_bin_offset` placed
    `s // n_sub_bin_offset` bins into a window `n_pad` bins wider, zero elsewhere.

    Args:
        template_harmonics: `tau[j, k]` per sub-bin offset, depth class and
            modulation depth.
        n_pad: Number of bins the window is widened by.

    Returns:
        The placed templates, `n_sub_bin_offset * (n_pad + 1)` of them.
    """
    return jnp.concatenate(
        [
            jnp.pad(
                template_harmonics,
                ((0, 0), (0, 0), (0, 0), (0, 0), (placement, n_pad - placement)),
            )
            for placement in range(n_pad + 1)
        ]
    )


@jax.jit
def template_norms_squared(
    template_harmonics: Complex[
        Array, "n_sub_bin_offset n_depth_class n_modulation_depth n_harmonic n_freq"
    ],
) -> Float[Array, "n_sub_bin_offset n_depth_class n_modulation_depth n_harmonic"]:
    """Squared norm of each template over its first harmonics, harmonic by harmonic.

        norms_squared[j] = sum_{j' <= j} sum_k |tau[j', k]|^2

    Args:
        template_harmonics: `tau[j, k]` per sub-bin offset, depth class and
            modulation depth.

    Returns:
        The squared norm over harmonics 1 to `j + 1` at index `j`.
    """
    return jnp.cumsum(jnp.sum(jnp.abs(template_harmonics) ** 2, axis=-1), axis=-1)


@partial(
    jax.jit,
    static_argnames=("n_matched_modulation_depths", "n_phi", "f0_batch_size"),
)
def matched_filter_grid(
    data_harmonics: Complex[
        Array, "n_doppler_reference n_period n_harmonic n_transformed_bin"
    ],
    placed_harmonics: Complex[
        Array, "n_placed n_depth_class n_modulation_depth n_harmonic n_padded_freq"
    ],
    norms_squared: Float[
        Array, "n_sub_bin_offset n_depth_class n_modulation_depth n_harmonic"
    ],
    window_start_idxs: Int[Array, " n_f0"],
    group_f0_idxs: Int[Array, "n_group n_member"],
    group_doppler_reference_idxs: Int[Array, " n_group"],
    group_sub_bin_offset_idxs: Int[Array, " n_group"],
    modulation_depth_start_idx: Int[Array, ""],
    whole_bin_shifts: Int[Array, "n_period n_matched_modulation_depth"],
    sub_bin_shifts: Int[Array, "n_period n_matched_modulation_depth"],
    depth_classes: Int[Array, " n_period"],
    *,
    n_matched_modulation_depths: int,
    n_phi: int,
    f0_batch_size: int,
) -> tuple[
    Float[Array, "n_f0 n_period n_matched_modulation_depth"],
    Int[Array, "n_f0 n_period n_matched_modulation_depth"],
]:
    """Matched filter on every (f0, period, modulation depth), maximized over phase.

    Args:
        data_harmonics: `F[j, k]` at each period over the transformed bins, per
            Doppler reference.
        placed_harmonics: placed_template_harmonics() of the templates at every
            modulation depth, per sub-bin offset and depth class.
        norms_squared: template_norms_squared() of the same templates.
        window_start_idxs: First bin of each f0 point's template window, among its
            reference's transformed bins.
        group_f0_idxs: The f0 points of each (reference, offset) group.
        group_doppler_reference_idxs: The reference each group reads.
        group_sub_bin_offset_idxs: The sub-bin offset each group takes.
        modulation_depth_start_idx: First modulation depth matched.
        whole_bin_shifts: Whole bins each (period, modulation depth) moves the f0
            grid by.
        sub_bin_shifts: Sub-bin offsets each (period, modulation depth) moves the f0
            grid by.
        depth_classes: The depth class of each period.
        n_matched_modulation_depths: Number of modulation depths matched.
        n_phi: Number of orbital phases the statistic is evaluated at.
        f0_batch_size: Number of f0 points matched at once.

    Returns:
        The maximum of `S` over the orbital phase and the phase index it sits at,
            per f0 point.
    """
    n_periods, n_harmonics = data_harmonics.shape[1], data_harmonics.shape[2]
    n_sub_bin_offsets = norms_squared.shape[0]
    n_padded_bins = placed_harmonics.shape[-1]
    depth_idxs = modulation_depth_start_idx + jnp.arange(n_matched_modulation_depths)

    def window_data_harmonics(
        reference_idx: Int[Array, ""], window_start_idx: Int[Array, ""]
    ) -> Complex[Array, "n_period n_harmonic n_padded_freq"]:
        return jax.lax.dynamic_slice(
            data_harmonics[reference_idx],
            (0, 0, window_start_idx),
            (n_periods, n_harmonics, n_padded_bins),
        )

    def placed_template(
        placed_idx: Int[Array, ""],
        depth_class: Int[Array, ""],
        depth_idx: Int[Array, ""],
    ) -> Complex[Array, "n_harmonic n_padded_freq"]:
        return jax.lax.dynamic_slice(
            placed_harmonics,
            jnp.stack([placed_idx, depth_class, depth_idx, 0, 0]),
            (1, 1, 1, n_harmonics, n_padded_bins),
        )[0, 0, 0]

    # the inner vmap runs over the modulation depths, the outer over the periods
    period_templates = jax.vmap(
        jax.vmap(placed_template, in_axes=(0, None, 0)), in_axes=(0, 0, None)
    )

    def match_group(
        group: tuple[Int[Array, " n_member"], Int[Array, ""], Int[Array, ""]],
    ) -> tuple[
        Float[Array, "n_member n_period n_matched_modulation_depth"],
        Int[Array, "n_member n_period n_matched_modulation_depth"],
    ]:
        f0_idxs, reference_idx, offset_idx = group
        member_data_harmonics = jax.vmap(window_data_harmonics, in_axes=(None, 0))(
            reference_idx, window_start_idxs[f0_idxs]
        )
        # a group's f0 points share one template per (period, modulation depth): at
        # offset index i moved by u sub-bin offsets and s whole bins, it is offset
        # (i + u) % n placed s + (i + u) // n bins into the widened window, which is
        # placed_harmonics[i + u + n s]
        placed_idxs = offset_idx + sub_bin_shifts + n_sub_bin_offsets * whole_bin_shifts
        templates = period_templates(placed_idxs, depth_classes, depth_idxs)
        norms = jnp.sqrt(
            norms_squared[
                placed_idxs % n_sub_bin_offsets,
                depth_classes[:, None],
                depth_idxs[None, :],
                n_harmonics - 1,
            ]
        )
        # c[j] of harmonic_overlap() at every (member, period, modulation depth)
        coefficients = (
            jnp.einsum("mpjk,pqjk->mpqj", member_data_harmonics, jnp.conj(templates))
            / norms[None, :, :, None]
        )
        statistic = jnp.vectorize(
            partial(statistic_over_phase, n_phi=n_phi), signature="(j)->(m)"
        )(coefficients)
        return jnp.max(statistic, axis=-1), jnp.argmax(statistic, axis=-1)

    n_members = group_f0_idxs.shape[1]
    # f0_batch_size rounded down to whole groups, at least one
    group_max_statistic, group_phase_idxs = jax.lax.map(
        match_group,
        (group_f0_idxs, group_doppler_reference_idxs, group_sub_bin_offset_idxs),
        batch_size=max(f0_batch_size // n_members, 1),
    )
    # each f0 point back in its own row; a repeated member writes the same values
    flat_f0_idxs = group_f0_idxs.ravel()
    grid_shape = (window_start_idxs.shape[0], *group_max_statistic.shape[2:])
    max_statistic = (
        jnp.zeros(grid_shape, group_max_statistic.dtype)
        .at[flat_f0_idxs]
        .set(group_max_statistic.reshape(-1, *grid_shape[1:]))
    )
    phase_idxs = (
        jnp.zeros(grid_shape, group_phase_idxs.dtype)
        .at[flat_f0_idxs]
        .set(group_phase_idxs.reshape(-1, *grid_shape[1:]))
    )
    return max_statistic, phase_idxs


@partial(jax.jit, static_argnames="n_top")
def top_k_templates(
    max_statistic: Float[Array, "n_f0 n_period n_matched_modulation_depth"],
    phase_idxs: Int[Array, "n_f0 n_period n_matched_modulation_depth"],
    ap_in_range: Bool[Array, "n_period n_matched_modulation_depth"],
    toplist_f0_idxs: Int[Array, "n_toplist n_member"],
    *,
    n_top: int,
) -> tuple[
    Float[Array, "n_toplist n_kept"],
    Int[Array, "n_toplist n_kept"],
    Int[Array, "n_toplist n_kept"],
]:
    """Each toplist's top in-range templates of a grid.

    Args:
        max_statistic: Maximum of `S` over the orbital phase at every template of
            the grid.
        phase_idxs: Index of the orbital phase each maximum sits at.
        ap_in_range: Whether each (period, modulation depth) has ap in range.
        toplist_f0_idxs: The f0 points each toplist covers.
        n_top: Number of templates kept per toplist.

    Returns:
        Per toplist, their statistic, top first and `-inf` past the in-range ones,
            their flat index into the grid, and their phase index.
    """
    n_f0, n_periods, n_modulation_depths = max_statistic.shape
    n_templates_per_f0 = n_periods * n_modulation_depths
    # top_k ranks NaN above every number, so a NaN statistic is dropped with the
    # out-of-box ones
    is_kept = ap_in_range[None] & ~jnp.isnan(max_statistic)
    ap_in_range_max_statistic = jnp.where(is_kept, max_statistic, -jnp.inf)
    # each toplist's f0 points, a padding point scoring -inf
    toplist_max_statistic = jnp.take(
        ap_in_range_max_statistic,
        toplist_f0_idxs,
        axis=0,
        mode="fill",
        fill_value=-jnp.inf,
    ).reshape(toplist_f0_idxs.shape[0], -1)
    # every template of the toplist when it holds fewer than n_top
    top_max_statistic, member_flat_idxs = jax.lax.top_k(
        toplist_max_statistic, min(n_top, toplist_max_statistic.shape[1])
    )
    f0_idxs = jnp.take_along_axis(
        toplist_f0_idxs, member_flat_idxs // n_templates_per_f0, axis=1
    )
    # a padding point's index lies past the grid and is clipped onto its last
    # template; its statistic is -inf
    flat_idxs = jnp.minimum(
        f0_idxs * n_templates_per_f0 + member_flat_idxs % n_templates_per_f0,
        n_f0 * n_templates_per_f0 - 1,
    )
    return top_max_statistic, flat_idxs, phase_idxs.ravel()[flat_idxs]
