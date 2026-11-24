"""The shared type alias, and the axis names shape annotations use."""

from jaxtyping import Array, Float

Scalar = int | float | Float[Array, ""]
"""One real number that may be traced: a Python number or a 0-d float array.
Not an integer array, which `jax.grad` cannot differentiate, and not a
length-one array.
"""

# Axis names used in shape annotations, so a shape reads as what it indexes:
#   n_sky       sky points; one sky position alone is a length-one array
#   n_sft       SFTs, the time axis of a plane or of the data's harmonics
#   n_time      arbitrary sample times, as a track function takes them
#   n_freq      frequency bins
#   n_harmonic  orbital harmonics j, from 1
#   n_theta     orbital phases a template plane is sampled at
#   n_phi       trial orbital phases the statistic is evaluated at
#   n_modulation_depth  modulation depths of a bank, one template shape each
#   n_window_modulation_depth  modulation depths one period batch is matched against
#   n_matched_modulation_depth  modulation depths one matched_filter_grid() call matches
#   n_sub_bin_offset  sub-bin offsets a bank's templates are built at
#   n_placed    placed templates: every sub-bin offset at every placement in a
#               window widened for the f0 shifts
#   n_padded_freq  frequency bins of a template window widened for the f0 shifts
#   n_omega2_class  omega^2 shift fractions a bank's sky points take
#   n_depth_class  modulation-depth shift fractions a bank's periods take
#   n_shift     shift fractions one generator row gives
#   n_outer     a generator row's entries past its diagonal, one per outer axis
#   n_axis      axes of a lattice and its metric
#   n_ap_block  ap blocks of a tiling
#   n_batch_sky  sky points of one sky batch
#   n_template_batch  period batches of a bank's layout
#   n_f0        f0 points of a bank
#   n_period    trial orbital periods of a bank
#   n_top       templates of a grid kept on the device, top first
#   n_doppler_reference  frequencies a band is Doppler-corrected at, shared by
#                        the f0 points near each
#   n_band      frequency bins of the loaded band
#   n_transformed_bin  bins of the band the template windows read, the ones
#                      transformed
#   n_lattice_point  points of a sky lattice in its eigenframe, before they are
#                    cut to the box
#   length      samples within one SFT, for a window
