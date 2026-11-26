"""The shared type alias, and the axis names shape annotations use."""

from typing import Union

from jaxtyping import Array, Float

Scalar = Union[float, Float[Array, ""]]
"""One real number that may be traced: a Python number or a 0-d float array.
Not an integer array, which `jax.grad` cannot differentiate, and not a
length-one array.
"""

# Index names:
#   *_lattice_idxs  integer coordinates n_k of A5* lattice points (jpec.bank only)
#   *_idx, *_idxs   positions in an array: a sky point in the bank, an f0 point on
#                   the base grid, a bin
#   *_class         index of a shift fraction (omega2_class, depth_class)
#   *_kept          positions a subset keeps
