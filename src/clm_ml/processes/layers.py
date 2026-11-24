"""Helpers for computing on every canopy layer slot of one patch (``clm_ml.state.PatchLayers``).

Profiles have ``max_layers`` slots on axis 0 (0-based). Processes compute on every slot and
keep results where a mask is true, so one compiled program serves any layer counts. Padding
slots first take the top layer's inputs (:func:`pad_with_top`): the top layer is above the
canopy and leafless, so padding computes a harmless leafless layer instead of arithmetic on
fill values (which would give infinities and NaN gradients even though the results are
discarded).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from clm_ml.state import PatchLayers


def expand(mask: Array, x: Array) -> Array:
    """``mask`` (layer,) broadcast against ``x`` (layer, ...)."""
    return mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim))


def keep(mask: Array, new: Array, old) -> Array:
    """``new`` where ``mask`` (layer,) is true, ``old`` elsewhere (layers on axis 0)."""
    return jnp.where(expand(mask, new), new, old)


def pad_with_top(layers: PatchLayers, x: Array) -> Array:
    """``x`` (layers on axis 0) with the padding slots given the top layer's value."""
    top = jnp.take(x, layers.n_layers - 1, axis=0)
    return keep(layers.in_use, x, top)


def leafy_or_top_leafy(layers: PatchLayers, x: Array) -> Array:
    """``x`` (layers on axis 0) with every slot outside the leafy layers given the top leafy
    layer's value (inputs of layer sweeps whose non-leafy results are discarded)."""
    return keep(layers.leafy, x, jnp.take(x, layers.n_within - 1, axis=0))
