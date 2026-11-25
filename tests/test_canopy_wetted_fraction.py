"""Production-path checks for the Fortran wet- and dry-area equations."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from multilayer_canopy.MLCanopyWaterMod import (
    CanopyWettedFraction,
    dewmx,
    fwet_exponent,
    maximum_leaf_wetted_fraction,
)


class CanopyProfiles(NamedTuple):
    dpai_profile: jnp.ndarray
    h2ocan_profile: jnp.ndarray
    dlai_profile: jnp.ndarray
    fwet_profile: jnp.ndarray
    fdry_profile: jnp.ndarray


def test_wetted_fraction_preserves_fortran_zero_and_positive_water():
    # First entry is the surface. The layers are dry, negative round-off,
    # barely wet, saturated, and inactive, respectively.
    area = jnp.asarray([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    water_ratio = jnp.asarray([[0.0, 0.0, -1e-12, 1e-35, 1.0, 1.0]])
    profiles = CanopyProfiles(
        area, water_ratio * dewmx, 0.5 * area, jnp.zeros_like(area), jnp.zeros_like(area)
    )

    result = CanopyWettedFraction(1, (0,), profiles)

    np.testing.assert_array_equal(result.fwet_profile[0, [1, 2, 5]], 0.0)
    np.testing.assert_array_equal(result.fdry_profile[0, [1, 2, 5]], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(
        result.fwet_profile[0, 3:5],
        [1e-35 ** fwet_exponent, maximum_leaf_wetted_fraction],
        rtol=1e-6,
        atol=0.0,
    )
