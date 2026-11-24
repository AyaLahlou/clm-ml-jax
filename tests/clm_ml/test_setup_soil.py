"""Soil properties at setup (``clm_ml.setup.soil``)."""

import jax
import numpy as np
import pytest

from clm_ml.setup import soil, tables

jax.config.update("jax_enable_x64", True)
NAN = float("nan")


@pytest.mark.parametrize("k, name", list(enumerate(tables.SOIL_TEXTURES)))
def test_texture_class_takes_its_own_row(k, name):
    """A texture class gets its own sand, clay and Clapp-Hornberger values
    (SoilTexMod.F90; Fortran's class number m indexes its 1-based tables), every class
    including the last (clay)."""
    z, dz, zi = soil.soil_layers()
    # no organic matter: the mineral values unchanged
    props = soil.soil_properties(np.array([NAN]), np.array([NAN]), np.array([0.0]), [name], z,
                                 soil.NLEVSOI)  # fmt: skip
    np.testing.assert_allclose(props["watsat"][0, 0], tables.WATSAT_TEX[k], rtol=1e-15)
    np.testing.assert_allclose(props["sucsat"][0, 0], -tables.SMPSAT_TEX[k], rtol=1e-15)
    np.testing.assert_allclose(props["bsw"][0, 0], tables.BSW_TEX[k], rtol=1e-15)
    np.testing.assert_allclose(props["hksat"][0, 0], tables.HKSAT_TEX[k] / 60.0, rtol=1e-15)


def test_percentages_do_not_use_the_class():
    """With sand and clay given, the hydraulic values come from them (CLM5's formulas), as in
    Fortran; the class is not used."""
    z, dz, zi = soil.soil_layers()
    args = (np.array([10.0]), np.array([35.0]), np.array([0.0]))
    a = soil.soil_properties(*args, ["sand"], z, soil.NLEVSOI)
    b = soil.soil_properties(*args, ["clay"], z, soil.NLEVSOI)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_allclose(a["watsat"][0, 0], 0.489 - 0.00126 * 10.0, rtol=1e-15)


def test_unknown_class():
    z, dz, zi = soil.soil_layers()
    with pytest.raises(ValueError, match="not found"):
        soil.soil_properties(np.array([NAN]), np.array([NAN]), np.array([0.0]), ["peat"], z,
                             soil.NLEVSOI)  # fmt: skip
