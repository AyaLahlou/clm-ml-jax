"""clm_ml.processes.longwave, checked against what it must conserve."""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import longwave as lw  # noqa: E402
from clm_ml.processes.constants import sb  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

N_SLOTS = 6  # layer slots; the profiles below have N_SLOTS + 1 entries (index 0 the ground)
NCAN, NTOP, NBOT = 5, 3, 2  # Fortran-style counts: leafy layers 2..3, padding slot 6
EMLEAF = 0.98


def _profiles():
    """Plant area and diffuse transmittance of the ground and layer slots (leafy layers only)."""
    dpai = np.zeros(N_SLOTS + 1)
    td = np.zeros(N_SLOTS + 1)
    dpai[NBOT : NTOP + 1] = [1.2, 1.5]
    td[NBOT : NTOP + 1] = [0.4, 0.3]
    return dpai, td


def _layers():
    return PatchLayers(
        n_layers=jnp.asarray(NCAN), n_within=jnp.asarray(NTOP),
        first_leafy=jnp.asarray(NBOT - 1), dz=jnp.ones(N_SLOTS), zs=jnp.ones(N_SLOTS),
        zw=jnp.ones(N_SLOTS + 1),
    )  # fmt: skip


def _norman(tleaf, lwsky, tg):
    dpai, td = _profiles()
    fracsun = jnp.full(N_SLOTS, 0.4)
    return lw.norman(tleaf, fracsun, jnp.asarray(td[1:]), jnp.asarray(dpai[1:]), lwsky, tg,
                     _layers(), EMLEAF)  # fmt: skip


def test_norman_conserves_energy():
    tleaf = jnp.asarray(np.random.default_rng(1).uniform(285.0, 300.0, (N_SLOTS, 2)))
    f = _norman(tleaf, 330.0, 292.0)
    np.testing.assert_allclose(f.lwveg + f.lwsoi, 330.0 - f.lwup, rtol=1e-12)
    assert abs(float(f.error)) < 1e-9
    leafy = np.zeros(N_SLOTS, bool)
    leafy[NBOT - 1 : NTOP] = True
    assert np.all(np.asarray(f.lwleaf)[~leafy] == 0.0)
    np.testing.assert_array_equal(f.lwleaf[:, 0], f.lwleaf[:, 1])  # same for sun and shade


def test_isothermal_canopy_is_in_radiative_equilibrium():
    # Everything at T under a sky emitting sigma T^4: no net exchange (Kirchhoff's law)
    t = 290.0
    f = _norman(jnp.full((N_SLOTS, 2), t), sb * t**4, t)
    np.testing.assert_allclose(f.lwup, sb * t**4, rtol=1e-12)
    assert abs(float(f.lwveg)) < 1e-9 and abs(float(f.lwsoi)) < 1e-9
    np.testing.assert_allclose(np.asarray(f.lwleaf), 0.0, atol=1e-9)


def test_registry():
    assert lw.LONGWAVE_RADIATION == {"norman": lw.norman}
