"""clm_ml.processes.canopy (Phase 7.2, process 10) and the sub-step options
(``clm_ml.options.substep_options``).

The physics is checked against hand-computed canopy totals, the conservation of plant area and
the definition of the sub-step average and the sunlit/shaded merge.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.options import DEFAULT_OPTIONS, substep_options  # noqa: E402
from clm_ml.processes import canopy  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

N_SLOTS = 6  # layer slots; legacy arrays have N_SLOTS + 1 entries (index 0 unused/ground)
N_CAN = 4  # layers in use (legacy slots 1..4; 0-based 0..3)


def test_substep_options_refuse_unsupported_variants():
    o = substep_options(DEFAULT_OPTIONS._replace(met_type=3))
    assert o.integrator == "rk4" and o.interpolation == "three_point"
    with pytest.raises(NotImplementedError):
        substep_options(DEFAULT_OPTIONS._replace(met_type=3, light_type=1))
    with pytest.raises(NotImplementedError):
        substep_options(DEFAULT_OPTIONS._replace(met_type=3, flux_profile_type=0))
    with pytest.raises(ValueError):
        substep_options(DEFAULT_OPTIONS._replace(met_type=2))


def test_n_stages():
    assert canopy.n_stages("euler") == 1
    assert canopy.n_stages("rk4") == 5


def _layers(n_layers=N_CAN, n_within=3, first_leafy=0):
    return PatchLayers(
        n_layers=jnp.asarray(n_layers),
        n_within=jnp.asarray(n_within),
        first_leafy=jnp.asarray(first_leafy),
        dz=jnp.ones(N_SLOTS),
        zs=jnp.arange(N_SLOTS, dtype=float),
        zw=jnp.arange(N_SLOTS + 1, dtype=float),
    )


def test_plant_area_conserves_lai_and_sai_and_keeps_the_floor():
    lay = _layers()
    dlai_frac = jnp.array([0.001, 0.6, 0.399, 0.0, 9.0, 9.0])  # padding holds junk
    dsai_frac = jnp.array([0.0, 0.5, 0.5, 0.0, 9.0, 9.0])
    r = canopy.plant_area(lay, dlai_frac, dsai_frac, 2.0, 0.5, 0.05)
    np.testing.assert_allclose(float(r.dpai[0]), 0.05)  # thin leafy layer raised to the floor
    np.testing.assert_allclose(float(r.sai_floor), 0.05 - 0.002)
    np.testing.assert_allclose(float(r.sai), 0.5 + 0.05 - 0.002)
    assert abs(float(r.error)) < 1e-15
    assert float(r.dpai[3]) == 0.0  # top layer (above the canopy) leafless
    # layers at or above the floor unchanged
    np.testing.assert_array_equal(np.asarray(r.dlai[1:3]), np.asarray(dlai_frac[1:3]) * 2.0)


def test_soil_surface_humidity():
    assert float(canopy.soil_surface_humidity(0.0, 290.0)) == 1.0
    assert 0.99 < float(canopy.soil_surface_humidity(-1.0e4, 290.0)) < 1.0


def _random_fluxes(rng, scale=1.0):
    def a(*shape):
        return jnp.asarray(rng.uniform(0.5, 1.5, shape) * scale)

    shapes = {
        "swsoi": (2,), "swskyb": (2,), "swskyd": (2,), "shair": (N_SLOTS,),
        "etair": (N_SLOTS,), "stair": (N_SLOTS,), "mflx": (N_SLOTS,), "kc_eddy": (N_SLOTS,),
        "gac": (N_SLOTS,), "swupw": (N_SLOTS + 1, 2), "swdwn": (N_SLOTS + 1, 2),
        "swbeam": (N_SLOTS + 1, 2), "lwupw": (N_SLOTS + 1,), "lwdwn": (N_SLOTS + 1,),
        "swleaf": (N_SLOTS, 2, 2),
    }  # fmt: skip
    return canopy.SubstepFluxes(
        **{
            n: a(*shapes.get(n, (N_SLOTS, 2) if n in _LEAF_NAMES else ()))
            for n in canopy.SubstepFluxes._fields
        }
    )


_LEAF_NAMES = {"lwleaf", "rnleaf", "shleaf", "lhleaf", "trleaf", "evleaf", "stleaf", "anet",
               "agross", "gs"}  # fmt: skip


def test_average_over_substeps():
    rng = np.random.default_rng(1)
    steps = [_random_fluxes(rng) for _ in range(6)]
    sums = canopy.zero_sums(steps[0])
    for s in steps:
        sums = canopy.accumulate(sums, s)
    mean = canopy.average(sums, 6)
    for name in canopy.SubstepFluxes._fields:
        expect = np.mean([np.asarray(getattr(s, name)) for s in steps], axis=0)
        np.testing.assert_allclose(np.asarray(getattr(mean, name)), expect, rtol=1e-14)


def test_canopy_totals_by_hand():
    rng = np.random.default_rng(2)
    lay = _layers()
    m = _random_fluxes(rng)
    dpai = jnp.array([0.3, 0.5, 0.4, 0.0, 1e36, 1e36])
    fracsun = jnp.linspace(0.2, 0.7, N_SLOTS)
    leaves = canopy.CanopyLeaves(
        dpai=dpai,
        fracsun=fracsun,
        fwet=jnp.full(N_SLOTS, 0.2),
        fdry=jnp.full(N_SLOTS, 0.6),
        wind=jnp.linspace(1.0, 3.0, N_SLOTS),
        tair=jnp.linspace(290.0, 292.0, N_SLOTS),
        vcmax25=jnp.full(N_SLOTS, 50.0),
        tleaf=jnp.stack([jnp.full(N_SLOTS, 295.0), jnp.full(N_SLOTS, 291.0)], axis=1),
        lwp=jnp.stack([jnp.full(N_SLOTS, -2.5), jnp.full(N_SLOTS, -1.0)], axis=1),
        vcmax25_leaf=jnp.full((N_SLOTS, 2), 50.0),
    )
    t = canopy.canopy_totals(lay, m, leaves, 0.9, 0.3, flux_profile_type=1)
    d = np.array([0.3, 0.5, 0.4])
    fs = np.asarray(fracsun)[:3]
    sh = np.asarray(m.shleaf)[:3]
    np.testing.assert_allclose(float(t.shveg), np.sum((sh[:, 0] * fs + sh[:, 1] * (1 - fs)) * d))
    np.testing.assert_allclose(float(t.shvegsun), np.sum(sh[:, 0] * fs * d))
    np.testing.assert_allclose(float(t.laisun + t.laisha), d.sum())
    np.testing.assert_allclose(float(t.vcmax25veg), 50.0 * d.sum())
    np.testing.assert_allclose(float(t.tlvegsun), 295.0)
    np.testing.assert_allclose(float(t.tlvegsha), 291.0)
    # fracgreen = fdry / (1 - fwet) = 0.75
    ag = np.asarray(m.agross)[:3]
    np.testing.assert_allclose(float(t.gppvegsun), np.sum(ag[:, 0] * fs * d * 0.75))
    # the canopy's fluxes at the top layer in use, its albedo at the canopy top (ntop)
    assert float(t.shflx) == float(m.shair[N_CAN - 1])
    assert float(t.lhflx) == float(m.lhtop)
    inc = np.asarray(m.swskyb) + np.asarray(m.swskyd)
    np.testing.assert_allclose(np.asarray(t.albcan), np.asarray(m.swupw)[3] / inc)
    # sunlit/shaded mean of the leaf water potential in leafy layers
    lp = -2.5 * fs + -1.0 * (1 - fs)
    np.testing.assert_allclose(np.asarray(t.lwp_mean)[:3], lp)
    np.testing.assert_allclose(float(t.fracminlwp), np.sum(np.where(lp <= -2.0, d, 0)) / 1.2)
    t0 = canopy.canopy_totals(lay, m, leaves, 0.9, 0.3, flux_profile_type=0)
    np.testing.assert_allclose(float(t0.shflx), float(t.shveg + m.shsoi))
    with pytest.raises(ValueError):
        canopy.canopy_totals(lay, m, leaves, 0.9, 0.3, flux_profile_type=2)


def test_merge_sun_shade():
    lay = _layers()
    dpai = jnp.array([0.3, 0.0, 0.4, 0.0, 1e36, 1e36])
    fracsun = jnp.full(N_SLOTS, 0.25)
    tleaf = jnp.stack([jnp.full(N_SLOTS, 300.0), jnp.full(N_SLOTS, 296.0)], axis=1)
    m = canopy.merge_sun_shade(lay, dpai, fracsun, tleaf, tleaf - 300.0)
    merged = np.asarray(m.tleaf)
    np.testing.assert_allclose(merged[[0, 2]], 297.0)  # layers with plant area
    np.testing.assert_array_equal(merged[[1, 3, 4, 5]], np.asarray(tleaf)[[1, 3, 4, 5]])
    np.testing.assert_array_equal(np.asarray(m.tleaf_hist), np.asarray(tleaf))
    np.testing.assert_allclose(np.asarray(m.lwp)[0], -3.0)


def test_derivative_rules_are_substep_options():
    """The derivative rules of decision 124: the water stress stabilized, the Obukhov length exact
    unless asked for."""
    o = substep_options(DEFAULT_OPTIONS._replace(met_type=3))
    assert (o.water_stress_derivative, o.obukhov_derivative) == ("stabilized", "exact")
    o = substep_options(
        DEFAULT_OPTIONS._replace(met_type=3),
        water_stress_derivative="exact",
        obukhov_derivative="stabilized",
    )
    assert (o.water_stress_derivative, o.obukhov_derivative) == ("exact", "stabilized")
