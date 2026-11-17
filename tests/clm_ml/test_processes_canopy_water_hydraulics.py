"""clm_ml.processes.canopy_water and plant_hydraulics (Phase 7.2, process 5), checked against
what they must conserve."""

from collections import namedtuple

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import canopy_water, plant_hydraulics  # noqa: E402
from clm_ml.processes.constants import (  # noqa: E402
    dewmx,
    fwet_exponent,
    maximum_leaf_wetted_fraction,
    mmh2o,
)

N_SLOTS = 6  # layer slots; legacy arrays have N_SLOTS + 1 entries (index 0 unused)
N_SOI = 4  # soil levels; legacy arrays have N_SOI + 1 entries
DTIME = 300.0
Params = namedtuple(
    "Params",
    "plant_conductance plant_capacitance root_radius root_density root_resistivity sla_top",
)
PARAMS = Params(4.0, 2500.0, 0.29e-3, 0.31e6, 14.0, 0.03)


def _dpai():
    v = np.zeros(N_SLOTS)
    v[1:3] = [1.2, 0.9]
    return jnp.asarray(v)


def test_interception_conserves_water():
    dpai = _dpai()
    bef = jnp.where(dpai > 0, 0.1, 0.0)
    r = canopy_water.interception(5e-3, 1e-3, 1.4, 0.7, dpai, bef, DTIME)
    drip = r.qflx_tflrain + r.qflx_tflsnow - (6e-3 - r.qflx_intr)  # beyond direct throughfall
    stored = float(jnp.sum(r.h2ocan - bef))
    np.testing.assert_allclose(stored + float(drip) * DTIME, float(r.qflx_intr) * DTIME, rtol=1e-12)
    assert np.all(np.asarray(r.h2ocan) <= np.asarray(dewmx * dpai) + 1e-15)  # capacity
    assert np.all(np.asarray(r.h2ocan)[np.asarray(dpai) == 0] == 0.0)


def test_evaporation_removes_what_the_leaves_evaporate():
    dpai, fracsun = _dpai(), jnp.full(N_SLOTS, 0.3)
    h2ocan = jnp.where(dpai > 0, 0.1, 0.05)
    ev = jnp.full((N_SLOTS, 2), 1e-5)
    new = canopy_water.evaporation(h2ocan, dpai, fracsun, ev, jnp.zeros((N_SLOTS, 2)), DTIME)
    expected = h2ocan - jnp.where(dpai > 0, 1e-5 * dpai * mmh2o * DTIME, 0.0)
    np.testing.assert_allclose(new, expected, rtol=1e-12)


def test_wetted_fraction_limits():
    dpai = _dpai()
    fwet, fdry = canopy_water.wetted_fraction(dewmx * dpai * 10.0, dpai, 0.5 * dpai)
    assert np.all(np.asarray(fwet)[np.asarray(dpai) > 0] == maximum_leaf_wetted_fraction)
    assert np.all(np.asarray(fdry)[np.asarray(dpai) == 0] == 0.0)


def _implicit_cases():
    """Water (as a fraction of the capacity) and demand (as a multiple of it) over many
    orders of magnitude, on a layer with plant area."""
    rel, dem = np.meshgrid(10.0 ** np.arange(-12.0, 0.5, 1.0), [0.0, 1e-3, 1.0, 30.0, 1e3])
    dpai = jnp.full(rel.size, 0.8)
    return dpai, jnp.asarray(rel.ravel() * dewmx * 0.8), jnp.asarray(dem.ravel() * dewmx * 0.8)


def test_implicit_wetted_fraction_solves_the_water_balance():
    """Decision 122: the water left (by the explicit relation) plus the water evaporated is
    the water there was, so a layer never evaporates more than it holds; at the cap where
    the water allows it; the explicit value without demand; 0 without plant area."""
    dpai, water, demand = _implicit_cases()
    on = jnp.ones(dpai.shape, bool)
    fwet, fdry = canopy_water.wetted_fraction_implicit(water, dpai, 0.5 * dpai, demand, on)
    fwet, water, demand = np.asarray(fwet), np.asarray(water), np.asarray(demand)
    left = dewmx * 0.8 * fwet ** (1.0 / fwet_exponent)
    capped = fwet == maximum_leaf_wetted_fraction
    assert capped.any() and (~capped).any() and np.all(fwet <= maximum_leaf_wetted_fraction)
    np.testing.assert_allclose((left + demand * fwet)[~capped], water[~capped], rtol=1e-10)
    assert np.all((left + demand * fwet)[capped] <= water[capped])
    assert np.all(demand * fwet <= water * (1 + 1e-10))
    np.testing.assert_allclose(fdry, (1.0 - fwet) * 0.5, rtol=1e-14)
    explicit, _ = canopy_water.wetted_fraction(water, dpai, 0.5 * dpai)
    np.testing.assert_allclose(fwet[demand == 0], np.asarray(explicit)[demand == 0], rtol=1e-12)
    assert np.all(fwet <= np.asarray(explicit) * (1 + 1e-12))
    # negative water counts as none; no plant area, or a slot not in use: 0
    dpai0 = _dpai()
    f, d = canopy_water.wetted_fraction_implicit(
        jnp.full(N_SLOTS, -1e-3), dpai0, dpai0, jnp.full(N_SLOTS, 0.1), jnp.ones(N_SLOTS, bool)
    )
    assert np.all(np.asarray(f) == 0.0) and np.all(np.asarray(d)[np.asarray(dpai0) == 0] == 0.0)
    f, _ = canopy_water.wetted_fraction_implicit(
        jnp.full(N_SLOTS, 0.01), dpai0, dpai0, jnp.full(N_SLOTS, 0.1), jnp.zeros(N_SLOTS, bool)
    )
    assert np.all(np.asarray(f) == 0.0)


def test_implicit_wetted_fraction_gradient():
    """The implicit-function gradient against central finite differences, in the water, the
    demand and the plant area; 0 at the cap; finite for a dry layer and without plant area;
    and the forward value is the solver's own (the gradient step adds exactly 0)."""
    dpai, water, demand = _implicit_cases()
    on = jnp.ones(dpai.shape, bool)

    def f(scale):
        w, dm, pai = water * scale[0], demand * scale[1], dpai * scale[2]
        return canopy_water.wetted_fraction_implicit(w, pai, 0.5 * pai, dm, on)[0]

    one = jnp.ones(3)
    jac = np.asarray(jax.jacfwd(f)(one))
    rev = np.asarray(jax.jacrev(f)(one))
    np.testing.assert_allclose(rev, jac, rtol=1e-10, atol=1e-300)
    fwet = np.asarray(f(one))
    capped = fwet == maximum_leaf_wetted_fraction
    h = 1e-6
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        fd = (np.asarray(f(one + e)) - np.asarray(f(one - e))) / (2 * h)
        np.testing.assert_allclose(jac[~capped, k], fd[~capped], rtol=1e-5, atol=1e-12)
    assert np.all(jac[capped] == 0.0) and np.all(np.asarray(f(one))[capped] == 0.05)
    assert np.all(jac[:, 0] >= 0.0) and np.all(jac[:, 1] <= 0.0)  # more water, less demand
    dpai0 = _dpai()
    g = jax.jacrev(
        lambda w: canopy_water.wetted_fraction_implicit(
            w, dpai0, dpai0, jnp.full(N_SLOTS, 0.1), jnp.ones(N_SLOTS, bool)
        )[0]
    )(jnp.zeros(N_SLOTS))
    assert np.all(np.isfinite(np.asarray(g)))


def test_wet_leaf_demand():
    """The water a fully wet layer evaporates in a sub-step: the evaporation term of the
    leaf fluxes; 0 with dew, without plant area and in slots not in use (whose inputs may
    be fill values)."""
    from clm_ml.processes.water_vapor import sat_vap

    dpai, fracsun = _dpai(), jnp.full(N_SLOTS, 0.3)
    tleaf = jnp.stack([jnp.full(N_SLOTS, 300.0), jnp.full(N_SLOTS, 295.0)], axis=1)
    gbv = jnp.full((N_SLOTS, 2), 1.5)
    on = jnp.arange(N_SLOTS) < 2  # slot 2 has plant area but is not in use
    spval = jnp.where(on, 0.0, 1e36)
    d = np.asarray(
        canopy_water.wet_leaf_demand(
            tleaf + spval[:, None], 1500.0 + spval, gbv, fracsun, dpai, 1e5, DTIME, on
        )
    )
    esat = np.asarray(sat_vap(jnp.asarray([300.0, 295.0]))[0])
    one = 1.2 * mmh2o * DTIME * 1.5 * (0.3 * (esat[0] - 1500.0) + 0.7 * (esat[1] - 1500.0)) / 1e5
    np.testing.assert_allclose(d, [0.0, one, 0.0, 0.0, 0.0, 0.0], rtol=1e-12)
    dew = canopy_water.wet_leaf_demand(
        tleaf, jnp.full(N_SLOTS, 1e4), gbv, fracsun, dpai, 1e5, DTIME, on
    )
    assert np.all(np.asarray(dew) == 0.0)


def test_soil_uptake_fractions_sum_to_one_above_bedrock():
    rng = np.random.default_rng(2)
    r = plant_hydraulics.soil_resistance(
        jnp.asarray(rng.uniform(1e-6, 1e-4, N_SOI)), jnp.asarray(rng.uniform(-3e4, -1e3, N_SOI)),
        jnp.full(N_SOI, 0.2), jnp.full(N_SOI, 0.25), jnp.zeros(N_SOI), 3, 1.4, 500.0,
        *PARAMS[2:5],
    )  # fmt: skip
    np.testing.assert_allclose(float(jnp.sum(r.soil_et_loss)), 1.0, rtol=1e-12)
    assert float(r.soil_et_loss[3]) == 0.0  # below bedrock


def test_leaf_water_potential_relaxes_to_steady_state():
    dpai = _dpai()
    lsc = plant_hydraulics.plant_resistance(dpai, 0.5, 4.0)
    tr = jnp.full((N_SLOTS, 2), 1e-3)
    zs = jnp.arange(N_SLOTS, dtype=float)
    lwp = plant_hydraulics.leaf_water_potential(
        dpai, zs, lsc, tr, jnp.full((N_SLOTS, 2), -0.5), -0.3, 2500.0, 1e9
    )
    head = plant_hydraulics.HEAD
    steady = -0.3 - head * zs[:, None] - 1000.0 * tr / jnp.maximum(lsc, 1e-30)[:, None]
    np.testing.assert_allclose(np.asarray(lwp)[1:3], np.asarray(steady)[1:3], rtol=1e-12)
    assert np.all(np.asarray(lwp)[np.asarray(dpai) == 0] == 0.0)
