"""clm_ml.processes.turbulence (Phase 7.2, process 6).

The physics is checked against what its definitions imply (the wind profile recovers the
forcing wind and is continuous at the canopy top, the solvers find their roots) and the Obukhov
length's implicit-function gradient against finite differences.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.io.rsl_tables import default_rsl_tables  # noqa: E402
from clm_ml.processes import turbulence as turb  # noqa: E402
from clm_ml.processes.constants import vkc  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

# A 10 m canopy under a 23 m reference height: 4 layers within, 3 above, 1 padding slot
ZTOP, ZREF = 10.0, 23.0
ZW = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 14.0, 18.5, 23.0, 23.0])  # ground + layer tops
N_SLOTS, NCAN, NTOP, NBOT = 8, 7, 4, 2  # legacy 1-based counts


def _layers():
    dz = np.diff(ZW)
    zs = ZW[:-1] + 0.5 * dz
    return PatchLayers(
        n_layers=jnp.asarray(NCAN), n_within=jnp.asarray(NTOP),
        first_leafy=jnp.asarray(NBOT - 1), dz=jnp.asarray(dz), zs=jnp.asarray(zs),
        zw=jnp.asarray(ZW),
    )  # fmt: skip


def _forcing(thref=296.0, taf=294.0, uref=3.0):
    qref = 0.010
    return dict(
        pref=101325.0, ztop=ZTOP, lai=2.0, sai=0.7, zref=ZREF, uref=uref, thref=thref,
        thvref=thref * (1.0 + 0.61 * qref), qref=qref, rhomol=41.0,
        tair=jnp.full(N_SLOTS, taf), eair=jnp.full(N_SLOTS, 1619.0),
    )  # fmt: skip


def _run(ground="log_profile", unroll=False, zeta_max_stable=turb.NO_ZETA_LIMIT, **forcing):
    f = _forcing(**forcing)
    return turb.harman_finnigan(
        _layers(), f["tair"], f["eair"], f["pref"], f["ztop"], f["lai"], f["sai"], f["zref"],
        f["uref"], f["thref"], f["thvref"], f["qref"], f["rhomol"], default_rsl_tables(),
        sparse_canopy=True, ground_conductance=ground, unroll=unroll,
        zeta_max_stable=zeta_max_stable,
    )  # fmt: skip


# canopy-top air colder (stable) .. warmer (unstable) than the reference air at 296 K, the
# same humidity; 294 K and 299 K give roots beyond the theory's range -2 <= Lc/L <= 1
@pytest.mark.parametrize("taf", [294.0, 295.5, 295.8, 296.2, 296.5, 297.0, 299.0])
def test_obukhov_length_is_the_root_or_the_limit(taf):
    f = _forcing(taf=taf)
    t = _run(taf=taf)
    assert (float(t.obu) > 0) == (taf < 296.0)
    LcL = float(t.Lc / t.obu)
    if taf in (294.0, 299.0):
        assert LcL == pytest.approx(1.0 if taf < 296.0 else -2.0, rel=1e-12)
        return
    assert -2.0 < LcL < 1.0
    inp = turb.ObuInputs(t.Lc, ZTOP, f["lai"], f["sai"], ZREF, f["uref"], f["thref"],
                         f["thvref"], f["qref"], f["rhomol"], t.taf, t.qaf)  # fmt: skip
    obu_new = turb.obukhov_terms(t.obu, inp, default_rsl_tables(), True).obu_new
    assert abs(float(obu_new - t.obu)) < 1e-9 * abs(float(t.obu))


def test_stable_zeta_limit():
    """Decision 123: in stable conditions the stability at the reference height,
    (zref - zdisp)/L with the displacement height of the Lc-limited length, is at most the
    PFT's zeta_max_stable; unstable conditions, and a limit that is not reached, change
    nothing; the limit has a derivative that agrees with finite differences."""
    strong = dict(taf=290.0, uref=1.0)
    free = _run(**strong)  # at the theory's limit Lc/L = 1
    zeta_free = float((ZREF - free.zdisp) / free.obu)
    assert zeta_free > 0.5 and float(free.Lc / free.obu) == pytest.approx(1.0, rel=1e-12)
    t = _run(**strong, zeta_max_stable=0.5)
    assert float((ZREF - free.zdisp) / t.obu) == pytest.approx(0.5, rel=1e-12)
    assert float(t.obu) > float(free.obu) and float(t.ustar) > float(free.ustar)
    for taf in (294.0, 295.5, 295.8):
        assert float((ZREF - _run(taf=taf).zdisp) / _run(taf=taf, zeta_max_stable=0.5).obu) <= 0.5
    for kind, kw in (("not reached", dict(**strong, zeta_max_stable=2.0 * zeta_free)),
                     ("unstable", dict(taf=297.0, zeta_max_stable=0.5))):  # fmt: skip
        a = _run(**kw)
        b = _run(**{k: v for k, v in kw.items() if k != "zeta_max_stable"})
        for name in ("obu", "ustar", "beta", "zdisp", "z0m"):
            assert float(getattr(a, name)) == float(getattr(b, name)), (kind, name)

    def ustar(z):
        return _run(**strong, zeta_max_stable=z).ustar

    d = float(jax.grad(ustar)(0.5))
    fd = float(ustar(0.5 + 1e-5) - ustar(0.5 - 1e-5)) / 2e-5
    assert d < 0.0 and d == pytest.approx(fd, rel=1e-5)
    assert float(jax.grad(ustar)(turb.NO_ZETA_LIMIT)) == 0.0


def test_wind_profile_recovers_forcing_wind_and_is_continuous():
    t = _run()
    tables = default_rsl_tables()
    psim_ref = turb.psi_rsl(ZREF, ZTOP, t.zdisp, t.obu, t.beta, t.PrSc, tables)[0]
    u_ref = t.ustar / vkc * (jnp.log((ZREF - t.zdisp) / (ZTOP - t.zdisp)) + psim_ref)
    np.testing.assert_allclose(u_ref, 3.0, rtol=1e-12)
    psim_top = turb.psi_rsl(ZTOP, ZTOP, t.zdisp, t.obu, t.beta, t.PrSc, tables)[0]
    np.testing.assert_allclose(t.ustar / vkc * psim_top, t.uaf, rtol=1e-12)  # above = below
    wind = np.asarray(t.wind)[:NCAN]
    assert np.all(np.diff(wind) > 0)  # increases with height
    assert np.all(np.asarray(t.mflx)[:NCAN] <= 0) and float(t.mflx[NTOP]) == -float(t.ustar**2)


def test_roughness_length_is_the_root():
    t = _run()
    _, _, psim_hc, psim_hat_hc = turb.psi_rsl(
        ZREF, ZTOP, t.zdisp, t.obu, t.beta, t.PrSc, default_rsl_tables()
    )
    r = turb._z0m_residual(t.z0m, t.obu, ZTOP - t.zdisp, jnp.exp(-vkc / t.beta),
                           jnp.exp(psim_hat_hc), psim_hc)  # fmt: skip
    assert 0.0 < float(t.z0m) < ZTOP and abs(float(r)) < 0.002


def test_conductances_positive_and_limited():
    for ground in turb.GROUND_CONDUCTANCE:
        t = _run(ground=ground)
        gac = np.asarray(t.gac)[:NCAN]
        assert np.all(gac >= 41.0 / 500.0 * (1 - 1e-12)) and float(t.gac0) > 0
        zs = np.asarray(_layers().zs)[:NCAN]
        to_next = np.append(np.diff(zs), ZREF - zs[-1])  # mid-layer to mid-layer, top to zref
        np.testing.assert_allclose(np.asarray(t.kc_eddy)[:NCAN], gac / 41.0 * to_next)
    with pytest.raises(ValueError):
        _run(ground="other")


def test_lookup_psihat_on_nodes_and_outside():
    m = default_rsl_tables().momentum
    i, j = 100, 20
    assert float(turb.lookup_psihat(m.zdt[i], m.dtl[j], m)) == pytest.approx(m.psi[i, j])
    # beyond the grid both neighbours are the edge value
    assert float(turb.lookup_psihat(m.zdt[0] + 1.0, m.dtl[-1] + 1.0, m)) == pytest.approx(
        m.psi[0, -1]
    )


@pytest.mark.parametrize("search", ["unrolled", "compare_all"])
def test_lookup_psihat_search_methods_agree(search):
    """Every table search (numerics.table_search, decision 114) finds the same grid cell:
    the lookup is identical inside the grid, on its nodes, on its edges and outside."""
    m = default_rsl_tables().momentum
    rng = np.random.default_rng(0)
    zdt = np.concatenate([rng.uniform(-1.0, 52.0, 500), m.zdt, m.zdt[:1] + 1.0, [0.0]])
    dtl = np.concatenate([rng.uniform(-3.0, 3.0, 500), m.dtl, m.dtl[-1:] + 1.0, [0.0]])
    zz, ll = (jnp.asarray(a) for a in np.meshgrid(zdt, dtl, indexing="ij"))
    lookup = jax.vmap(jax.vmap(turb.lookup_psihat, (0, 0, None, None)), (0, 0, None, None))
    np.testing.assert_array_equal(lookup(zz, ll, m, search), lookup(zz, ll, m, "loop"))


def test_psi_rsl_shares_the_table_searches():
    """psi_rsl finds the places of its two heights and of (h - d)/L once for the four
    lookups (decision 115): the same psihat values as four separate lookups."""
    tables = default_rsl_tables()
    za, hc, disp, obu, beta, PrSc = 23.0, 10.0, 7.3, -41.0, 0.33, 0.6
    psim, psic, _, psim_hat2 = turb.psi_rsl(za, hc, disp, obu, beta, PrSc, tables)
    dt = hc - disp
    hat = [
        turb.lookup_psihat(zdt, dt / obu, table)
        for table in (tables.momentum, tables.scalar)
        for zdt in ((za - hc) / dt, 0.0)
    ]
    c1_m = (1.0 - vkc / (2.0 * beta * turb.phim_monin_obukhov(dt / obu))) * jnp.exp(0.5 * turb.c2)
    c1_c = (1.0 - PrSc * vkc / (2.0 * beta * turb.phic_monin_obukhov(dt / obu))) * jnp.exp(
        0.5 * turb.c2
    )
    assert float(psim_hat2) == float(hat[1] * c1_m)
    mo_m = -turb.psim_monin_obukhov((za - disp) / obu) + turb.psim_monin_obukhov(dt / obu)
    mo_c = -turb.psic_monin_obukhov((za - disp) / obu) + turb.psic_monin_obukhov(dt / obu)
    assert float(psim) == float(mo_m + hat[0] * c1_m - hat[1] * c1_m + vkc / beta)
    assert float(psic) == float(mo_c + hat[2] * c1_c - hat[3] * c1_c)


def test_tables_must_share_their_grids(tmp_path):
    """The momentum and scalar tables are on one pair of grids (decision 115): a file where
    they differ is refused."""
    import xarray as xr

    from clm_ml.io.rsl_tables import read_rsl_tables, rsl_table_file

    with xr.open_dataset(rsl_table_file()) as ds:
        ds = ds.load()
    ds["dtLgridH"] = ds["dtLgridH"] * 1.5
    ds.to_netcdf(tmp_path / "psihat.nc")
    with pytest.raises(ValueError, match="same grids"):
        read_rsl_tables(tmp_path / "psihat.nc")


def test_table_search_follows_the_device():
    from clm_ml import driver

    assert driver.table_search("compare_all") == "compare_all"
    expected = "unrolled" if jax.default_backend() == "gpu" else "loop"
    assert driver.table_search("auto") == expected


# moderately stable .. strongly unstable; at 298.3 K the Obukhov length is -8 m, where the
# residual bends within a metre (a slope by finite differences with a 1 m step was 12 % off
# there, 0.5 % at 298.0 K: decision 119)
@pytest.mark.parametrize("taf", [295.0, 295.8, 296.5, 297.5, 298.0, 298.3])
def test_obukhov_gradient_is_the_implicit_one(taf):
    """d obu / d thref and d ustar / d thref by jax.grad (implicit-function correction, with
    the exact slope of the residual) vs central differences of the solver's forward value."""
    for name in ("obu", "ustar"):

        def value(thref):
            return getattr(_run(thref=thref, taf=taf), name)

        g = float(jax.grad(value)(296.0))
        h = 1e-4
        fd = (float(value(296.0 + h)) - float(value(296.0 - h))) / (2 * h)
        assert np.isfinite(g) and g != 0.0 and g == pytest.approx(fd, rel=1e-6), name


def test_registry():
    assert turb.ROUGHNESS_SUBLAYER == {"harman_finnigan": turb.harman_finnigan}


def test_solver_loops_follow_the_device():
    from clm_ml import driver

    assert driver.solver_loops("unrolled") is True and driver.solver_loops("loop") is False
    assert driver.solver_loops("auto") == (jax.default_backend() == "gpu")


@pytest.mark.parametrize("taf", [294.0, 295.8, 296.5])
def test_unrolled_iterations_match_the_loops(taf):
    """The Obukhov and roughness-length iterations written out (numerics.solver_loops,
    decision 117) are the loops' iterations."""
    loop, unrolled = _run(taf=taf), _run(taf=taf, unroll=True)
    for name in ("obu", "z0m", "ustar", "wind", "gac"):
        np.testing.assert_allclose(getattr(unrolled, name), getattr(loop, name), rtol=1e-12)
