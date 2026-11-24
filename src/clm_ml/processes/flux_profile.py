"""Flux-profile solution: the implicit coupled solution for canopy air temperature and water
vapor, leaf temperature and ground temperature, and the leaf and soil fluxes that go with it.

One patch. Layers are 0-based slots on axis 0 (``clm_ml.state.PatchLayers``); leaves 0
sunlit, 1 shaded (axis 1). Every slot is computed: padding slots take the top layer's inputs
(:func:`clm_ml.processes.layers.pad_with_top`) and become identity rows of the tridiagonal
system after the top layer, whose upward coupling is zero, so the solution is the one for
the layers in use. The solution is chosen by name (``FLUX_PROFILE``, model.yaml
``processes.canopy_air.flux_profile_solution``): ``implicit`` (Fortran ``flux_profile_type = 1``);
``well_mixed`` is registered but not supported (Fortran stops after it too).

Fortran reference: CLM-ml v2 ``MLFluxProfileSolutionMod.F90``, ``MLLeafFluxesMod.F90``,
``MLSoilFluxesMod.F90``. Bonan et al. (2018) Geosci. Model Dev. 11, 1467-1496.

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from clm_ml.processes.layers import pad_with_top
from clm_ml.processes.math import tridiag_2eq
from clm_ml.processes.water_vapor import lat_vap, sat_vap
from clm_ml.state import PatchLayers


class Reference(NamedTuple):
    """Forcing at the reference height (scalars)."""

    pref: Array  # air pressure, Pa
    rhomol: Array  # molar density of air, mol/m3
    cpair: Array  # specific heat of air at constant pressure, J/mol/K
    tref: Array  # air temperature, K
    thref: Array  # potential temperature, K
    eref: Array  # vapor pressure, Pa


class Ground(NamedTuple):
    """The soil surface (scalars)."""

    tg_bef: Array  # soil surface temperature at the start of the sub-step, K
    rhg: Array  # relative humidity of the airspace at the soil surface (fraction)
    rnsoi: Array  # net radiation, W/m2
    soilres: Array  # soil evaporative resistance, s/m
    soil_t: Array  # temperature of the first soil layer, K
    soil_dz: Array  # depth from the surface to the first soil layer's node, m
    soil_tk: Array  # thermal conductivity from the surface to that node, W/m/K
    gac0: Array  # aerodynamic conductance from the ground to the first layer, mol/m2/s


class Canopy(NamedTuple):
    """The canopy layers: (layer,) and (layer, leaf) arrays."""

    dpai: Array  # (layer,) plant area index, m2/m2
    fwet: Array  # (layer,) wetted fraction of the plant area
    fdry: Array  # (layer,) dry fraction of the leaves (transpiring)
    fracsun: Array  # (layer,) sunlit fraction
    cpleaf: Array  # (layer,) leaf heat capacity, J/m2 leaf/K
    gac: Array  # (layer,) aerodynamic conductance to the layer above, mol/m2/s
    tair_bef: Array  # (layer,) air temperature at the start of the sub-step, K
    eair_bef: Array  # (layer,) vapor pressure at the start of the sub-step, Pa
    gbh: Array  # (layer, leaf) leaf boundary layer conductance for heat, mol/m2 leaf/s
    gbv: Array  # (layer, leaf) leaf boundary layer conductance for water vapor
    gs: Array  # (layer, leaf) stomatal conductance, mol H2O/m2 leaf/s
    rnleaf: Array  # (layer, leaf) leaf net radiation, W/m2 leaf
    tleaf_bef: Array  # (layer, leaf) leaf temperature at the start of the sub-step, K


class LeafFluxes(NamedTuple):
    """Leaf temperature and fluxes per unit leaf area (layer, leaf); 0 (and ``tleaf = tair``)
    where ``dpai == 0``."""

    tleaf: Array  # K
    stleaf: Array  # heat storage, W/m2 leaf
    shleaf: Array  # sensible heat, W/m2 leaf
    lhleaf: Array  # latent heat, W/m2 leaf
    evleaf: Array  # evaporation of intercepted water, mol H2O/m2 leaf/s
    trleaf: Array  # transpiration, mol H2O/m2 leaf/s
    error: Array  # energy balance error, W/m2 leaf (Fortran stops above 1e-3)


class SoilFluxes(NamedTuple):
    """Soil surface temperature and fluxes (scalars)."""

    tg: Array  # soil surface temperature, K
    eg: Array  # vapor pressure at the soil surface, Pa
    shsoi: Array  # sensible heat, W/m2
    lhsoi: Array  # latent heat, W/m2
    gsoi: Array  # soil heat flux, W/m2
    etsoi: Array  # water vapor flux, mol H2O/m2/s
    error: Array  # energy balance error, W/m2 (Fortran stops above 1e-3)


class ImplicitFluxes(NamedTuple):
    """The fluxes as the implicit solution gives them, for Fortran's error checks
    (:func:`implicit_errors`; ``ErrorCheck01``, ``ErrorCheck02``)."""

    lambda_: Array  # () latent heat of vaporization at tref, J/mol
    shsrc: Array  # (layer,) leaf sensible heat source, W/m2
    etsrc: Array  # (layer,) leaf water vapor source, mol H2O/m2/s
    stveg: Array  # (layer,) leaf heat storage, W/m2
    tleaf: Array  # (layer, leaf) leaf temperature, K
    avail_energy: Array  # (layer, leaf) leaf net radiation times plant area, W/m2
    storage_sh: Array  # (layer,) air heat storage, W/m2
    storage_et: Array  # (layer,) air water vapor storage, mol H2O/m2/s
    sh0: Array  # () ground sensible heat, W/m2
    et0: Array  # () ground water vapor flux, mol H2O/m2/s
    g0: Array  # () soil heat flux, W/m2
    t0: Array  # () soil surface temperature, K
    e0: Array  # () vapor pressure at the soil surface, Pa


class FluxProfile(NamedTuple):
    """Result of the flux-profile solution (layer values valid in the layers in use)."""

    tair: Array  # (layer,) air temperature, K
    eair: Array  # (layer,) vapor pressure, Pa
    shair: Array  # (layer,) sensible heat flux to the layer above, W/m2
    etair: Array  # (layer,) water vapor flux to the layer above, mol H2O/m2/s
    stair: Array  # (layer,) air heat storage (sensible + latent), W/m2
    leaf: LeafFluxes
    soil: SoilFluxes
    implicit: ImplicitFluxes


def leaf_fluxes(
    tair, eair, cpleaf, fwet, fdry, gbh, gbv, gs, rnleaf, tleaf_bef, dpai, pref, cpair, tref,
    dtime,
) -> LeafFluxes:  # fmt: skip
    """Leaf temperature and fluxes for given air temperature and vapor pressure, from the
    leaf energy balance linearized about ``tleaf_bef`` (Bonan et al. 2018, eqs. 10-13).

    Elementwise: ``gbh``, ``gbv``, ``gs``, ``rnleaf``, ``tleaf_bef`` per leaf; ``tair``,
    ``eair`` (Pa), ``cpleaf``, ``fwet``, ``fdry``, ``dpai`` broadcast against them (e.g.
    (layer, 1) against (layer, leaf)); ``pref``, ``cpair``, ``tref`` scalars; ``dtime`` (s).

    Fortran reference: MLLeafFluxesMod.F90::LeafFluxes.
    Reads: ``tair_profile``, ``eair_profile``, ``cpleaf_profile``, ``fwet_profile``,
    ``fdry_profile``, ``gbh_leaf``, ``gbv_leaf``, ``gs_leaf``, ``rnleaf_leaf``,
    ``tleaf_bef_leaf``, ``dpai_profile``, ``pref_forcing``, ``cpair_forcing``,
    ``tref_forcing``. Writes: ``tleaf_leaf``, ``stleaf_leaf``, ``shleaf_leaf``,
    ``lhleaf_leaf``, ``evleaf_leaf``, ``trleaf_leaf``; the leaf energy balance check.
    """
    lambda_ = lat_vap(tref)
    active = dpai > 0.0
    esat, desat = sat_vap(tleaf_bef)
    qsat = esat / pref
    dqsat = desat / pref

    # Leaf conductance for transpiration and total conductance for water vapor;
    # jnp.maximum (not a select) keeps 0/0 out of leafless layers
    gleaf = gs * gbv / jnp.maximum(gs + gbv, 1.0e-30)
    gw = gleaf * fdry + gbv * fwet

    num1 = 2.0 * cpair * gbh
    num2 = lambda_ * gw
    num3 = rnleaf - lambda_ * gw * (qsat - dqsat * tleaf_bef) + cpleaf / dtime * tleaf_bef
    den = jnp.maximum(cpleaf / dtime + num1 + num2 * dqsat, 1.0e-30)
    tleaf = jnp.where(active, (num1 * tair + num2 * (eair / pref) + num3) / den, tair)

    stleaf = jnp.where(active, (tleaf - tleaf_bef) * cpleaf / dtime, 0.0)
    shleaf = jnp.where(active, 2.0 * cpair * (tleaf - tair) * gbh, 0.0)
    vapor = qsat + dqsat * (tleaf - tleaf_bef) - eair / pref
    trleaf = jnp.where(active, gleaf * fdry * vapor, 0.0)
    evleaf = jnp.where(active, gbv * fwet * vapor, 0.0)
    lhleaf = jnp.where(active, (trleaf + evleaf) * lambda_, 0.0)
    error = jnp.where(active, rnleaf - shleaf - lhleaf - stleaf, 0.0)
    return LeafFluxes(tleaf, stleaf, shleaf, lhleaf, evleaf, trleaf, error)


def soil_fluxes(tair1, eair1, ground: Ground, ref: Reference) -> SoilFluxes:
    """Soil surface temperature and fluxes for given air temperature ``tair1`` (K) and vapor
    pressure ``eair1`` (Pa) of the lowest layer, from the surface energy balance linearized
    about ``tg_bef`` (Bonan et al. 2018, eqs. 14-15).

    Fortran reference: MLSoilFluxesMod.F90::SoilFluxes.
    Reads: ``tair_profile`` and ``eair_profile`` of layer 1, ``tref_forcing``,
    ``pref_forcing``, ``rhomol_forcing``, ``cpair_forcing``, ``rnsoi_soil``, ``rhg_soil``,
    ``soilres_soil``, ``gac0_soil``, ``soil_t_soil``, ``soil_dz_soil``, ``soil_tk_soil``,
    ``tg_bef_soil``. Writes: ``shsoi_soil``, ``lhsoi_soil``, ``gsoi_soil``, ``etsoi_soil``,
    ``tg_soil``, ``eg_soil``; the soil energy balance check.
    """
    g, r = ground, ref
    lam = lat_vap(r.tref)
    gws = (1.0 / g.soilres) * r.rhomol  # soil conductance for water vapor, mol H2O/m2/s
    gw = g.gac0 * gws / (g.gac0 + gws)  # in series with the aerodynamic conductance
    esat, desat = sat_vap(g.tg_bef)
    qsat = esat / r.pref
    dqsat = desat / r.pref

    num1 = r.cpair * g.gac0
    num2 = lam * gw
    num3 = g.soil_tk / g.soil_dz
    num4 = g.rnsoi - num2 * g.rhg * (qsat - dqsat * g.tg_bef) + num3 * g.soil_t
    den = num1 + num2 * dqsat * g.rhg + num3
    tg = (num1 * tair1 + num2 * (eair1 / r.pref) + num4) / den

    shsoi = r.cpair * (tg - tair1) * g.gac0
    eg = g.rhg * (esat + desat * (tg - g.tg_bef))
    lhsoi = lam * (eg - eair1) / r.pref * gw
    gsoi = g.soil_tk * (tg - g.soil_t) / g.soil_dz
    error = g.rnsoi - shsoi - lhsoi - gsoi
    etsoi = lhsoi / lam
    return SoilFluxes(tg, eg, shsoi, lhsoi, gsoi, etsoi, error)


def implicit(
    canopy: Canopy,
    ground: Ground,
    ref: Reference,
    layers: PatchLayers,
    dtime,
    unroll: bool = False,
) -> FluxProfile:
    """Implicit flux-profile solution (Bonan et al. 2018): leaf temperatures and the ground
    temperature are linear in the air temperature and water vapor of their layer
    (``T_leaf = alpha T + beta q + delta``), so air temperature and water vapor mole fraction
    of all layers solve one coupled tridiagonal system (``tridiag_2eq``), with the ground
    below the first layer and the reference height above the top layer. The leaf and soil
    fluxes are then those of :func:`leaf_fluxes` and :func:`soil_fluxes` at the solved
    profiles; vertical fluxes between layers and the air storage follow. ``unroll``: the
    solve's loops over layers written out (``math.tridiag``).

    Fortran reference: MLFluxProfileSolutionMod.F90::ImplicitFluxProfileSolution (with
    ``LeafFluxes`` and ``SoilFluxes``; the CO2 profile set to ``co2ref`` is done in ``canopy``).
    Reads: ``dpai_profile``, ``fwet_profile``, ``fdry_profile``, ``fracsun_profile``,
    ``cpleaf_profile``, ``gac_profile``, ``tair_bef_profile``, ``eair_bef_profile``,
    ``gbh_leaf``, ``gbv_leaf``, ``gs_leaf``, ``rnleaf_leaf``, ``tleaf_bef_leaf``, the soil
    surface (:class:`Ground`), the reference forcing (:class:`Reference`), the layer grid.
    Writes: ``tair_profile``, ``eair_profile``, ``shair_profile``, ``etair_profile``,
    ``stair_profile``, the leaf fluxes (:func:`leaf_fluxes`) and soil fluxes
    (:func:`soil_fluxes`).
    """
    c, g, r = canopy, ground, ref
    n_slots = c.dpai.shape[0]
    use = layers.in_use

    def pad(x):
        return pad_with_top(layers, x)

    dz = pad(layers.dz)
    dpai, fwet, fdry, fracsun, cpleaf = (pad(x) for x in (c.dpai, c.fwet, c.fdry, c.fracsun,
                                                           c.cpleaf))  # fmt: skip
    gac, tair_bef, eair_bef = pad(c.gac), pad(c.tair_bef), pad(c.eair_bef)
    gbh, gbv, gs, rnleaf, tleaf_bef = (pad(x) for x in (c.gbh, c.gbv, c.gs, c.rnleaf,
                                                        c.tleaf_bef))  # fmt: skip

    lambda_ = lat_vap(r.tref)

    # Ground temperature coefficients: t0 = alpha0 T(1) + beta0 q(1) + delta0
    esat, desat = sat_vap(g.tg_bef)
    qsat0 = esat / r.pref
    dqsat0 = desat / r.pref
    gsw = (1.0 / g.soilres) * r.rhomol
    gs0 = g.gac0 * gsw / (g.gac0 + gsw)
    c02 = g.soil_tk / g.soil_dz
    c01 = -c02 * g.soil_t
    den = r.cpair * g.gac0 + lambda_ * g.rhg * gs0 * dqsat0 + c02
    alpha0 = r.cpair * g.gac0 / den
    beta0 = lambda_ * gs0 / den
    delta0 = (g.rnsoi - lambda_ * g.rhg * gs0 * (qsat0 - dqsat0 * g.tg_bef) - c01) / den

    # Leaf temperature coefficients (Bonan et al. 2018, eqs. 10-13, A1-A5), (layer, leaf)
    active = (dpai > 0.0)[:, None]
    gsh_raw = 2.0 * gbh
    get_raw = gs * gbv / jnp.maximum(gs + gbv, 1.0e-30) * fdry[:, None] + gbv * fwet[:, None]
    esat_l, desat_l = sat_vap(tleaf_bef)
    qsat_l = esat_l / r.pref
    dqsat_l = desat_l / r.pref
    qsat_term_val = qsat_l - dqsat_l * tleaf_bef
    hcap_raw = cpleaf[:, None]
    den_l = jnp.maximum(hcap_raw / dtime + gsh_raw * r.cpair + get_raw * lambda_ * dqsat_l, 1.0e-30)
    alpha_val = gsh_raw * r.cpair / den_l
    beta_val = get_raw * lambda_ / den_l
    delta_val = (
        rnleaf / den_l
        - lambda_ * get_raw * qsat_term_val / den_l
        + hcap_raw / dtime * tleaf_bef / den_l
    )
    pai = jnp.stack([dpai * fracsun, dpai * (1.0 - fracsun)], axis=1)  # sunlit, shaded
    gleaf_sh = jnp.where(active, gsh_raw * pai, 0.0)
    gleaf_et = jnp.where(active, get_raw * pai, 0.0)
    heatcap = jnp.where(active, hcap_raw * pai, 0.0)
    avail_energy = jnp.where(active, rnleaf * pai, 0.0)
    dqsat = jnp.where(active, dqsat_l, 0.0)
    qsat_term = jnp.where(active, qsat_term_val, 0.0)
    alpha = jnp.where(active, alpha_val, 0.0)
    beta = jnp.where(active, beta_val, 0.0)
    delta = jnp.where(active, delta_val, 0.0)

    # Tridiagonal coefficients (eqs. 16-17, S10-S31): every slot, then the bottom layer
    # (ground below) and the top layer (reference height above) as in Fortran
    rdt = r.rhomol * dz / dtime
    gac_below_sh = jnp.concatenate([jnp.reshape(g.gac0, (1,)), gac[: n_slots - 1]])
    gac_below_et = jnp.concatenate([jnp.reshape(gs0, (1,)), gac[: n_slots - 1]])
    gsh_s, gsh_h = gleaf_sh[:, 0], gleaf_sh[:, 1]
    get_s, get_h = gleaf_et[:, 0], gleaf_et[:, 1]
    al_s, al_h = alpha[:, 0], alpha[:, 1]
    be_s, be_h = beta[:, 0], beta[:, 1]
    de_s, de_h = delta[:, 0], delta[:, 1]
    dq_s, dq_h = dqsat[:, 0], dqsat[:, 1]
    qt_s, qt_h = qsat_term[:, 0], qsat_term[:, 1]

    a1 = -gac_below_sh
    b11 = rdt + gac_below_sh + gac + gsh_s * (1.0 - al_s) + gsh_h * (1.0 - al_h)
    b12 = -gsh_s * be_s - gsh_h * be_h
    c1 = -gac
    d1 = rdt * tair_bef + gsh_s * de_s + gsh_h * de_h

    a2 = -gac_below_et
    b21 = -get_s * dq_s * al_s - get_h * dq_h * al_h
    b22 = rdt + gac_below_et + gac + get_s * (1.0 - dq_s * be_s) + get_h * (1.0 - dq_h * be_h)
    c2 = -gac
    d2 = rdt * (eair_bef / r.pref) + get_s * (dq_s * de_s + qt_s) + get_h * (dq_h * de_h + qt_h)

    top = layers.n_layers - 1  # exchange with the reference height
    c1 = c1.at[top].set(0.0)
    d1 = d1.at[top].set(d1[top] + gac[top] * r.thref)
    c2 = c2.at[top].set(0.0)
    d2 = d2.at[top].set(d2[top] + gac[top] * (r.eref / r.pref))
    # Bottom layer: exchange with the ground
    a1 = a1.at[0].set(0.0)
    b11 = b11.at[0].set(b11[0] - g.gac0 * alpha0)
    b12 = b12.at[0].set(b12[0] - g.gac0 * beta0)
    d1 = d1.at[0].set(d1[0] + g.gac0 * delta0)
    a2 = a2.at[0].set(0.0)
    b21 = b21.at[0].set(b21[0] - gs0 * g.rhg * dqsat0 * alpha0)
    b22 = b22.at[0].set(b22[0] - gs0 * g.rhg * dqsat0 * beta0)
    d2 = d2.at[0].set(d2[0] + gs0 * g.rhg * (qsat0 + dqsat0 * (delta0 - g.tg_bef)))

    # Padding slots: identity rows (x = 0), not coupled to the top layer (c(top) = 0)
    zero, one = jnp.zeros_like(a1), jnp.ones_like(a1)
    a1, c1, d1, a2, c2, d2, b12, b21 = (
        jnp.where(use, x, zero) for x in (a1, c1, d1, a2, c2, d2, b12, b21)
    )
    b11, b22 = jnp.where(use, b11, one), jnp.where(use, b22, one)

    # Air temperature (K) and water vapor (mol/mol); padding: the top layer's (finite)
    t_sol, q_sol = tridiag_2eq(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n_slots, unroll)
    tair = pad(jnp.stack(t_sol))
    qair = pad(jnp.stack(q_sol))

    # Soil surface temperature and vapor pressure (mol/mol), leaf temperatures
    t0 = alpha0 * tair[0] + beta0 * qair[0] + delta0
    e0 = g.rhg * (qsat0 + dqsat0 * (t0 - g.tg_bef))
    tleaf_implic = alpha * tair[:, None] + beta * qair[:, None] + delta

    # Water vapor from mol/mol to Pa
    eair = qair * r.pref
    e0 = e0 * r.pref

    # Leaf and soil fluxes at the solved profiles (Fortran calls LeafFluxes and SoilFluxes
    # here: the same arithmetic as the implicit solution, compared in ErrorCheck01)
    leaf = leaf_fluxes(
        tair[:, None], eair[:, None], cpleaf[:, None], fwet[:, None], fdry[:, None], gbh, gbv,
        gs, rnleaf, tleaf_bef, dpai[:, None], r.pref, r.cpair, r.tref, dtime,
    )  # fmt: skip
    soil = soil_fluxes(tair[0], eair[0], g, r)

    # Vertical fluxes between layers: to the layer above, the reference height from the top
    is_top = layers.index == top
    tair_next = jnp.where(is_top, r.thref, jnp.concatenate([tair[1:], tair[-1:]]))
    eair_next = jnp.where(is_top, r.eref, jnp.concatenate([eair[1:], eair[-1:]]))
    shair = -r.cpair * (tair_next - tair) * gac
    etair = -(eair_next - eair) / r.pref * gac

    # Canopy air storage
    storage_sh = r.cpair * (tair - tair_bef) * rdt
    storage_et = (eair - eair_bef) / r.pref * rdt
    stair = storage_sh + storage_et * lambda_

    # Source fluxes of the implicit solution (per ground area: leaf fluxes times plant area),
    # summed over the leaves in Fortran order (sunlit, then shaded)
    sh_l = r.cpair * (tleaf_implic - tair[:, None]) * gleaf_sh
    et_l = (esat_l + desat_l * (tleaf_implic - tleaf_bef) - eair[:, None]) / r.pref * gleaf_et
    stv_l = heatcap * (tleaf_implic - tleaf_bef) / dtime
    sh_l, et_l, stv_l = (jnp.where(active, x, 0.0) for x in (sh_l, et_l, stv_l))
    shsrc = sh_l[:, 0] + sh_l[:, 1]
    etsrc = et_l[:, 0] + et_l[:, 1]
    stveg = stv_l[:, 0] + stv_l[:, 1]

    # Soil fluxes of the implicit solution
    sh0 = -r.cpair * (tair[0] - t0) * g.gac0
    et0 = -(eair[0] - e0) / r.pref * gs0
    g0 = -g.soil_tk / g.soil_dz * g.soil_t + g.soil_tk / g.soil_dz * t0

    implicit_fluxes = ImplicitFluxes(
        lambda_, shsrc, etsrc, stveg, tleaf_implic, avail_energy, storage_sh, storage_et, sh0,
        et0, g0, t0, e0,
    )  # fmt: skip
    return FluxProfile(tair, eair, shair, etair, stair, leaf, soil, implicit_fluxes)


class ImplicitErrors(NamedTuple):
    """Largest absolute error of each of Fortran's checks of the implicit solution
    (``ErrorCheck01``: against ``LeafFluxes`` and ``SoilFluxes``; ``ErrorCheck02``:
    conservation). Fortran stops above 1e-3 (W/m2) for fluxes and 1e-6 for temperatures
    (K) and vapor pressure (Pa)."""

    leaf_sh: Array
    leaf_lh: Array
    leaf_storage: Array
    leaf_temperature: Array
    soil_sh: Array
    soil_lh: Array
    soil_g: Array
    soil_temperature: Array
    soil_vapor_pressure: Array
    leaf_energy: Array
    layer_sh: Array
    layer_lh: Array
    canopy_sh: Array
    canopy_lh: Array
    ground_energy: Array


def implicit_errors(f: FluxProfile, canopy: Canopy, ground: Ground, layers: PatchLayers):
    """Fortran's checks of the implicit solution (:class:`ImplicitErrors`). The time step does
    not run them; the tests do.

    Fortran reference: MLFluxProfileSolutionMod.F90::ErrorCheck01, ErrorCheck02.
    """
    imp, lf, sl = f.implicit, f.leaf, f.soil
    use = layers.in_use
    leafy = use & (canopy.dpai > 0.0)
    fs = jnp.stack([canopy.fracsun, 1.0 - canopy.fracsun], axis=1)

    def per_area(x):
        return jnp.sum(x * fs, axis=1) * canopy.dpai

    def largest(err, mask):
        return jnp.max(jnp.where(mask, jnp.abs(err), 0.0))

    lam = imp.lambda_
    below_sh = jnp.concatenate([jnp.reshape(imp.sh0, (1,)), f.shair[:-1]])
    below_et = jnp.concatenate([jnp.reshape(imp.et0, (1,)), f.etair[:-1]])
    ntop = layers.n_within - 1
    in_canopy = layers.index <= ntop
    leaf_energy = (
        imp.avail_energy[:, 0] + imp.avail_energy[:, 1] - imp.shsrc - lam * imp.etsrc - imp.stveg
    )
    canopy_sh = imp.sh0 + jnp.sum(jnp.where(in_canopy, imp.shsrc - imp.storage_sh, 0.0))
    canopy_et = imp.et0 + jnp.sum(jnp.where(in_canopy, imp.etsrc - imp.storage_et, 0.0))
    return ImplicitErrors(
        leaf_sh=largest(imp.shsrc - per_area(lf.shleaf), leafy),
        leaf_lh=largest(lam * (imp.etsrc - per_area(lf.trleaf + lf.evleaf)), leafy),
        leaf_storage=largest(imp.stveg - per_area(lf.stleaf), leafy),
        leaf_temperature=largest(jnp.max(jnp.abs(lf.tleaf - imp.tleaf), axis=1), leafy),
        soil_sh=jnp.abs(sl.shsoi - imp.sh0),
        soil_lh=jnp.abs(lam * (sl.etsoi - imp.et0)),
        soil_g=jnp.abs(sl.gsoi - imp.g0),
        soil_temperature=jnp.abs(sl.tg - imp.t0),
        soil_vapor_pressure=jnp.abs(sl.eg - imp.e0),
        leaf_energy=largest(leaf_energy, use),
        layer_sh=largest(imp.storage_sh - (below_sh + imp.shsrc - f.shair), use),
        layer_lh=largest((imp.storage_et - (below_et + imp.etsrc - f.etair)) * lam, use),
        canopy_sh=jnp.abs(canopy_sh - f.shair[ntop]),
        canopy_lh=jnp.abs((canopy_et - f.etair[ntop]) * lam),
        ground_energy=jnp.abs(ground.rnsoi - imp.sh0 - lam * imp.et0 - imp.g0),
    )


def well_mixed(*args, **kwargs):
    """Well-mixed canopy air (or profiles from a dataset): registered but not supported.

    Fortran reference: MLFluxProfileSolutionMod.F90::WellMixed. Fortran stops after it
    ("WellMixed option not valid").
    """
    raise NotImplementedError(
        "flux_profile_solution: well_mixed is not supported (Fortran stops for it too); "
        "use implicit"
    )


#: Flux-profile solutions by name (model.yaml ``processes.canopy_air.flux_profile_solution``).
FLUX_PROFILE = {"implicit": implicit, "well_mixed": well_mixed}
