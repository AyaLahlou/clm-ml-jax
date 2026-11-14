"""The step-level canopy pieces of the multilayer canopy time step: the plant area of each
layer for the step's LAI and SAI, the soil surface humidity, the averages of the fluxes over
the sub-steps, the canopy totals and conservation checks, and the merge of sunlit and shaded
leaves at the end of the step.

One patch. Layers are 0-based slots on axis 0 (``clm_ml.state.PatchLayers``); levels (the
ground and the layers above it) on axis 0 of the ``level`` arrays; leaves 0 sunlit, 1 shaded
on axis 1; bands 0 visible, 1 near-infrared on the last axis.

Fortran reference: CLM-ml v2 ``MLCanopyFluxesMod.F90`` (``MLCanopyFluxes``,
``MLTimeStepFluxIntegration``, ``CanopyFluxesDiagnostics``).

Reads and writes are the arguments and results; the docstrings name the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import grav, mmh2o, rgas
from clm_ml.processes.layers import pad_with_top
from clm_ml.processes.water_vapor import lat_vap
from clm_ml.state import PatchLayers

# ---------------------------------------------------------------------------
# Plant area of each layer — Fortran MLCanopyFluxes, lines 220-243
# ---------------------------------------------------------------------------


class PlantArea(NamedTuple):
    """Leaf and stem area of the canopy and its layers for the step's LAI and SAI."""

    lai: Array  # () leaf area index, m2/m2 (lai_canopy)
    sai: Array  # () stem area index with sai_floor, m2/m2 (sai_canopy)
    sai_floor: Array  # () stem area added by the per-layer minimum, m2/m2 (sai_floor_canopy)
    dlai: Array  # (layer,) leaf area index of the layer (dlai_profile)
    dsai: Array  # (layer,) stem area index of the layer (dsai_profile)
    dpai: Array  # (layer,) plant area index of the layer (dpai_profile)
    error: Array  # () sum of dpai minus (lai + sai): the check PLANT_AREA


def plant_area(
    layers: PatchLayers,
    dlai_frac: Array,
    dsai_frac: Array,
    lai: Array,
    sai: Array,
    layer_pai_floor: float,
) -> PlantArea:
    """The layers' leaf and stem area: the canopy's LAI and SAI (``elai``, ``esai`` of the
    CLM history) spread with the layers' fractions (``dlai_frac_profile``,
    ``dsai_frac_profile``, set at initialization).

    Not Fortran (Phase 6b, issue 14): every leafy layer keeps at least ``layer_pai_floor`` of
    plant area, the missing part added as stem area, and the canopy's stem area includes it.
    CLM-ml's 9-angle diffuse transmittance exceeds 1 below ~0.005 m2/m2, which makes a layer's
    net longwave grow with its temperature and the leaf temperature run away. Layers at or
    above the floor are unchanged (``x + 0.0``), so CLM-ml v2 results are kept where no layer
    is that thin (CHATS7). Value: CLM-ml v1's ``dpai_min`` (m2/m2).

    Every slot is computed (padding with the top layer's fractions); the caller keeps the
    layers in use. Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 220-243).
    """
    dlai = pad_with_top(layers, dlai_frac) * lai
    dsai = pad_with_top(layers, dsai_frac) * sai
    add = jnp.where(layers.leafy, jnp.maximum(layer_pai_floor - (dlai + dsai), 0.0), 0.0)
    dsai = dsai + add
    sai_floor = jnp.sum(add)
    sai = sai + sai_floor
    dpai = dlai + dsai
    error = jnp.sum(jnp.where(layers.in_use, dpai, 0.0)) - (lai + sai)
    return PlantArea(lai, sai, sai_floor, dlai, dsai, dpai, error)


def soil_surface_humidity(smp: Array, t_soil: Array) -> Array:
    """Relative humidity of the air at the soil surface (``rhg_soil``) from the matric
    potential (mm) and temperature (K) of the first soil layer.

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 259-262).
    """
    return jnp.exp(grav * mmh2o * smp * 1.0e-3 / (rgas * t_soil))


# ---------------------------------------------------------------------------
# Averages over the sub-steps — Fortran MLTimeStepFluxIntegration
# ---------------------------------------------------------------------------


class SubstepFluxes(NamedTuple):
    """The values averaged over the sub-steps of a step (the legacy container's field of the
    same name, without its suffix). Every slot of the layer, level and leaf axes."""

    ustar: Array  # () friction velocity, m/s (_canopy)
    beta: Array  # () u*/U(h) (_canopy)
    obu: Array  # () Obukhov length, m (_canopy)
    z0m: Array  # () roughness length for momentum, m (_canopy)
    zdisp: Array  # () displacement height, m (_canopy)
    lwup: Array  # () upward longwave above the canopy, W/m2 (_canopy)
    swsoi: Array  # (band,) shortwave absorbed by the ground, W/m2 (_soil)
    lwsoi: Array  # () longwave absorbed by the ground, W/m2 (_soil)
    rnsoi: Array  # () net radiation of the ground, W/m2 (_soil)
    shsoi: Array  # () sensible heat flux of the ground, W/m2 (_soil)
    lhsoi: Array  # () latent heat flux of the ground, W/m2 (_soil)
    etsoi: Array  # () water vapor flux of the ground, mol H2O/m2/s (_soil)
    gsoi: Array  # () soil heat flux, W/m2 (_soil)
    gac0: Array  # () aerodynamic conductance of the ground, mol/m2/s (_soil)
    qflx_intr: Array  # () intercepted precipitation, kg H2O/m2/s (_canopy)
    qflx_tflrain: Array  # () rain throughfall, kg H2O/m2/s (_canopy)
    qflx_tflsnow: Array  # () snow throughfall, kg H2O/m2/s (_canopy)
    swskyb: Array  # (band,) direct beam shortwave, W/m2 (_forcing)
    swskyd: Array  # (band,) diffuse shortwave, W/m2 (_forcing)
    lwsky: Array  # () atmospheric longwave, W/m2 (_forcing)
    # Not Fortran (issue 12): the canopy-top latent heat flux of the sub-step, with the
    # sub-step's own latent heat (Fortran multiplies the averaged water vapor flux by the last
    # sub-step's LatVap(tref), which differs when tref crosses freezing within the step)
    lhtop: Array  # () W/m2 (_canopy)
    shair: Array  # (layer,) air sensible heat flux, W/m2 (_profile)
    etair: Array  # (layer,) air water vapor flux, mol H2O/m2/s (_profile)
    stair: Array  # (layer,) air storage heat flux, W/m2 (_profile)
    mflx: Array  # (layer,) momentum flux, m2/s2 (_profile)
    kc_eddy: Array  # (layer,) eddy diffusivity, m2/s (_profile)
    gac: Array  # (layer,) aerodynamic conductance for scalars, mol/m2/s (_profile)
    swupw: Array  # (level, band) upward diffuse shortwave, W/m2 (_profile)
    swdwn: Array  # (level, band) downward diffuse shortwave, W/m2 (_profile)
    swbeam: Array  # (level, band) direct beam shortwave, W/m2 (_profile)
    lwupw: Array  # (level,) upward longwave, W/m2 (_profile)
    lwdwn: Array  # (level,) downward longwave, W/m2 (_profile)
    swleaf: Array  # (layer, leaf, band) leaf absorbed shortwave, W/m2 leaf (_leaf)
    lwleaf: Array  # (layer, leaf) leaf absorbed longwave, W/m2 leaf (_leaf)
    rnleaf: Array  # (layer, leaf) leaf net radiation, W/m2 leaf (_leaf)
    shleaf: Array  # (layer, leaf) leaf sensible heat flux, W/m2 leaf (_leaf)
    lhleaf: Array  # (layer, leaf) leaf latent heat flux, W/m2 leaf (_leaf)
    trleaf: Array  # (layer, leaf) leaf transpiration, mol H2O/m2 leaf/s (_leaf)
    evleaf: Array  # (layer, leaf) leaf evaporation, mol H2O/m2 leaf/s (_leaf)
    stleaf: Array  # (layer, leaf) leaf storage heat flux, W/m2 leaf (_leaf)
    anet: Array  # (layer, leaf) net photosynthesis, umol CO2/m2 leaf/s (_leaf)
    agross: Array  # (layer, leaf) gross photosynthesis, umol CO2/m2 leaf/s (_leaf)
    gs: Array  # (layer, leaf) stomatal conductance, mol H2O/m2 leaf/s (_leaf)


def canopy_top_latent_heat(layers: PatchLayers, etair: Array, tref: Array) -> Array:
    """The sub-step's latent heat flux at the canopy top (top layer in use), W/m2: the
    ``lhtop`` entry of :class:`SubstepFluxes` (not Fortran, issue 12)."""
    return jnp.take(etair, layers.n_layers - 1, axis=0) * lat_vap(tref)


def zero_sums(fluxes: SubstepFluxes) -> SubstepFluxes:
    """Sums of no sub-steps, shaped as ``fluxes``."""
    return jax.tree.map(jnp.zeros_like, fluxes)


def accumulate(sums: SubstepFluxes, fluxes: SubstepFluxes) -> SubstepFluxes:
    """``sums`` with one sub-step's ``fluxes`` added.

    Fortran reference: MLCanopyFluxesMod.F90::MLTimeStepFluxIntegration (the sums).
    """
    return jax.tree.map(jnp.add, sums, fluxes)


def average(sums: SubstepFluxes, n_substeps: int) -> SubstepFluxes:
    """The averages of ``sums`` over ``n_substeps`` sub-steps (multiplied by
    ``1 / n_substeps``, as Fortran).

    Fortran reference: MLCanopyFluxesMod.F90::MLTimeStepFluxIntegration (the averages).
    """
    scale = 1.0 / float(n_substeps)
    return jax.tree.map(lambda s: s * scale, sums)


# ---------------------------------------------------------------------------
# Canopy totals and checks — Fortran CanopyFluxesDiagnostics
# ---------------------------------------------------------------------------


class CanopyLeaves(NamedTuple):
    """End-of-step values of the layers and leaves read by the canopy totals (not
    averaged)."""

    dpai: Array  # (layer,) plant area index (dpai_profile)
    fracsun: Array  # (layer,) sunlit fraction (fracsun_profile)
    fwet: Array  # (layer,) wet fraction of the plant area (fwet_profile)
    fdry: Array  # (layer,) green and dry fraction of the plant area (fdry_profile)
    wind: Array  # (layer,) wind speed, m/s (wind_profile)
    tair: Array  # (layer,) air temperature, K (tair_profile)
    vcmax25: Array  # (layer,) Vcmax at 25 C of the layer (vcmax25_profile)
    tleaf: Array  # (layer, leaf) leaf temperature, K (tleaf_leaf)
    lwp: Array  # (layer, leaf) leaf water potential, MPa (lwp_leaf)
    vcmax25_leaf: Array  # (layer, leaf) Vcmax at 25 C (vcmax25_leaf)


class CanopyTotals(NamedTuple):
    """Canopy totals of a step (``*_canopy`` fields; per m2 ground) and the layers' mean leaf
    water potential, with the errors of Fortran's four energy checks. Only what outputs and
    checks read (D6; the other totals and profiles of Fortran's ``CanopyFluxesDiagnostics``
    are listed in ``docs/design/phase7_removed_diagnostics.md``)."""

    swveg: Array  # (band,) absorbed shortwave: vegetation, W/m2
    swvegsun: Array  # (band,) absorbed shortwave: sunlit canopy, W/m2
    swvegsha: Array  # (band,) absorbed shortwave: shaded canopy, W/m2
    lwveg: Array  # () absorbed longwave: vegetation, W/m2
    stflx_veg: Array  # () vegetation heat storage flux, W/m2
    shveg: Array  # () sensible heat: vegetation, W/m2
    shvegsun: Array  # () sensible heat: sunlit canopy, W/m2
    shvegsha: Array  # () sensible heat: shaded canopy, W/m2
    lhveg: Array  # () latent heat: vegetation, W/m2
    lhvegsun: Array  # () latent heat: sunlit canopy, W/m2
    lhvegsha: Array  # () latent heat: shaded canopy, W/m2
    trveg: Array  # () transpiration, mol H2O/m2/s
    evveg: Array  # () evaporation of intercepted water, mol H2O/m2/s
    gppveg: Array  # () GPP: vegetation, umol CO2/m2/s
    gppvegsun: Array  # () GPP: sunlit canopy, umol CO2/m2/s
    gppvegsha: Array  # () GPP: shaded canopy, umol CO2/m2/s
    vcmax25veg: Array  # () Vcmax25: canopy, umol/m2/s
    vcmax25sun: Array  # () Vcmax25: sunlit canopy, umol/m2/s
    vcmax25sha: Array  # () Vcmax25: shaded canopy, umol/m2/s
    gsveg: Array  # () stomatal conductance: canopy, mol H2O/m2/s
    gsvegsun: Array  # () stomatal conductance: sunlit canopy, mol H2O/m2/s
    gsvegsha: Array  # () stomatal conductance: shaded canopy, mol H2O/m2/s
    albcan: Array  # (band,) canopy albedo
    shflx: Array  # () total sensible heat flux, W/m2
    etflx: Array  # () total water vapor flux, mol H2O/m2/s (CLM coupling, mlcan_to_clm)
    lhflx: Array  # () total latent heat flux, W/m2
    stflx_air: Array  # () canopy air storage heat flux, W/m2
    rnet: Array  # () net radiation, W/m2
    laisun: Array  # () sunlit plant area index, m2/m2
    laisha: Array  # () shaded plant area index, m2/m2
    windveg: Array  # () wind speed: vegetation (plant-area weighted), m/s
    windvegsun: Array  # () wind speed: sunlit canopy, m/s
    windvegsha: Array  # () wind speed: shaded canopy, m/s
    tlveg: Array  # () leaf temperature: vegetation, K
    tlvegsun: Array  # () leaf temperature: sunlit canopy, K
    tlvegsha: Array  # () leaf temperature: shaded canopy, K
    taveg: Array  # () air temperature: vegetation, K
    tavegsun: Array  # () air temperature: sunlit canopy, K
    tavegsha: Array  # () air temperature: shaded canopy, K
    fracminlwp: Array  # () fraction of the canopy at leaf water potential <= -2 MPa
    lwp_mean: Array  # (layer,) leaf water potential, sunlit/shaded mean, MPa (lwp_mean_profile)
    errors: Array  # (4,) the checks CANOPY_ENERGY_1 .. CANOPY_ENERGY_4 (signed)


def canopy_totals(
    layers: PatchLayers,
    means: SubstepFluxes,
    leaves: CanopyLeaves,
    lai: Array,
    sai: Array,
    *,
    flux_profile_type: int,
) -> CanopyTotals:
    """Canopy totals from the step's averaged fluxes (``means``) and the end-of-step layers
    and leaves; ``lai``, ``sai`` the canopy's leaf and stem area (``sai`` with the per-layer
    minimum). Sunlit and shaded leaves are weighted by the sunlit fraction, layers by their
    plant area; layers without plant area and the padding contribute nothing.

    ``flux_profile_type`` (static): 1 the canopy's turbulent fluxes are those at the top layer
    (``lhflx`` the sub-step average of the canopy-top latent heat flux, issue 12; Fortran
    ``etair(ncan) * LatVap(tref)`` with the averaged ``etair`` and the last sub-step's
    ``tref``); 0 or -1 the vegetation's plus the ground's.

    Fortran reference: MLCanopyFluxesMod.F90::CanopyFluxesDiagnostics.
    """
    if flux_profile_type not in (-1, 0, 1):
        raise ValueError(f"canopy_totals: flux_profile_type {flux_profile_type} not valid")

    def pad(x):
        return pad_with_top(layers, x)

    dpai = pad(leaves.dpai)
    fs = pad(leaves.fracsun)
    fsh = 1.0 - fs
    fwet = pad(leaves.fwet)
    fdry = pad(leaves.fdry)
    has_pai = (dpai > 0.0) & layers.in_use
    dpai_s = jnp.where(has_pai, dpai, 0.0)

    # fracgreen = fdry / (1 - fwet), guarded against fwet == 1
    fwet_safe = jnp.where(fwet < 1.0, fwet, 0.0)
    fracgreen = jnp.where(has_pai, fdry / (1.0 - fwet_safe), 0.0)

    swleaf = pad(means.swleaf)
    sw_sun_vis, sw_sun_nir = swleaf[:, 0, 0], swleaf[:, 0, 1]
    sw_sha_vis, sw_sha_nir = swleaf[:, 1, 0], swleaf[:, 1, 1]

    def sun_sha(x):
        x = pad(x)
        return x[:, 0], x[:, 1]

    lw_sun, lw_sha = sun_sha(means.lwleaf)
    st_sun, st_sha = sun_sha(means.stleaf)
    sh_sun, sh_sha = sun_sha(means.shleaf)
    lh_sun, lh_sha = sun_sha(means.lhleaf)
    ev_sun, ev_sha = sun_sha(means.evleaf)
    tr_sun, tr_sha = sun_sha(means.trleaf)
    ag_sun, ag_sha = sun_sha(means.agross)
    gs_sun, gs_sha = sun_sha(means.gs)
    tl_sun, tl_sha = sun_sha(leaves.tleaf)
    lp_sun, lp_sha = sun_sha(leaves.lwp)
    vc_sun, vc_sha = sun_sha(leaves.vcmax25_leaf)

    # Layer means per leaf area, then sources per ground area
    def blend(a_sun, a_sha):
        return jnp.where(has_pai, a_sun * fs + a_sha * fsh, 0.0)

    lw_m = blend(lw_sun, lw_sha)
    sw_m_vis = blend(sw_sun_vis, sw_sha_vis)
    sw_m_nir = blend(sw_sun_nir, sw_sha_nir)
    st_m = blend(st_sun, st_sha)
    sh_m = blend(sh_sun, sh_sha)
    lh_m = blend(lh_sun, lh_sha)
    ev_m = blend(ev_sun, ev_sha)
    tr_m = blend(tr_sun, tr_sha)
    et_m = ev_m + tr_m
    gs_m = blend(gs_sun, gs_sha)
    tl_m = blend(tl_sun, tl_sha)
    lp_m = blend(lp_sun, lp_sha)

    # Vegetation totals
    swveg_vis = jnp.sum(sw_m_vis * dpai_s)
    swveg_nir = jnp.sum(sw_m_nir * dpai_s)
    lwveg = jnp.sum(lw_m * dpai_s)
    stflx_veg = jnp.sum(st_m * dpai_s)
    shveg = jnp.sum(sh_m * dpai_s)
    lhveg = jnp.sum(lh_m * dpai_s)
    etveg = jnp.sum(et_m * dpai_s)
    trveg = jnp.sum(tr_m * dpai_s)
    evveg = jnp.sum(ev_m * dpai_s)
    vcmax25veg = jnp.sum(pad(leaves.vcmax25) * dpai_s)
    gppveg = jnp.sum((ag_sun * fs + ag_sha * fsh) * dpai_s * fracgreen)
    gsveg = jnp.sum(gs_m * dpai_s)

    # Vegetation energy balance
    err1 = swveg_vis + swveg_nir + lwveg - shveg - lhveg - stflx_veg

    # Albedo (0 without incoming shortwave)
    inc_vis = means.swskyb[0] + means.swskyd[0]
    inc_nir = means.swskyb[1] + means.swskyd[1]
    swupw_top = jnp.take(means.swupw, layers.n_within, axis=0)
    alb_vis = jnp.where(inc_vis > 0.0, swupw_top[0] / jnp.maximum(inc_vis, 1e-30), 0.0)
    alb_nir = jnp.where(inc_nir > 0.0, swupw_top[1] / jnp.maximum(inc_nir, 1e-30), 0.0)

    # Turbulent fluxes
    if flux_profile_type in (0, -1):
        shflx = shveg + means.shsoi
        etflx = etveg + means.etsoi
        lhflx = lhveg + means.lhsoi
    else:
        shflx = jnp.take(means.shair, layers.n_layers - 1, axis=0)
        etflx = jnp.take(means.etair, layers.n_layers - 1, axis=0)
        lhflx = means.lhtop

    # Air heat storage
    stflx_air = jnp.sum(jnp.where(layers.in_use, pad(means.stair), 0.0))

    # Energy balance checks
    rnet = swveg_vis + swveg_nir + means.swsoi[0] + means.swsoi[1] + lwveg + means.lwsoi
    radin = means.swskyb[0] + means.swskyd[0] + means.swskyb[1] + means.swskyd[1] + means.lwsky
    radout = (
        alb_vis * (means.swskyb[0] + means.swskyd[0])
        + alb_nir * (means.swskyb[1] + means.swskyd[1])
        + means.lwup
    )
    err2 = rnet - (radin - radout)
    avail = radin - radout - means.gsoi
    flux = shflx + lhflx + stflx_air + stflx_veg
    err3 = avail - flux
    swbeam_top = jnp.take(means.swbeam, layers.n_within, axis=0)
    swdwn_top = jnp.take(means.swdwn, layers.n_within, axis=0)
    lwdwn_top = jnp.take(means.lwdwn, layers.n_within, axis=0)
    lwupw_top = jnp.take(means.lwupw, layers.n_within, axis=0)
    radin_top = swbeam_top[0] + swbeam_top[1] + swdwn_top[0] + swdwn_top[1] + lwdwn_top
    radout_top = swupw_top[0] + swupw_top[1] + lwupw_top
    err4 = (radin_top - radout_top) - rnet

    # Sunlit and shaded canopy
    laisun = jnp.sum(fs * dpai_s)
    laisha = jnp.sum(fsh * dpai_s)
    swvegsun_vis = jnp.sum(sw_sun_vis * fs * dpai_s)
    swvegsun_nir = jnp.sum(sw_sun_nir * fs * dpai_s)
    swvegsha_vis = jnp.sum(sw_sha_vis * fsh * dpai_s)
    swvegsha_nir = jnp.sum(sw_sha_nir * fsh * dpai_s)
    shvegsun = jnp.sum(sh_sun * fs * dpai_s)
    shvegsha = jnp.sum(sh_sha * fsh * dpai_s)
    lhvegsun = jnp.sum(lh_sun * fs * dpai_s)
    lhvegsha = jnp.sum(lh_sha * fsh * dpai_s)
    gppvegsun = jnp.sum(ag_sun * fs * dpai_s * fracgreen)
    gppvegsha = jnp.sum(ag_sha * fsh * dpai_s * fracgreen)
    vcmax25sun = jnp.sum(vc_sun * fs * dpai_s)
    vcmax25sha = jnp.sum(vc_sha * fsh * dpai_s)
    gsvegsun = jnp.sum(gs_sun * fs * dpai_s)
    gsvegsha = jnp.sum(gs_sha * fsh * dpai_s)

    # Plant-area weighted temperatures and wind (0 without plant area)
    total = laisun + laisha
    wind = pad(leaves.wind)
    tair = pad(leaves.tair)
    total_safe = jnp.maximum(total, 1e-30)
    laisun_safe = jnp.maximum(laisun, 1e-30)
    laisha_safe = jnp.maximum(laisha, 1e-30)

    def mean_all(x):
        return jnp.where(total > 0.0, jnp.sum(x * dpai_s) / total_safe, 0.0)

    def mean_sun(x):
        return jnp.where(laisun > 0.0, jnp.sum(x * fs * dpai_s) / laisun_safe, 0.0)

    def mean_sha(x):
        return jnp.where(laisha > 0.0, jnp.sum(x * fsh * dpai_s) / laisha_safe, 0.0)

    # Fraction of the canopy at the minimum leaf water potential
    fracminlwp = jnp.sum(jnp.where(lp_m <= -2.0, dpai_s, 0.0))
    fracminlwp = jnp.where(lai + sai > 0.0, fracminlwp / (lai + sai), fracminlwp)

    return CanopyTotals(
        swveg=jnp.stack([swveg_vis, swveg_nir]),
        swvegsun=jnp.stack([swvegsun_vis, swvegsun_nir]),
        swvegsha=jnp.stack([swvegsha_vis, swvegsha_nir]),
        lwveg=lwveg,
        stflx_veg=stflx_veg,
        shveg=shveg,
        shvegsun=shvegsun,
        shvegsha=shvegsha,
        lhveg=lhveg,
        lhvegsun=lhvegsun,
        lhvegsha=lhvegsha,
        trveg=trveg,
        evveg=evveg,
        gppveg=gppveg,
        gppvegsun=gppvegsun,
        gppvegsha=gppvegsha,
        vcmax25veg=vcmax25veg,
        vcmax25sun=vcmax25sun,
        vcmax25sha=vcmax25sha,
        gsveg=gsveg,
        gsvegsun=gsvegsun,
        gsvegsha=gsvegsha,
        albcan=jnp.stack([alb_vis, alb_nir]),
        shflx=shflx,
        etflx=etflx,
        lhflx=lhflx,
        stflx_air=stflx_air,
        rnet=rnet,
        laisun=laisun,
        laisha=laisha,
        windveg=mean_all(wind),
        windvegsun=mean_sun(wind),
        windvegsha=mean_sha(wind),
        tlveg=mean_all(tl_m),
        tlvegsun=mean_sun(tl_sun),
        tlvegsha=mean_sha(tl_sha),
        taveg=mean_all(tair),
        tavegsun=mean_sun(tair),
        tavegsha=mean_sha(tair),
        fracminlwp=fracminlwp,
        lwp_mean=lp_m,
        errors=jnp.stack([err1, err2, err3, err4]),
    )


# ---------------------------------------------------------------------------
# Sunlit and shaded leaves merged — Fortran MLCanopyFluxes, lines 355-375
# ---------------------------------------------------------------------------


class MergedLeaves(NamedTuple):
    """Leaf state at the end of a step."""

    tleaf: Array  # (layer, leaf) leaf temperature, merged, K (tleaf_leaf)
    lwp: Array  # (layer, leaf) leaf water potential, merged, MPa (lwp_leaf)
    tleaf_hist: Array  # (layer, leaf) leaf temperature before the merge, K (tleaf_hist_leaf)
    lwp_hist: Array  # (layer, leaf) leaf water potential before the merge (lwp_hist_leaf)


def merge_sun_shade(
    layers: PatchLayers, dpai: Array, fracsun: Array, tleaf: Array, lwp: Array
) -> MergedLeaves:
    """Leaf temperature and water potential of the sunlit and shaded leaves replaced by their
    layer mean (weighted by the sunlit fraction) in layers with plant area: the state is
    prognostic and the sunlit fraction changes between steps. The values before the merge
    are kept for output (``*_hist``).

    Every slot is computed (layers without plant area and the padding unchanged); the caller
    keeps the layers in use. Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes
    (lines 355-375).
    """
    has_pai = (dpai > 0.0) & layers.in_use

    def merge(x):
        mean = x[:, 0] * fracsun + x[:, 1] * (1.0 - fracsun)
        return jnp.stack(
            [jnp.where(has_pai, mean, x[:, 0]), jnp.where(has_pai, mean, x[:, 1])], axis=1
        )

    return MergedLeaves(tleaf=merge(tleaf), lwp=merge(lwp), tleaf_hist=tleaf, lwp_hist=lwp)


# ---------------------------------------------------------------------------
# The sub-step — Fortran MLCanopyFluxes, lines 261-330
# ---------------------------------------------------------------------------


class SubstepOptions(NamedTuple):
    """The static choices of the sub-step (model.yaml names; hashable, a static argument)."""

    interpolation: str  # forcing time interpolation (processes.atm_forcing.TIME_INTERPOLATION)
    acclimation: bool  # temperature acclimation of photosynthesis
    kn_val: float  # nitrogen decay coefficient (-999: from Vcmax25)
    sparse_canopy: bool  # turbulence: sparse canopy
    ground_conductance: str  # turbulence: processes.turbulence.GROUND_CONDUCTANCE
    regime: str  # leaf boundary layer: processes.leaf_boundary_layer.REGIMES
    stomatal_conductance: str  # processes.photosynthesis.STOMATAL_CONDUCTANCE
    water_stress: bool  # stomatal conductance limited by leaf water potential
    colimitation: bool  # photosynthesis colimitation
    wue_solver: str  # stomatal optimization solver
    dtime: float  # sub-step length, s
    integrator: str  # processes.integrators.TABLEAUS
    table_search: str = "loop"  # psihat table search: processes.turbulence.TABLE_SEARCH
    #: the loops over layers (radiation, longwave and flux-profile solves) and the Obukhov and
    #: roughness-length iterations written out instead of compiled loops (decision 117)
    unroll_loops: bool = False


class SubstepInputs(NamedTuple):
    """Values fixed over a step's sub-steps (legacy container field in parentheses)."""

    layers: PatchLayers
    forcing_bef: object  # processes.atm_forcing.ForcingRecord at the previous step
    forcing_cur: object  # ... at this step
    forcing_next: object  # ... at the next step
    time_bef: Array  # () calendar days of the three records for the interpolation
    time_cur: Array
    time_next: Array
    zref: Array  # () reference height, m (zref_forcing)
    solar_zen: Array  # () solar zenith angle (solar_zen_forcing)
    dlai: Array  # (layer,) (dlai_profile)
    dsai: Array  # (layer,) (dsai_profile)
    dpai: Array  # (layer,) (dpai_profile)
    albsoib: Array  # (band,) direct beam albedo of the ground (albsoib_soil)
    albsoid: Array  # (band,) diffuse albedo of the ground (albsoid_soil)
    lai: Array  # () (lai_canopy)
    sai: Array  # () (sai_canopy)
    tacclim: Array  # () acclimation temperature, K (tacclim_forcing)
    ztop: Array  # () canopy height, m (ztop_canopy)
    o2ref: Array  # () O2, mmol/mol (o2ref_forcing)
    qflx_rain: Array  # () rainfall, kg H2O/m2/s (qflx_rain_forcing)
    qflx_snow: Array  # () snowfall, kg H2O/m2/s (qflx_snow_forcing)
    rhg: Array  # () relative humidity at the soil surface (rhg_soil)
    soilres: Array  # () soil evaporative resistance, s/m (soilres_soil)
    soil_t: Array  # () first soil layer temperature, K (soil_t_soil)
    soil_dz: Array  # () depth to the first soil layer, m (soil_dz_soil)
    soil_tk: Array  # () first soil layer thermal conductivity (soil_tk_soil)
    cpleaf: Array  # (layer,) leaf heat capacity (cpleaf_profile)
    lsc: Array  # (layer,) leaf-specific conductance (lsc_profile)
    psis: Array  # () soil water potential, MPa (psis_soil)


#: Fields of :class:`Substep` by kind (their legacy container names; the kind gives the
#: shape: scalar (), band (band,), layer (layer,), level (level,), level_band (level, band),
#: leaf (layer, leaf), leaf_band (layer, leaf, band), stage (stage,), layer_stage
#: (layer, stage), leaf_stage (layer, leaf, stage), checks (check,))
SUBSTEP_KINDS = {
    "scalar": ("tg_soil", "tg_bef_soil", "uref_forcing", "tref_forcing", "qref_forcing",
               "pref_forcing", "co2ref_forcing", "lwsky_forcing", "eref_forcing",
               "rhomol_forcing", "rhoair_forcing", "mmair_forcing", "cpair_forcing",
               "thref_forcing", "thvref_forcing", "qflx_intr_canopy", "qflx_tflrain_canopy",
               "qflx_tflsnow_canopy", "lwsoi_soil", "lwveg_canopy", "lwup_canopy",
               "rnsoi_soil", "obu_canopy", "beta_canopy", "zdisp_canopy", "ustar_canopy",
               "z0m_canopy", "gac0_soil", "btran_soil", "shsoi_soil", "lhsoi_soil",
               "gsoi_soil", "etsoi_soil"),
    "band": ("swskyb_forcing", "swskyd_forcing", "swsoi_soil", "swveg_canopy",
             "swvegsun_canopy", "swvegsha_canopy", "albcan_canopy"),
    "layer": ("tair_profile", "eair_profile", "cair_profile", "h2ocan_profile",
              "tair_bef_profile", "eair_bef_profile", "cair_bef_profile", "h2ocan_bef_profile",
              "fracsun_profile", "kb_profile", "tb_profile", "td_profile", "vcmax25_profile",
              "jmax25_profile", "rd25_profile", "kp25_profile", "fwet_profile", "fdry_profile",
              "wind_profile", "mflx_profile", "gac_profile", "kc_eddy_profile",
              "shair_profile", "etair_profile", "stair_profile"),
    "level": ("tbi_profile", "lwupw_profile", "lwdwn_profile"),
    "level_band": ("swbeam_profile", "swdwn_profile", "swupw_profile"),
    "leaf": ("tleaf_leaf", "lwp_leaf", "tleaf_bef_leaf", "lwp_bef_leaf", "apar_leaf",
             "vcmax25_leaf", "jmax25_leaf", "rd25_leaf", "kp25_leaf", "lwleaf_leaf",
             "rnleaf_leaf", "gbh_leaf", "gbv_leaf", "gbc_leaf", "gs_leaf", "anet_leaf",
             "agross_leaf", "stleaf_leaf", "shleaf_leaf", "lhleaf_leaf", "evleaf_leaf",
             "trleaf_leaf"),
    "leaf_band": ("swleaf_leaf",),
    "stage": ("dtg_soil",),
    "layer_stage": ("dtair_profile", "deair_profile", "dh2ocan_profile"),
    "leaf_stage": ("dtleaf_leaf", "dlwp_leaf"),
    "checks": ("balance_error",),
}  # fmt: skip

#: The values a sub-step carries: everything the sub-step's processes read or write except
#: :class:`SubstepInputs` (the prognostic variables, their values at the start of the
#: sub-step, the Runge-Kutta increments, and the processes' results that later processes,
#: the averages, the canopy totals or the outputs read). Every slot of each axis; padding
#: slots keep their values where the legacy routines kept them.
Substep = NamedTuple("Substep", [(f, Array) for fields in SUBSTEP_KINDS.values() for f in fields])


#: The fill value of the legacy containers (clm_varcon.spval): slots no routine has written
SPVAL = 1.0e36


def initial_substep(n_layers: int, n_stage_slots: int, n_checks: int) -> Substep:
    """The sub-step carry at the start of a step, before the prognostic state is put in:
    ``spval`` in every slot (the work values are written before they are read; the slots no
    step writes, above the canopy or the top layer, keep it, as the legacy containers did)
    and no conservation errors. ``n_layers``: layer slots; ``n_stage_slots``: the
    Runge-Kutta increments' slots (the method's stages + 1). Phase 7.4 (C26)."""
    shapes = {
        "scalar": (), "band": (2,), "layer": (n_layers,), "level": (n_layers + 1,),
        "level_band": (n_layers + 1, 2), "leaf": (n_layers, 2), "leaf_band": (n_layers, 2, 2),
        "stage": (n_stage_slots,), "layer_stage": (n_layers, n_stage_slots),
        "leaf_stage": (n_layers, 2, n_stage_slots), "checks": (n_checks,),
    }  # fmt: skip
    values = {}
    for kind, fields in SUBSTEP_KINDS.items():
        for f in fields:
            fill = 0.0 if kind == "checks" else SPVAL
            values[f] = jnp.full(shapes[kind], fill, dtype=jnp.float64)
    return Substep(**values)


def stage_slots(integrator: str) -> int:
    """Slots of the Runge-Kutta increments in the carry: the method's stages + 1 (the
    legacy ``nrk + 1``)."""
    from clm_ml.processes.integrators import TABLEAUS

    return len(TABLEAUS[integrator].b) + 1


def _record(errors: Array, check: int, err) -> Array:
    """``errors`` with ``|err|`` (largest element) recorded for ``check`` (as
    ``MLCanopyFluxesType.record_balance_error``)."""
    return errors.at[check].max(jax.lax.stop_gradient(jnp.max(jnp.abs(jnp.asarray(err)))))


def _keep(mask, new, old):
    from clm_ml.processes.layers import keep

    return keep(mask, new, old)


def save_start(c: Substep, layers: PatchLayers) -> Substep:
    """The values at the start of the sub-step (``*_bef``), layers in use (padding kept).

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 272-285).
    """
    use = layers.in_use
    return c._replace(
        tg_bef_soil=c.tg_soil,
        tair_bef_profile=_keep(use, c.tair_profile, c.tair_bef_profile),
        eair_bef_profile=_keep(use, c.eair_profile, c.eair_bef_profile),
        cair_bef_profile=_keep(use, c.cair_profile, c.cair_bef_profile),
        h2ocan_bef_profile=_keep(use, c.h2ocan_profile, c.h2ocan_bef_profile),
        tleaf_bef_leaf=_keep(use, c.tleaf_leaf, c.tleaf_bef_leaf),
        lwp_bef_leaf=_keep(use, c.lwp_leaf, c.lwp_bef_leaf),
    )


def _radiation(c: Substep, x: SubstepInputs, params, calday, opt: SubstepOptions, c3: bool):
    """Forcing, solar radiation and the nitrogen profile of the sub-step (once per
    sub-step, before the Runge-Kutta stages)."""
    from clm_ml.checks import NITROGEN, SOLAR, SUNLIT_FRACTION
    from clm_ml.processes import atm_forcing, nitrogen_profile, solar_radiation

    lay = x.layers
    use = lay.in_use
    errors = c.balance_error

    # Forcing at the reference height — Fortran GetAtmForcing
    f = atm_forcing.atm_forcing(
        x.forcing_bef, x.forcing_cur, x.forcing_next, x.zref,
        x.time_bef, x.time_cur, x.time_next, calday, interpolation=opt.interpolation,
    )  # fmt: skip
    c = c._replace(**{name + "_forcing": v for name, v in f._asdict().items()})

    # Solar radiation — Fortran SolarRadiation (two-stream; values in the layers in use,
    # padding kept)
    optics = solar_radiation.canopy_optics(
        x.solar_zen, x.dlai, x.dsai, x.dpai, lay,
        params.leaf_reflectance, params.leaf_transmittance,  # [vis, nir]
        params.stem_reflectance, params.stem_transmittance,
        params.leaf_angle_index, params.clumping_index, opt.unroll_loops,
    )  # fmt: skip
    errors = _record(errors, SUNLIT_FRACTION, optics.sunlit_fraction_error)
    c = c._replace(
        kb_profile=_keep(use, optics.kb, c.kb_profile),
        fracsun_profile=_keep(use, optics.fracsun, c.fracsun_profile),
        tb_profile=_keep(use, optics.tb, c.tb_profile),
        td_profile=_keep(use, optics.td, c.td_profile),
        tbi_profile=jnp.concatenate(
            [optics.tbi_ground[None], _keep(use, optics.tbi, c.tbi_profile[1:])]
        ),
    )
    s = solar_radiation.two_stream(
        optics, x.dpai, c.swskyb_forcing, c.swskyd_forcing, x.albsoib, x.albsoid, lay,
        opt.unroll_loops,
    )  # fmt: skip
    errors = _record(errors, SOLAR, s.error)
    upto = jnp.arange(lay.dz.shape[0] + 1) <= lay.n_layers  # levels 0..ncan
    swleaf = _keep(use, s.swleaf, c.swleaf_leaf)
    c = c._replace(
        swbeam_profile=_keep(upto, s.swbeam, c.swbeam_profile),
        swdwn_profile=_keep(upto, s.swdwn, c.swdwn_profile),
        swupw_profile=_keep(upto, s.swupw, c.swupw_profile),
        swleaf_leaf=swleaf,
        swsoi_soil=s.swsoi,
        swveg_canopy=s.swveg,
        swvegsun_canopy=s.swvegsun,
        swvegsha_canopy=s.swvegsha,
        albcan_canopy=s.albcan,
        apar_leaf=_keep(use, solar_radiation.absorbed_par(swleaf), c.apar_leaf),
    )

    # Leaf nitrogen and photosynthetic capacity — Fortran CanopyNitrogenProfile (every slot)
    n = nitrogen_profile.canopy_nitrogen_profile(
        lay, x.dpai, c.kb_profile, c.tbi_profile[1:], c.fracsun_profile, x.lai, x.sai,
        x.tacclim, params.vcmax25, params.clumping_index,
        c3=c3, acclimation=opt.acclimation, kn_val=opt.kn_val, vary_with_height=False,
    )  # fmt: skip
    errors = _record(errors, NITROGEN, n.integration_error)
    c = c._replace(**{k: v for k, v in n._asdict().items() if k != "integration_error"})
    return c._replace(balance_error=errors)


def _stage(c: Substep, x: SubstepInputs, params, tables, opt: SubstepOptions, c3: bool, stage):
    """One Runge-Kutta stage (1-based ``stage``; the last one, after the method's stages,
    only computes the fluxes): the canopy processes, then the state update. ``tables``: the
    psihat lookup tables (``turbulence.RSLTables``).

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 295-328).
    """
    from clm_ml.checks import LEAF_ENERGY, LONGWAVE, SOIL_ENERGY
    from clm_ml.processes import (
        canopy_water,
        flux_profile,
        integrators,
        leaf_boundary_layer,
        longwave,
        photosynthesis,
        plant_hydraulics,
        turbulence,
    )

    lay = x.layers
    use = lay.in_use
    errors = c.balance_error

    # Wetted fraction of the plant area — Fortran CanopyWettedFraction (every slot)
    fwet, fdry = canopy_water.wetted_fraction(c.h2ocan_profile, x.dpai, x.dlai)
    c = c._replace(fwet_profile=fwet, fdry_profile=fdry)

    # Longwave radiation — Fortran LongwaveRadiation (layers and levels up to the canopy top)
    lw = longwave.norman(
        c.tleaf_leaf, c.fracsun_profile, c.td_profile, x.dpai, c.lwsky_forcing, c.tg_soil,
        lay, params.leaf_emissivity, opt.unroll_loops,
    )  # fmt: skip
    errors = _record(errors, LONGWAVE, lw.error)
    upto_top = jnp.arange(lay.dz.shape[0] + 1) <= lay.n_within  # levels 0..ntop
    c = c._replace(
        lwup_canopy=lw.lwup,
        lwveg_canopy=lw.lwveg,
        lwsoi_soil=lw.lwsoi,
        lwleaf_leaf=_keep(upto_top[1:], lw.lwleaf, c.lwleaf_leaf),
        lwupw_profile=jnp.where(upto_top, lw.lwupw, c.lwupw_profile),
        lwdwn_profile=jnp.where(upto_top, lw.lwdwn, c.lwdwn_profile),
    )

    # Net radiation — Fortran lines 301-307 (every slot)
    c = c._replace(
        rnleaf_leaf=c.swleaf_leaf[..., 0] + c.swleaf_leaf[..., 1] + c.lwleaf_leaf,
        rnsoi_soil=c.swsoi_soil[0] + c.swsoi_soil[1] + c.lwsoi_soil,
    )

    # Canopy turbulence — Fortran CanopyTurbulence (profiles in the layers in use)
    t = turbulence.harman_finnigan(
        lay, c.tair_profile, c.eair_profile, c.pref_forcing, x.ztop, x.lai, x.sai, x.zref,
        c.uref_forcing, c.thref_forcing, c.thvref_forcing, c.qref_forcing, c.rhomol_forcing,
        tables,
        sparse_canopy=opt.sparse_canopy, ground_conductance=opt.ground_conductance,
        table_search=opt.table_search, unroll=opt.unroll_loops,
    )  # fmt: skip
    c = c._replace(
        obu_canopy=t.obu,
        beta_canopy=t.beta,
        zdisp_canopy=t.zdisp,
        ustar_canopy=t.ustar,
        z0m_canopy=t.z0m,
        gac0_soil=t.gac0,
        wind_profile=_keep(use, t.wind, c.wind_profile),
        mflx_profile=_keep(use, t.mflx, c.mflx_profile),
        gac_profile=_keep(use, t.gac, c.gac_profile),
        kc_eddy_profile=_keep(use, t.kc_eddy, c.kc_eddy_profile),
    )

    # Leaf boundary layer conductances — Fortran LeafBoundaryLayer (every slot)
    gbh, gbv, gbc = leaf_boundary_layer.leaf_boundary_layer(
        x.dpai, c.wind_profile, c.tair_profile, c.tleaf_leaf, c.pref_forcing, c.tref_forcing,
        c.rhomol_forcing, params.leaf_dimension, regime=opt.regime,
    )  # fmt: skip
    c = c._replace(gbh_leaf=gbh, gbv_leaf=gbv, gbc_leaf=gbc)

    # Photosynthesis and stomatal conductance — Fortran LeafPhotosynthesis (layers in use)
    stomatal = {}
    if opt.stomatal_conductance != "wue":
        g0, g1 = photosynthesis.stomatal_parameters(params, opt.stomatal_conductance)
        stomatal = {"g0": g0, "g1": g1}
    ps = photosynthesis.leaf_photosynthesis(
        lay, x.dpai, c.eair_profile, c.cair_profile, c.tleaf_leaf, c.vcmax25_leaf,
        c.jmax25_leaf, c.rd25_leaf, c.kp25_leaf, c.apar_leaf, c.gbc_leaf, c.gbv_leaf,
        c.lwp_leaf, x.o2ref, c.pref_forcing, x.tacclim,
        iota=params.iota, gs_min=params.gs_min, psi50_gs=params.psi50_gs,
        shape_gs=params.shape_gs, **stomatal, c3=c3,
        stomatal_conductance=opt.stomatal_conductance, water_stress=opt.water_stress,
        colimitation=opt.colimitation, acclimation=opt.acclimation, wue_solver=opt.wue_solver,
    )  # fmt: skip
    c = c._replace(
        gs_leaf=_keep(use, ps.gs, c.gs_leaf),
        anet_leaf=_keep(use, ps.anet, c.anet_leaf),
        agross_leaf=_keep(use, ps.agross, c.agross_leaf),
        btran_soil=jnp.ones_like(c.btran_soil),
    )

    # Flux-profile solution — Fortran FluxProfileSolution (implicit; layers in use)
    fp = flux_profile.implicit(
        flux_profile.Canopy(
            x.dpai, c.fwet_profile, c.fdry_profile, c.fracsun_profile, x.cpleaf,
            c.gac_profile, c.tair_bef_profile, c.eair_bef_profile, c.gbh_leaf, c.gbv_leaf,
            c.gs_leaf, c.rnleaf_leaf, c.tleaf_bef_leaf,
        ),
        flux_profile.Ground(
            c.tg_bef_soil, x.rhg, c.rnsoi_soil, x.soilres, x.soil_t, x.soil_dz, x.soil_tk,
            c.gac0_soil,
        ),
        flux_profile.Reference(
            c.pref_forcing, c.rhomol_forcing, c.cpair_forcing, c.tref_forcing,
            c.thref_forcing, c.eref_forcing,
        ),
        lay,
        opt.dtime,
        opt.unroll_loops,
    )  # fmt: skip
    errors = _record(errors, SOIL_ENERGY, fp.soil.error)
    errors = _record(errors, LEAF_ENERGY, fp.leaf.error)
    # cair: the reference CO2 in layers 0..ncan-2, as the legacy translation set it
    # (Fortran: all layers; no effect while cair is constant, numerics doc §4.5)
    below_top = lay.index < lay.n_layers - 1
    c = c._replace(
        tair_profile=_keep(use, fp.tair, c.tair_profile),
        eair_profile=_keep(use, fp.eair, c.eair_profile),
        shair_profile=_keep(use, fp.shair, c.shair_profile),
        etair_profile=_keep(use, fp.etair, c.etair_profile),
        stair_profile=_keep(use, fp.stair, c.stair_profile),
        **{name + "_leaf": _keep(use, getattr(fp.leaf, name), getattr(c, name + "_leaf"))
           for name in ("tleaf", "stleaf", "shleaf", "lhleaf", "evleaf", "trleaf")},
        shsoi_soil=fp.soil.shsoi,
        lhsoi_soil=fp.soil.lhsoi,
        gsoi_soil=fp.soil.gsoi,
        etsoi_soil=fp.soil.etsoi,
        tg_soil=fp.soil.tg,
        cair_profile=jnp.where(below_top, c.co2ref_forcing, c.cair_profile),
    )  # fmt: skip

    # Leaf water potential — Fortran LeafWaterPotential (every slot)
    c = c._replace(
        lwp_leaf=plant_hydraulics.leaf_water_potential(
            x.dpai, lay.zs, x.lsc, c.trleaf_leaf, c.lwp_bef_leaf, x.psis,
            params.plant_capacitance, opt.dtime,
        )
    )  # fmt: skip

    # Intercepted water — Fortran CanopyInterception, CanopyEvaporation (every slot)
    i = canopy_water.interception(
        x.qflx_rain, x.qflx_snow, x.lai, x.sai, x.dpai, c.h2ocan_bef_profile, opt.dtime
    )
    c = c._replace(
        h2ocan_profile=canopy_water.evaporation(
            i.h2ocan, x.dpai, c.fracsun_profile, c.evleaf_leaf, c.trleaf_leaf, opt.dtime
        ),
        qflx_intr_canopy=i.qflx_intr,
        qflx_tflrain_canopy=i.qflx_tflrain,
        qflx_tflsnow_canopy=i.qflx_tflsnow,
        balance_error=errors,
    )

    # Runge-Kutta update — Fortran RungeKuttaUpdate (no update for Euler, nor after the
    # last stage)
    if opt.integrator == "euler":
        return c
    fields = {"tair": "_profile", "eair": "_profile", "h2ocan": "_profile",
              "tleaf": "_leaf", "lwp": "_leaf", "tg": "_soil"}  # fmt: skip

    def gather(fmt):
        return integrators.Integrated(
            **{n: getattr(c, fmt.format(n=n, s=s)) for n, s in fields.items()}
        )

    new, increment, k = integrators.runge_kutta_update(
        stage, integrators.TABLEAUS[opt.integrator], lay,
        gather("{n}_bef{s}"), gather("{n}{s}"), gather("d{n}{s}"),
    )  # fmt: skip
    updates = {}
    for n, s in fields.items():
        updates[n + s] = getattr(new, n)
        delta = getattr(c, "d" + n + s)
        updates["d" + n + s] = delta.at[..., k].set(getattr(increment, n))
    return c._replace(**updates)


def substep_fluxes(c: Substep, layers: PatchLayers) -> SubstepFluxes:
    """The values of a sub-step that the step averages."""
    names = {
        "ustar": "ustar_canopy", "beta": "beta_canopy", "obu": "obu_canopy",
        "z0m": "z0m_canopy", "zdisp": "zdisp_canopy", "lwup": "lwup_canopy",
        "swsoi": "swsoi_soil", "lwsoi": "lwsoi_soil", "rnsoi": "rnsoi_soil",
        "shsoi": "shsoi_soil", "lhsoi": "lhsoi_soil", "etsoi": "etsoi_soil",
        "gsoi": "gsoi_soil", "gac0": "gac0_soil", "qflx_intr": "qflx_intr_canopy",
        "qflx_tflrain": "qflx_tflrain_canopy", "qflx_tflsnow": "qflx_tflsnow_canopy",
        "swskyb": "swskyb_forcing", "swskyd": "swskyd_forcing", "lwsky": "lwsky_forcing",
    }  # fmt: skip
    values = {}
    for name in SubstepFluxes._fields:
        if name == "lhtop":
            values[name] = canopy_top_latent_heat(layers, c.etair_profile, c.tref_forcing)
        else:
            field = names.get(name) or next(
                f for f in (name + "_profile", name + "_leaf") if f in Substep._fields
            )
            values[name] = getattr(c, field)
    return SubstepFluxes(**values)


def n_stages(integrator: str) -> int:
    """Passes of the canopy processes per sub-step: the method's stages and a last pass
    for the fluxes (Euler: one pass). Fortran: ``nrk_steps + 1``."""
    from clm_ml.processes.integrators import TABLEAUS

    return 1 if integrator == "euler" else len(TABLEAUS[integrator].b) + 1


def substep(
    carry: tuple[Substep, SubstepFluxes],
    calday,
    inputs: SubstepInputs,
    params,
    tables,
    *,
    options: SubstepOptions,
    c3: bool,
) -> tuple[Substep, SubstepFluxes]:
    """One sub-step of the multilayer canopy at calendar day ``calday``: the values at the
    start saved, the forcing interpolated, solar radiation and the nitrogen profile, then
    the canopy processes and the state update for each Runge-Kutta stage (one
    ``lax.scan``), and the sub-step's fluxes added to the sums. ``carry``: the
    :class:`Substep` values and the sums of :class:`SubstepFluxes`. ``tables``: the psihat
    lookup tables (``turbulence.RSLTables``), an input like ``params``.

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 261-330) and
    MLTimeStepFluxIntegration (the sums).
    """
    c, sums = carry
    c = save_start(c, inputs.layers)
    c = _radiation(c, inputs, params, calday, options, c3)
    c, _ = jax.lax.scan(
        lambda c, stage: (_stage(c, inputs, params, tables, options, c3, stage), None),
        c,
        jnp.arange(1, n_stages(options.integrator) + 1),
    )
    return c, accumulate(sums, substep_fluxes(c, inputs.layers))
